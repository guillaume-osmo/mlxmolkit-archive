#!/usr/bin/env python3
"""Search and audit every Claude Code session, across every project.

The transcripts under `~/.claude/projects/` are the record of what was
actually decided, measured and rejected — 988 MB across 148 files and seven
projects at the time of writing. Nothing could read them but `grep`, which
fails on them for a specific reason: each line is a JSON object whose text is
buried in `message.content` as either a string or a list of blocks, with
newlines escaped. A regex over the raw lines matches across field boundaries
and misses text that is split across blocks, so it returns confident nonsense
or, more often, nothing.

This builds an SQLite FTS5 index (stdlib, no dependencies) over the extracted
text and gives three commands:

    index    — build or refresh; only re-reads files whose mtime changed
    search   — ranked full-text search, with project/date/role filters
    threads  — what is running where: recent sessions per project, so a
               parallel thread in another checkout is visible from here

Examples:

    python tools/session_search.py index
    python tools/session_search.py search "sigma profile 15k dataset"
    python tools/session_search.py search "cosmo-sac" --project osmo --since 2026-06-01
    python tools/session_search.py search "odieu out of domain" --role user
    python tools/session_search.py threads --days 7
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"
INDEX = Path.home() / ".claude" / "session_index.sqlite"

# Extra transcript roots, colon-separated, for a second account or a second
# machine whose ~/.claude/projects has been synced here. Both accounts are the
# same person; only the store is split, so they belong in one index. Each root
# is tagged so a hit says which account it came from.
#
#   export CLAUDE_SESSION_ROOTS="/Volumes/other-mac/.claude/projects:personal"
EXTRA_ROOTS_ENV = "CLAUDE_SESSION_ROOTS"


def transcript_roots() -> list[tuple[Path, str]]:
    """(directory, account tag) for every transcript store to index."""
    roots = [(PROJECTS, "")]
    for entry in os.environ.get(EXTRA_ROOTS_ENV, "").split(":"):
        entry = entry.strip()
        if not entry:
            continue
        # "path" or "path:tag" — but ':' is also the separator, so a tag is
        # attached with '=' to keep the split unambiguous.
        path, _, tag = entry.partition("=")
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            roots.append((candidate, tag or candidate.parent.name))
        else:
            print(f"  warning: {EXTRA_ROOTS_ENV} entry not a directory: {path}",
                  file=sys.stderr)
    return roots

# Tool results are the bulk of the bytes and almost none of the meaning — they
# are file dumps, test output and directory listings that drown any search.
# Prompts and assistant prose are what carry decisions.
SKIP_BLOCK_TYPES = {"tool_result", "image", "thinking"}


def project_label(path: Path) -> str:
    """`-Users-guillaume-osmo-Github-osmo` -> `osmo`."""
    name = path.name.lstrip("-").replace("-Users-guillaume-osmo-", "")
    return name.replace("Github-", "").replace("-", "/") or name


def extract_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in SKIP_BLOCK_TYPES:
            continue
        if kind == "text" and block.get("text"):
            parts.append(block["text"])
        elif kind == "tool_use":
            # The command or query itself is meaningful; its output is not.
            inp = block.get("input") or {}
            for field in ("command", "query", "prompt", "file_path", "pattern"):
                if isinstance(inp.get(field), str):
                    parts.append(f"[{block.get('name')}] {inp[field][:400]}")
                    break
    return "\n".join(parts)


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(INDEX)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY, mtime REAL, size INTEGER, n_rows INTEGER,
            -- Bytes already consumed. Transcripts are append-only while a
            -- session is live, so the file being written to right now is
            -- exactly the one that would otherwise be re-parsed in full on
            -- every refresh. Seeking to this offset makes a refresh cost the
            -- new lines only.
            offset INTEGER DEFAULT 0
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(
            text, project UNINDEXED, session UNINDEXED, role UNINDEXED,
            ts UNINDEXED, path UNINDEXED, tokenize='porter unicode61'
        );
        -- Trigram index for fuzzy recall. FTS5's porter tokenizer matches
        -- whole stemmed words, so "cosmosac" never finds "COSMO-SAC" and a
        -- typo finds nothing at all. Trigram matches on 3-character runs, so
        -- substrings and misspellings still hit. It is not semantic — no
        -- tokenizer is — but it covers the "I half-remember the word" case
        -- that lexical search otherwise fails outright.
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fuzzy USING fts5(
            text, session UNINDEXED, tokenize='trigram'
        );
        """
    )
    return con


def index_all(con: sqlite3.Connection, verbose: bool = True) -> None:
    roots = transcript_roots()
    files = [(f, tag) for root, tag in roots for f in sorted(root.glob("*/*.jsonl"))]
    if verbose and len(roots) > 1:
        print(f"indexing {len(roots)} transcript roots: "
              + ", ".join(str(r) + (f" [{t}]" if t else "") for r, t in roots))
    seen = {row[0]: (row[1], row[2], row[3], row[4]) for row in
            con.execute("SELECT path, mtime, size, offset, n_rows FROM files")}
    added = skipped = 0
    t0 = time.perf_counter()

    for path, account in files:
        stat = path.stat()
        known = seen.get(str(path))
        if known and abs(known[0] - stat.st_mtime) < 1e-6 and known[1] == stat.st_size:
            skipped += 1
            continue

        # Append-only growth: keep what is indexed and read from the offset.
        # Anything else — a shrunk file, a rewritten one, a row with no
        # recorded offset — falls back to a full reparse, because a wrong
        # offset silently indexes from the middle of a line.
        start = 0
        if known and known[2] and stat.st_size > known[2]:
            start = known[2]
        else:
            con.execute("DELETE FROM messages WHERE path = ?", (str(path),))
            con.execute("DELETE FROM messages_fuzzy WHERE session = ?", (path.stem,))

        project = project_label(path.parent)
        if account:
            project = f"{account}:{project}"
        session = path.stem
        rows = []
        with path.open(errors="replace") as fh:
            if start:
                fh.seek(start)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = rec.get("message")
                if not isinstance(message, dict):
                    continue
                text = extract_text(message)
                if len(text) < 12:
                    continue
                rows.append((text, project, session, message.get("role") or "?",
                             rec.get("timestamp") or "", str(path)))
        con.executemany(
            "INSERT INTO messages (text, project, session, role, ts, path) "
            "VALUES (?,?,?,?,?,?)", rows)
        # Trigram indexes are ~3x the size of a word index, so only the user's
        # own prompts go in: that is what a half-remembered phrase came from.
        con.executemany(
            "INSERT INTO messages_fuzzy (text, session) VALUES (?,?)",
            [(r[0][:2000], r[2]) for r in rows if r[3] == "user"])

        # `tell()` on an iterating text file is unreliable, so the offset that
        # gets stored is the size that was just fully consumed.
        end = stat.st_size
        # known = (mtime, size, offset, n_rows); keep the prior count only
        # when this was an incremental read, since a full reparse replaced it.
        prior = (known[3] if known and start else 0) or 0
        con.execute(
            "INSERT OR REPLACE INTO files (path, mtime, size, n_rows, offset) "
            "VALUES (?,?,?,?,?)",
            (str(path), stat.st_mtime, stat.st_size, prior + len(rows), end))
        added += 1
        if verbose:
            how = f"+{len(rows)} new" if start else f"{len(rows)} messages"
            print(f"  indexed {project}/{session[:8]}  {how}")

    con.commit()
    total = con.execute("SELECT count(*) FROM messages").fetchone()[0]
    if verbose:
        print(f"\n{added} file(s) indexed, {skipped} unchanged, "
              f"{total} messages total, {time.perf_counter()-t0:.1f}s")


def snippet(text: str, terms: list[str], width: int = 260) -> str:
    flat = " ".join(text.split())
    low = flat.lower()
    pos = min((low.find(t.lower()) for t in terms if t.lower() in low), default=-1)
    if pos < 0:
        return flat[:width]
    start = max(0, pos - width // 3)
    out = flat[start:start + width]
    return ("…" if start else "") + out + ("…" if start + width < len(flat) else "")


def cmd_search(args) -> None:
    con = connect()
    where, params = ["messages MATCH ?"], [args.query]
    if args.project:
        where.append("project LIKE ?")
        params.append(f"%{args.project}%")
    if args.role:
        where.append("role = ?")
        params.append(args.role)
    if args.since:
        where.append("ts >= ?")
        params.append(args.since)
    if args.until:
        where.append("ts <= ?")
        params.append(args.until)

    # bm25 alone keeps surfacing three-month-old reasoning that a later
    # session already overturned. Recency is evidence: a more recent statement
    # about the same thing is usually the corrected one. rank is negative and
    # more-negative is better, so age is *added* as a penalty.
    if args.hybrid:
        hits, n_lex, n_dense = hybrid_search(con, args.query, args.limit,
                                             args.recency)
        print(f"{len(hits)} hybrid match(es) for {args.query!r}  "
              f"(bm25 {n_lex}, cosine {n_dense})\n")
        for h in hits:
            text, project, session, role, ts = h["row"]
            print(f"\033[1m{(ts or '')[:16].replace('T',' ')}  {project}  "
                  f"[{role}]\033[0m  {session[:8]}  "
                  f"rrf {h['rrf']:.4f}  {'+'.join(h['via'])}")
            print(f"    {' '.join(text.split())[:250]}\n")
        return

    if args.semantic:
        hits = semantic_search(con, args.query, args.limit)
        print(f"{len(hits)} semantic match(es) for {args.query!r}\n")
        for text, project, session, role, ts, score in hits:
            print(f"\033[1m{(ts or '')[:16].replace('T',' ')}  {project}  "
                  f"[{role}]\033[0m  {session[:8]}  cos {score:.3f}")
            print(f"    {' '.join(text.split())[:260]}\n")
        return

    if args.fuzzy:
        hits = con.execute(
            "SELECT f.text, m.project, f.session, 'user', max(m.ts) "
            "FROM messages_fuzzy f JOIN messages m ON m.session = f.session "
            "WHERE f.text LIKE ? GROUP BY f.rowid ORDER BY max(m.ts) DESC LIMIT ?",
            (f"%{args.query}%", args.limit)).fetchall()
        if not hits:
            print(f"no fuzzy match for {args.query!r}")
            return
        terms = [t for t in re.findall(r"\w+", args.query) if len(t) > 2]
        print(f"{len(hits)} fuzzy match(es) for {args.query!r}\n")
        for text, project, session, role, ts in hits:
            print(f"\033[1m{(ts or '')[:16].replace('T',' ')}  {project}  [{role}]\033[0m  {session[:8]}")
            print(f"    {snippet(text, terms or [args.query])}\n")
        return

    sql = (f"SELECT text, project, session, role, ts, "
           f"  rank + ? * (julianday('now') - julianday(substr(ts,1,19))) AS score "
           f"FROM messages WHERE {' AND '.join(where)} "
           f"AND ts != '' ORDER BY score LIMIT ?")
    # the recency '?' sits in the SELECT, so it binds before the WHERE params
    params = [args.recency] + params + [args.limit]
    try:
        rows = con.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"query error: {exc}\n(FTS5 syntax: quote phrases, "
                         f"use OR / NOT / NEAR, prefix with *)")

    if not rows:
        print("no matches — try fewer terms, or `index` if transcripts are new")
        return
    terms = [t for t in re.findall(r"\w+", args.query) if len(t) > 2]
    print(f"{len(rows)} match(es) for {args.query!r}\n")
    for text, project, session, role, ts, _score in rows:
        stamp = (ts or "")[:16].replace("T", " ")
        print(f"\033[1m{stamp}  {project}  [{role}]\033[0m  {session[:8]}")
        print(f"    {snippet(text, terms)}\n")


def cmd_threads(args) -> None:
    """What is active where — the parallel threads running in other checkouts."""
    con = connect()
    cutoff = time.time() - args.days * 86400
    rows = con.execute(
        "SELECT project, session, count(*), min(ts), max(ts) FROM messages "
        "GROUP BY project, session HAVING max(ts) != '' ORDER BY max(ts) DESC"
    ).fetchall()

    shown = 0
    print(f"sessions active in the last {args.days} day(s)\n")
    for project, session, n, first, last in rows:
        try:
            if time.mktime(time.strptime(last[:19], "%Y-%m-%dT%H:%M:%S")) < cutoff:
                continue
        except (ValueError, TypeError):
            continue
        opening = con.execute(
            "SELECT text FROM messages WHERE session = ? AND role = 'user' "
            "ORDER BY ts LIMIT 1", (session,)).fetchone()
        gist = " ".join((opening[0] if opening else "").split())[:150]
        print(f"\033[1m{project:22s}\033[0m {session[:8]}  {n:5d} msgs  "
              f"{first[:16].replace('T',' ')} → {last[:16].replace('T',' ')}")
        print(f"    {gist}\n")
        shown += 1
        if shown >= args.limit:
            break
    if not shown:
        print("  (none — widen with --days)")



# Signals that tie two sessions to the same piece of work. File paths and
# branch names are far more discriminating than words: two sessions touching
# `mlxmolkit/cosmo/sigma.py` are related whatever vocabulary they used.
_PATH_RE = re.compile(r"(?:/Users/[\w.-]+/)?(?:[\w.-]+/){1,6}[\w.-]+\.(?:py|md|toml|json|csv|cpp|metal)")
_BRANCH_RE = re.compile(r"\b(?:guillaume|wip|feature|fix)/[\w.-]+")
_ISSUE_RE = re.compile(r"(?:#|PR |issue )(\d{2,5})\b")


def session_signals(con, session: str) -> set[str]:
    rows = con.execute("SELECT text FROM messages WHERE session = ?", (session,))
    sig: set[str] = set()
    for (text,) in rows:
        for m in _PATH_RE.findall(text)[:40]:
            sig.add("f:" + m.rsplit("/", 2)[-1] if "/" in m else "f:" + m)
        for m in _BRANCH_RE.findall(text)[:20]:
            sig.add("b:" + m)
        for m in _ISSUE_RE.findall(text)[:20]:
            sig.add("#" + m)
    return sig


def cmd_related(args) -> None:
    """Sessions connected to a starting point, and what connects them."""
    con = connect()
    if len(args.seed) >= 8 and all(c in "0123456789abcdef-" for c in args.seed):
        seed = con.execute("SELECT DISTINCT session FROM messages WHERE session LIKE ?",
                           (args.seed + "%",)).fetchone()
        if not seed:
            raise SystemExit(f"no session starting {args.seed!r}")
        seed = seed[0]
    else:
        hit = con.execute("SELECT session FROM messages WHERE messages MATCH ? "
                          "ORDER BY rank LIMIT 1", (args.seed,)).fetchone()
        if not hit:
            raise SystemExit(f"nothing matches {args.seed!r}")
        seed = hit[0]

    base = session_signals(con, seed)
    if not base:
        raise SystemExit("seed session has no file/branch/issue signals to link on")

    sessions = [r[0] for r in con.execute(
        "SELECT DISTINCT session FROM messages WHERE ts != ''")]
    scored = []
    for other in sessions:
        if other == seed:
            continue
        sig = session_signals(con, other)
        shared = base & sig
        if len(shared) >= args.min_shared:
            jaccard = len(shared) / len(base | sig)
            scored.append((jaccard, len(shared), other, shared))
    scored.sort(reverse=True)

    meta = dict(con.execute(
        "SELECT session, project || '|' || max(ts) FROM messages GROUP BY session"))
    print(f"seed {seed[:8]}  ({len(base)} signals)\n")
    for jac, n, other, shared in scored[:args.limit]:
        project, last = (meta.get(other, "?|?").split("|") + ["?"])[:2]
        gist = con.execute("SELECT text FROM messages WHERE session=? AND role='user' "
                           "ORDER BY ts LIMIT 1", (other,)).fetchone()
        gist = " ".join((gist[0] if gist else "").split())[:90]
        print(f"\033[1m{other[:8]}\033[0m  {project:26s} {last[:10]}  "
              f"jaccard {jac:.2f}  {n} shared")
        print(f"    via: {', '.join(sorted(shared)[:6])}")
        print(f"    {gist}\n")


def cmd_recall(args) -> None:
    """Compact prior context for a prompt — designed for a hook, not a human.

    Prints nothing when nothing scores well. A hook that always injects
    something trains the reader to ignore it, and burns context on every turn
    for the many prompts that need no history at all.
    """
    con = connect()
    terms = [t for t in re.findall(r"[A-Za-z][\w-]{3,}", args.prompt)][:12]
    if len(terms) < 2:
        return
    # Recall runs on every prompt, so it must stay in the tens of
    # milliseconds. Dense retrieval cannot: `import sentence_transformers` is
    # 2.6 s and loading the model another 2.9 s, paid per invocation because a
    # hook is a fresh subprocess — 7 s wall against 0.4 s of actual encoding.
    # So the default fuses the two cheap retrievers, bm25 and trigram, which
    # cost milliseconds and disagree usefully: bm25 ranks by term weight,
    # trigram survives the misspelling or the punctuation. `--semantic` opts
    # into the dense leg for interactive use, where 7 s is affordable.
    query = " OR ".join(f'"{t}"' for t in terms)
    try:
        lex = con.execute(
            "SELECT text, project, session, ts, "
            "  rank + 0.02 * (julianday('now') - julianday(substr(ts,1,19))) AS score "
            "FROM messages WHERE messages MATCH ? AND ts != '' "
            "ORDER BY score LIMIT ?", (query, args.limit * 6)).fetchall()
    except sqlite3.OperationalError:
        return

    fuzz = con.execute(
        "SELECT f.text, m.project, f.session, max(m.ts), 0 "
        "FROM messages_fuzzy f JOIN messages m ON m.session = f.session "
        "WHERE " + " OR ".join(["f.text LIKE ?"] * len(terms[:4])) +
        " GROUP BY f.rowid ORDER BY max(m.ts) DESC LIMIT ?",
        [f"%{t}%" for t in terms[:4]] + [args.limit * 3]).fetchall() if terms else []

    if args.semantic:
        try:
            fused, _, _ = hybrid_search(con, args.prompt, args.limit * 6, 0.02)
            lex = [(f["row"][0], f["row"][1], f["row"][2], f["row"][4], 0)
                   for f in fused]
        except (sqlite3.OperationalError, SystemExit):
            pass

    seen_keys, rows = set(), []
    for rank, row in enumerate(list(lex) + list(fuzz)):
        key = (row[2], row[3])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)

    # Require several query terms in the hit, or every prompt drags in
    # something loosely related.
    keep = []
    for text, project, session, ts, _ in rows:
        # A tool invocation that merely *searched* for a term is not knowledge
        # about it. Without this, recall for any topic returns the greps run
        # while looking for the answer instead of the answer.
        if text.lstrip().startswith("["):
            continue
        low = text.lower()
        overlap = sum(1 for t in terms if t.lower() in low)
        if overlap >= max(2, len(terms) // 3):
            keep.append((text, project, session, ts, overlap))
    if not keep:
        return
    print("Relevant prior sessions (auto-recall):")
    for text, project, session, ts, _ in keep[:args.limit]:
        print(f"- [{(ts or '')[:10]} {project} {session[:8]}] {snippet(text, terms, 200)}")


def all_signals(con, refresh: bool = False) -> dict[str, set[str]]:
    """Signals per session, cached — recomputing costs a full table scan."""
    con.execute("CREATE TABLE IF NOT EXISTS signals "
                "(session TEXT PRIMARY KEY, sig TEXT, last_ts TEXT)")
    known = {r[0]: (r[1], r[2]) for r in
             con.execute("SELECT session, sig, last_ts FROM signals")}
    live = dict(con.execute(
        "SELECT session, max(ts) FROM messages WHERE ts != '' GROUP BY session"))
    out, dirty = {}, []
    for session, last in live.items():
        cached = known.get(session)
        if cached and cached[1] == last and not refresh:
            out[session] = set(filter(None, cached[0].split("\x1f")))
            continue
        sig = session_signals(con, session)
        out[session] = sig
        dirty.append((session, "\x1f".join(sorted(sig)), last))
    if dirty:
        con.executemany("INSERT OR REPLACE INTO signals VALUES (?,?,?)", dirty)
        con.commit()
    return out


def cmd_goals(args) -> None:
    """Cluster sessions into goals.

    A session is a unit of *time*; a goal is a unit of *work*, and the two do
    not line up. One goal runs across several sessions, several days and often
    several repositories — the mlxmolkit work here reaches into osmo, and the
    f2b thread spans three checkouts. Grouping by project directory therefore
    splits goals apart and merges unrelated ones, because the directory is
    just where the shell happened to be.

    So the grouping is the artifact graph: sessions sharing files, branches or
    issue numbers are joined, and connected components are the goals. Edges
    are thresholded on Jaccard so one shared README does not fuse everything.
    """
    con = connect()
    sigs = {k: v for k, v in all_signals(con).items() if len(v) >= args.min_signals}

    # Drop hub signals. `README.md`, `setup.py` and a handful of issue numbers
    # appear in most sessions, and a shared README says nothing about two
    # sessions being the same work. Left in, they fused 75 unrelated sessions
    # spanning two months into a single "goal". Anything in more than
    # `--max-df` of sessions carries no information and is removed.
    from collections import Counter
    df = Counter(sig for s_ in sigs.values() for sig in s_)
    cutoff = max(2, int(args.max_df * len(sigs)))
    hubs = {sig for sig, n in df.items() if n > cutoff}
    sigs = {k: (v - hubs) for k, v in sigs.items()}
    sigs = {k: v for k, v in sigs.items() if len(v) >= args.min_signals}
    names = sorted(sigs)
    if args.verbose and hubs:
        print(f"  dropped {len(hubs)} hub signal(s) seen in >{cutoff} sessions: "
              f"{', '.join(sorted(hubs)[:8])}\n")

    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = sigs[a], sigs[b]
            shared = sa & sb
            if not shared:
                continue
            if len(shared) / len(sa | sb) >= args.threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
                edges += 1

    clusters: dict[str, list[str]] = {}
    for n in names:
        clusters.setdefault(find(n), []).append(n)

    meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute(
        "SELECT session, min(ts), max(ts), count(*) FROM messages "
        "WHERE ts != '' GROUP BY session")}
    projects = {r[0]: r[1] for r in con.execute(
        "SELECT session, project FROM messages GROUP BY session")}

    ranked = sorted(clusters.values(),
                    key=lambda c: max(meta[s][1] for s in c), reverse=True)
    print(f"{len(ranked)} goal(s) from {len(names)} sessions, "
          f"{edges} edges at Jaccard >= {args.threshold}\n")

    for group in ranked[:args.limit]:
        group = sorted(group, key=lambda s: meta[s][1])
        first, last = meta[group[0]][0], max(meta[s][1] for s in group)
        repos = sorted({projects[s].split("/")[-1] or projects[s] for s in group})
        msgs = sum(meta[s][2] for s in group)
        opening = con.execute(
            "SELECT text FROM messages WHERE session=? AND role='user' "
            "ORDER BY ts LIMIT 1", (group[0],)).fetchone()
        gist = " ".join((opening[0] if opening else "").split())[:130]

        shared = set.intersection(*(sigs[s] for s in group)) if len(group) > 1 else sigs[group[0]]
        print(f"\033[1m{first[:10]} → {last[:10]}\033[0m  "
              f"{len(group)} session(s), {msgs} msgs")
        print(f"    repos: {', '.join(repos[:4])}")
        if shared:
            print(f"    shared: {', '.join(sorted(shared)[:6])}")
        print(f"    {gist}")
        print(f"    ids: {' '.join(s[:8] for s in group[:8])}\n")


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device), device


def cmd_embed(args) -> None:
    """Dense vectors for every message, so recall survives different wording.

    FTS5 and trigram are both lexical: they need the words, or at least the
    characters. "the dataset that was too big for perfumery" finds nothing
    lexically even though the answer exists, because the answer says
    "applicability domain" and "MW > 300 Da".

    Stored as float16 in SQLite — 57,880 x 384 x 2 B is ~44 MB, and the
    precision loss is far below the noise in a cosine ranking. Only user
    prompts and assistant prose are embedded; tool invocations are excluded
    for the same reason `recall` skips them.
    """
    import numpy as np

    con = connect()
    con.execute("CREATE TABLE IF NOT EXISTS embeddings ("
                "rowid_ INTEGER PRIMARY KEY, vec BLOB)")
    have = {r[0] for r in con.execute("SELECT rowid_ FROM embeddings")}
    rows = [(r[0], r[1]) for r in con.execute(
        "SELECT rowid, text FROM messages WHERE ts != ''")
        if r[0] not in have and not r[1].lstrip().startswith("[")]
    if not rows:
        print("embeddings up to date")
        return

    model, device = _load_embedder()
    print(f"embedding {len(rows)} message(s) on {device}…")
    t0 = time.perf_counter()
    B = 512
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        vecs = model.encode([t[:1200] for _, t in chunk],
                            batch_size=128, normalize_embeddings=True,
                            show_progress_bar=False)
        con.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?)",
                        [(rid, np.asarray(v, dtype=np.float16).tobytes())
                         for (rid, _), v in zip(chunk, vecs)])
        con.commit()
        done = min(i + B, len(rows))
        print(f"  {done}/{len(rows)}  {done/(time.perf_counter()-t0):.0f}/s", end="\r")
    print(f"\n{len(rows)} embedded in {time.perf_counter()-t0:.0f}s")


def semantic_search(con, query: str, limit: int):
    import numpy as np

    rows = con.execute(
        "SELECT e.rowid_, e.vec, m.text, m.project, m.session, m.role, m.ts "
        "FROM embeddings e JOIN messages m ON m.rowid = e.rowid_").fetchall()
    if not rows:
        raise SystemExit("no embeddings — run `embed` first")
    mat = np.frombuffer(b"".join(r[1] for r in rows),
                        dtype=np.float16).reshape(len(rows), EMBED_DIM)
    model, _ = _load_embedder()
    q = np.asarray(model.encode([query], normalize_embeddings=True)[0],
                   dtype=np.float32)
    scores = mat.astype(np.float32) @ q
    top = np.argpartition(-scores, min(limit, len(scores) - 1))[:limit]
    top = top[np.argsort(-scores[top])]
    return [(rows[i][2], rows[i][3], rows[i][4], rows[i][5], rows[i][6],
             float(scores[i])) for i in top]


def hybrid_search(con, query: str, limit: int, recency: float, k: int = 60):
    """Reciprocal-rank fusion of lexical and semantic retrieval.

    The two fail in opposite directions and neither dominates:

      * BM25 needs the words. "the dataset molecules were too big to smell"
        returns nothing, though the answer is there under "applicability
        domain" and "MW > 300 Da".
      * Cosine needs no words but has no notion of an exact token, so a
        specific identifier — a branch name, `MF_ALPHA_PM6`, issue #4268 —
        ranks no better than a paraphrase of it.

    RRF merges the two *rankings* rather than their scores, which is the point:
    a bm25 rank and a cosine similarity are not on a common scale, and
    normalising them is a tuning exercise that has to be redone per corpus.
    Rank position needs no calibration. Each list contributes 1/(k + rank),
    k = 60 as in Cormack et al.
    """
    lex_sql = ("SELECT text, project, session, role, ts, "
               "  rank + ? * (julianday('now') - julianday(substr(ts,1,19))) AS score "
               "FROM messages WHERE messages MATCH ? AND ts != '' "
               "ORDER BY score LIMIT ?")
    try:
        lexical = con.execute(lex_sql, (recency, query, limit * 4)).fetchall()
    except sqlite3.OperationalError:
        # Unparseable as FTS5 (bare punctuation, stray quotes) — semantic alone
        # still answers, which is better than failing the whole query.
        lexical = []

    try:
        dense = semantic_search(con, query, limit * 4)
    except SystemExit:
        dense = []

    fused: dict[tuple, dict] = {}
    for rank, row in enumerate(lexical):
        key = (row[2], row[4])
        fused.setdefault(key, {"row": row[:5], "rrf": 0.0, "via": []})
        fused[key]["rrf"] += 1.0 / (k + rank)
        fused[key]["via"].append(f"bm25#{rank+1}")
    for rank, row in enumerate(dense):
        key = (row[2], row[4])
        fused.setdefault(key, {"row": row[:5], "rrf": 0.0, "via": []})
        fused[key]["rrf"] += 1.0 / (k + rank)
        fused[key]["via"].append(f"cos#{rank+1}")

    out = sorted(fused.values(), key=lambda d: -d["rrf"])[:limit]
    return out, len(lexical), len(dense)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="build or refresh the index")

    s = sub.add_parser("search", help="ranked full-text search")
    s.add_argument("query")
    s.add_argument("--project", help="substring of the project name, e.g. osmo")
    s.add_argument("--role", choices=["user", "assistant"])
    s.add_argument("--since", help="ISO date, e.g. 2026-06-01")
    s.add_argument("--until")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--hybrid", action="store_true",
                   help="reciprocal-rank fusion of bm25 and cosine — use this "
                        "unless you want to see one method alone")
    s.add_argument("--semantic", action="store_true",
                   help="dense-vector search — finds by meaning, needs `embed`")
    s.add_argument("--fuzzy", action="store_true",
                   help="trigram substring match — survives typos and "
                        "punctuation (cosmosac finds COSMO-SAC)")
    s.add_argument("--recency", type=float, default=0.02,
                   help="penalty per day of age; 0 disables the tilt "
                        "(default 0.02, ~1 bm25 point per 50 days)")
    s.set_defaults(func=cmd_search)

    t = sub.add_parser("threads", help="recent sessions per project")
    t.add_argument("--days", type=int, default=7)
    t.add_argument("--limit", type=int, default=25)
    t.set_defaults(func=cmd_threads)

    r = sub.add_parser("related", help="sessions linked to a seed by shared files/branches/issues")
    r.add_argument("seed", help="session id prefix, or a search query to find one")
    r.add_argument("--min-shared", type=int, default=2)
    r.add_argument("--limit", type=int, default=10)
    r.set_defaults(func=cmd_related)

    rc = sub.add_parser("recall", help="compact prior context for a prompt (for hooks)")
    rc.add_argument("prompt")
    rc.add_argument("--limit", type=int, default=3)
    rc.add_argument("--semantic", action="store_true",
                    help="add the dense leg — accurate but ~7s per call, so "
                         "not for a per-prompt hook")
    rc.set_defaults(func=cmd_recall)

    g = sub.add_parser("goals", help="cluster sessions into units of work, across repos")
    g.add_argument("--threshold", type=float, default=0.10,
                   help="Jaccard needed to join two sessions (default 0.10)")
    g.add_argument("--min-signals", type=int, default=4)
    g.add_argument("--max-df", type=float, default=0.15,
                   help="drop signals appearing in more than this fraction of "
                        "sessions; they are hubs and carry no information")
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--limit", type=int, default=12)
    g.set_defaults(func=cmd_goals)

    e = sub.add_parser("embed", help="build dense vectors for semantic search")
    e.set_defaults(func=cmd_embed)

    args = ap.parse_args()
    if args.cmd == "index":
        index_all(connect())
    else:
        args.func(args)


if __name__ == "__main__":
    main()
