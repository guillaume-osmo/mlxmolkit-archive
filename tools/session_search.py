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
            path TEXT PRIMARY KEY, mtime REAL, size INTEGER, n_rows INTEGER
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(
            text, project UNINDEXED, session UNINDEXED, role UNINDEXED,
            ts UNINDEXED, path UNINDEXED, tokenize='porter unicode61'
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
    seen = {row[0]: (row[1], row[2]) for row in
            con.execute("SELECT path, mtime, size FROM files")}
    added = skipped = 0
    t0 = time.perf_counter()

    for path, account in files:
        stat = path.stat()
        known = seen.get(str(path))
        if known and abs(known[0] - stat.st_mtime) < 1e-6 and known[1] == stat.st_size:
            skipped += 1
            continue

        con.execute("DELETE FROM messages WHERE path = ?", (str(path),))
        project = project_label(path.parent)
        if account:
            project = f"{account}:{project}"
        session = path.stem
        rows = []
        with path.open(errors="replace") as fh:
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
        con.execute(
            "INSERT OR REPLACE INTO files (path, mtime, size, n_rows) VALUES (?,?,?,?)",
            (str(path), stat.st_mtime, stat.st_size, len(rows)))
        added += 1
        if verbose:
            print(f"  indexed {project}/{session[:8]}  {len(rows)} messages")

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

    args = ap.parse_args()
    if args.cmd == "index":
        index_all(connect())
    else:
        args.func(args)


if __name__ == "__main__":
    main()
