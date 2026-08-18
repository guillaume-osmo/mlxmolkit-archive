#!/bin/zsh
# Replay the frozen benchmark across the commit series. Sequential by design:
# one GPU, so parallel runs would contend and corrupt the timings.
W=/Users/tgg/Github/mlxmolkit-bench
BENCH=/Users/tgg/Github/_mlxmolkit_safety/bench
OUT=$BENCH/results.jsonl

POINTS=(
  "504f5b8:pre-mmff"
  "8ac77e9:pr55-armijo"
  "e95e615:pr56-iters500"
  "296193c:pr61-batch1000"
  "626511e:pr64-gxtb-merge"
  "5112468:pr65-conv-fix"
  "a7082eb:pr66-eigvec-follow"
  "d374f6e:pr67-overlap-norm"
  "966a74c:pr70-grad3x"
  "96ad080:pr71-fock-shape"
  "6107eeb:pr72-kernel-off"
  "275617b:pr73-perpair-grad"
  "0a1f2b3:pr74-head"
)

for pt in $POINTS; do
  sha=${pt%%:*}
  lab=${pt##*:}
  cd $W
  git checkout -q $sha 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "{\"label\":\"$lab\",\"sha\":\"$sha\",\"checkout_failed\":true}" >> $OUT
    continue
  fi
  echo "### $lab ($sha)" >&2
  BENCH_LABEL=$lab \
  BENCH_SHA=$sha \
  PYTHONPATH=$W \
  PYTHONPYCACHEPREFIX=$BENCH/pyc/$sha \
  python $BENCH/run_bench.py 2>/dev/null | grep "^BENCHJSON " | sed 's/^BENCHJSON //' >> $OUT
done
echo "DRIVER DONE" >&2
