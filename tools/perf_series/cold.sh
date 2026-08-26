#!/bin/zsh
# Cold-start: one call per fresh interpreter, repeated from outside.
W=/Users/tgg/Github/mlxmolkit-bench
BENCH=/Users/tgg/Github/_mlxmolkit_safety/bench
OUT=$BENCH/cold.jsonl

POINTS=("504f5b8:pre-mmff" "966a74c:pr70-grad3x" "96ad080:pr71-fock-shape" "275617b:pr73-perpair-grad" "0a1f2b3:pr74-head")

for pt in $POINTS; do
  sha=${pt%%:*}
  lab=${pt##*:}
  cd $W
  git checkout -q $sha 2>/dev/null
  for what in scf grad; do
    for i in 1 2 3; do
      BENCH_LABEL=$lab COLD_WHAT=$what COLD_MOL=cholesterol \
      PYTHONPATH=$W PYTHONPYCACHEPREFIX=$BENCH/pyc/$sha \
      python $BENCH/cold.py 2>/dev/null >> $OUT
    done
  done
  echo "### cold $lab done" >&2
done
echo "COLD DONE" >&2
