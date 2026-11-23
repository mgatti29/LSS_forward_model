#!/bin/bash
ulimit -s unlimited
export OMP_NUM_THREADS=128 PYTHONNOUSERSITE=1
PY=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
cd /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/kcmb_test
/usr/bin/time -v $PY -u kcmb_test.py kappa Lonly 2>&1 | grep -vE "^\s+(Average|Socket|Major|Minor|Volun|Invol|Swaps|File|Signals|Page|Exit|Percent|System|User|Command)"
/usr/bin/time -v $PY -u kcmb_test.py kappa L 2>&1 | grep -vE "^\s+(Average|Socket|Major|Minor|Volun|Invol|Swaps|File|Signals|Page|Exit|Percent|System|User|Command)"
$PY -u kcmb_test.py stats
$PY -u replication_test.py Lonly
