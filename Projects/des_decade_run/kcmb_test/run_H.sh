#!/bin/bash
ulimit -s unlimited
export OMP_NUM_THREADS=128 NUMEXPR_MAX_THREADS=64 PYTHONNOUSERSITE=1
PY=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
cd /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/kcmb_test
for v in Lonly L Honly H; do /usr/bin/time -f "$v wall %e s maxrss %M kB" $PY -u kcmb_test.py kappa $v; done
$PY -u kcmb_test.py stats
$PY -u profile_ladder.py
