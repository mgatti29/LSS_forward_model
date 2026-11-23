#!/bin/bash
export OMP_NUM_THREADS=128 PYTHONNOUSERSITE=1
PY=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
cd /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/kcmb_test

/usr/bin/time -f 'wall %e s maxrss %M kB' $PY -u test_make_kappa.py
