#!/bin/bash
export OMP_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128 PYTHONNOUSERSITE=1
PY=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
cd /pscratch/sd/m/mgatti/des_decade_test
$PY -u /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/tests/check_footprint_population.py > logs/check_footprint.log 2>&1
bash /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/tests/run_e2e_gower.sh
