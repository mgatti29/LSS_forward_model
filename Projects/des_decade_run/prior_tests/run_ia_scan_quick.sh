#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J ia_scan_q
#SBATCH -o /pscratch/sd/m/mgatti/des_decade_test/logs/ia_scan_quick_%j.log
# timing + validation test of ia_acceptance_scan.py: NGC, one population node, validation at b0 r1
export OMP_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128 PYTHONNOUSERSITE=1
PY=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
/usr/bin/time -v $PY -u /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests/ia_acceptance_scan.py \
    /pscratch/sd/m/mgatti/des_decade_test/Fiducial/1_big 0 1 /pscratch/sd/m/mgatti/des_decade_test/ia_scan/quick_b0_r1 --quick
