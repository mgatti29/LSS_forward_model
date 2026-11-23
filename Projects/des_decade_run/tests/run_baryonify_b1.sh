#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J dd_bary1
#SBATCH -o /pscratch/sd/m/mgatti/des_decade_test/logs/baryonify_b1_%j.log
export OMP_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128 PYTHONNOUSERSITE=1
/global/cfs/cdirs/m5099/env/pyccl_env/bin/python -u /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/tests/baryonify_draw.py /pscratch/sd/m/mgatti/des_decade_test/Fiducial/1_big 1
