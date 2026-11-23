#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J ia_over_D
#SBATCH -o /pscratch/sd/m/mgatti/des_decade_test/logs/test_ia_over_D_%j.log
export OMP_NUM_THREADS=128 PYTHONNOUSERSITE=1
/global/cfs/cdirs/m5099/env/pyccl_env/bin/python -u /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/tests/test_ia_over_D.py bfd
