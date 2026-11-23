#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J kcmb_Bfast
#SBATCH -o /pscratch/sd/m/mgatti/kcmb_test/run_Bfast_%j.log
ulimit -s unlimited
export OMP_STACKSIZE=1G OMP_NUM_THREADS=128 PYTHONNOUSERSITE=1
cd /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/kcmb_test
/global/cfs/cdirs/m5099/env/pyccl_env/bin/python -X faulthandler -u kcmb_test.py kappa Bfast
/global/cfs/cdirs/m5099/env/pyccl_env/bin/python -u kcmb_test.py stats
