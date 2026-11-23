#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J ia_prior_noise
#SBATCH -o /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests/logs/gower_noise_%j.log
export OMP_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128 PYTHONNOUSERSITE=1
/global/cfs/cdirs/m5099/env/pyccl_env/bin/python -u /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests/gower_ia_noise.py
