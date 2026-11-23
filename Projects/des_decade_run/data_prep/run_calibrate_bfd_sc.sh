#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J bfd_sc_calib
#SBATCH -o /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/data_prep/logs/bfd_sc_calib_%j.log
# SC v2 noise calibration for BFD: the four tomographic bins in parallel, then the polynomial fit
export OMP_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64 PYTHONNOUSERSITE=1
P=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
D=/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/data_prep
for t in 0 1 2 3; do $P -u $D/calibrate_bfd_sc.py bin $t > $D/logs/bfd_sc_bin$t.log 2>&1 & done
wait
$P -u $D/calibrate_bfd_sc.py merge
