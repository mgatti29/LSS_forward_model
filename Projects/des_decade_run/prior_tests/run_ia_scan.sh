#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 08:00:00
#SBATCH -J ia_scan
#SBATCH -o /pscratch/sd/m/mgatti/des_decade_test/logs/ia_scan_%j.log
# Halo-IA acceptance scan (ia_acceptance_scan.py): Gower Fiducial/1_big, baryon draw 0, rotations 0-3 as four
# tasks, two per node (64 cores each).  Four per node ran out of memory (pool workers are forked copies).
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64
PY=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
SCAN=/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests/ia_acceptance_scan.py
OUT=/pscratch/sd/m/mgatti/des_decade_test/ia_scan
for r in 0 1 2 3; do
    srun --exclusive -N1 -n1 -c 128 --cpu-bind=cores $PY -u $SCAN /pscratch/sd/m/mgatti/des_decade_test/Fiducial/1_big 0 $r \
        $OUT/scan_b0_r$r --nproc 32 > /pscratch/sd/m/mgatti/des_decade_test/logs/ia_scan_b0_r${r}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State
