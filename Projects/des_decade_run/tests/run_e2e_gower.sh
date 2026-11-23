#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J dd_e2e
#SBATCH -o /pscratch/sd/m/mgatti/des_decade_test/logs/e2e_%j.log
# End-to-end test of make_mocks.py on Gower St LCDM Fiducial/1_big (symlinked to pscratch): both IA models,
# four samples, two "baryon draws" (both = the Gower DMO delta_1024.npy, symlinked), four rotations, kappa_CMB.
export OMP_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128 PYTHONNOUSERSITE=1
PY=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
/usr/bin/time -v $PY -u /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/make_mocks.py \
    /pscratch/sd/m/mgatti/des_decade_test/Fiducial/1_big --models halo nla \
    --halo-catalog /pscratch/sd/m/mgatti/source_clustering_v2/halo_ia/halo_catalog_ia_z2_m12_Fiducial_1_big.parquet
