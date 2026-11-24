#!/usr/bin/env bash
# First job on Hypatia: one simulation, one baryon draw, one rotation, all samples, halo IA, kappa_CMB.
# Exercises everything that has only been run at NERSC on parquet FoF: FoF tar extraction (native
# run.NNNNN.fofstats.0), the IA halo lightcone, run.NNNNN.lightcone.npy reading for kappa_CMB, the Hypatia
# delta_b / sys_baryo files and the staged inputs.  Output goes to a scratch directory, not the simulation.
# usage: sbatch -p CORES64 --ntasks=1 --cpus-per-task=64 --mem=480G smoke_test.sh SUITE RUN_DIR
#SBATCH -J dd_smoke
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
suite="${1:?usage: smoke_test.sh SUITE RUN_DIR}"
sim="${2:?usage: smoke_test.sh SUITE RUN_DIR}"
LFM=${LFM:-/share/lustre/mgatti/LSS_forward_model}
FOF_ROOT=${FOF_ROOT:-/share/gowerst/ucapwhi/GowerStreetExtendedSims/runs${suite}}
OUT=${OUT:-/share/lustre/mgatti/des_decade_smoke/$(basename "$sim")}
export DES_DECADE_INPUTS=${DES_DECADE_INPUTS:-/share/lustre/mgatti/des_decade_run_inputs}
module purge
source /share/rcifdata/mgatti/miniforge3/etc/profile.d/conda.sh
conda activate /share/rcifdata/mgatti/conda_envs/lss_py
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_MAX_THREADS="$SLURM_CPUS_PER_TASK"
cd "$LFM/Projects/des_decade_run"
# staged inputs present (sizes from config/large_inputs.json; cat_BFD, the full BFD catalogue, is not used)
python - <<'PY'
import json, os
from pathlib import Path
stage = Path(os.environ['DES_DECADE_INPUTS'])
files = json.loads(Path('config/large_inputs.json').read_text())['files']
bad = [f"{k}: {v['file']}" for k, v in files.items() if k != 'cat_BFD' and
       (not (stage/v['file']).exists() or (stage/v['file']).stat().st_size != v['size'])]
assert not bad, f'missing or wrong size in {stage}: {bad}'
print('staged inputs OK')
PY
mkdir -p "$OUT"
extra=()
[[ -d "$sim/fof" ]] || extra+=(--fof-tar "$FOF_ROOT/$(basename "$sim").fof.tar.gz")
start=$(date +%s)
python -u make_mocks.py "$sim" --out "$OUT" --sim-id "smoke/runs${suite}/$(basename "$sim")" --models halo \
    --baryons 0 --rots 0 --nproc "$SLURM_CPUS_PER_TASK" "${extra[@]}"
echo "wall $(( $(date +%s) - start )) s"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,TotalCPU,State
