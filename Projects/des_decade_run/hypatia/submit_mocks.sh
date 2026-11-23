#!/usr/bin/env bash
# One Slurm job = one node = SIMS_PER_NODE simulations in parallel, one srun task per simulation (no MPI inside
# the driver: it forks process pools, which is unsafe after MPI_Init).  Each task runs make_mocks.py on the
# simulation of its manifest line; finished simulations (mocks_done.json) are skipped, so a resubmission resumes.
#
# usage: sbatch -p CORES64 --ntasks=2 --cpus-per-task=32 --mem=480G submit_mocks.sh SUITE MANIFEST
#   memory: measured peak RSS 164 GB per simulation (NERSC e2e test, halo + NLA, 4 samples): 2 per 480 GB node.
#   time: ~3.2 h per simulation for halo-only on 128 NERSC cores; scale for fewer / slower cores and add the
#         FoF extraction + IA halo lightcone (~20-40 min) -> request 12 h until the first job is measured.
#SBATCH -J dd_mocks
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
suite="${1:?usage: submit_mocks.sh SUITE MANIFEST}"
manifest="${2:?usage: submit_mocks.sh SUITE MANIFEST}"

LFM=${LFM:-/share/lustre/mgatti/LSS_forward_model}
FOF_ROOT=${FOF_ROOT:-/share/gowerst/ucapwhi/GowerStreetExtendedSims/runs${suite}}
MODELS=${MODELS:-halo}
export DES_DECADE_INPUTS=${DES_DECADE_INPUTS:-/share/lustre/mgatti/des_decade_run_inputs}

module purge
source /share/rcifdata/mgatti/miniforge3/etc/profile.d/conda.sh
conda activate /share/rcifdata/mgatti/conda_envs/lss_py
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_MAX_THREADS="$SLURM_CPUS_PER_TASK"

cd "$LFM/Projects/des_decade_run/hypatia"
mkdir -p logs/sims
mapfile -t sims < "$manifest"
echo "job $SLURM_JOB_ID on $SLURM_NODELIST: ${#sims[@]} simulations, $SLURM_NTASKS tasks x $SLURM_CPUS_PER_TASK cpus"
echo "package $(git -C "$LFM" rev-parse HEAD)"

for i in "${!sims[@]}"; do
    sim="${sims[$i]}"
    run=$(basename "$sim")
    if [[ -f "$sim/mocks_done.json" ]]; then echo "skip $sim (done)"; continue; fi
    tar="$FOF_ROOT/${run}.fof.tar.gz"
    extra=()
    [[ -d "$sim/fof" ]] || extra+=(--fof-tar "$tar")
    (
        start=$(date +%s); rc=0
        srun --exclusive -N1 -n1 -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
            python -u "$LFM/Projects/des_decade_run/make_mocks.py" "$sim" --sim-id "runs${suite}/${run}" \
            --models $MODELS --nproc "$SLURM_CPUS_PER_TASK" --cleanup "${extra[@]}" || rc=$?
        echo "$sim exit $rc in $(( $(date +%s) - start )) s"
    ) > "logs/sims/${suite}_${run}_${SLURM_JOB_ID}.log" 2>&1 &
    # at most SLURM_NTASKS simulations at a time
    while (( $(jobs -rp | wc -l) >= SLURM_NTASKS )); do wait -n || true; done
done
wait
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,TotalCPU,State
