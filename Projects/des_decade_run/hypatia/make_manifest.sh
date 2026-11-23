#!/usr/bin/env bash
# List the simulations of one suite that still need mocks (no mocks_done.json) and have the inputs.
# usage: bash make_manifest.sh SUITE [N_PER_FILE]     e.g.  bash make_manifest.sh X 2
#   -> manifests/runsX/todo.txt and todo_NN chunks of N_PER_FILE simulations (one chunk = one job)
set -euo pipefail
suite="${1:?usage: make_manifest.sh SUITE [N_PER_FILE]}"
per="${2:-2}"
base="${MOCKS_BASE:-/share/lustre/mgatti/mocks}/runs${suite}"
out="manifests/runs${suite}"
mkdir -p "$out"
: > "$out/todo.txt"
: > "$out/missing_inputs.txt"
for d in "$base"/run*; do
    [[ -d "$d" ]] || continue
    [[ -f "$d/mocks_done.json" ]] && continue
    ok=1
    for f in delta_b_1024_0.npy delta_b_1024_1.npy run.00100.lightcone.npy control.par z_values.txt; do
        [[ -e "$d/$f" ]] || { echo "$d missing $f" >> "$out/missing_inputs.txt"; ok=0; }
    done
    (( ok )) && echo "$d" >> "$out/todo.txt"
done
rm -f "$out"/todo_[0-9]*
split -l "$per" -d -a 3 "$out/todo.txt" "$out/todo_"
echo "$(wc -l < "$out/todo.txt") simulations to do in $(ls "$out"/todo_[0-9]* 2>/dev/null | wc -l) chunks;" \
     "$(wc -l < "$out/missing_inputs.txt") missing inputs (see $out/missing_inputs.txt)"
