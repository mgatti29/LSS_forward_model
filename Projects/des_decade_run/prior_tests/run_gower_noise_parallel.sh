#!/bin/bash
# all Gower noise-budget populations in parallel on one node
export OMP_NUM_THREADS=48 NUMEXPR_MAX_THREADS=48 PYTHONNOUSERSITE=1
D=/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests
for pair in 0.5:0.8 0.2:0.3 0.2:0.8 0.7:0.3 0.7:0.8; do
  fc=${pair%%:*}; fs=${pair##*:}
  /global/cfs/cdirs/m5099/env/pyccl_env/bin/python -u $D/gower_ia_noise.py $fc $fs > $D/logs/gower_noise_${fc}_${fs}.log 2>&1 &
done
wait
