#!/bin/bash
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J ia_prior_tng
#SBATCH -o /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests/logs/tng_%j.log
# usage: sbatch run_tng_populations.sh "FC:FSAT FC:FSAT ..." [nla]
# TNG300 DES virtual-halo populations for the IA prior test (eta_c -2..2, eta_s -0.5..0.5), all in parallel
export OMP_NUM_THREADS=24 PYTHONNOUSERSITE=1
P=/global/cfs/cdirs/m5099/env/pyccl_env/bin/python
V=/global/u2/m/mgatti/LSS_forward_model/Projects/IA_non_gaussian/cheap_model_validation
L=/global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests/logs
for pair in $1; do
  fc=${pair%%:*}; fs=${pair##*:}
  (cd $V && $P -u cheap_ia_validation.py hod --survey DES --virtual sim --f-c $fc --f-sat $fs \
     --eta-c-grid -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 --eta-s-grid -0.5 -0.25 0 0.25 0.5 --tag _prior \
     > $L/hod_fc${fc}_fsat${fs}.log 2>&1) &
done
if [ "$2" == "nla" ]; then
  $P -u /global/u2/m/mgatti/LSS_forward_model/Projects/des_decade_run/prior_tests/nla_tng_fields.py > $L/nla_fields.log 2>&1 &
fi
wait
