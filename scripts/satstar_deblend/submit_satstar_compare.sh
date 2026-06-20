#!/bin/bash
# Submit baseline + deblend satstar comparison on one gc2211 frame as two slurm
# jobs (run in parallel, ~2-4h each).  QOS/account = astronomy-dept-b (project rule).
set -euo pipefail
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-satdeblend
SCR=$WT/scripts/satstar_deblend
LOG=$SCR/out_compare
mkdir -p "$LOG"
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python

for MODE in baseline deblend; do
  sbatch --account=astronomy-dept --qos=astronomy-dept-b \
         --partition=hpg-default \
         --job-name="satcmp_${MODE}" \
         --output="$LOG/satcmp_${MODE}_%j.log" \
         --time=8:00:00 --mem=48G --cpus-per-task=4 --nodes=1 --ntasks=1 \
         --wrap "export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/; \
                 $PY $SCR/run_satstar_compare.py $MODE"
done
echo "submitted baseline + deblend"
