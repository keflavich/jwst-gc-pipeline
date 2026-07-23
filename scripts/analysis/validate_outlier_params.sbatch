#!/bin/bash
#SBATCH --job-name=outlier-param-validate-brick-f200w-nrca
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/outlier-param-validate-brick-f200w-nrca_%j.log

# Validate keflavich/jwst-gc-pipeline#161: does raising outlier_detection
# snr/scale cut the bright-star PSF halo/spike over-rejection? Runs the step at
# default vs proposed params on the SAME post-tweakreg brick o004 F200W nrca
# frames and compares OUTLIER flagging. Read-only wrt the production products
# (writes to a scratch dir).

set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export VALIDATE_OUTDIR=/blue/adamginsburg/adamginsburg/tmp/outlier_validate
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
"$PY" scripts/analysis/validate_outlier_params.py
echo "[validate] DONE"
