#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=8:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/satstar-fitsweep-%x_%j.log

# Env-driven fit-footprint sweep (for the flux-correction calibration array).
# Reads SWEEP_CRF / SWEEP_PSFS / SWEEP_OUT / SWEEP_SIZES from the environment
# (pass with sbatch --export and set --job-name to the frame label).
set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
echo "[sweep-env] CRF=$SWEEP_CRF"
echo "[sweep-env] PSFS=$SWEEP_PSFS  OUT=$SWEEP_OUT  SIZES=${SWEEP_SIZES:-default}"
"$PY" scripts/satstar_deblend/fit_footprint_sweep.py
echo "[sweep-env] DONE"
