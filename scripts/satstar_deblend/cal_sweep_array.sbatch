#!/bin/bash
#SBATCH --job-name=satstar-fitsweep-cal
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=10:00:00
#SBATCH --array=1-8
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/satstar-fitsweep-cal_%A_%a.log

# Calibration array for the fit-footprint flux correction: one task per frame in
# cal_manifest.tsv, each running the full size sweep (default 11..81). Produces
# per-size satstar catalogs under each frame's OUT dir for the correction fit.
set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline

M=scripts/satstar_deblend/cal_manifest.tsv
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$M")
LABEL=$(echo "$LINE" | cut -f1)
export SWEEP_CRF=$(echo "$LINE" | cut -f2)
export SWEEP_PSFS=$(echo "$LINE" | cut -f3)
export SWEEP_OUT=$(echo "$LINE" | cut -f4)
echo "[cal $SLURM_ARRAY_TASK_ID] $LABEL"
echo "[cal] CRF=$SWEEP_CRF  PSFS=$SWEEP_PSFS  OUT=$SWEEP_OUT"
"$PY" scripts/satstar_deblend/fit_footprint_sweep.py
echo "[cal $SLURM_ARRAY_TASK_ID] DONE $LABEL"
