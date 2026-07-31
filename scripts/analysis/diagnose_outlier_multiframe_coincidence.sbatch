#!/bin/bash
#SBATCH --job-name=outlier161-multiframe-coincidence
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=6:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/outlier161-multiframe-coincidence_%j.log

# keflavich/jwst-gc-pipeline#161 DIAGNOSIS: run outlier_detection's own flagging
# test against a blot of the exposure's OWN resampled image, so no other
# exposure can contribute.  Whatever it flags is pure resample/blot round-trip.
# Read-only with respect to /orange.

set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export OUTLIER_DIST_DIR=${OUTLIER_DIST_DIR:-/blue/adamginsburg/adamginsburg/tmp/outlier_dist}
# repo root: submit from the repo root, or set OUTLIER_REPO
REPO=${OUTLIER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
export PYTHONPATH="$REPO/scripts/analysis:${PYTHONPATH:-}"
"$PY" "$REPO/scripts/analysis/diagnose_outlier_multiframe_coincidence.py"
echo "[coin] DONE"
