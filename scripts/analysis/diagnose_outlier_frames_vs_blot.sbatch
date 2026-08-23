#!/bin/bash
#SBATCH --job-name=outlier161-diagnose-frames-vs-blot
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=4:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/outlier161-diagnose-frames-vs-blot_%j.log

# No core dumps.  A python SIGBUS in a 60+ GB reduction writes a core file
# the size of the process (43 GiB measured, 2026-08-06) into the working
# directory, and /orange runs at 97% full.  HiPerGator sets
# PropagateResourceLimits=NONE, so a `ulimit -c 0` in the submitting shell
# does NOT reach the compute node -- it has to be set here, inside the batch
# script itself (jwst-gc-pipeline#428).
ulimit -c 0

# keflavich/jwst-gc-pipeline#161 DIAGNOSIS (not tuning): at the pixels
# outlier_detection flags around bright stars, do the exposures disagree with
# EACH OTHER, or only with the blotted median?  Reads the intermediate products
# of the earlier OutlierDetectionStep run (save_intermediate_results=True);
# read-only with respect to /orange.

set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export OUTLIER_DIST_DIR=${OUTLIER_DIST_DIR:-/blue/adamginsburg/adamginsburg/tmp/outlier_dist}
# repo root: submit from the repo root, or set OUTLIER_REPO
REPO=${OUTLIER_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
"$PY" "$REPO/scripts/analysis/diagnose_outlier_frames_vs_blot.py"
echo "[diag] DONE"
