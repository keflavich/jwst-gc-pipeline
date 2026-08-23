#!/bin/bash
#SBATCH --job-name=satstar-adaptive-fitshape-ab-gc2211-f200w
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=4:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/satstar-adaptive-fitshape-ab_%j.log

# No core dumps.  A python SIGBUS in a 60+ GB reduction writes a core file
# the size of the process (43 GiB measured, 2026-08-06) into the working
# directory, and /orange runs at 97% full.  HiPerGator sets
# PropagateResourceLimits=NONE, so a `ulimit -c 0` in the submitting shell
# does NOT reach the compute node -- it has to be set here, inside the batch
# script itself (jwst-gc-pipeline#428).
ulimit -c 0

# A/B: adaptive per-star fit_shape (=f(core radius)) vs the flat size=81 baseline
# catalog, on one gc2211 F200W frame. Validates that scaling the fit box to each
# star's saturated core keeps the flux accurate (esp. for the faint/small-core
# stars that get the smallest boxes) while cutting wall-clock. Read-only wrt
# /orange products.

set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
"$PY" scripts/satstar_deblend/adaptive_fit_shape_ab.py
echo "[adaptive-ab] DONE"
