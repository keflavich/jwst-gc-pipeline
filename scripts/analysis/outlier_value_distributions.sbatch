#!/bin/bash
#SBATCH --job-name=outlier-value-dist-brick-f200w-nrca
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --time=06:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/outlier-value-dist-brick-f200w-nrca_%j.log

# No core dumps.  A python SIGBUS in a 60+ GB reduction writes a core file
# the size of the process (43 GiB measured, 2026-08-06) into the working
# directory, and /orange runs at 97% full.  HiPerGator sets
# PropagateResourceLimits=NONE, so a `ulimit -c 0` in the submitting shell
# does NOT reach the compute node -- it has to be set here, inside the batch
# script itself (jwst-gc-pipeline#428).
ulimit -c 0
set -euo pipefail
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export DIST_OUTDIR=/blue/adamginsburg/adamginsburg/tmp/outlier_dist
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python scripts/analysis/outlier_value_distributions.py
echo "[dist] DONE"
