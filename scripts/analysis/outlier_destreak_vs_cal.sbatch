#!/bin/bash
#SBATCH --job-name=outlier-destreak-vs-cal-brick-f200w
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --time=06:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/outlier-destreak-vs-cal-brick-f200w_%j.log
set -euo pipefail
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export CALTEST_OUTDIR=/blue/adamginsburg/adamginsburg/tmp/outlier_caltest
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python scripts/analysis/outlier_destreak_vs_cal.py
echo "[caltest] DONE"
