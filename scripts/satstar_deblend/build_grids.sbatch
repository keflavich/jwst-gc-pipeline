#!/bin/bash
#SBATCH --job-name=nircam-psf-grid-build
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=3:00:00
#SBATCH --array=1-3
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/nircam-psf-grid_%A_%a.log
set -euo pipefail
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/
WT=/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/81778923-7a53-4903-85de-4e1a21cfef0f/scratchpad/wt-satinject
cd "$WT"
PAIRS=("nrca1 F115W" "nrca1 F150W" "nrcalong F405N")
IFS=' ' read -r DET BAND <<< "${PAIRS[$((SLURM_ARRAY_TASK_ID-1))]}"
$PY scripts/satstar_deblend/build_nircam_psf_grids.py /blue/adamginsburg/adamginsburg/jwst/brick/psfs "$DET" "$BAND"
