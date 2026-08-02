#!/bin/bash
#SBATCH --job-name=satstar-fbf-calibrate
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=5:00:00
#SBATCH --array=1-4
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/satstar-fbf-cal_%A_%a.log
set -euo pipefail
WT=/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/81778923-7a53-4903-85de-4e1a21cfef0f/scratchpad/wt-satinject
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/ CRDS_PATH=/orange/adamginsburg/jwst/crds PYTHONPATH=$WT
cd "$WT"
FBFS=(0.0 0.5 1.0 2.0)
$PY jwst_gc_pipeline/photometry/calibrate_fbf.py "${FBFS[$((SLURM_ARRAY_TASK_ID-1))]}"
