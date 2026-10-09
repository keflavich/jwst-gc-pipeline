#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=08:00:00
#SBATCH --job-name=s4fit
#SBATCH --array=0-1
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f150x/samp4/fit_%A_%a.log
export PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-gridT
export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
export OMP_NUM_THREADS=1
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f150x/samp4
B=(150W 200W)
python fit_s4.py ${B[$SLURM_ARRAY_TASK_ID]} nrcb1
