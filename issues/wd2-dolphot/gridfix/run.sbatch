#!/bin/bash
#SBATCH --job-name=gridfix
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix/log_%a.txt
BANDS=(200W 212N 182M 150W 300M 250M 323N 410M)
export PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-gridT
export OMP_NUM_THREADS=2
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python closure.py ${BANDS[$SLURM_ARRAY_TASK_ID]} 4000
