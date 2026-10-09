#!/bin/bash
#SBATCH --job-name=gridfix-rc
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix/recentre/log_%a.txt
BANDS=(150W 200W)
DETS=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
B=${BANDS[$((SLURM_ARRAY_TASK_ID / 8))]}
DT=${DETS[$((SLURM_ARRAY_TASK_ID % 8))]}
export PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-gridT
export OMP_NUM_THREADS=2
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python recentre_ab.py $B $DT
