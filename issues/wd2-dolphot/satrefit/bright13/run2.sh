#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/bright13/log2_%a.txt
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/bright13
B=(250M 250M 250M 250M 300M 300M 300M 300M)
V=(04101 04101 04101 04101 12101 12101 12101 12101)
E=$(printf "%05d" $(( SLURM_ARRAY_TASK_ID % 4 + 1 )))
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u stage2.py ${B[$SLURM_ARRAY_TASK_ID]} jw03523005001_${V[$SLURM_ARRAY_TASK_ID]}_${E}_nrcblong_align_o005_crf
