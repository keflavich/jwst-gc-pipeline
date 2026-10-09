#!/bin/bash
#SBATCH --job-name=gridfix-rep
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix/rep_log.txt
export PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-gridT
export OMP_NUM_THREADS=2
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix
P=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
$P replicate_m7.py 150W nrcb1 00001 > replicate_m7_150W_nrcb1.txt
$P replicate_m7.py 200W nrcb1 00001 > replicate_m7_200W_nrcb1.txt
