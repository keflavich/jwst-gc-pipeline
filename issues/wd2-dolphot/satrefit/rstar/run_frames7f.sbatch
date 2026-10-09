#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --array=1-24
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/rstar/out7f/log7f_%a.txt
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/rstar
read BAND DET E < <(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs7f.txt)
echo $BAND $DET $E
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u run_frames7f.py $BAND $DET $E
