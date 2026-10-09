#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --job-name=wd2-satrefit7g
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --array=1-24
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/flatarea/out7g/log7g_%a.txt
cd /orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/flatarea
read BAND DET E < <(sed -n "${SLURM_ARRAY_TASK_ID}p" ../satrefit/rstar/jobs7f.txt)
echo $BAND $DET $E
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -u run_frames7g.py $BAND $DET $E
