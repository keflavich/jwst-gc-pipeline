#!/bin/bash
# usage: sbatch --array=0-1 --dependency=afterok:... merge_dvx.sbatch  (task = arm)
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --job-name=wd2-dvx-merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix/logs/%x_%A_%a.out
ulimit -c 0
source ~/.bash_profile 2>/dev/null
H=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix
ARMS=(dvx0 dvx1)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2gt
export GC_BASEPATH_OVERRIDE=$H/tree_$ARM
cd $H
echo "ARM $ARM BAND F150W"
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python $H/merge_dvx.py $ARM F150W
