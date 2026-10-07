#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --job-name=wd2-pb-f212n
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/sat1to1/pb/%x_%a_%A.out
#SBATCH --array=0-1
B=/blue/adamginsburg/adamginsburg/repos
H=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/sat1to1/pb
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then ARM=pb0; REPO=$B/jwst-gc-pipeline-wd2main2; else ARM=pb1; REPO=$B/jwst-gc-pipeline-sat1to1ab; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$REPO GC_BASEPATH_OVERRIDE=$H/tree_$ARM
echo "ARM=$ARM REPO=$REPO HEAD=$(git -C $REPO rev-parse --short HEAD)"
cd $H
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python run_pb.py $ARM
