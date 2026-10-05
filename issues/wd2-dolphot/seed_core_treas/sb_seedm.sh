#!/bin/bash
#SBATCH --account=astronomy-dept --qos=astronomy-dept-b
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=12G --time=10:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/R_rcal/logs/sb_%x_%j.out
# usage: sbatch --job-name=NAME --export=ALL,BAND=F187N,FR=<frame> sb_seedm.sh
# c1m = c1s + seed-core fragment cut (filtered NaN-variance pieces < 3 px leave the core).
# Code: jwst-gc-pipeline-wt-seedab2 (276d4d29 + the wt-seedab patch + the cut, uncommitted).
W=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/R_rcal
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export REPO_PATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-seedab2
for v in $(env | grep -E '^(SATSTAR_|NIRCAM_SATSTAR_|RUN_DEBLEND)' | cut -d= -f1); do unset $v; done
ARM=c1m
b=${FR:?}_align_o005_crf.fits
echo "=== $ARM ${BAND:?} $b $(date +%T) $(hostname)"
python $W/run_frame_rc.py $W/tree_$ARM $BAND $b > $W/logs/${ARM}_${FR}.log 2>&1
echo "rc=$? $(date +%T)"
