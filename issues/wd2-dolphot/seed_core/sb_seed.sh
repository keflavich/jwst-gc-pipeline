#!/bin/bash
#SBATCH --account=astronomy-dept --qos=astronomy-dept-b
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=12G --time=10:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/R_rcal/logs/sb_%x_%j.out
# usage: sbatch --job-name=NAME --export=ALL,ARM=c1s|c1ks,BAND=F187N,FR=<frame> sb_seed.sh
# Seed-core DQ filter (SATSTAR_SEED_CORE_DQ, default on) on top of 276d4d29: c1s = c1 + filter,
# c1ks = c1k + filter.  Code: jwst-gc-pipeline-wt-seedab (276d4d29 + uncommitted patch).
W=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/R_rcal
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export REPO_PATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-seedab
for v in $(env | grep -E '^(SATSTAR_|NIRCAM_SATSTAR_|RUN_DEBLEND)' | cut -d= -f1); do unset $v; done
case ${ARM:?} in
 c1s) ;;
 c1ks) export SATSTAR_ZF_KEEP_FINITE=1 ;;
 *) echo bad arm; exit 1 ;;
esac
b=${FR:?}_align_o005_crf.fits
echo "=== $ARM ${BAND:?} $b $(date +%T) $(hostname)"
python $W/run_frame_rc.py $W/tree_$ARM $BAND $b > $W/logs/${ARM}_${FR}.log 2>&1
echo "rc=$? $(date +%T)"
