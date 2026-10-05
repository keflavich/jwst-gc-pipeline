#!/bin/bash
#SBATCH --account=astronomy-dept --qos=astronomy-dept-b
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=16G --time=03:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/S_seedtreas/logs/sb_%x_%j.out
# usage: sbatch --job-name=NAME --export=ALL,ARM=s0|s1,FR=<frame base> sb_st.sh
# Seed-core DQ filter A/B on gc-treasury F480M nrcalong o040 (#1101 review).
# s0 = SATSTAR_SEED_CORE_DQ=0, s1 = default (on).  Code: jwst-gc-pipeline-wt-seedfix (e4eac2c3), read-only.
S=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/S_seedtreas
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export REPO_PATH=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-seedfix
for v in $(env | grep -E '^(SATSTAR_|NIRCAM_SATSTAR_|RUN_DEBLEND)' | cut -d= -f1); do unset $v; done
case ${ARM:?} in
 s0) export SATSTAR_SEED_CORE_DQ=0 ;;
 s1) ;;
 *) echo bad arm; exit 1 ;;
esac
b=${FR:?}_destreak_o040_crf.fits
echo "=== $ARM $b $(date +%T) $(hostname) HEAD $(git -C $REPO_PATH rev-parse --short HEAD)"
python $S/run_frame_st.py $S/tree_$ARM F480M $b > $S/logs/${ARM}_${FR}.log 2>&1
echo "rc=$? $(date +%T)"
