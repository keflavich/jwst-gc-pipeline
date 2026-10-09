#!/bin/bash
# m7 fan-out shard for the DVA-consistent seed A/B (#1128) on the main2gt configuration:
# jwst-gc-pipeline-wd2main2gt (727e583a = main2 + grid-loader fix), sub_gt.sh flags and
# environment.  dvx0 = production seed inputs, dvx1 = 15 non-F150W m6 catalogs shifted
# into the F150W DVA frame (stage_dvx.py).  Bands F150W,F323N (F323N only makes the run
# multi-filter so the m7 phase exists).  usage:  ARM=dvx0|dvx1 sbatch --array=0-31 run_dvx.sbatch
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --job-name=wd2-dvx
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix/logs/%x_%A_%a.out
ulimit -c 0
source ~/.bash_profile 2>/dev/null
set -e
H=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix
export PIPE_ROOT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2gt
export CRDS_PATH=/orange/adamginsburg/jwst/crds CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export CRDS_CLIENT_RETRY_COUNT=8 CRDS_CLIENT_RETRY_DELAY_SECONDS=20 GC_ALLOW_DEV=1
export PYTHONDONTWRITEBYTECODE=1 STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
export DAOPHOT_HANDOFF_UNACCEPTED_SAT=1
unset SATSTAR_ZF_KEEP_FINITE DAOPHOT_HANDOFF_PEAK_MIN_AREA SATSTAR_ZF_R_HEADER
XA="--residual-bg-median-size=0 --no-manual-gc-superseded-perframe --manual-keep-intermediate-smoothed-bg"
case $ARM in
  dvx0|dvx1) ;;
  *) echo "bad ARM $ARM"; exit 1 ;;
esac
export GC_BASEPATH_OVERRIDE=$H/tree_$ARM
cd $H
SHARD=${SLURM_ARRAY_TASK_ID}
# The manual-iteration driver refuses to run with SLURM_ARRAY_TASK_ID set.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID
sleep $(( (RANDOM % 25) + 1 ))
echo "ARM $ARM shard ${SHARD}/32 repo $PIPE_ROOT $(git -C $PIPE_ROOT log --oneline -1) XA=$XA"
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python $H/run_m7.py \
  --filternames=F150W,F323N --modules=merged --each-exposure --proposal_id=3523 --field=005 \
  --target=wd2 --each-suffix=align_o005_crf --max-group-size=unlimited --parallel-workers=2 \
  --manual-start-phase=m7 --manual-stop-after-phase=m7 --manual-skip-finalize \
  --manual-frame-shard=${SHARD}/32 $XA
