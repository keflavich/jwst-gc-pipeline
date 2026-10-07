#!/bin/bash
# LW (F250M, F277W, F300M) m7 fan-out shard for the peak hand-off arms
# (dbl2: peaks; dbl3: peaks + deblend), pinned at jwst-gc-pipeline-hpeaksrun
# (048299af).  The m7 phase
# exists only for multi-band runs, so the three LW bands run together; shard i
# fits frame i of each band.  usage:
#   ARM=dbl2|dbl3 sbatch --array=0-7 run_pk.sbatch
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --job-name=wd2-lw-pk
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=10:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap/dbl/logs/%x_%A_%a.out
ulimit -c 0
source ~/.bash_profile 2>/dev/null
set -e
H=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap/dbl
B=/blue/adamginsburg/adamginsburg/repos
export CRDS_PATH=/orange/adamginsburg/jwst/crds CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export CRDS_CLIENT_RETRY_COUNT=8 CRDS_CLIENT_RETRY_DELAY_SECONDS=20 GC_ALLOW_DEV=1
export PYTHONDONTWRITEBYTECODE=1 STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
export DAOPHOT_HANDOFF_UNACCEPTED_SAT=1 DAOPHOT_HANDOFF_PEAK_MIN_AREA=50
unset SATSTAR_ZF_KEEP_FINITE
XA="--residual-bg-median-size=0 --no-manual-gc-superseded-perframe --manual-keep-intermediate-smoothed-bg"
case $ARM in
  dbl2) export PIPE_ROOT=$B/jwst-gc-pipeline-hpeaksrun ;;
  dbl3) export PIPE_ROOT=$B/jwst-gc-pipeline-hpeaksrun; XA="$XA --deblend-satstars" ;;
  *) echo "bad ARM $ARM"; exit 1 ;;
esac
export GC_BASEPATH_OVERRIDE=$H/tree_$ARM
cd $H
SHARD=${SLURM_ARRAY_TASK_ID}
# The manual-iteration driver refuses to run with SLURM_ARRAY_TASK_ID set.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID
sleep $(( (RANDOM % 25) + 1 ))
echo "PEAK_MIN_AREA $DAOPHOT_HANDOFF_PEAK_MIN_AREA ARM $ARM shard ${SHARD}/8 repo $PIPE_ROOT $(git -C $PIPE_ROOT log --oneline -1) XA=$XA"
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python $H/run_m7.py \
  --filternames=F250M,F277W,F300M --modules=merged --each-exposure --proposal_id=3523 --field=005 \
  --target=wd2 --each-suffix=align_o005_crf --max-group-size=unlimited --parallel-workers=2 \
  --manual-start-phase=m7 --manual-stop-after-phase=m7 --manual-skip-finalize \
  --manual-frame-shard=${SHARD}/8 $XA
