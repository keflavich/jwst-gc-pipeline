#!/bin/bash
# m7 fan-out shard for the frame-lineage A/B (pedestal): dst0 = DAOPHOT_REFIT_NBRSUB
# off, F200W on destreak_o005_crf frames (as m1-m6), pinned at jwst-gc-pipeline-frnbrrun (dafe57d7) with the
# mainfcbg configuration (hand-off on, --residual-bg-median-size=0).  Shard i
# fits frames with index % 32 == i of each band.  usage:
#   ARM=dst0 sbatch --array=0-31 run_dst.sbatch
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --job-name=wd2-dst
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
#SBATCH --output=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pedestal/dst_ab/logs/%x_%A_%a.out
ulimit -c 0
source ~/.bash_profile 2>/dev/null
set -e
H=/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pedestal/dst_ab
export PIPE_ROOT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-frnbrrun
export CRDS_PATH=/orange/adamginsburg/jwst/crds CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export CRDS_CLIENT_RETRY_COUNT=8 CRDS_CLIENT_RETRY_DELAY_SECONDS=20 GC_ALLOW_DEV=1
export PYTHONDONTWRITEBYTECODE=1 STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
export DAOPHOT_HANDOFF_UNACCEPTED_SAT=1
unset SATSTAR_ZF_KEEP_FINITE
XA="--residual-bg-median-size=0 --no-manual-gc-superseded-perframe --manual-keep-intermediate-smoothed-bg"
case $ARM in
  dst0) export DAOPHOT_REFIT_NBRSUB=0 ;;
  *) echo "bad ARM $ARM"; exit 1 ;;
esac
export GC_BASEPATH_OVERRIDE=$H/tree_$ARM
cd $H
SHARD=${SLURM_ARRAY_TASK_ID}
# The manual-iteration driver refuses to run with SLURM_ARRAY_TASK_ID set.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID
sleep $(( (RANDOM % 25) + 1 ))
echo "ARM $ARM NBRSUB $DAOPHOT_REFIT_NBRSUB shard ${SHARD}/32 repo $PIPE_ROOT $(git -C $PIPE_ROOT log --oneline -1) XA=$XA"
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python $H/run_m7.py \
  --filternames=F200W --modules=merged --each-exposure --proposal_id=3523 --field=005 \
  --target=wd2 --each-suffix=destreak_o005_crf --max-group-size=unlimited --parallel-workers=2 \
  --manual-start-phase=m7 --manual-stop-after-phase=m7 --manual-skip-finalize \
  --manual-frame-shard=${SHARD}/32 $XA
