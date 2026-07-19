#!/bin/bash
# ---------------------------------------------------------------------------
# Submit artificial-star (injection-recovery) completeness tests for the
# Brick: 2221 F212N + 1182 F200W, one crf frame per detector (8 each),
# ~1250 injected stars per frame => ~10k per band, spanning the number-count
# turnover (F212N 15-21, F200W 16-22).
#
# Task array 0-15: 0-7 = F212N nrc{a,b}{1-4}; 8-15 = F200W same order.
# A dependent job aggregates completeness curves once all tasks finish.
#
#   bash scripts/analysis/submit_artificial_star_completeness.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PIPE_ROOT=${PIPE_ROOT:-/orange/adamginsburg/repos/jwst-gc-pipeline-wt-artstar}
PY=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
WORKDIR=${WORKDIR:-/blue/adamginsburg/adamginsburg/jwst/brick/artificial_star_tests}
LOGDIR=${LOGDIR:-/blue/adamginsburg/adamginsburg/brick_logs}
NSTARS=${NSTARS:-1250}
SEED=${SEED:-0}

mkdir -p "$WORKDIR" "$LOGDIR"

JID=$(sbatch --parsable \
    --job-name=brick-artstar-completeness \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --array=0-15 \
    --ntasks=1 --cpus-per-task=4 --mem=48gb --time=12:00:00 \
    --output="$LOGDIR/brick-artstar-completeness_%A_%a.log" \
    --wrap "
DETS=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
IDX=\$SLURM_ARRAY_TASK_ID
if [ \$IDX -lt 8 ]; then BAND=F212N; else BAND=F200W; fi
DET=\${DETS[\$((IDX % 8))]}
PYTHONPATH=$PIPE_ROOT $PY -m jwst_gc_pipeline.photometry.artificial_stars run \
    --band \$BAND --detector \$DET --n-stars $NSTARS --seed $SEED \
    --workdir $WORKDIR
")
echo "injection-recovery array job: $JID"

AID=$(sbatch --parsable \
    --job-name=brick-artstar-analyze \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --dependency=afterany:$JID \
    --ntasks=1 --cpus-per-task=1 --mem=8gb --time=0:30:00 \
    --output="$LOGDIR/brick-artstar-analyze_%j.log" \
    --wrap "PYTHONPATH=$PIPE_ROOT $PY -m jwst_gc_pipeline.photometry.artificial_stars analyze --workdir $WORKDIR")
echo "analysis job (afterany:$JID): $AID"
