#!/bin/bash
# ---------------------------------------------------------------------------
# Single-band m8 forced cross-band fill -> partial catalog
# (..._resbgsub_m8_partial_<FILT>.fits).
#
# The monolithic m8 fill sweeps EVERY frame in one job and can overrun the wall
# (on sickle: F187N+F210M = 192 frames ate an 18h budget, F470N/F480M never ran).
# forced_fill_band writes only this band's columns, so the bands fill
# independently; scripts/reduction/m8_merge_partials.py column-merges the
# partials into the final ..._resbgsub_m8.fits (+ dedup).
#
# Reads the m7 merged catalog + per-frame m7 products already on disk
# (--manual-start-phase=m8 reconstructs the m6/m7 context).
#
# Args:  $1 = FILTER (one band, e.g. F335M)   $2 = each-suffix for that band
# Usually launched by submit_cataloging_m8.sh, which knows the per-filter suffix.
# ---------------------------------------------------------------------------
#SBATCH --job-name=catalog_m8p
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
# Low cpu ask backfills into small queue holes (queue delay = large-cpu-node
# scarcity, not memory); mem/time kept generous against an OOM/timeout kill.
#SBATCH --cpus-per-task=6
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%j.out

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=${CRDS_PATH:-/orange/adamginsburg/jwst/crds}
export CRDS_SERVER_URL=${CRDS_SERVER_URL:-https://jwst-crds.stsci.edu}

FILT=${1:?need a single FILTER as arg 1}
SUFFIX=${2:?need each-suffix as arg 2}
PROPOSAL=${PROPOSAL:-3958}
FIELD=${FIELD:-007}
TARGET=${TARGET:-sickle}
MODULES=${MODULES:-nrcb}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-10}
GROUP_MIN_SEP=${GROUP_MIN_SEP:-3.0}
EXTRA_ARGS=${EXTRA_ARGS:-}

PIPE_ROOT=${PIPE_ROOT:-}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"
PY=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}

[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}-catalog-m8p-${FILT}" 2>/dev/null || true

echo "NODE=$(hostname) nproc=$(nproc) FILT=$FILT SUFFIX=$SUFFIX start=$(date -u +%FT%TZ)"
"$PY" -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames="$FILT" --modules="$MODULES" \
    --proposal_id="$PROPOSAL" --field="$FIELD" --target="$TARGET" \
    --each-suffix="$SUFFIX" --each-exposure --daophot --skip-crowdsource \
    --manual-iterations --manual-start-phase=m8 --manual-m8-partial \
    --group --max-group-size="$MAX_GROUP_SIZE" \
    --manual-group-min-sep-fwhm="$GROUP_MIN_SEP" \
    --parallel-workers="${SLURM_CPUS_PER_TASK:-6}" \
    $EXTRA_ARGS
echo "CATALOG_M8_PARTIAL_${FILT}_DONE rc=$? end=$(date -u +%FT%TZ)"
