#!/bin/bash
# ---------------------------------------------------------------------------
# OPTIONAL low-resource cataloging stream (dependency-chained).
#
#   stage 1:  per-filter array  (one filter per task; runs m12..m6 only,
#             because a single-filter run has no cross-band m7 phase)
#   stage 2:  m7 cross-band finalize  (ALL filters; --dependency=afterok on
#             the stage-1 array; reuses on-disk m6 products)
#
# This trades the one fat monolithic multifilter job (32 cores / 48 h, hard to
# schedule) for N small per-filter jobs + one finalize -- each a small ask that
# fits small queue holes.  Tune the per-filter slice with PERFILTER_CPUS.
#
# The fast/high-resource path is still available: just run the monolithic
# multifilter job (submit_cataloging.sbatch with FILTERS="all" on a single
# task), or this chain with large PERFILTER_CPUS.  See README.md.
#
# Granularity note: this splits at the *filter* boundary (the finest split that
# needs no core-code change).  Finer per-phase / per-frame splitting requires
# persisting cross-phase state (residual i2d, off-FOV satstar pins, iter_found)
# and is not yet implemented -- see README.md "Finer splitting".
#
# Usage:
#   scripts/reduction/submit_cataloging_chain.sh                 # Sgr C defaults
#   PROPOSAL=2221 FIELD=001 TARGET=brick FILTERS="F405N F410M F466N" \
#       PERFILTER_CPUS=4 scripts/reduction/submit_cataloging_chain.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}
CROSSBAND_REF=${CROSSBAND_REF:-}
PIPE_ROOT=${PIPE_ROOT:-}
# Passthrough to both stages (e.g. "--cutout-region=RA,DEC,SIZE --cutout-label=foo").
# Exported (not listed in --export) because its value may contain commas, which
# would corrupt SLURM's comma-separated --export list; --export=ALL inherits it.
export EXTRA_ARGS=${EXTRA_ARGS:-}

# Per-filter (stage 1) resource slice -- keep small so it fits queue holes.
PERFILTER_CPUS=${PERFILTER_CPUS:-8}
PERFILTER_MEM=${PERFILTER_MEM:-64gb}
PERFILTER_TIME=${PERFILTER_TIME:-24:00:00}
# m7 finalize (stage 2) slice.
M7_CPUS=${M7_CPUS:-8}
M7_MEM=${M7_MEM:-64gb}
M7_TIME=${M7_TIME:-24:00:00}

read -r -a FILT_ARR <<< "$FILTERS"
NF=${#FILT_ARR[@]}
if [ "$NF" -lt 1 ]; then echo "No filters given (FILTERS='$FILTERS')."; exit 1; fi

COMMON_EXPORT="ALL,PROPOSAL=$PROPOSAL,FIELD=$FIELD,TARGET=$TARGET,MODULES=$MODULES"
COMMON_EXPORT="$COMMON_EXPORT,EACH_SUFFIX=$EACH_SUFFIX,MAX_GROUP_SIZE=$MAX_GROUP_SIZE"
COMMON_EXPORT="$COMMON_EXPORT,FILTERS=$FILTERS"
[ -n "$PIPE_ROOT" ] && COMMON_EXPORT="$COMMON_EXPORT,PIPE_ROOT=$PIPE_ROOT"

echo "Stage 1: per-filter array (0-$((NF-1))) over: $FILTERS"
ARR=$(sbatch --parsable \
    --array=0-$((NF-1)) \
    --cpus-per-task="$PERFILTER_CPUS" --mem="$PERFILTER_MEM" --time="$PERFILTER_TIME" \
    --export="$COMMON_EXPORT,PARALLEL_WORKERS=$PERFILTER_CPUS" \
    "$HERE/submit_cataloging.sbatch")
echo "  submitted per-filter array job: $ARR"

echo "Stage 2: m7 cross-band finalize (afterok:$ARR)"
M7_EXPORT="$COMMON_EXPORT,PARALLEL_WORKERS=$M7_CPUS"
[ -n "$CROSSBAND_REF" ] && M7_EXPORT="$M7_EXPORT,CROSSBAND_REF=$CROSSBAND_REF"
M7=$(sbatch --parsable \
    --dependency=afterok:"$ARR" \
    --cpus-per-task="$M7_CPUS" --mem="$M7_MEM" --time="$M7_TIME" \
    --export="$M7_EXPORT" \
    "$HERE/submit_cataloging_m7.sbatch")
echo "  submitted m7 finalize job: $M7  (runs after array $ARR completes OK)"

echo "DONE: array=$ARR  m7=$M7"
echo "Watch:  squeue -j $ARR,$M7"
