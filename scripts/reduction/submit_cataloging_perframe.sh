#!/bin/bash
# ---------------------------------------------------------------------------
# Per-frame fan-out cataloging chain (option C) -- the finest split.
#
# For EACH phase (m12 -> m3 -> m4 -> m5 -> m6 [-> m7]):
#   stage A: a per-frame fan-out ARRAY (NSHARDS small tasks, each fits a frame
#            shard and writes completion markers; --manual-skip-finalize), then
#   stage B: ONE finalize job (afterok stage A) that verifies all markers and
#            runs the per-phase barrier (--manual-finalize-only).
# Phase p+1's stage A waits (afterok) on phase p's stage B.
#
# Why: cataloging's queue delay is large-cpu node scarcity.  Each fan-out task
# asks FANOUT_CPUS (default 2) so it backfills into tiny holes; the barrier is
# I/O-bound so the finalize asks FINALIZE_CPUS (default 4).  NSHARDS only tunes
# granularity -- the shard predicate covers every frame for any NSHARDS.
#
# This is the same SCIENCE as the monolithic --each-exposure run (validate with
# tests/test_perframe_equivalence or scripts/.../validate_perframe_equivalence).
#
# Usage:
#   PIPE_ROOT=/path/to/checkout scripts/reduction/submit_cataloging_perframe.sh
#   PROPOSAL=2221 FIELD=001 TARGET=brick FILTERS="F405N F410M F466N" \
#       NSHARDS=24 scripts/reduction/submit_cataloging_perframe.sh
#
# NOTE (MIRI): all-MIRI multifilter runs drop m7 internally.  Set PHASES
# explicitly (e.g. PHASES="m12 m3 m4 m5 m6") for those.
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
export EXTRA_ARGS=${EXTRA_ARGS:-}   # may contain commas -> inherit via --export=ALL

# Granularity + per-stage resource slices (cpu only; mem/time stay generous so a
# job is never killed mid-run -- see scripts/reduction/README.md scheduling note).
NSHARDS=${NSHARDS:-16}
FANOUT_CPUS=${FANOUT_CPUS:-2}
FANOUT_MEM=${FANOUT_MEM:-32gb}
FANOUT_TIME=${FANOUT_TIME:-12:00:00}
FINALIZE_CPUS=${FINALIZE_CPUS:-4}
FINALIZE_MEM=${FINALIZE_MEM:-64gb}
FINALIZE_TIME=${FINALIZE_TIME:-12:00:00}

# Phase list.  Default: full NIRCam set, with m7 only when multifilter.
read -r -a _FA <<< "$FILTERS"
if [ -z "${PHASES:-}" ]; then
    PHASES="m12 m3 m4 m5 m6"
    [ "${#_FA[@]}" -gt 1 ] && PHASES="$PHASES m7"
fi

COMMON="ALL,PROPOSAL=$PROPOSAL,FIELD=$FIELD,TARGET=$TARGET,MODULES=$MODULES"
COMMON="$COMMON,EACH_SUFFIX=$EACH_SUFFIX,MAX_GROUP_SIZE=$MAX_GROUP_SIZE,NSHARDS=$NSHARDS"
COMMON="$COMMON,FILTERS=$FILTERS"
[ -n "$PIPE_ROOT" ] && COMMON="$COMMON,PIPE_ROOT=$PIPE_ROOT"
[ -n "$CROSSBAND_REF" ] && COMMON="$COMMON,CROSSBAND_REF=$CROSSBAND_REF"

# Optional upstream dependency (e.g. a reduction array job id): "<jobid>" or
# "afterok:<jobid>".  The first phase's fan-out waits on it.
DEP=${DEP:-}
prev_dep=""
if [ -n "$DEP" ]; then
    case "$DEP" in after*:*) prev_dep="$DEP";; *) prev_dep="afterok:$DEP";; esac
fi

echo "Per-frame chain: target=$TARGET $PROPOSAL/$FIELD modules=$MODULES"
echo "  phases: $PHASES   NSHARDS=$NSHARDS   filters: $FILTERS"
SB="$HERE/submit_cataloging_perframe_phase.sbatch"

for ph in $PHASES; do
    dep_arg=""; [ -n "$prev_dep" ] && dep_arg="--dependency=$prev_dep"
    A=$(sbatch --parsable $dep_arg \
        --array=0-$((NSHARDS-1)) \
        --cpus-per-task="$FANOUT_CPUS" --mem="$FANOUT_MEM" --time="$FANOUT_TIME" \
        --export="$COMMON,PHASE=$ph,MODE=fanout,PARALLEL_WORKERS=$FANOUT_CPUS" \
        "$SB")
    echo "  $ph fan-out array : $A  (0-$((NSHARDS-1)))${dep_arg:+  [$dep_arg]}"
    B=$(sbatch --parsable --dependency=afterok:"$A" \
        --cpus-per-task="$FINALIZE_CPUS" --mem="$FINALIZE_MEM" --time="$FINALIZE_TIME" \
        --export="$COMMON,PHASE=$ph,MODE=finalize,PARALLEL_WORKERS=$FINALIZE_CPUS" \
        "$SB")
    echo "  $ph finalize      : $B  (afterok:$A)"
    prev_dep="afterok:$B"
done

echo "DONE.  Final phase finalize job is the last printed B; watch: squeue -u \$USER -n gc_pf"
