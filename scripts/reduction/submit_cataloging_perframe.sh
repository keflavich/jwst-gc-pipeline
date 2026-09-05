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
#
# RESTARTING a chain that hit its wall clock:
#
#   SKIP_IF_DONE=1 scripts/reduction/submit_cataloging_perframe.sh ...
#
# The fan-out then resumes from the completion markers the killed tasks left
# behind instead of refitting the shard (wd1 #570: 4:26-8:44 per shard down to
# ~50 s).  It is OPT-IN because no marker changes when the photometry code
# does, so a re-catalog run to apply a fit fix must NOT inherit it -- that run
# would skip every frame and report green with the old photometry.  Markers
# that are older than their `_crf`, or than the phase's seed inputs, are
# refitted even with the flag on, and counted in the run's REFITTING line.
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
# Opt-in fan-out resume (see the header).  Exported rather than folded into
# COMMON so a wrapper that sets it without `export` still reaches the array;
# COMMON begins with ALL, so the phase script reads it from the environment.
export SKIP_IF_DONE=${SKIP_IF_DONE:-}
# MODULES is comma-valued for any multi-module field ("nrca,nrcb"), and sbatch
# --export is itself comma-separated: inlining it as MODULES=$MODULES makes
# sbatch read "nrcb" as a separate KEY=VALUE-less token and the variable arrives
# truncated to "nrca".  The m2 checkpoint then only ever sees one module, so the
# other half of the field stays untied while the run reports success -- the
# "half the mosaic offset" failure class.  Export it and let --export=ALL carry
# it, exactly as EXTRA_ARGS above.
export MODULES

# Granularity + per-stage resource slices.  cpu is flat; memory and wall time
# both scale with the FIELD (crf count, measured once just below) and wall time
# additionally with the STAGE.
NSHARDS=${NSHARDS:-16}
FANOUT_CPUS=${FANOUT_CPUS:-2}
FANOUT_MEM=${FANOUT_MEM:-32gb}
FINALIZE_CPUS=${FINALIZE_CPUS:-4}

# FIELD SIZE, measured once and used twice: finalize memory (issue #611) and
# wall time (issue #737).  crf count is the proxy available at submit time and
# it tracks what both actually scale with -- the number of frames fitted and
# the size of the merge.
#
# `ls` exits non-zero when the glob matches nothing, and this script runs under
# `set -euo pipefail`, so a bare `ls ... | wc -l` ABORTS the whole submit on any
# host without the data tree -- which is exactly CI.  Swallow the status inside
# the pipeline so a missing tree counts 0 and falls to the smallest tier instead
# of killing the run.
#
# A split-tree field is NOT under $TARGET.  The driver is invoked
# TARGET=gc2211 FIELD=023 (886 job logs use that spelling) while the data sits
# in gc2211_o023/; bare gc2211/ holds 0 crf.  Keyed on $TARGET alone every
# gc2211 observation measured 0 and took the SMALLEST tier -- including o046
# (240 crf -> 128gb), one of the OOMs that motivated the memory sizing.  So try
# <target>_o<field> first and fall back to <target>, which is what a single-tree
# field (sgrb2/001) and an already-per-obs TARGET (gc2211_o046, whose
# _o046_o046 does not exist) both need.
_crf_base=${BASEPATH:-/orange/adamginsburg/jwst}
_crf_dir="$_crf_base/${TARGET}_o${FIELD}"
_crf_tried="$_crf_dir"
_crf_count=$({ ls "$_crf_dir"/*/pipeline/*crf.fits 2>/dev/null || true; } | wc -l)
if [ "$_crf_count" -eq 0 ]; then
    _crf_dir="$_crf_base/$TARGET"
    _crf_tried="$_crf_tried then $_crf_dir"
    _crf_count=$({ ls "$_crf_dir"/*/pipeline/*crf.fits 2>/dev/null || true; } | wc -l)
fi
if   [ "$_crf_count" -ge 1000 ]; then FIELD_TIER=large
elif [ "$_crf_count" -ge 200 ];  then FIELD_TIER=mid
else                                  FIELD_TIER=small
fi

# FINALIZE MEMORY SCALES WITH THE FIELD (issue #611).  A flat 64gb was the same
# request for sgrb2 -- 1540 crf over 16 filters, merging 14.2M detections -- as
# for m92's 80.  sgrb2's m3-finalize was OOM-killed at 64gb (accounting recorded
# MaxRSS 155G) and the kill landed *inside* a multiprocessing queue write, so the
# survivors deadlocked and the job sat at ~0 CPU for 44 h before anyone looked;
# the afterok chain behind it died with it.  It completed in 12 h 26 at 256gb.
#
#     small  m92 80 / arches 110 / ngc6397 120 / m4 150      -> 64gb  (m92 ran at 96)
#     mid    sgrc 240 / sickle 536 / cloudef 640 / wd1 696   -> 128gb
#     large  w51 1120 / cloudc 1208 / sgrb2 1540 / brick 2016 -> 256gb (sgrb2 measured)
#
# Fan-out is per-shard and unaffected.  FINALIZE_MEM in the environment still
# wins, so a one-off can override without editing this.
if [ -z "${FINALIZE_MEM:-}" ]; then
    case "$FIELD_TIER" in
        large) FINALIZE_MEM=256gb ;;
        mid)   FINALIZE_MEM=128gb ;;
        *)     FINALIZE_MEM=64gb ;;
    esac
    if [ "$_crf_count" -eq 0 ]; then
        # 0 is two different states now that an absent tree is supported: the
        # data is genuinely not here (CI, a fresh checkout, a field before its
        # first reduce), or the count looked in the wrong place.  Only the
        # second is a memory decision, so name the paths -- a job log has to be
        # able to tell them apart.
        echo "finalize memory: $FINALIZE_MEM (crf count came back EMPTY --" \
             "tried $_crf_tried; smallest tier by default, not a measurement)"
    else
        echo "finalize memory: $FINALIZE_MEM (${_crf_count} crf under $_crf_dir)"
    fi
fi
FINALIZE_MEM=${FINALIZE_MEM:-64gb}

# WALL TIME SCALES WITH THE FIELD *AND* THE STAGE (issue #737).  A flat
# 12:00:00 was handed to every fan-out and every finalize of every field.  A
# stage killed on its time limit takes the whole `afterok` chain down with it:
# crowded_l3's eight m12-fanout shards all hit 12:00:03 on 2026-09-04 -- after
# writing 280 per-frame catalogs, so the work was done -- and left the field's
# remaining 11 jobs stranded on Dependency.  Every TIMEOUT in the 14 days to
# 2026-09-04 (35 of them, 31 m12-fanout + 4 m12-finalize) was submitted at that
# 12 h default.
#
# The stage alone is NOT the right axis, and this is why the tier above is
# reused rather than a stage table applied flat.  14 days of sacct
# (COMPLETED + TIMEOUT), longest run in hours, by crf tier:
#
#     fan-out         small       mid       large        finalize    small   mid   large
#     m12               6.5      23.5        21.8        m12           3.6  20.0    55.3
#     m3                3.3       5.8        13.1        m3            2.4  18.2    19.0
#     m4                3.3       6.5        13.5        m4            3.0  16.2    24.0 *
#     m5                1.5       5.0        10.0        m5            2.8  12.7    21.3
#     m6                1.5       6.1         9.8        m6            3.5  18.6    18.8
#     m7                1.5      21.8         7.0        m7            3.1  13.9    20.7
#
# The whole small column is under 7 h, over 452 runs of arches / m92 /
# ngc6397 / m4 / gc2211-o023 / gc2211-o050 (mid is 1574 runs over 11 fields,
# large 1000 over 5).  Giving the small fields the large
# column's limits is the OTHER failure mode, measured on this queue: a 4-cpu
# job asking a 3-day wall can only be placed in a 3-day-wide gap, and sgrb2's
# m6-finalize -- submitted at 3-00:00:00 for a stage whose measured maximum is
# 13.7 h -- waited exactly its own walltime on an otherwise finished field.
# With 52 treasury tiles starting 2026-09-10 that cost is paid 52 times over.
#
# * m4-finalize's 24.0 h is cloudc/o002 TIMED OUT at a 24 h limit, so that one
#   value is censored and the true runtime is unmeasured; 2-00:00:00 sizes above
#   it rather than to it.  The other maxima are COMPLETED runs, i.e. real.
#
# Nothing here is shorter than the 12:00:00 it replaces.  4-00:00:00 is the
# ceiling: astronomy-dept-b (the QOS the phase script submits under) has
# MaxWall=4-00:00:00 with DenyOnLimit, so a longer request is REJECTED at submit
# time rather than queued.  An m12-finalize that outgrows four days needs a
# finer split, not a bigger number here.
#
# The SMALL tier is left at exactly the 12:00:00 it has today.  Not one of the
# 35 TIMEOUTs in the window belongs to a small field -- they are 8 sgrb2, 8
# crowded_l3, 8 brick/1182 and 7 w51 m12-fanout shards plus 4 brick/cloudc
# m12-finalizes, every one of them mid or large.  A treasury tile that turns
# out slower than arches either crosses 200 crf into the mid tier on its own or
# gets FANOUT_TIME_M12 from its runner; guessing upward for all 52 of them is
# the cost this tier exists to avoid.
_stage_time() {          # $1=phase  $2=fanout|finalize  ->  a SLURM --time
    case "$FIELD_TIER:$2:$1" in
        small:*)                          echo 12:00:00 ;;
        mid:fanout:m12|mid:fanout:m7)     echo 1-12:00:00 ;;
        mid:fanout:*)                     echo 12:00:00 ;;
        mid:finalize:*)                   echo 1-12:00:00 ;;
        large:fanout:m12|large:fanout:m7) echo 1-12:00:00 ;;
        large:fanout:*)                   echo 1-00:00:00 ;;
        large:finalize:m12)               echo 4-00:00:00 ;;
        large:finalize:m4)                echo 2-00:00:00 ;;
        large:finalize:*)                 echo 1-12:00:00 ;;
        *)                                echo 1-00:00:00 ;;
    esac
}

# OVERRIDES, per phase first.  FANOUT_TIME / FINALIZE_TIME are single knobs for
# SIX phases each, which is how the campaign's two longest fields ended up
# asking their m12 number for every stage: pipeline-runners' run_sgrb2 sets
# FINALIZE_TIME=72:00:00 -- sized in its own comment for the m12 finalize --
# and sgrb2's m3..m7 finalizes, which run 12.5-17.3 h, inherited it; that
# 3-00:00:00 m6-finalize is the backfill-exclusion case above.  run_gc2211 sets
# FINALIZE_TIME=24:00:00 the same way.
#
# So each phase now reads FANOUT_TIME_<PHASE> / FINALIZE_TIME_<PHASE> (e.g.
# FINALIZE_TIME_M12) first.  The flat spellings keep working EXACTLY as before
# -- an explicit ask is never silently shortened -- but when one is set the
# submit line says so and prints what the table would have given, so a runner
# carrying a stale blanket override is visible in its own log.
_phase_time() {          # $1=phase  $2=fanout|finalize  ->  time, then source
    local per flat table
    per=$(echo "${2}_TIME_${1}" | tr '[:lower:]' '[:upper:]')
    flat=$(echo "${2}_TIME" | tr '[:lower:]' '[:upper:]')
    table=$(_stage_time "$1" "$2")
    if [ -n "${!per:-}" ]; then
        echo "${!per}" "($per; $FIELD_TIER-field table: $table)"
    elif [ -n "${!flat:-}" ]; then
        echo "${!flat}" "($flat, ALL phases; $FIELD_TIER-field table: $table)"
    else
        echo "$table" "($FIELD_TIER field)"
    fi
}

# Phase list.  Default: full NIRCam set, with m7 only when multifilter.
read -r -a _FA <<< "$FILTERS"
if [ -z "${PHASES:-}" ]; then
    PHASES="m12 m3 m4 m5 m6"
    [ "${#_FA[@]}" -gt 1 ] && PHASES="$PHASES m7"
fi

# Where _pipe_root.sh lives.  sbatch copies the batch script to a spool
# dir, so the job cannot always find its own siblings; hand it the path.
export GC_SCRIPTS_DIR="$HERE"
COMMON="ALL,PROPOSAL=$PROPOSAL,FIELD=$FIELD,TARGET=$TARGET"
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

# ---------------------------------------------------------------------------
# Refuse a submission duplicating a phase already queued for this field.
. "$HERE/_refuse_duplicate_chain.sh"

echo "Per-frame chain: target=$TARGET $PROPOSAL/$FIELD modules=$MODULES"
echo "  phases: $PHASES   NSHARDS=$NSHARDS   filters: $FILTERS"
SB="$HERE/submit_cataloging_perframe_phase.sbatch"

for ph in $PHASES; do
    dep_arg=""; [ -n "$prev_dep" ] && dep_arg="--dependency=$prev_dep"
    read -r ph_fanout_time ph_fanout_why <<< "$(_phase_time "$ph" fanout)"
    read -r ph_finalize_time ph_finalize_why <<< "$(_phase_time "$ph" finalize)"
    A=$(sbatch --parsable $dep_arg \
        --job-name="${TARGET}${PROPOSAL}-o${FIELD}-${ph}-fanout" \
        --array=0-$((NSHARDS-1)) \
        --cpus-per-task="$FANOUT_CPUS" --mem="$FANOUT_MEM" --time="$ph_fanout_time" \
        --export="$COMMON,PHASE=$ph,MODE=fanout,PARALLEL_WORKERS=$FANOUT_CPUS" \
        "$SB")
    echo "  $ph fan-out array : $A  (0-$((NSHARDS-1)))  --time=$ph_fanout_time $ph_fanout_why${dep_arg:+  [$dep_arg]}"
    B=$(sbatch --parsable --dependency=afterok:"$A" \
        --job-name="${TARGET}${PROPOSAL}-o${FIELD}-${ph}-finalize" \
        --cpus-per-task="$FINALIZE_CPUS" --mem="$FINALIZE_MEM" --time="$ph_finalize_time" \
        --export="$COMMON,PHASE=$ph,MODE=finalize,PARALLEL_WORKERS=$FINALIZE_CPUS" \
        "$SB")
    echo "  $ph finalize      : $B  (afterok:$A)  --time=$ph_finalize_time $ph_finalize_why"
    prev_dep="afterok:$B"
done

echo "DONE.  Final phase finalize job is the last printed B; watch: squeue -u \$USER -n gc_pf"
