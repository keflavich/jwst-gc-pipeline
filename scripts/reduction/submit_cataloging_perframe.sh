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
# PER-FILTER FINALIZE (opt-in) -- PER_FILTER_FINALIZE=1.
#
# The finalize is ONE job that walks every (filter x module) combo serially, so
# a big field's barrier is the sum of its filters: sgrb2 5365/001's m3 finalize
# measured 43.2 h over 33 combos, of which F187N alone was 13.1 h.  With the
# flag set, m3-m6 submit ONE finalize PER FILTER instead, all of them afterok on
# the same fan-out, and the next phase's fan-out waits on afterok of ALL of them.
# The phase barrier is unchanged -- it is now a list dependency rather than a
# single job id -- and so is the science: nothing in m3-m6 reads another
# filter's products (the cross-band merge and the cross-filter astrometry gate
# are gated to m7).
#
# Each per-filter finalize is SIZED for its own filter (--mem and, when the
# wall clock comes from the table rather than an explicit override, --time),
# from that filter's crf count.  The largest filter keeps the whole-field
# request; the small ones stop asking for it.
#
# A phase's per-filter finalizes are submitted --hold and released together, so
# a phase is never left half-finalized: a failed sbatch mid-loop cancels the
# held jobs and the phase's fan-out and prints the command that resumes the
# chain.
#
# m7 is NEVER split and neither is m12: m7 IS the cross-band merge, and a
# single-filter run does not even build an m7 phase; m12 is the CORRECTING
# astrometry checkpoint, whose whole job on a real misalignment is to STOP the
# run -- and one job raising cannot stop the other ten.  Asking for either is
# refused rather than quietly ignored.
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
# Whether the operator NAMED the finalize memory.  An explicit ask is never
# silently changed -- the same rule the wall-clock overrides below follow -- so
# the per-filter sizing further down leaves it exactly as given.
FINALIZE_MEM_EXPLICIT=0
[ -n "${FINALIZE_MEM:-}" ] && FINALIZE_MEM_EXPLICIT=1
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

# PER-FILTER FINALIZE (opt-in).  See the header.  OFF by default: sgrb2 is
# mid-chain and the 10678 tiles are landing, so a run that does not ask for the
# split submits exactly the jobs it submits today.
PER_FILTER_FINALIZE=${PER_FILTER_FINALIZE:-0}

# The phases whose finalize MAY be split, and the ones that may never be.  This
# is a fixed list rather than a free-form knob because the two exclusions are
# correctness, not taste:
#
#   m7   IS the cross-band merge (cataloging._do_crossband) and the cross-filter
#        astrometry anchor gate, both of which read every filter's m6 vetted
#        catalog.  A single-filter run does not even build an m7 phase --
#        `phases` gets m7 only when len(filternames) > 1 -- so a split m7 dies
#        with "--manual-start-phase='m7' not in phases ['m12','m3','m4','m5','m6']".
#
#   m12  runs the CORRECTING astrometry checkpoint (stage in CORRECTION_STAGES),
#        and a correcting checkpoint's verdict is to STOP THE FIELD: it corrects
#        the shared offsets table, renames the first-pass mosaics to
#        `*_i2d_im0_badastrom.fits` and raises AstrometryCorrectionRequiredError
#        saying "the current crf frames/catalogs are stale".  ONE process is
#        what that raise ends.  Whole, the m12 finalize stops the field the
#        instant any filter measures a misalignment; split, the stop reaches
#        only the filter that raised it and the other ten keep fitting against
#        frames the checkpoint has just declared stale.
#
#        That is not hypothetical: sgrb2's eleven per-filter m12 finalizes of
#        2026-08-22 (39933194-39933204) are the measurement.  Seven corrected
#        Offsets_JWST_Brick5365_VIRAC2locked.csv and stale-tagged im0 between
#        12:04:08 and 12:36:41 UTC; the other four carried on and COMPLETED --
#        F210M at 07:27, F150W at 09:35, F182M at 09:46 and F187N at 23:11,
#        i.e. ~15 h after the first stop signal -- writing m12 products into a
#        field the checkpoint had already quarantined.
#
#        (The offsets CSV itself is NOT the reason: update_offsets_table has
#        wrapped its whole read-modify-write in `with locked(offsets_path)`
#        since 9f73c05, 2026-08-02, three weeks BEFORE that run, so those seven
#        writes were serialised.  The ledger those renames append to,
#        astrometry_checkpoint.mark_i2d_stale's stale_i2d_renames.json, is the
#        one shared write with no lock on it -- it survived the seven-way
#        append, which is luck rather than a guarantee, and it is the file a
#        bad run is undone from.)
#
# m3-m6 have neither property: every path the per-phase body touches is keyed
# (module, filter) -- bg_for_next, resid_i2d_for_next, prev_merged_for,
# satstar_overrides, the merged-catalog and data-i2d paths -- the marker verify
# and the module-coverage check are both scoped to the run's own filters, and
# the astrometry checkpoint at a FROZEN stage records to a per-filter file
# (checkpoint_<stage>_<filter><obs>_latest.json, written via a per-filter tmp
# name) and corrects nothing.
_SPLITTABLE_PHASES="m3 m4 m5 m6"

PER_FILTER_FINALIZE_PHASES=${PER_FILTER_FINALIZE_PHASES:-$_SPLITTABLE_PHASES}

_in_list() {             # $1 = needle, $2 = space-separated haystack
    case " $2 " in *" $1 "*) return 0;; *) return 1;; esac
}

if [ "$PER_FILTER_FINALIZE" = "1" ]; then
    for _ph in $PER_FILTER_FINALIZE_PHASES; do
        if ! _in_list "$_ph" "$_SPLITTABLE_PHASES"; then
            echo "REFUSING: PER_FILTER_FINALIZE_PHASES names '$_ph', which cannot" >&2
            echo "  be split per filter.  Splittable: $_SPLITTABLE_PHASES" >&2
            echo "  m7 is the cross-band merge (a one-filter run has no m7 phase at" >&2
            echo "  all); m12 is the CORRECTING astrometry checkpoint, whose verdict on" >&2
            echo "  a misalignment is to STOP THE FIELD -- correct the shared offsets" >&2
            echo "  table, stale-tag im0, raise.  That raise ends ONE process, so split" >&2
            echo "  it stops one filter while the other ten keep fitting frames it has" >&2
            echo "  just declared stale (sgrb2 2026-08-22: seven corrected and stopped," >&2
            echo "  four ran on for up to 15 h and COMPLETED)." >&2
            exit 4
        fi
    done
fi

_split_finalize() {      # $1 = phase -> 0 when this phase's finalize is split
    [ "$PER_FILTER_FINALIZE" = "1" ] || return 1
    _in_list "$1" "$PER_FILTER_FINALIZE_PHASES"
}

# ---------------------------------------------------------------------------
# PER-FILTER RESOURCE SIZING (split finalizes only).
#
# A split finalize does ONE filter's work, and it must not keep asking for the
# whole field's slice.  Measured why, on sgrb2 5365/001:
#
#   memory does not pool.  The finalize loop is `for module: for filt:` and the
#   state it carries between filters is paths and a SkyCoord per combo, so the
#   peak is the biggest COMBO, not the sum.  The whole 11-filter m3 finalize
#   (41424864) recorded MaxRSS 148 GiB and the single-filter F187N m12 finalize
#   (39933196) recorded 151 GiB -- the same number, because F187N IS the peak.
#   So the largest filter needs the field's memory and the other ten do not.
#
#   time is very nearly linear in frames.  F187N has 384 crf to F182M's 192 and
#   took 13.09 h to its 6.79 h in the m3 finalize (1.93x on 2.00x the frames).
#
# So both are scaled by the filter's own crf count over the largest filter's:
# the biggest filter keeps exactly the request the pooled job used, and a
# filter with a quarter of the frames asks for about a quarter.  Floors keep a
# small filter above the hand-launch floor in the phase sbatch
# (--mem=64gb --time=12:00:00), which is the smallest slice anything here has
# ever been run at, and the field value is a ceiling in both directions.
#
# What the operator NAMED is never scaled: an explicit FINALIZE_MEM, or an
# explicit FINALIZE_TIME / FINALIZE_TIME_<PHASE>, reaches every split job
# verbatim.  With no counts on disk (CI, a fresh checkout) every filter gets
# the field value, which is what it gets today.
PER_FILTER_MEM_FLOOR_GB=${PER_FILTER_MEM_FLOOR_GB:-64}
PER_FILTER_TIME_FLOOR=${PER_FILTER_TIME_FLOOR:-12:00:00}

_filter_crf_count() {    # $1 = filter -> its crf count under the tier's dir
    { ls "$_crf_dir/$1"/pipeline/*crf.fits 2>/dev/null || true; } | wc -l
}

_mem_gb() {              # "256gb" -> 256
    local v="${1//[!0-9]/}"
    echo "${v:-0}"
}

_time_to_min() {         # "1-12:00:00" / "12:00:00" -> minutes, rounding up
    local t="$1" d=0 hms h m sec
    case "$t" in *-*) d="${t%%-*}"; hms="${t#*-}";; *) hms="$t";; esac
    IFS=: read -r h m sec <<< "$hms"
    echo $(( 10#${d:-0} * 1440 + 10#${h:-0} * 60 + 10#${m:-0} \
             + $(( 10#${sec:-0} > 0 ? 1 : 0 )) ))
}

_min_to_time() {         # minutes -> "HH:MM:00" / "D-HH:MM:00"
    if [ "$1" -lt 1440 ]; then
        printf '%02d:%02d:00' $(( $1 / 60 )) $(( $1 % 60 ))
    else
        printf '%d-%02d:%02d:00' $(( $1 / 1440 )) $(( ($1 % 1440) / 60 )) \
               $(( $1 % 60 ))
    fi
}

_scaled_mem() {          # $1 = this filter's crf count -> a --mem value
    local n="$1" full gb floor
    full=$(_mem_gb "$FINALIZE_MEM")
    if [ "$FINALIZE_MEM_EXPLICIT" = "1" ] || [ "$_PF_MAX_CRF" -le 0 ] \
       || [ "$n" -le 0 ] || [ "$full" -le 0 ]; then
        echo "$FINALIZE_MEM"; return
    fi
    gb=$(( (full * n + _PF_MAX_CRF - 1) / _PF_MAX_CRF ))
    floor=$PER_FILTER_MEM_FLOOR_GB
    [ "$floor" -gt "$full" ] && floor=$full
    [ "$gb" -lt "$floor" ] && gb=$floor
    [ "$gb" -gt "$full" ] && gb=$full
    echo "${gb}gb"
}

_scaled_time() {         # $1 = crf count, $2 = the phase's finalize --time
    local n="$1" whole="$2" full mins floor
    if _time_is_explicit "$_PF_PHASE" || [ "$_PF_MAX_CRF" -le 0 ] \
       || [ "$n" -le 0 ]; then
        echo "$whole"; return
    fi
    full=$(_time_to_min "$whole")
    mins=$(( (full * n + _PF_MAX_CRF - 1) / _PF_MAX_CRF ))
    floor=$(_time_to_min "$PER_FILTER_TIME_FLOOR")
    [ "$floor" -gt "$full" ] && floor=$full
    [ "$mins" -lt "$floor" ] && mins=$floor
    [ "$mins" -gt "$full" ] && mins=$full
    _min_to_time "$mins"
}

_time_is_explicit() {    # $1 = phase -> 0 when the operator NAMED this time
    local per
    per=$(echo "FINALIZE_TIME_${1}" | tr '[:lower:]' '[:upper:]')
    [ -n "${!per:-}" ] || [ -n "${FINALIZE_TIME:-}" ]
}

# Per-filter crf counts, measured once.  Only when the split is on: with the
# flag off this script must touch the disk exactly as it does today.
_PF_MAX_CRF=0
declare -A _PF_CRF=()
if [ "$PER_FILTER_FINALIZE" = "1" ]; then
    for _f in "${_FA[@]}"; do
        _PF_CRF[$_f]=$(_filter_crf_count "$_f")
        [ "${_PF_CRF[$_f]}" -gt "$_PF_MAX_CRF" ] && _PF_MAX_CRF=${_PF_CRF[$_f]}
    done
    if [ "$_PF_MAX_CRF" -le 0 ]; then
        echo "per-filter sizing: NO crf counts under $_crf_dir -- every split" \
             "finalize keeps the whole-field --mem/--time (not a measurement)"
    fi
fi

# Where _pipe_root.sh lives.  sbatch copies the batch script to a spool
# dir, so the job cannot always find its own siblings; hand it the path.
export GC_SCRIPTS_DIR="$HERE"
COMMON="ALL,PROPOSAL=$PROPOSAL,FIELD=$FIELD,TARGET=$TARGET"
COMMON="$COMMON,EACH_SUFFIX=$EACH_SUFFIX,MAX_GROUP_SIZE=$MAX_GROUP_SIZE,NSHARDS=$NSHARDS"
# FILTERS is NOT in COMMON: a split finalize overrides it with its own single
# filter, and two FILTERS= entries in one --export list is not a documented
# precedence.  Every submit site below appends the one it wants.
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
if [ "$PER_FILTER_FINALIZE" = "1" ]; then
    echo "  per-filter finalize: ON for phases [$PER_FILTER_FINALIZE_PHASES]" \
         "-- ${#_FA[@]} finalize jobs each; every other phase stays whole"
fi
SB="$HERE/submit_cataloging_perframe_phase.sbatch"

# ---------------------------------------------------------------------------
# A PHASE IS SUBMITTED ALL-OR-NOTHING.
#
# This script runs under `set -euo pipefail`, so a failed `sbatch` aborts it
# where it stands.  Whole, that leaves at worst a fan-out with no finalize.
# Split, it leaves a state that did not exist before: phase p with SOME of its
# filters finalizing and the rest never submitted, nothing queued behind them,
# and every job that WAS submitted reaching COMPLETED -- a field that looks
# finished and is missing seven filters' barriers.  Nothing downstream can see
# that, because a finalize's strict marker verify only covers the filters its
# own job was given.
#
# So the per-filter finalizes of a phase are submitted `--hold` and released in
# ONE `scontrol release` once all of them exist.  Until that release none of
# them can start, so an abort mid-loop can still undo the whole phase: the EXIT
# trap cancels the held finalizes AND the phase's own fan-out (neither has run),
# and prints the command that resumes the chain from the last COMPLETE phase --
# including the barrier dependency, which is otherwise printed and then lost.
#
# Being killed outright (SIGKILL) skips the trap and leaves the phase's
# finalizes HELD.  Held jobs do not run, so that state is a stalled chain the
# queue shows as JobHeldUser -- recoverable with `scontrol release`, and not a
# half-finalized phase.
_submitted=""        # every id this invocation submitted, in order
_held=""             # the current phase's finalizes: submitted, not yet released
_held_phase=""
_held_fanout=""
_resume_dep=""       # what the current phase was going to wait on
_resume_phases=""    # the current phase and everything after it

_recover() {
    local rc=$?
    trap - EXIT
    [ "$rc" -eq 0 ] && exit 0
    [ -z "$_submitted" ] && exit "$rc"
    {
        echo ""
        echo "SUBMIT ABORTED (rc=$rc).  Jobs submitted by this run: $_submitted"
        if [ -n "$_held" ]; then
            echo "  Phase $_held_phase was left PART-SUBMITTED: its per-filter"
            echo "  finalizes are held and were never released, so none of them ran."
            echo "  Cancelling them and phase $_held_phase's fan-out, so no phase is"
            echo "  left half-finalized:"
            if scancel $_held $_held_fanout; then
                echo "    scancel $_held $_held_fanout -- done"
            else
                echo "    scancel FAILED.  Cancel by hand, BEFORE resubmitting:"
                echo "      scancel $_held $_held_fanout"
            fi
        fi
        echo "  Everything before phase ${_resume_phases%% *} is queued and intact."
        echo "  Resume the rest of the chain with:"
        echo "    DEP='$_resume_dep' PHASES='$_resume_phases' \\"
        echo "      <the same environment as this run> $0"
    } >&2
    exit "$rc"
}
trap _recover EXIT

_remaining="$PHASES"
for ph in $PHASES; do
    _resume_dep="$prev_dep"
    _resume_phases="$_remaining"
    _remaining="${_remaining#"$ph"}"; _remaining="${_remaining# }"
    _PF_PHASE="$ph"
    dep_arg=""; [ -n "$prev_dep" ] && dep_arg="--dependency=$prev_dep"
    read -r ph_fanout_time ph_fanout_why <<< "$(_phase_time "$ph" fanout)"
    read -r ph_finalize_time ph_finalize_why <<< "$(_phase_time "$ph" finalize)"
    A=$(sbatch --parsable $dep_arg \
        --job-name="${TARGET}${PROPOSAL}-o${FIELD}-${ph}-fanout" \
        --array=0-$((NSHARDS-1)) \
        --cpus-per-task="$FANOUT_CPUS" --mem="$FANOUT_MEM" --time="$ph_fanout_time" \
        --export="$COMMON,FILTERS=$FILTERS,PHASE=$ph,MODE=fanout,PARALLEL_WORKERS=$FANOUT_CPUS" \
        "$SB")
    _submitted="${_submitted:+$_submitted }$A"
    echo "  $ph fan-out array : $A  (0-$((NSHARDS-1)))  --time=$ph_fanout_time $ph_fanout_why${dep_arg:+  [$dep_arg]}"

    # The fan-out is NOT split: one array already covers every frame of every
    # filter, and its shard predicate needs the whole exposure list to partition.
    if _split_finalize "$ph"; then
        # One finalize per filter, all afterok on the SAME fan-out, and the
        # PHASE BARRIER is the list of every one of them.  Dropping any id from
        # this dependency would let phase p+1 start reading a filter whose
        # barrier had not run -- a stale merged catalog and residual/bg seed for
        # that filter, silently, with no missing-marker crash to catch it,
        # because the marker verify only covers the filters a job was given.
        fin_ids=""
        _held=""; _held_phase="$ph"; _held_fanout="$A"
        for filt in "${_FA[@]}"; do
            _n=${_PF_CRF[$filt]:-0}
            _mem=$(_scaled_mem "$_n")
            _time=$(_scaled_time "$_n" "$ph_finalize_time")
            _why="$ph_finalize_why"
            if [ "$_mem" != "$FINALIZE_MEM" ] || [ "$_time" != "$ph_finalize_time" ]; then
                _why="(${_n}/${_PF_MAX_CRF} crf; whole-field finalize asks --mem=$FINALIZE_MEM --time=$ph_finalize_time)"
            fi
            B=$(sbatch --parsable --hold --dependency=afterok:"$A" \
                --job-name="${TARGET}${PROPOSAL}-o${FIELD}-${ph}-finalize-${filt}" \
                --cpus-per-task="$FINALIZE_CPUS" --mem="$_mem" --time="$_time" \
                --export="$COMMON,FILTERS=$filt,PHASE=$ph,MODE=finalize,PARALLEL_WORKERS=$FINALIZE_CPUS" \
                "$SB")
            _submitted="${_submitted:+$_submitted }$B"
            _held="${_held:+$_held }$B"
            fin_ids="${fin_ids}:$B"
            echo "  $ph finalize $filt : $B  (afterok:$A, held)  --mem=$_mem --time=$_time $_why"
        done
        # Every one of them exists -> release them together.  Before this line
        # the phase can still be undone; after it the barrier is whole.
        if ! scontrol release "$(echo "$_held" | tr ' ' ',')"; then
            echo "REFUSING to continue: phase $ph's ${#_FA[@]} per-filter" >&2
            echo "  finalizes were submitted but could NOT be released." >&2
            exit 5
        fi
        _held=""; _held_phase=""; _held_fanout=""
        prev_dep="afterok${fin_ids}"
        echo "  $ph barrier       : ${#_FA[@]} finalize jobs released -> next phase waits on $prev_dep"
    else
        B=$(sbatch --parsable --dependency=afterok:"$A" \
            --job-name="${TARGET}${PROPOSAL}-o${FIELD}-${ph}-finalize" \
            --cpus-per-task="$FINALIZE_CPUS" --mem="$FINALIZE_MEM" --time="$ph_finalize_time" \
            --export="$COMMON,FILTERS=$FILTERS,PHASE=$ph,MODE=finalize,PARALLEL_WORKERS=$FINALIZE_CPUS" \
            "$SB")
        _submitted="${_submitted:+$_submitted }$B"
        echo "  $ph finalize      : $B  (afterok:$A)  --time=$ph_finalize_time $ph_finalize_why"
        prev_dep="afterok:$B"
    fi
done

echo "DONE.  Final phase finalize job is the last printed B; watch: squeue -u \$USER -n gc_pf"
# The split's barrier is a LIST of ids, printed once as it is built and then
# gone; a recovery that has to re-point the chain needs it back.  Only on the
# split path, so a run with the flag off prints exactly what it prints today.
# An `if`, not `cond && echo`: this is the LAST statement, so a false `cond &&`
# list makes the script exit 1 -- which the EXIT trap would then report as an
# aborted submission on a run that submitted the whole chain.
if [ "$PER_FILTER_FINALIZE" = "1" ]; then
    echo "  The chain ends on $prev_dep -- pass that as DEP= to hang more phases off it."
fi
