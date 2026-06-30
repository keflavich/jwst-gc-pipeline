#!/bin/bash
# ---------------------------------------------------------------------------
# Fan the m8 forced cross-band fill out into one per-filter partial job + one
# merge, instead of the monolithic inline m8 (which sweeps every frame serially
# and can overrun the wall).  Use AFTER m7 is complete (the m7 merged catalog +
# per-frame m7 products must exist on disk).
#
#   stage A: one submit_cataloging_m8_partial.sbatch per filter (each fills only
#            its band -> ..._resbgsub_m8_partial_<FILT>.fits);
#   stage B: submit_cataloging_m8_merge.sbatch -- afterok on all of stage A,
#            column-merges the partials into ..._resbgsub_m8.fits and dedups it.
#
# Pair the m7 job with --no-forced-fill-m8 so the inline m8 is skipped and this
# split path owns the fill.
#
# Config via env (defaults = sickle):
#   TARGET PROPOSAL FIELD MODULES   -- field identity
#   FILTERS                         -- space-separated bands to fill
#   SUFFIXES                        -- space-separated each-suffix per FILTER,
#                                      SAME length+order as FILTERS (SW and LW
#                                      bands usually differ, e.g. destreak_* vs
#                                      align_*).  Defaults all to EACH_SUFFIX.
#   EACH_SUFFIX                     -- fallback suffix when SUFFIXES unset
#   PIPE_ROOT                       -- pin a non-installed checkout
#   WAIT_JOBID / DEP_TYPE           -- optional dependency on a preceding m7 job
#                                      (DEP_TYPE afterok [default] | afterany)
#
# Example (sickle, SW=destreak / LW=align):
#   TARGET=sickle PROPOSAL=3958 FIELD=007 MODULES=nrcb \
#   FILTERS="F187N F210M F335M F470N F480M" \
#   SUFFIXES="destreak_o007_crf destreak_o007_crf align_o007_crf align_o007_crf align_o007_crf" \
#   scripts/reduction/submit_cataloging_m8.sh
# ---------------------------------------------------------------------------
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET=${TARGET:-sickle}
PROPOSAL=${PROPOSAL:-3958}
FIELD=${FIELD:-007}
MODULES=${MODULES:-nrcb}
FILTERS=${FILTERS:-"F187N F210M F335M F470N F480M"}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o007_crf}
PIPE_ROOT=${PIPE_ROOT:-}
WAIT_JOBID=${WAIT_JOBID:-}
DEP_TYPE=${DEP_TYPE:-afterok}

read -r -a _F <<< "$FILTERS"
if [ -n "${SUFFIXES:-}" ]; then
    read -r -a _S <<< "$SUFFIXES"
    if [ "${#_S[@]}" -ne "${#_F[@]}" ]; then
        echo "ERROR: SUFFIXES has ${#_S[@]} entries but FILTERS has ${#_F[@]}; "\
             "they must align 1:1." >&2
        exit 2
    fi
else
    _S=(); for _ in "${_F[@]}"; do _S+=("$EACH_SUFFIX"); done
fi

DEP=""
[ -n "$WAIT_JOBID" ] && DEP="--dependency=${DEP_TYPE}:$WAIT_JOBID"
echo "target=$TARGET ${PROPOSAL}/${FIELD} modules=$MODULES wait=${WAIT_JOBID:-none} (${DEP_TYPE})"

export TARGET PROPOSAL FIELD MODULES PIPE_ROOT
PARTIAL_IDS=()
for i in "${!_F[@]}"; do
    F="${_F[$i]}"; S="${_S[$i]}"
    JID=$(sbatch --parsable $DEP --export=ALL \
          "$HERE/submit_cataloging_m8_partial.sbatch" "$F" "$S")
    echo "  submitted m8 partial $F (suffix=$S) -> $JID"
    PARTIAL_IDS+=("$JID")
done

DEPLIST=$(IFS=:; echo "${PARTIAL_IDS[*]}")
export FILTERS
MID=$(sbatch --parsable --dependency=afterok:"$DEPLIST" --export=ALL \
      "$HERE/submit_cataloging_m8_merge.sbatch")
echo "  submitted m8 merge -> $MID (afterok:$DEPLIST)"
echo "DONE: ${#_F[@]} partials + merge armed."
