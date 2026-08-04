#!/bin/bash
# ---------------------------------------------------------------------------
# Automated reduce -> catalog-to-m2 -> re-tie loop (Fix C, 2026-07-16).
#
# Closes the astrometry checkpoint the hands-off way for tweakreg/fix_alignment
# fields (sgrc, cloudef, ... -- anything outside the brick VIRAC2locked path):
#
#   iter i:
#     1. REDUCE (Image3 + fix_alignment).  On iter>=2 fix_alignment applies the
#        consensus offsets table seeded by the previous iter's m2 checkpoint.
#     2. CATALOG TO m2 ONLY (PHASES="m12") with ASTROM_CHECKPOINT_APPLY=1.
#        - checkpoint PASSES (no exposure > 2 mas off consensus)  -> converged.
#        - checkpoint measures misalignment -> it SEEDS/updates the consensus
#          offsets table (seed_offsets_table_from_consensus / update_offsets_
#          table), stale-tags the im0 mosaics, and the finalize exits non-zero.
#     3. Converged?  break.  Else re-reduce (the applied table removes the
#        per-exposure jitter) and repeat, capped at MAXITER.
#   after convergence: run the FULL cataloging chain (m3..m7).
#
# The m2 checkpoint stays a hard gate throughout -- this loop does NOT demote it
# (never sets ASTROM_CHECKPOINT_WARN_ONLY); it makes the SANCTIONED remediation
# (seed table -> re-reduce -> re-catalog) automatic instead of manual.
#
# Usage:
#   PROPOSAL=4147 FIELD=012 TARGET=sgrc FILTERS="F115W F162M F182M F212N F360M F405N F470N F480M" \
#     PIPE_ROOT=/orange/adamginsburg/repos/jwst-gc-pipeline \
#     scripts/reduction/run_field_retie_loop.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROPOSAL=${PROPOSAL:?set PROPOSAL}
FIELD=${FIELD:?set FIELD}
TARGET=${TARGET:?set TARGET}
FILTERS=${FILTERS:?set FILTERS}
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o${FIELD}_crf}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}
PIPE_ROOT=${PIPE_ROOT:-}
MAXITER=${MAXITER:-3}
QOS=${QOS:-astronomy-dept-b}
BASE=${BASE:-/orange/adamginsburg/jwst/${TARGET}}
# Actionability floor for the m2 checkpoint (see cataloging.py ~L3064): per-detector
# residuals of a few mas are SIAF/DVA-class systematics the module-locked/consensus
# offsets table cannot express, so correcting on their detector means never converges.
# Setting this ABOVE the residual scatter lets the loop stop on a sub-floor PASS while
# still measuring+recording every residual. Default 0 = strict 2 mas (unchanged).
ASTROM_M2_CORRECTION_FLOOR_MAS=${ASTROM_M2_CORRECTION_FLOOR_MAS:-0}
export ASTROM_M2_CORRECTION_FLOOR_MAS
# Which offsets table does m2 REWRITE for this field?  The before/after md5sum
# check below is only meaningful against that one file, and the answer depends on
# the field's CONFIGURED CHANNEL -- not on which tables happen to exist.
#
# Guessing has failed twice.  A hardcoded "_consensus.csv" missed every locked
# field (sgrc, cloudc, sgrb2, quintuplet, gc2211).  A locked-before-consensus
# preference order then missed cloudef obs002, whose bulk is a recorded constant
# but whose per-exposure jitter the checkpoint writes to "_consensus.csv" -- a
# stale "_VIRAC2locked.csv" sat beside it and won the preference.  Both produced
# the same silent symptom: watch a file nobody writes, see no change, stop with
# "this is NOT a checkpoint re-tie" while the correction had in fact been made.
#
# alignment_config.offsets_table_path is the single source of truth, and it is
# what cataloging's own _astrom_offsets_channel delegates to.  OFFSETS_TBL
# overrides it for a field that uses some third table.
CONSENSUS_TBL="${OFFSETS_TBL:-}"
if [ -z "$CONSENSUS_TBL" ]; then
    CONSENSUS_TBL=$(PYTHONPATH="${PIPE_ROOT:-}:${PYTHONPATH:-}" python -c "
from jwst_gc_pipeline.reduction.alignment_config import offsets_table_path
print(offsets_table_path('$BASE', '$PROPOSAL', '$FIELD'))" 2>/dev/null)
fi
if [ -z "$CONSENSUS_TBL" ]; then
    echo "REFUSING: alignment_config declares no table-driven correction channel for"
    echo "  proposal=$PROPOSAL field=$FIELD.  The m2 checkpoint cannot record a"
    echo "  correction for this field, so a re-tie loop can never converge."
    echo "  Add an entry to jwst_gc_pipeline/reduction/alignment_config.py, or set"
    echo "  OFFSETS_TBL explicitly if it uses some other table."
    exit 2
fi
echo "offsets table watched for changes: $CONSENSUS_TBL"

read -r -a _FA <<< "$FILTERS"
NF=${#_FA[@]}
export_common="ALL,PROPOSAL=$PROPOSAL,FIELD=$FIELD,TARGET=$TARGET,FILTERS=$FILTERS"
[ -n "$PIPE_ROOT" ] && export_common="$export_common,PIPE_ROOT=$PIPE_ROOT"

# Poll a SLURM job to a terminal state; echo COMPLETED/FAILED/...
wait_job () {
    local jid="$1"
    while :; do
        local st
        st=$(sacct -j "$jid" --format=State -Pn 2>/dev/null | head -1 | tr -d ' ')
        case "$st" in
            ''|RUNNING|PENDING|REQUEUED|RESIZING|SUSPENDED|COMPLETING) sleep 30;;
            *) echo "$st"; return 0;;
        esac
    done
}

for ((it=1; it<=MAXITER; it++)); do
    echo "=================  RE-TIE ITER $it / $MAXITER  ($TARGET $PROPOSAL/$FIELD)  ================="

    # --- 1. reduce (blocks until the whole array finishes) ---
    echo "[iter $it] reducing (fix_alignment applies consensus table if present: $([ -f "$CONSENSUS_TBL" ] && echo yes || echo no))"
    # --job-name at SUBMIT time (standing rule, CLAUDE.md): the in-script runtime
    # rename only fires when the job STARTS, and a quota-bound retie sits PENDING
    # for hours under the generic name -- which is exactly when the queue is being
    # watched, and when several reduce arrays are in flight at once.
    sbatch --wait --array=0-$((NF-1)) --qos="$QOS" \
        --job-name="${TARGET}${PROPOSAL}-o${FIELD}-reduce-retie${it}" \
        --export="${export_common}" \
        "$HERE/submit_reduction.sbatch"

    # --- 2. catalog to m2 only, with auto-apply ON ---
    echo "[iter $it] cataloging to m2 (ASTROM_CHECKPOINT_APPLY=1)"
    tbl_before=$( [ -f "$CONSENSUS_TBL" ] && md5sum "$CONSENSUS_TBL" | cut -d' ' -f1 || echo none )
    # submit_cataloging_perframe.sh self-sbatches the chain; PHASES="m12" stops at
    # the m2 merge+checkpoint.  Capture the finalize (stage B) job id it prints.
    export ASTROM_CHECKPOINT_APPLY=1
    chain_out=$(PROPOSAL=$PROPOSAL FIELD=$FIELD TARGET=$TARGET MODULES=$MODULES \
        EACH_SUFFIX=$EACH_SUFFIX FILTERS="$FILTERS" MAX_GROUP_SIZE=$MAX_GROUP_SIZE \
        PHASES="m12" PIPE_ROOT=$PIPE_ROOT \
        ASTROM_M2_CORRECTION_FLOOR_MAS=$ASTROM_M2_CORRECTION_FLOOR_MAS \
        bash "$HERE/submit_cataloging_perframe.sh")
    echo "$chain_out"
    fin_jid=$(echo "$chain_out" | grep -oE 'finalize[^0-9]*[0-9]+' | grep -oE '[0-9]+' | tail -1)
    if [ -z "$fin_jid" ]; then
        fin_jid=$(echo "$chain_out" | grep -oE '[0-9]{6,}' | tail -1)
    fi
    echo "[iter $it] waiting on m2 finalize job $fin_jid"
    st=$(wait_job "$fin_jid")
    echo "[iter $it] m2 finalize state: $st"

    # --- 3. converged? ---
    tbl_after=$( [ -f "$CONSENSUS_TBL" ] && md5sum "$CONSENSUS_TBL" | cut -d' ' -f1 || echo none )
    if [ "$st" = "COMPLETED" ]; then
        echo "[iter $it] m2 checkpoint PASSED -- converged after $it iter(s)."
        break
    fi
    if [ "$tbl_after" = "$tbl_before" ]; then
        echo "[iter $it] m2 finalize failed ($st) but the consensus table did NOT change."
        echo "           This is NOT a checkpoint re-tie (some other failure) -- STOPPING."
        echo "           Inspect logs/catalog_pf_${fin_jid}*.out before retrying."
        exit 1
    fi
    echo "[iter $it] consensus table updated -> re-reduce + re-catalog."
    if [ "$it" -eq "$MAXITER" ]; then
        echo "REACHED MAXITER=$MAXITER without the checkpoint passing."
        echo "The per-exposure scatter is not closing via the consensus tie alone;"
        echo "inspect ${CONSENSUS_TBL} and the m2 checkpoint records under"
        echo "${BASE}/astrometry_checkpoints/ (residual centroiding/distortion?)."
        exit 2
    fi
done

# --- final: full cataloging (m3..m7) now that im0 is self-consistent ---
echo "=================  FULL CATALOGING (m3..m7)  ================="
unset ASTROM_CHECKPOINT_APPLY   # m3+ must be a FROZEN solution; no more corrections
# Named explicitly rather than left to inheritance.  The chain re-runs m12 and m2
# runs inside it, so this value decides whether the checkpoint raises -- and a
# variable that changes a gate's verdict should be visible at the call site
# rather than reaching the job by a route the reader has to reconstruct.  It also
# makes this invocation symmetric with the m12 one above.
#
# NOT a fix for a propagation failure: the value already arrives.
# submit_cataloging_perframe.sh's --export list begins with ALL, so the job
# inherits the submitting environment plus the named variables, and `export` at
# the top of this script is sufficient on its own.  Verified directly:
#
#   $ export ASTROM_M2_CORRECTION_FLOOR_MAS=4.0
#   $ sbatch --export="ALL,PROPOSAL=p,FIELD=f" --wrap='echo "FLOOR=[$ASTROM_M2_CORRECTION_FLOOR_MAS]"'
#   FLOOR=[4.0] PROPOSAL=[p]
#
# cloudef obs005's m3..m7 chain died here on 2026-08-04, and this is NOT why --
# see #281.  Do not treat this line as having closed that.
PROPOSAL=$PROPOSAL FIELD=$FIELD TARGET=$TARGET MODULES=$MODULES \
    EACH_SUFFIX=$EACH_SUFFIX FILTERS="$FILTERS" MAX_GROUP_SIZE=$MAX_GROUP_SIZE \
    PIPE_ROOT=$PIPE_ROOT \
    ASTROM_M2_CORRECTION_FLOOR_MAS=$ASTROM_M2_CORRECTION_FLOOR_MAS \
    bash "$HERE/submit_cataloging_perframe.sh"
echo "Submitted full cataloging chain for $TARGET $PROPOSAL/$FIELD."
