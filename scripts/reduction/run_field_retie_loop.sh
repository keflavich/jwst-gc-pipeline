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
CONSENSUS_TBL="${BASE}/offsets/Offsets_JWST_Brick${PROPOSAL}_consensus.csv"

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
    sbatch --wait --array=0-$((NF-1)) --qos="$QOS" \
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
PROPOSAL=$PROPOSAL FIELD=$FIELD TARGET=$TARGET MODULES=$MODULES \
    EACH_SUFFIX=$EACH_SUFFIX FILTERS="$FILTERS" MAX_GROUP_SIZE=$MAX_GROUP_SIZE \
    PIPE_ROOT=$PIPE_ROOT \
    bash "$HERE/submit_cataloging_perframe.sh"
echo "Submitted full cataloging chain for $TARGET $PROPOSAL/$FIELD."
