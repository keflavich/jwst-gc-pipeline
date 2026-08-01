#!/bin/bash
#SBATCH --job-name=satstar-inject-bias
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --array=1-9
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/satstar-inject-bias_%A_%a.log

# Multi-field saturated-star injection->recovery bias run. One task per line in
# inject_manifest.tsv. F_BF (charge-migration coupling) passed via env; N_STARS
# via env. Runs from the WORKTREE (PYTHONPATH) so the worktree code is imported,
# not the main-checkout editable install (worktrees-only rule).
set -euo pipefail
WT=/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/81778923-7a53-4903-85de-4e1a21cfef0f/scratchpad/wt-satinject
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
export STPSF_PATH=/orange/adamginsburg/jwst/stpsf-data/
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export PYTHONPATH=$WT
cd "$WT"
F_BF=${F_BF:-0.3}
N_STARS=${N_STARS:-120}
OUTROOT=${OUTROOT:-/blue/adamginsburg/adamginsburg/tmp/satinject_fleet/fbf${F_BF}}

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/satstar_deblend/inject_manifest.tsv)
FLD=$(echo "$LINE"|cut -f1); FILT=$(echo "$LINE"|cut -f2); DET=$(echo "$LINE"|cut -f3)
CRF=$(echo "$LINE"|cut -f4); PSFS=$(echo "$LINE"|cut -f5)
OUT="$OUTROOT/${FLD}_${FILT}_${DET}"
echo "[inject-array $SLURM_ARRAY_TASK_ID] $FLD $FILT $DET f_bf=$F_BF n=$N_STARS"
CFG=$($PY -c "import json,sys; print(json.dumps(dict(crf=sys.argv[1],band=sys.argv[2],detector=sys.argv[3],field=sys.argv[4],psfs=sys.argv[5],out=sys.argv[6],f_bf=float(sys.argv[7]),n_stars=int(sys.argv[8]))))" "$CRF" "$FILT" "$DET" "$FLD" "$PSFS" "$OUT" "$F_BF" "$N_STARS")
$PY jwst_gc_pipeline/photometry/saturated_injection_recovery.py "$CFG"
echo "[inject-array $SLURM_ARRAY_TASK_ID] DONE $FLD $FILT $DET"
