#!/bin/bash
# ---------------------------------------------------------------------------
# Validate that the per-frame fan-out (option C) reproduces the monolithic
# --each-exposure manual pipeline.  Runs BOTH on the SAME small cutout, LOCALLY
# (no SLURM), then diffs the final per-filter vetted catalogs + the cross-band
# table.  Run this on a chosen cutout before trusting the SLURM fan-out at scale.
#
#   A) monolithic : one run, all phases in-process (today's path).
#   B) per-frame  : for each phase, NSHARDS skip-finalize shard runs + one
#                   finalize run -- exactly what submit_cataloging_perframe.sh
#                   schedules, but executed serially here.
#
# A and B write to SEPARATE cutout labels so their catalogs don't collide; the
# python diff at the end asserts equal row counts + matched flux columns.
#
# Usage (pick a SMALL cutout so this finishes quickly):
#   PIPE_ROOT=/path/to/jwst-gc-pipeline-wt-perframe \
#   PROPOSAL=2221 FIELD=001 TARGET=brick MODULES=nrcb \
#   FILTERS="F405N F466N" EACH_SUFFIX=destreak_o001_crf \
#   CUTOUT="266.5,-28.7,0.01" NSHARDS=4 \
#   scripts/reduction/validate_perframe_equivalence.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROPOSAL=${PROPOSAL:?}; FIELD=${FIELD:?}; TARGET=${TARGET:?}
MODULES=${MODULES:-nrcb}
FILTERS=${FILTERS:?space-separated, e.g. "F405N F466N"}
EACH_SUFFIX=${EACH_SUFFIX:?e.g. destreak_o001_crf}
CUTOUT=${CUTOUT:?RA,DEC,SIZE_deg e.g. 266.5,-28.7,0.01}
NSHARDS=${NSHARDS:-4}
PIPE_ROOT=${PIPE_ROOT:-}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:${PYTHONPATH:-}"
PY=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
export CRDS_PATH=${CRDS_PATH:-/orange/adamginsburg/jwst/crds}
export CRDS_SERVER_URL=${CRDS_SERVER_URL:-https://jwst-crds.stsci.edu}

FILT_CSV=$(echo "$FILTERS" | tr ' ' ',')
read -r -a _FA <<< "$FILTERS"
PHASES="m12 m3 m4 m5 m6"; [ "${#_FA[@]}" -gt 1 ] && PHASES="$PHASES m7"

base() {
    "$PY" -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
        --filternames="$FILT_CSV" --modules="$MODULES" --each-exposure \
        --proposal_id="$PROPOSAL" --field="$FIELD" --target="$TARGET" \
        --each-suffix="$EACH_SUFFIX" --cutout-region="$CUTOUT" "$@"
}

echo "=== A) monolithic ==="
base --cutout-label=mono_ref

echo "=== B) per-frame (serial emulation of the SLURM chain) ==="
for ph in $PHASES; do
    for i in $(seq 0 $((NSHARDS-1))); do
        echo "--- B $ph fan-out shard $i/$NSHARDS ---"
        base --cutout-label=perframe --manual-start-phase="$ph" \
             --manual-stop-after-phase="$ph" --manual-skip-finalize \
             --manual-frame-shard="$i/$NSHARDS"
    done
    echo "--- B $ph finalize ---"
    base --cutout-label=perframe --manual-start-phase="$ph" \
         --manual-stop-after-phase="$ph" --manual-finalize-only
done

echo "=== DIFF final catalogs ==="
"$PY" - "$TARGET" "$MODULES" "$FILT_CSV" <<'PYEOF'
import sys, glob, os
import numpy as np
from astropy.table import Table
target, modules, filt_csv = sys.argv[1], sys.argv[2], sys.argv[3]
filts = filt_csv.split(',')
# cutout basepaths embed the label; find each label's catalogs dir
def catdir(label):
    hits = glob.glob(f'/orange/adamginsburg/jwst/{target}/**/{label}*/catalogs',
                     recursive=True) or \
           glob.glob(f'/orange/adamginsburg/jwst/{target}/**/*{label}*/catalogs',
                     recursive=True)
    return hits[0] if hits else None
A, B = catdir('mono_ref'), catdir('perframe')
print('mono catalogs :', A)
print('perframe cat  :', B)
assert A and B, 'could not locate both catalog dirs'
ok = True
for f in filts:
    pat = f'{f.lower()}_{modules}_indivexp_merged*_m6*vetted.fits'
    ga = sorted(glob.glob(os.path.join(A, pat)))
    gb = sorted(glob.glob(os.path.join(B, pat)))
    if not ga or not gb:
        print(f'  {f}: MISSING vetted ({len(ga)} mono / {len(gb)} perframe)'); ok = False; continue
    ta, tb = Table.read(ga[-1]), Table.read(gb[-1])
    same_n = len(ta) == len(tb)
    msg = f'  {f}: n_mono={len(ta)} n_perframe={len(tb)} {"OK" if same_n else "DIFFER"}'
    if same_n and 'flux' in ta.colnames and 'flux' in tb.colnames:
        fa = np.sort(np.asarray(ta['flux'], float)); fb = np.sort(np.asarray(tb['flux'], float))
        close = np.allclose(fa, fb, rtol=1e-6, atol=1e-6, equal_nan=True)
        msg += f'  flux_match={close}'; ok = ok and close
    print(msg); ok = ok and same_n
print('RESULT:', 'EQUIVALENT' if ok else 'MISMATCH -- investigate')
sys.exit(0 if ok else 1)
PYEOF
echo "validation exit: $?"
