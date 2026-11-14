#!/usr/bin/env python
"""Build the per-detector fixed-pattern (hot/cosmetic defect) group-0-core
position registry consumed by
``jwst_gc_pipeline.reduction.satstar_deblend_gate.load_defect_registry``.

Provenance: generated once (2026-10-15) from data-qa's read-only, one-off
10678 diagnostic scan of every DQ-SATURATED blob in all 4040 crf frames
(``scan_satblend.py`` in the data-qa ``satblend`` investigation scratch dir --
that scan is NOT a permanent pipeline artifact and the directory it lives in
is a Claude session scratchpad, not a stable path; this script documents how
the shipped registry (``jwst_gc_pipeline/reduction/data/satstar_g0_defect_registry.json``)
was derived and lets it be regenerated against a fresh scan, not that the
original scan directory is guaranteed to still exist).

A detector hot/cosmetic defect saturates at the SAME detector (x, y) every
frame regardless of dither, so a group-0-GROUPDQ-saturated CORE position
(the deblended ``centers`` entries in the scan's per-frame table -- finer
grained than data-qa's own ``g0_clean.py``, which flags at the whole-COMPONENT
level) that recurs at the same 4-px detector bin across >=MIN_FRAMES distinct
frames is flagged as a fixed-pattern defect rather than a real star, and is
excluded from ``satstar_deblend_gate.group0_core_positions``.

usage: build_satstar_g0_defect_registry.py <scan_dir> [<out.json>]
"""
import sys
import glob
import json
import os
import collections

import numpy as np
from astropy.table import Table

BIN_PX = 4
MIN_FRAMES = 5

DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), '..', '..',
    'jwst_gc_pipeline', 'reduction', 'data', 'satstar_g0_defect_registry.json')


def build(scan_dir, out_path):
    files = sorted(glob.glob(os.path.join(scan_dir, '*_satblend.fits')))
    print(f'{len(files)} scan files', flush=True)

    # (detector, ybin, xbin) -> set of frame basenames that produced a core there
    seen = collections.defaultdict(set)
    for i, fn in enumerate(files):
        try:
            t = Table.read(fn)
        except OSError:
            continue
        if len(t) == 0 or 'centers' not in t.colnames:
            continue
        base = os.path.basename(fn)
        parts = [p for p in base.split('_') if p.startswith('nrc')]
        if not parts:
            continue
        det = parts[0]
        frame = base.replace('_satblend.fits', '')
        for row in t:
            try:
                cens = json.loads(row['centers'])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            for (cy, cx) in cens:
                if not (np.isfinite(cy) and np.isfinite(cx)):
                    continue
                ybin = int(np.floor(cy / BIN_PX))
                xbin = int(np.floor(cx / BIN_PX))
                seen[(det, ybin, xbin)].add(frame)
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(files)} files', flush=True)

    registry = collections.defaultdict(list)
    for (det, ybin, xbin), frames in seen.items():
        if len(frames) >= MIN_FRAMES:
            registry[det].append([ybin, xbin])
    for det in sorted(registry):
        print(f'  {det}: {len(registry[det])} flagged bins', flush=True)

    with open(out_path, 'w') as fh:
        json.dump({det: sorted(bins) for det, bins in registry.items()}, fh)
    print(f'wrote {out_path}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    scan_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    build(scan_dir, out_path)
