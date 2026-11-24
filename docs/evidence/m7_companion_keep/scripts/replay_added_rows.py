"""Rows a one-frame m7 replay with the exemption seed adds to the frame catalog, and the residual at each.

usage: python replay_added_rows.py CONTROL_DIR TREATMENT_DIR [--match 1.0] [--r 2] [--json OUT.json]

CONTROL_DIR and TREATMENT_DIR are the replay band directories (they hold ``*_m7_daophot_basic.fits``, the frame catalog, and ``pipeline/`` with the
residual and model images).  A treatment row is "added" when no control row lies within --match pixels of its fitted position; a control row is
"lost" when no treatment row lies within --match pixels.  For every added row, the residual image (MJy/sr) in the (2R+1) x (2R+1) pixel window
at the row is reported for both runs: the control peak (what the frame left there), the treatment peak, and the treatment minimum (a hole deeper
than 50 MJy/sr would show here).  Matched rows whose fitted flux changed by more than 10% are counted.
"""
import glob
import json
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

args = sys.argv[1:]
MATCH, R, OUT = 1.0, 2, None
for flag in ('--match', '--r', '--json'):
    if flag in args:
        i = args.index(flag)
        val = args[i + 1]
        del args[i:i + 2]
        if flag == '--match':
            MATCH = float(val)
        elif flag == '--r':
            R = int(val)
        else:
            OUT = val
CTRL, TREAT = args


def one(d, pattern):
    paths = glob.glob(f'{d}/{pattern}')
    if len(paths) != 1:
        raise SystemExit(f'{d}: expected one {pattern}, found {len(paths)}')
    return paths[0]


def image(d, kind):
    with fits.open(one(d, f'pipeline/*_m7_daophot_basic_{kind}.fits')) as h:
        return np.asarray(h['SCI'].data, dtype=np.float64)


ct = Table.read(one(CTRL, '*_m7_daophot_basic.fits'))
tt = Table.read(one(TREAT, '*_m7_daophot_basic.fits'))
cxy = np.column_stack([ct['x_fit'], ct['y_fit']])
txy = np.column_stack([tt['x_fit'], tt['y_fit']])
d_tc, i_tc = cKDTree(cxy).query(txy)
d_ct, i_ct = cKDTree(txy).query(cxy)
added = np.nonzero(d_tc > MATCH)[0]
lost = np.nonzero(d_ct > MATCH)[0]
matched = np.nonzero(d_tc <= MATCH)[0]
dflux = np.abs(tt['flux_fit'][matched] / np.where(ct['flux_fit'][i_tc[matched]] != 0, ct['flux_fit'][i_tc[matched]], np.nan) - 1)
moved = int(np.sum(dflux > 0.1))
rc, rt = image(CTRL, 'residual'), image(TREAT, 'residual')
print(f'control {len(ct)} rows, treatment {len(tt)} rows: {len(added)} added, {len(lost)} lost; {moved} of {len(matched)} matched rows changed flux by more than 10%')
rec = dict(n_control=len(ct), n_treatment=len(tt), n_added=int(len(added)), n_lost=int(len(lost)), n_flux_changed=moved, added=[])
for j in added[np.argsort(-tt['flux_fit'][added])]:
    x, y = float(tt['x_fit'][j]), float(tt['y_fit'][j])
    ix, iy = int(round(x)), int(round(y))
    sl = (slice(max(iy - R, 0), iy + R + 1), slice(max(ix - R, 0), ix + R + 1))
    row = dict(x=x, y=y, flux=float(tt['flux_fit'][j]), snr=float(tt['flux_fit'][j] / tt['flux_err'][j]) if tt['flux_err'][j] > 0 else float('nan'),
               qfit=float(tt['qfit'][j]), flags=int(tt['flags'][j]),
               control_peak=float(np.nanmax(rc[sl])), treatment_peak=float(np.nanmax(rt[sl])), treatment_min=float(np.nanmin(rt[sl])))
    rec['added'].append(row)
    print(f'  added ({x:.1f},{y:.1f}) flux {row["flux"]:.0f} S/N {row["snr"]:.0f} qfit {row["qfit"]:.2f} flags {row["flags"]}: '
          f'residual peak within {R} px {row["control_peak"]:.1f} -> {row["treatment_peak"]:.1f}, treatment minimum {row["treatment_min"]:.1f} MJy/sr')
for j in lost:
    print(f'  lost ({ct["x_fit"][j]:.1f},{ct["y_fit"][j]:.1f}) flux {ct["flux_fit"][j]:.0f}')
if OUT:
    json.dump(rec, open(OUT, 'w'), indent=1)
