"""Per-frame m7 residual of a one-frame replay with the exemption seed against the control replay of the same frame.

usage: python replay_resid_compare.py CONTROL_PIPELINE_DIR TREATMENT_PIPELINE_DIR 'X,Y;X,Y;...' [--r 3] [--json OUT.json]

Each directory holds one ``*_m7_daophot_basic_residual.fits`` and one ``*_m7_daophot_basic_model.fits`` (SCI extension, MJy/sr) written by the replay.
X,Y are 0-based frame pixels of the target stars.  Prints:
  per target: peak and minimum residual within R pixels (control -> treatment), and the peak as a fraction of the data peak (residual + model);
  per frame: the number of pixels where the two residuals differ by more than 1e-3 MJy/sr, the number below -50 and above +50 MJy/sr in the difference
  (treatment - control), the number of pixels with a residual below -50 MJy/sr in each run, and the new craters: pixels with a treatment residual below -50 MJy/sr whose control residual is above -50 MJy/sr and whose depth is
  more than 0.3 of the treatment model at that pixel (the frame-level form of the mosaic crater metric: depth > 50 MJy/sr and depth / model peak > 0.3).
"""
import glob
import json
import sys

import numpy as np
from astropy.io import fits

args = sys.argv[1:]
R = 3
OUT = None
if '--r' in args:
    i = args.index('--r')
    R = int(args[i + 1])
    del args[i:i + 2]
if '--json' in args:
    i = args.index('--json')
    OUT = args[i + 1]
    del args[i:i + 2]
CTRL, TREAT, TARGETS = args[0], args[1], args[2]


def load(d):
    out = {}
    for kind in ('residual', 'model'):
        paths = glob.glob(f'{d}/*_m7_daophot_basic_{kind}.fits')
        if len(paths) != 1:
            raise SystemExit(f'{d}: expected one {kind} file, found {len(paths)}')
        with fits.open(paths[0]) as h:
            out[kind] = np.asarray(h['SCI'].data, dtype=np.float64)
    return out


c, t = load(CTRL), load(TREAT)
rec = dict(targets=[])
for tg in TARGETS.split(';'):
    x, y = (float(v) for v in tg.split(','))
    ix, iy = int(round(x)), int(round(y))
    sl = (slice(max(iy - R, 0), iy + R + 1), slice(max(ix - R, 0), ix + R + 1))
    row = {}
    for name, d in (('control', c), ('treatment', t)):
        res = d['residual'][sl]
        data = (d['residual'] + d['model'])[sl]
        row[name] = dict(resid_peak=float(np.nanmax(res)), resid_min=float(np.nanmin(res)), data_peak=float(np.nanmax(data)),
                         resid_over_data=float(np.nanmax(res) / np.nanmax(data)))
    rec['targets'].append(dict(xy=[x, y], **row))
    print(f'target ({x:.1f},{y:.1f}) within {R} px: residual peak {row["control"]["resid_peak"]:.1f} -> {row["treatment"]["resid_peak"]:.1f} MJy/sr, '
          f'residual / data {row["control"]["resid_over_data"]:.2f} -> {row["treatment"]["resid_over_data"]:.2f}, '
          f'residual minimum {row["control"]["resid_min"]:.1f} -> {row["treatment"]["resid_min"]:.1f}')
dres = t['residual'] - c['residual']
fin = np.isfinite(dres)
changed = fin & (np.abs(dres) > 1e-3)
neg, pos = fin & (dres < -50), fin & (dres > 50)
crater = np.isfinite(t['residual']) & (t['residual'] < -50) & ~(c['residual'] < -50) & (-t['residual'] > 0.3 * np.where(t['model'] > 0, t['model'], np.inf))
low_c, low_t = int((c['residual'] < -50).sum()), int((t['residual'] < -50).sum())
rec['frame'] = dict(pixels_changed=int(changed.sum()), delta_below_m50=int(neg.sum()), delta_above_p50=int(pos.sum()), new_craters=int(crater.sum()),
                    pixels_below_m50_control=low_c, pixels_below_m50_treatment=low_t)
print(f'frame: {int(changed.sum())} pixels differ by more than 1e-3 MJy/sr; difference (treatment - control) below -50: {int(neg.sum())}, above +50: {int(pos.sum())}; '
      f'pixels with residual below -50: {low_c} -> {low_t}; new craters: {int(crater.sum())}')
if neg.any():
    ys, xs = np.nonzero(neg)
    o = np.argsort(dres[ys, xs])[:5]
    for j in o:
        print(f'  most negative difference at ({xs[j]},{ys[j]}): {dres[ys[j], xs[j]]:.1f}; residual {c["residual"][ys[j], xs[j]]:.1f} -> {t["residual"][ys[j], xs[j]]:.1f}, '
              f'treatment model {t["model"][ys[j], xs[j]]:.1f}')
if crater.any():
    ys, xs = np.nonzero(crater)
    for j in np.argsort(t['residual'][ys, xs])[:5]:
        print(f'  new crater at ({xs[j]},{ys[j]}): residual {c["residual"][ys[j], xs[j]]:.1f} -> {t["residual"][ys[j], xs[j]]:.1f}, treatment model {t["model"][ys[j], xs[j]]:.1f}')
if OUT:
    json.dump(rec, open(OUT, 'w'), indent=1)
