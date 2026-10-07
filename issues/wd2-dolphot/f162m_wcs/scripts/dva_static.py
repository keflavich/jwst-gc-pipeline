"""F150W - F212N: subtract the inter-detector DVA shift that only F150W
carries (dva_pred.py), turn the remainder into instrument-frame rows with
the PR #1132 solver, and test leave-one-out transfer wd2 <-> wd1."""
import importlib.util
import sys

import numpy as np

WT = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-ffsign'
sys.path.insert(0, WT)
from jwst_gc_pipeline.reduction import filter_frame_correction as ffc  # noqa: E402
spec = importlib.util.spec_from_file_location('solver', f'{WT}/scripts/analysis/solve_filter_frame_offsets.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
import glob  # noqa: E402
from astropy.io import fits  # noqa: E402
import dva_pred as dp  # noqa: E402

ROLL = {'wd2': 141.01, 'wd1': 284.71}
data = {}
for field, (pat, src) in dp.FIELDS.items():
    pred = {}
    for d in dp.DETS:
        h = fits.getheader(sorted(glob.glob(pat.format(d=d)))[0], 'SCI')
        s = float(h['VA_SCALE']) - 1.0
        cosd = np.cos(np.radians(h['DEC_REF']))
        pred[d] = s * np.array([(h['RA_REF'] - h['RA_V1']) * cosd, h['DEC_REF'] - h['DEC_V1']]) * 3.6e6
    pred = dp.demean(pred)
    meas = dp.demean(dp.c0(src))
    rem = {d.upper(): tuple(meas[d] - pred[d]) for d in dp.DETS}
    full = {d.upper(): tuple(meas[d]) for d in dp.DETS}
    data[field] = (rem, full, solver.sky_residual_to_instrument_correction(rem, ROLL[field]))


def rms(v):
    a = np.array([v[d] for d in sorted(v)])
    for mod in (slice(0, 4), slice(4, 8)):
        a[mod] -= a[mod].mean(0)
    return float(np.sqrt((a ** 2).sum(1).mean()))


print('instrument-frame rows of the DVA-subtracted remainder (mas)')
for d in sorted(data['wd2'][2]):
    print(f"  {d}  wd2 ({data['wd2'][2][d][0]:+5.2f},{data['wd2'][2][d][1]:+5.2f})"
          f"  wd1 ({data['wd1'][2][d][0]:+5.2f},{data['wd1'][2][d][1]:+5.2f})")
for k, o in (('wd2', 'wd1'), ('wd1', 'wd2')):
    rem, full, _ = data[k]
    sky = ffc.instrument_to_sky(data[o][2], ROLL[k])
    after = {d: (rem[d][0] + sky[d][0], rem[d][1] + sky[d][1]) for d in rem}
    print(f'  {k}: raw {rms(full):.2f} -> DVA-subtracted {rms(rem):.2f} -> + static rows from {o} {rms(after):.2f} mas')
