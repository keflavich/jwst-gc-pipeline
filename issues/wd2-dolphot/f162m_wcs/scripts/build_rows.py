"""Instrument-frame correction rows per (band, field) and a leave-one-out
test of their transfer across fields and rolls.

Input per (field, band): the c0 table of datascale(_field).py -- explicit
(frame - F212N anchor) per detector at the detector centre, module median
removed, median over exposures.  Each is turned into a correction with the
solver's own ``sky_residual_to_instrument_correction`` (rotate by +ROLL_REF,
negate, module-mean gauge) from the PR #1132 worktree.  For every field k the
mean of the OTHER fields' rows is applied to k's measured residual through
``filter_frame_correction.instrument_to_sky`` at k's roll; the module-mean
removed 2-D rms is printed before and after.  Writes rows_<band>.ecsv-style
lines (mean over all fields) to proposed_rows.ecsv."""
import importlib.util
import glob
import re
import sys

import numpy as np
from astropy.io import fits

WT = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-ffsign'
sys.path.insert(0, WT)
from jwst_gc_pipeline.reduction import filter_frame_correction as ffc  # noqa: E402
assert ffc.__file__.startswith(WT), ffc.__file__
spec = importlib.util.spec_from_file_location('solver', f'{WT}/scripts/analysis/solve_filter_frame_offsets.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

DETS = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']
FIELDS = {
    'wd2': ('/orange/adamginsburg/jwst/wd2', '{R}/{band}/pipeline/jw03523005001_*_00001_nrca1_align_o005_crf.fits',
            'jw03523-o005'),
    'sgrc': ('/orange/adamginsburg/jwst/sgrc', '{R}/{band}/pipeline/jw04147012001_*_00001_nrca1_destreak_o012_crf.fits',
             'jw04147-o012'),
    'wd1': ('/orange/adamginsburg/jwst/wd1', '{R}/{band}/pipeline/jw01905001001_*_00001_nrca1_destreak_o001_crf.fits',
            'jw01905-o001'),
}
SOURCES = {
    'F162M': [('wd2', 'datascale_f162m_run1.txt'), ('sgrc', 'datascale_sgrc_f162m_e2.txt')],
    'F164N': [('wd2', 'datascale_other.txt'), ('wd1', 'datascale_wd1_f164n.txt')],
    'F150W': [('wd2', 'datascale_other.txt'), ('wd1', 'datascale_wd1_other.txt')],
    'F200W': [('wd2', 'datascale_other.txt'), ('wd1', 'datascale_wd1_f200w.txt')],
    'F115W': [('wd2', 'datascale_other.txt'), ('wd1', 'datascale_wd1_f115w.txt')],
    'F187N': [('wd2', 'datascale_other.txt'), ('wd1', 'datascale_wd1_f187n.txt')],
    'F182M': [('wd2', 'datascale_other.txt')],
}


def c0(path, band):
    txt = open(path).read()
    m = re.search(rf'### {band.lower()} - f212n.*?\n.*?\n(.*?)(?:\n\s*\n|\n###|\Z)', txt, re.S)
    if m is None:
        return None
    out = {}
    for line in m.group(1).splitlines():
        p = line.split()
        if p and re.fullmatch(r'nrc[ab][1-4]', p[0]):
            out[p[0].upper()] = (float(p[2]), float(p[3]))
    return out if len(out) == 8 else None


def roll_of(field, band):
    R, pat, _ = FIELDS[field]
    f = sorted(glob.glob(pat.format(R=R, band=band)))[0]
    h = fits.getheader(f, 'SCI')
    return float(h['ROLL_REF']) if 'ROLL_REF' in h else float(fits.getheader(f, 0)['ROLL_REF'])


def rms(vecs):
    a = np.array([vecs[d] for d in DETS])
    for mod in (slice(0, 4), slice(4, 8)):
        a[mod] -= a[mod].mean(0)
    return float(np.sqrt((a ** 2).sum(1).mean()))


rows_out = []
for band, srcs in SOURCES.items():
    data = []
    for field, path in srcs:
        meas = c0(path, band)
        if meas is None:
            print(f'{band} {field}: no table in {path}')
            continue
        roll = roll_of(field, band)
        corr = solver.sky_residual_to_instrument_correction(meas, roll)
        data.append((field, roll, meas, corr))
    if not data:
        continue
    print(f'\n### {band} - F212N  (fields: {", ".join(f"{d[0]} roll {d[1]:.2f}" for d in data)})')
    print('  instrument-frame correction rows (mas):')
    print('  det    ' + '  '.join(f'{d[0]:>16s}' for d in data) + '       mean')
    mean = {d: tuple(np.mean([x[3][d] for x in data], axis=0)) for d in DETS}
    for det in DETS:
        print(f'  {det}  ' + '  '.join(f'({x[3][det][0]:+6.2f},{x[3][det][1]:+6.2f})' for x in data)
              + f'   ({mean[det][0]:+6.2f},{mean[det][1]:+6.2f})')
    if len(data) > 1:
        diff = np.array([[np.subtract(data[0][3][d], data[1][3][d]) for d in DETS]])[0]
        print(f'  field-to-field row difference rms: {np.sqrt((diff ** 2).sum(1).mean()):.2f} mas')
    for k, (field, roll, meas, corr) in enumerate(data):
        others = [x for j, x in enumerate(data) if j != k]
        if not others:
            pred = corr
            tag = 'in-sample'
        else:
            pred = {d: tuple(np.mean([x[3][d] for x in others], axis=0)) for d in DETS}
            tag = 'held out (rows from ' + ', '.join(x[0] for x in others) + ')'
        sky = ffc.instrument_to_sky(pred, roll)
        after = {d: (meas[d][0] + sky[d][0], meas[d][1] + sky[d][1]) for d in DETS}
        print(f'  {field:5s} roll {roll:7.2f}: rms {rms(meas):5.2f} -> {rms(after):5.2f} mas   {tag}')
    n = len(data)
    for det in DETS:
        rows_out.append((det, band, mean[det][0], mean[det][1], n,
                         '+'.join(FIELDS[x[0]][2] for x in data)))

with open('proposed_rows.tsv', 'w') as fh:
    fh.write('detector\tfilter\tanchor\tframe\tdx_mas\tdy_mas\tn\tsources\n')
    for det, band, dx, dy, n, srcs in rows_out:
        fh.write(f'{det}\t{band}\tF212N\tinstrument\t{dx:+.3f}\t{dy:+.3f}\t{n}\t{srcs}\n')
print('\nwrote proposed_rows.tsv')
