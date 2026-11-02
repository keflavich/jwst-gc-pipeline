"""Second realness test for the exemption region: does the data i2d have a
local maximum at the catalog position?

A star fitted at (x, y) puts the brightest pixel of a 7x7 box within 1 px of
(x, y).  An emission knot edge, a PSF-wing/spike feature or a duplicate in a
bright star's halo usually does not.  Run on qfit <= 0.2, prominence 2-3
sources (the exemption region) in S/N_prop bins, split by continuum match.

usage: python peak_check.py   (reads the pl_*.fits tables in the cwd)
"""
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
from astropy.coordinates import SkyCoord

warnings.simplefilter('ignore', wcs.FITSFixedWarning)
SETS = [('w51', 'F187N', 'pl_w51.fits'), ('w51', 'F480M', 'pl_w51_f480m.fits'),
        ('wd2', 'F187N', 'pl_wd2_f187n.fits'), ('wd2', 'F405N', 'pl_wd2_f405n.fits'),
        ('ngc6334', 'F187N', 'pl_ngc6334_f187n.fits'), ('sickle', 'F187N', 'pl_sickle_f187n.fits'),
        ('sgrb2', 'F187N', 'pl_sgrb2.fits'), ('sgrb2', 'F480M', 'pl_sgrb2_f480m.fits')]
BINS = [(20, 30), (30, 60), (60, np.inf)]


def peak_at(d, x, y):
    out = np.zeros(len(x), bool)
    for j, (xi, yi) in enumerate(zip(x, y)):
        xi0, yi0 = int(round(xi)), int(round(yi))
        if not (3 <= xi0 < d.shape[1] - 3 and 3 <= yi0 < d.shape[0] - 3):
            continue
        box = d[yi0 - 3:yi0 + 4, xi0 - 3:xi0 + 4]
        if np.isfinite(box).any():
            out[j] = np.nanmax(box[2:5, 2:5]) >= np.nanmax(box)
    return out


lines = []
for field, band, tab in SETS:
    t = Table.read(tab)
    # the data i2d the prominence was measured on (build_pl.py records it)
    fns = [t.meta['DATAI2D']] if 'DATAI2D' in t.meta else sorted(glob.glob(
        f'/orange/adamginsburg/jwst/{field}/{band}/pipeline/'
        f'jw*-o00?_t001_nircam_clear-{band.lower()}-merged_data_i2d.fits'))
    with fits.open(fns[0]) as h:
        d = h['SCI'].data.astype(float)
        w = wcs.WCS(h['SCI'].header)
    snrp = np.asarray(t['flux'] / t['flux_err'], float) * np.sqrt(np.clip(np.asarray(t['nmatch'], float), 1, None))
    qf, pr = np.asarray(t['qfit'], float), np.asarray(t['prominence'], float)
    m = np.asarray(t['cont_match'], bool)
    reg = (qf <= 0.2) & (pr >= 2) & (pr < 3)
    ctl = (qf <= 0.2) & (pr >= 5) & (snrp >= 60)
    x, y = w.world_to_pixel(SkyCoord(t['skycoord']))
    pk = np.zeros(len(t), bool)
    sel = reg | ctl
    pk[sel] = peak_at(d, x[sel], y[sel])
    lines.append(f'== {field} {band} ({fns[0].split("/")[-1]}); control: qfit<=0.2, prom>=5, S/N>=60: '
                 f'peak {pk[ctl].sum()}/{ctl.sum()} = {pk[ctl].mean():.2f}')
    for lo, hi in BINS:
        k = reg & (snrp >= lo) & (snrp < hi)
        if not k.any():
            lines.append(f'   S/N {lo}-{hi}: --')
            continue
        km, ku = k & m, k & ~m
        lines.append(f'   S/N {lo}-{hi}: peak {pk[k].sum()}/{k.sum()} = {pk[k].mean():.2f}   '
                     f'(cont-matched {pk[km].sum()}/{km.sum()}, unmatched {pk[ku].sum()}/{ku.sum()})')
open('evid1021/peak_check.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
