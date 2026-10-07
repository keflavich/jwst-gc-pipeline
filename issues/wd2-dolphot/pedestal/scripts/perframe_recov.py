"""Per-frame recovery of dolphot stars (within 1.5 px of the offset-corrected
dolphot position) by the m6 and m7 per-frame catalogs, exposure 00001.
usage: python perframe_recov.py DET"""
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
import astropy.units as u

warnings.filterwarnings('ignore')
R = '/orange/adamginsburg/jwst/wd2'
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
DET = sys.argv[1]
OFF = {'F150W': (-28.1, -12.7)}
VG = {'F115W': '08101', 'F150W': '10101', 'F162M': '14101', 'F182M': '16101', 'F200W': '12101', 'F212N': '04101'}
dm = Table.read(f'{D}/matched_Q_mainfcbg.fits')
dsk = SkyCoord(np.asarray(dm['RA'], float) * u.deg, np.asarray(dm['DEC'], float) * u.deg)
bins = [(10, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26)]
print(f'{DET} exp00001: fraction of dolphot stars with a per-frame row within 1.5 px (n dolphot)')
print('band  phase rows    ' + '  '.join(f'{lo}-{hi}'.rjust(12) for lo, hi in bins))
for b, vg in VG.items():
    p = f'{R}/{b}/pipeline/jw03523005001_{vg}_00001_{DET}_align_o005_crf.fits'
    h = fits.getheader(p, 'SCI')
    w = WCS(h)
    ox, oy = OFF.get(b, (-42.4, 5.4))
    dsh = SkyCoord(dsk.ra + (ox * u.mas) / np.cos(dsk.dec.rad), dsk.dec + oy * u.mas)
    xd, yd = w.world_to_pixel(dsh)
    ins = (xd > 5) & (xd < h['NAXIS1'] - 5) & (yd > 5) & (yd < h['NAXIS2'] - 5)
    mag = np.asarray(dm['ref_' + b[1:]], float)
    for m in ('m6', 'm7'):
        t = Table.read(f'{R}/{b}/{b.lower()}_{DET}_visit001_vgroup{vg}_exp00001_resbgsub_{m}_daophot_basic.fits')
        tr = cKDTree(np.c_[np.asarray(t['x_fit'], float), np.asarray(t['y_fit'], float)])
        d, _ = tr.query(np.c_[xd[ins], yd[ins]])
        hit = d < 1.5
        mm = mag[ins]
        cells = []
        for lo, hi in bins:
            k = np.isfinite(mm) & (mm >= lo) & (mm < hi)
            cells.append(f'{hit[k].mean():.2f} ({k.sum():4d})' if k.sum() else '   -        ')
        print(f'{b} {m}  {len(t):6d}  ' + '  '.join(c.rjust(12) for c in cells))
