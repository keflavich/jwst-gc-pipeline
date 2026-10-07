"""Offline check of the peak hand-off (jwst-gc-pipeline-hpeaksrun, 048299af):
for every LW frame of BAND, hand-off positions with DAOPHOT_HANDOFF_PEAK_MIN_AREA
0 (centre of mass) and 50, and for the no-row and control dolphot stars on a
SATURATED pixel, whether a hand-off position lies within the exemption radius
and whether the star's pixel is restored to the fit.  No fitting.
usage: python pkcheck.py ARM BAND"""
import glob
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402
REPO = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-hpeaksrun'
sys.path.insert(0, REPO)
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402
assert C.__file__.startswith(REPO), C.__file__

arm, b = sys.argv[1], sys.argv[2]
FWHM = {'277W': 1.48, '250M': 1.33, '300M': 1.58}[b]
RAD = max(1.0, 0.5 * FWHM)
P = f'{an.Q}/tree_{arm}/F{b}/pipeline'
frames = sorted(glob.glob(f'{P}/jw03523005001_*_nrc?long_align_o005_crf.fits'))
tr3 = Table.read(f'trace3_{arm}_{b}.ecsv')
A = an.Arm(arm)
rs_all = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
groups = {g: np.unique(np.asarray(tr3['dolphot_idx'][tr3['grp'] == g])) for g in ('no-row', 'control')}
tot = {}
out = []
for f in frames:
    sci = fits.getdata(f, 'SCI').astype(float)
    dq = fits.getdata(f, 'DQ')
    err = fits.getdata(f, 'ERR').astype(float)
    bad = ~np.isfinite(err) | (err <= 0)
    w = WCS(fits.getheader(f, 'SCI'))
    acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
    sat = (dq & 2) != 0
    lab, n = ndimage.label(sat)
    line = [f.split('/')[-1][:40]]
    res = {}
    for area in (0, 50):
        xy = C._unaccepted_sat_component_xy(dq, acc, FWHM, sci=sci, data_floor=0.0,
                                            label=f'a{area}', peak_min_area=area)
        if xy is None:
            xy = np.empty((0, 2))
        rest = C._handoff_restore_pixels(dq, sci, bad, xy, acc, FWHM)
        res[area] = (xy, rest)
        line.append(f'area{area}: {len(xy)} pos, {int(rest.sum())} px restored')
    for g, idx in groups.items():
        x, y = w.world_to_pixel(rs_all[idx])
        ins = (x > 2) & (x < sat.shape[1] - 3) & (y > 2) & (y < sat.shape[0] - 3)
        x, y = x[ins], y[ins]
        ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
        on = sat[iy, ix]
        x, y, ix, iy = x[on], y[on], ix[on], iy[on]
        for area in (0, 50):
            xy, rest = res[area]
            if len(xy):
                d, _ = cKDTree(xy).query(np.column_stack([x, y]))
            else:
                d = np.full(len(x), np.inf)
            ex = d <= RAD
            rs = rest[iy, ix]
            k = (g, area)
            a = tot.setdefault(k, np.zeros(4, int))
            a += [len(x), ex.sum(), rs.sum(), (ex & rs).sum()]
    print(' | '.join(line), flush=True)
md = [f'# pkcheck {arm} F{b} (FWHM {FWHM} px, exemption radius {RAD:g} px)', '',
      '| group | area | star-frames on SAT | exempt | restored | both |', '|---|---|---|---|---|---|']
for (g, area), a in sorted(tot.items()):
    md.append(f'| {g} | {area} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |')
print('\n'.join(md))
open(f'pkcheck_{arm}_{b}.md', 'w').write('\n'.join(md) + '\n')
