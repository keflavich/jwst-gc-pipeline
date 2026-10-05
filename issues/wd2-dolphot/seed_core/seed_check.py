"""Seed offsets of satstar fits against the dolphot benchmark.
For each benchmark star matched to a satstar row (0.15"), report the seed (x_init, y_init) distance
to the dolphot position, the fit distance, whether the fit sits on the position bound (flag 16 =
no covariance), and whether the seed came from the NaN-VAR_POISSON sub-cluster branch of
_refine_coms_by_data (>= 3 NaN-variance pixels in the star's SATURATED component).
usage: python seed_check.py ARM [ARM ...]"""
import sys, numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from scipy import ndimage
arms = sys.argv[1:]
sys.argv = sys.argv[:1]
import score_rc as S

rows = []
for line in open(f'{S.H}/frames.txt'):
    band, fr = line.split()
    F = S.frame_data(band, fr)
    crf = fits.open(f'{S.R}/{band}/pipeline/{fr}_align_o005_crf.fits')
    sat = (crf['DQ'].data & 2) != 0
    unrec = np.isnan(crf['VAR_POISSON'].data)
    lab, _ = ndimage.label(sat)
    rx, ry = F['w'].world_to_pixel(F['rsc'])
    for arm in arms:
        t = Table.read(f'{S.H}/tree_{arm}/{band}/pipeline/{fr}_align_o005_crf_rctest_satstar_catalog.fits')
        sc = SkyCoord(t['skycoord_fit'])
        fin = np.isfinite(sc.ra.deg)
        i1, i2, sep, _ = search_around_sky(F['rsc'], sc[fin], 0.15 * u.arcsec)
        o = np.argsort(sep)[::-1]
        ia = np.full(len(F['rsc']), -1); ia[i1[o]] = np.where(fin)[0][i2[o]]
        f = np.asarray(np.ma.filled(t['flux_fit'], np.nan), float)
        for i in np.where(F['bench'] & (ia >= 0))[0]:
            r = t[ia[i]]
            ox, oy = float(r['x_0'] - r['x_fit']), float(r['y_0'] - r['y_fit'])
            sx, sy = float(r['x_init']) + ox, float(r['y_init']) + oy
            l = lab[int(round(sy)), int(round(sx))] if 0 <= round(sy) < sat.shape[0] and 0 <= round(sx) < sat.shape[1] else 0
            if l == 0:
                # seed off the component: take the component nearest the dolphot position
                l = lab[int(round(ry[i])), int(round(rx[i]))]
            ncore = int((unrec & (lab == l)).sum()) if l else 0
            dm = F['zp'] - 2.5 * np.log10(f[ia[i]]) - F['rm'][i] if f[ia[i]] > 0 else np.nan
            rows.append((arm, band, fr.split('_', 1)[1], float(rx[i]), float(ry[i]), F['rm'][i],
                         np.hypot(sx - rx[i], sy - ry[i]), np.hypot(float(r['x_0']) - rx[i], float(r['y_0']) - ry[i]),
                         int(r['flags']) & 16 > 0, ncore, dm, str(r['seed_kind'])))
T = Table(rows=rows, names=('arm', 'band', 'frame', 'x', 'y', 'mag', 'd_seed', 'd_fit', 'nocov', 'n_nanvar', 'dm', 'seed_kind'))
T.write(f'{S.H}/seed_check.ecsv', overwrite=True)
for arm in arms:
    a = T[T['arm'] == arm]
    good = np.abs(a['dm']) < 0.3
    print(f'\n## {arm}: {len(a)} matched benchmark stars, good {good.sum()}')
    print('| seed offset (px) | N | good | median dm | flag 16 (no covariance) | NaN-var branch (>=3 px) |')
    print('|---|---|---|---|---|---|')
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 99)):
        k = (a['d_seed'] >= lo) & (a['d_seed'] < hi)
        if k.sum():
            print(f'| {lo}-{hi} | {k.sum()} | {(good & k).sum()} | {np.nanmedian(a["dm"][k]):+.3f} | {a["nocov"][k].sum()} | {(a["n_nanvar"][k] >= 3).sum()} |')
    k = a['n_nanvar'] >= 3
    print(f'NaN-var branch: {k.sum()} stars, median seed offset {np.median(a["d_seed"][k]):.2f} px, good {(good & k).sum()}; '
          f'other: {(~k).sum()} stars, median seed offset {np.median(a["d_seed"][~k]):.2f} px, good {(good & ~k).sum()}')
