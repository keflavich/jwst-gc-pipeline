"""Task 2 statistics for fr1 forced rows by class a/b/c, vs unforced control."""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '.')
from classify import load, dsk, A, H
bins = [(-99, 18.6), (18.6, 21), (21, 23), (23, 25), (25, 99)]
for b in ['187N', '200W']:
    z = np.load(f'classify_{b}.npz')
    t0, s0, f0 = load('fr0', b)
    t1, s1, f1 = load('fr1', b)
    cls = z['cls']; m1 = z['m1']; dd = z['ddolphot']
    ref = A.ref[b]
    print(f'## F{b}: dolphot rows with finite mag {np.isfinite(ref).sum()}; 50/90/99 pct dolphot mag {np.nanpercentile(ref, [50, 90, 99]).round(2)}; '
          f'fr1 catalog mag median {np.median(m1):.2f}, 90/99 pct {np.percentile(m1, [90, 99]).round(2)}')
    # fr0 nmatch for matched rows
    j, d, _ = s1.match_to_catalog_sky(s0)
    nm0 = np.where(d.arcsec <= 0.04, np.asarray(t0['nmatch'], float)[j], np.nan)
    nm1 = z['nmatch']
    flux0 = np.where(d.arcsec <= 0.04, f0[j], np.nan)
    for name, sel in (('a forced both', cls == 'a'), ('b forced fr1 only', cls == 'b'), ('c absent fr0', cls == 'c'), ('control: fr1 unforced', z['ff1'] == 0)):
        n = sel.sum()
        print(f'  [{name}] N={n}; mag median {np.median(m1[sel]):.2f} (16/84 {np.percentile(m1[sel], 16):.2f}/{np.percentile(m1[sel], 84):.2f}); '
              f'dolphot within 0.1": {np.mean(dd[sel] < 0.1):.3f}; median nmatch fr1 {np.nanmedian(nm1[sel]):.1f}'
              + (f', fr0 {np.nanmedian(nm0[sel]):.1f}' if name.startswith('b') or name.startswith('a') else '')
              + f'; median forced_nframes {np.nanmedian(z["nfr1"][sel]):.1f}; frac of frames forced median {np.median(z["ff1"][sel]):.2f}')
        nn = z['nnb'][sel]
        print(f'      NN brighter row: median {np.nanmedian(nn):.3f}", frac<0.1" {np.nanmean(nn < 0.1):.3f}, frac<0.2" {np.nanmean(nn < 0.2):.3f}')
        row = []
        for lo, hi in bins:
            s = sel & (m1 >= lo) & (m1 < hi)
            row.append(f'{lo}-{hi}: N={s.sum()} dol={np.mean(dd[s] < 0.1) if s.sum() else float("nan"):.2f}')
        print('      ' + '; '.join(row))
    # flux ratio fr0/fr1 for b
    sel = cls == 'b'
    print('  b: flux fr0/fr1 median %.2f' % np.nanmedian(flux0[sel] / f1[sel]))
