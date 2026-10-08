"""Cutouts of F162M rows against dolphot: three free fits and three forced-refit rows fainter than dolphot.
Columns: F162M data, F162M m7 residual, F212N data, F212N m7 residual (all mainfcbg mosaics;
residual panels have the cutout median removed).
Markers: red circle = dolphot star moved by the band's median free-fit offset (where the band's
own WCS puts the star); cyan x = F162M m7 merged row; lime + = F212N m7 merged row.
usage: python fig_cutouts.py"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

T = f'{an.Q}/tree_mainfcbg'
P = 'jw03523-o005_t001_nircam_clear-{b}-merged{s}_i2d.fits'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)


def load(b):
    t = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    t, sk, fl = t[ok], sk[ok], fl[ok]
    mi = -2.5 * np.log10(fl)
    rep = np.asarray(t['replaced_saturated'], bool)
    ff = np.asarray(t['forced_refit_frac'], float)
    ref = A.ref[b]
    have = np.where(np.isfinite(ref))[0]
    j, d, _ = dsk[have].match_to_catalog_sky(sk)
    hit = (d.arcsec < 0.08) & ~rep[j]
    mid = hit & (ref[have] >= 18.6) & (ref[have] < 21) & (d.arcsec < 0.05)
    zp = np.median(ref[have][mid] - mi[j][mid])
    dm = mi[j] + zp - ref[have]
    dra = (sk.ra.deg[j] - dsk.ra.deg[have]) * np.cos(np.deg2rad(dsk.dec.deg[have])) * 3.6e6
    dde = (sk.dec.deg[j] - dsk.dec.deg[have]) * 3.6e6
    z = hit & (ff[j] == 0)
    off = (np.median(dra[z]), np.median(dde[z]))
    r = np.hypot(dra - off[0], dde - off[1])
    return dict(sk=sk, ff=ff, have=have, j=j, hit=hit, dm=dm, r=r, off=off)


c150, c200 = load('162M'), load('212N')
# three free F162M rows (frac 0, |dm| < 0.03) and three forced rows (frac >= 0.5,
# dm > 0.15), 17-21 mag, sorted by magnitude
h = c150['have']
mag = A.ref['162M'][h]
base = c150['hit'] & (mag >= 17) & (mag < 21)
ffj = c150['ff'][c150['j']]
free = np.where(base & (ffj == 0) & (np.abs(c150['dm']) < 0.03))[0]
forced = np.where(base & (ffj >= 0.5) & (c150['dm'] > 0.15))[0]
print(f'{len(free)} free, {len(forced)} forced candidates')
def spread(c, n):
    c = c[np.argsort(mag[c])]
    return c[np.linspace(0, len(c) - 1, n).astype(int)]
pick = np.concatenate([spread(free, 3), spread(forced, 3)])

img = {}
for b in ('f162m', 'f212n'):
    for s, lab in (('_data', 'data'), ('_resbgsub_m7_daophot_basic_mergedcat_residual', 'm7 residual')):
        with fits.open(f'{T}/{b.upper()}/pipeline/' + P.format(b=b, s=s)) as hl:
            ext = 'SCI' if 'SCI' in hl else 0
            img[(b, lab)] = (hl[ext].data.astype(float), WCS(hl[ext].header))

def shifted(sk, off):
    dec = sk.dec.deg + off[1] / 3.6e6
    ra = sk.ra.deg + off[0] / 3.6e6 / np.cos(np.deg2rad(sk.dec.deg))
    return SkyCoord(ra * u.deg, dec * u.deg)

cols = [('f162m', 'data'), ('f162m', 'm7 residual'), ('f212n', 'data'), ('f212n', 'm7 residual')]
fig, axes = plt.subplots(len(pick), 4, figsize=(11, 2.75 * len(pick)))
size = 0.4 * u.arcsec
for i, k in enumerate(pick):
    di = h[k]
    star = dsk[di]
    row150 = c150['sk'][c150['j'][k]]
    s200 = SkyCoord(dsk.ra[di:di + 1], dsk.dec[di:di + 1])
    jj, dd, _ = s200.match_to_catalog_sky(c200['sk'])
    row200 = c200['sk'][jj[0]] if dd[0].arcsec < 0.08 else None
    for c, (b, lab) in enumerate(cols):
        ax = axes[i, c]
        data, w = img[(b, lab)]
        co = Cutout2D(data, star, size, wcs=w, mode='partial')
        cut = co.data
        if lab == 'data':
            norm = simple_norm(cut, 'asinh', percent=99.5)
            ax.imshow(cut, origin='lower', cmap='gray', norm=norm)
        else:
            cut = cut - np.nanmedian(cut)
            v = np.nanpercentile(np.abs(cut), 99)
            ax.imshow(cut, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
        off = c150['off'] if b == 'f162m' else c200['off']
        x, y = co.wcs.world_to_pixel(shifted(star, off))
        ax.scatter([x], [y], s=260, facecolors='none', edgecolors='red', lw=1.5)
        x, y = co.wcs.world_to_pixel(row150)
        ax.plot(x, y, 'x', color='cyan', ms=11, mew=2)
        if row200 is not None:
            x, y = co.wcs.world_to_pixel(row200)
            ax.plot(x, y, '+', color='lime', ms=13, mew=2)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(f'{b.upper()} {lab}', fontsize=10)
        if c == 0:
            ax.set_ylabel(f"F162M {A.ref['162M'][di]:.2f}\ndm {c150['dm'][k]:+.2f}, "
                          f"r {c150['r'][k]:.0f} mas\nfrac {c150['ff'][c150['j'][k]]:.2f}", fontsize=9)
fig.suptitle('wd2 F162M vs dolphot (0.4" cutouts): rows 1-3 free fits, rows 4-6 forced refits fainter than dolphot\n'
             'red o: dolphot star + band median offset; cyan x: F162M m7 row; lime +: F212N m7 row', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig('cutouts_f162m.png', dpi=110)
print('wrote cutouts_f162m.png')
