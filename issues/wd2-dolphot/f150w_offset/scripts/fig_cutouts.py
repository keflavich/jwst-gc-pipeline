"""Cutouts of F150W forced-refit rows that are fainter than dolphot.
Columns: F150W data, F150W m7 residual, F200W data, F200W m7 residual (all mainfcbg mosaics;
residual panels have the cutout median removed).
Markers: red circle = dolphot star moved by the band's median free-fit offset (where the band's
own WCS puts the star); cyan x = F150W m7 merged row; lime + = F200W m7 merged row.
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


c150, c200 = load('150W'), load('200W')
# F150W forced rows, 16-21 mag, 10-40 mas off the F150W star position, fainter by > 0.3
h = c150['have']
q = (c150['hit'] & (c150['ff'][c150['j']] >= 0.5) & (A.ref['150W'][h] >= 16) & (A.ref['150W'][h] < 21)
     & (c150['r'] >= 10) & (c150['r'] < 40) & (c150['dm'] > 0.3))
cand = np.where(q)[0]
print(f'{len(cand)} candidates')
rng = np.random.default_rng(1)
pick = cand[np.argsort(A.ref['150W'][h][cand])]
pick = pick[np.linspace(0, len(pick) - 1, 6).astype(int)]

img = {}
for b in ('f150w', 'f200w'):
    for s, lab in (('_data', 'data'), ('_resbgsub_m7_daophot_basic_mergedcat_residual', 'm7 residual')):
        with fits.open(f'{T}/{b.upper()}/pipeline/' + P.format(b=b, s=s)) as hl:
            ext = 'SCI' if 'SCI' in hl else 0
            img[(b, lab)] = (hl[ext].data.astype(float), WCS(hl[ext].header))

def shifted(sk, off):
    dec = sk.dec.deg + off[1] / 3.6e6
    ra = sk.ra.deg + off[0] / 3.6e6 / np.cos(np.deg2rad(sk.dec.deg))
    return SkyCoord(ra * u.deg, dec * u.deg)

cols = [('f150w', 'data'), ('f150w', 'm7 residual'), ('f200w', 'data'), ('f200w', 'm7 residual')]
fig, axes = plt.subplots(len(pick), 4, figsize=(11, 2.75 * len(pick)))
size = 0.5 * u.arcsec
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
        off = c150['off'] if b == 'f150w' else c200['off']
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
            ax.set_ylabel(f"F150W {A.ref['150W'][di]:.2f}\ndm {c150['dm'][k]:+.2f}, "
                          f"r {c150['r'][k]:.0f} mas\nfrac {c150['ff'][c150['j'][k]]:.2f}", fontsize=9)
fig.suptitle('wd2 F150W forced-refit rows fainter than dolphot (0.5" cutouts)\n'
             'red o: star in the band\'s own WCS; cyan x: F150W m7 row; lime +: F200W m7 row', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig('cutouts_f150w_forced.png', dpi=110)
print('wrote cutouts_f150w_forced.png')
