"""dm = pipeline - dolphot for the dvx0 / dvx1 F150W m7 merged catalogs, by forced_refit_frac (#1128).

Matching and ZP follow forced_refit/ab/cmp_fr.py: dolphot stars with a row within 0.08"
that is not replaced_saturated; ZP from 18.6-21 mag stars within 0.05".
Left: dm against dolphot magnitude for rows with frac >= 0.5.  Right: dm histograms
for frac >= 0.5 rows; the frac = 0 rows of dvx0 are the reference shape.
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.table import Table  # noqa: E402

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix'
CAT = 'catalogs/f150w_merged_indivexp_merged_resbgsub_m7_dao_basic.fits'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref['150W']
have = np.where(np.isfinite(ref))[0]


def score(path):
    t = Table.read(path)
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    t, sk, fl = t[ok], sk[ok], fl[ok]
    mi = -2.5 * np.log10(fl)
    rep = np.asarray(t['replaced_saturated'], bool)
    ff = np.asarray(t['forced_refit_frac'], float)
    mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
    j, d, _ = dsk[mid].match_to_catalog_sky(sk)
    sel = (d.arcsec < 0.05) & ~rep[j]
    zp = np.median(ref[mid][sel] - mi[j][sel])
    j, d, _ = dsk[have].match_to_catalog_sky(sk)
    hit = (d.arcsec < 0.08) & ~rep[j]
    return ref[have][hit], (mi[j] + zp - ref[have])[hit], ff[j][hit]


S = {a: score(f'{H}/tree_{a}/{CAT}') for a in ('dvx0', 'dvx1')}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
sty = {'dvx0': dict(c='tab:red', label='dvx0: production seeds'),
       'dvx1': dict(c='tab:blue', label='dvx1: DVA-consistent seeds')}
bins = np.linspace(-1, 2, 61)
for a, (m, dm, ff) in S.items():
    k = ff >= 0.5
    a1.scatter(m[k], dm[k], s=9, alpha=0.7, lw=0, **sty[a])
    a2.hist(np.clip(dm[k], bins[0], bins[-1]), bins=bins, histtype='step', lw=1.8,
            color=sty[a]['c'], label=f"{sty[a]['label']} (N={k.sum()}, median {np.median(dm[k]):+.3f}, "
                                     f"|dm|>0.3: {np.mean(np.abs(dm[k]) > 0.3):.2f})")
m, dm, ff = S['dvx0']
k0 = ff == 0
w = np.full(k0.sum(), (S['dvx1'][2] >= 0.5).sum() / k0.sum())
a2.hist(np.clip(dm[k0], bins[0], bins[-1]), bins=bins, weights=w, color='0.8', label='frac = 0 rows (scaled)')
a1.axhline(0, c='k', lw=0.6)
a1.axhline(0.3, c='0.5', lw=0.6, ls=':')
a1.axhline(-0.3, c='0.5', lw=0.6, ls=':')
a1.set_ylim(-1.2, 2.5)
a1.set_xlabel('dolphot F150W [mag]')
a1.set_ylabel('pipeline - dolphot [mag]')
a1.set_title('F150W m7 merged rows with forced_refit_frac >= 0.5')
a1.legend(fontsize=8, loc='upper left')
a2.set_xlabel('pipeline - dolphot [mag] (clipped to [-1, 2])')
a2.set_ylabel('N')
a2.set_title('dm distribution, forced_refit_frac >= 0.5')
a2.legend(fontsize=7.5, loc='upper right')
fig.savefig(f'{H}/dm_forced.png', dpi=100)
for a, (m, dm, ff) in S.items():
    for lab, k in (('frac=0', ff == 0), ('0<frac<0.5', (ff > 0) & (ff < 0.5)), ('frac>=0.5', ff >= 0.5)):
        print(f'{a} {lab:11s} N={k.sum():5d} median {np.median(dm[k]):+.4f} rstd '
              f'{1.4826 * np.median(np.abs(dm[k] - np.median(dm[k]))):.4f} bad {np.mean(np.abs(dm[k]) > 0.3):.3f}')
print('wrote dm_forced.png')
