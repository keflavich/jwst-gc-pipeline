"""#1015: restored seeds by separation to the nearest brighter seed, in PSF
FWHM units, for one band, with the in-production control.

usage: python seed_sep_band.py <band> <ref_catalog> <fwhm_arcsec>
"""
import os
import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from realness import match_fraction, in_footprint  # noqa: E402
from seed_cuts import rel_flux_matched  # noqa: E402

band, refpath, fwhm = sys.argv[1], sys.argv[2], float(sys.argv[3])
t = Table.read(f'{HERE}/brick/seed_{band}.fits')
sc = t['skycoord'] if isinstance(t['skycoord'], SkyCoord) else SkyCoord(t['skycoord'])
ref = Table.read(refpath)['skycoord']
ref = ref if isinstance(ref, SkyCoord) else SkyCoord(ref)
fp = in_footprint(sc, ref)
org = np.asarray(t['seed_origin']).astype(str)
inprod = np.asarray(t['sep_prod_m7_mas'], float) < 60
own = org == 'own_m6'
flux = np.asarray(t['flux'], float)
fl = np.where(own, np.asarray(t['m6_flux'], float), flux)
i1, i2, d, _ = sc.search_around_sky(sc, 8 * fwhm * u.arcsec)
d = d.to_value(u.arcsec) / fwhm
ok = (i1 != i2) & (fl[i2] > fl[i1])
sep_b = np.full(len(t), np.inf)
order = np.lexsort((d[ok], i1[ok]))
a, dd = i1[ok][order], d[ok][order]
first = np.r_[True, a[1:] != a[:-1]]
sep_b[a[first]] = dd[first]
lf = np.log10(np.clip(flux, 1e-30, None))
refgrp = own & inprod & fp
rest = own & ~inprod & fp
cache = {}
lines = [f'== Brick {band.upper()} vs {os.path.basename(refpath)}; FWHM {fwhm}"; restored n={rest.sum()}, '
         f'in-prod own n={refgrp.sum()}; separation to nearest brighter seed in FWHM']
for lo, hi in [(0, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 6), (6, np.inf)]:
    out = []
    for name, grp in (('restored', rest), ('in-prod', refgrp)):
        g = grp & (sep_b >= lo) & (sep_b < hi)
        if g.sum() < 20:
            out.append(f'{name} n={g.sum():6d}')
            continue
        m, ch = match_fraction(sc[g], ref)
        rel, _ = rel_flux_matched(sc, lf, g, refgrp, ref, cache)
        out.append(f'{name} n={g.sum():6d} match {m:.2f} (ch {ch:.2f}) rel {rel:.2f}')
    lines.append(f'  [{lo:.1f}, {hi:.1f}) FWHM: ' + '  |  '.join(out))
for R in [2.0, 2.5, 3.0]:
    g = rest & (sep_b >= R)
    rel, _ = rel_flux_matched(sc, lf, g, refgrp, ref, cache)
    lines.append(f'  restore only if >= {R} FWHM from a brighter seed: n={g.sum():6d} rel {rel:.2f}')
txt = '\n'.join(lines)
print(txt)
open(f'{HERE}/seed_sep_band_{band}.txt', 'w').write(txt + '\n')
