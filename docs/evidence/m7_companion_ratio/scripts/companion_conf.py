"""Are the m6 companions that the m7 companion cut leaves out real?

Seed table: production Brick 2221/o001 F182M m7 seed with the own-band union
on, before the companion cut (faint2/evid1015/brick/seed_f182m.fits, built
by docs/evidence/faint_m7_seed_union/scripts/seeds_fullfield.py).  Restored
own-band sources (m6 vetted, > 60 mas from every production m7 vetted
source) within 2.5 FWHM of a brighter seed source are the companion-cut
group.  For each (separation, flux ratio) bin:

* rel against three independent-visit catalogs (Brick 1182/o004 F200W):
  its m7 vetted catalog (the one the #1015 evidence used), its m6 vetted
  catalog and the union of its m2..m6 vetted catalogs.  o004's m7 seed is
  the production cross-band seed, so o004 m7 lacks the same kind of
  single-band close companions; m6 and earlier are not filtered by it.
* position-angle harmonics of the companion around its brighter neighbour
  (|<exp(i m PA)>|, m = 2, 6): fits to a PSF feature of one visit sit at
  fixed angles; real companions are isotropic.  Expected |.| for isotropic
  positions: ~ sqrt(pi / 4n).

usage: python companion_conf.py <seed_f182m.fits> <outdir>
writes <outdir>/companion_conf.json and <outdir>/companion_pa.npz
(companion_fig.py reads both).
"""
import json
import os
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'faint_m7_seed_union', 'scripts'))
from realness import match_fraction, in_footprint  # noqa: E402
from seed_cuts import rel_flux_matched  # noqa: E402

FWHM = 1.99 * 0.0309  # F182M, arcsec
CAT = '/orange/adamginsburg/jwst/brick/catalogs'
PH = ['m2', 'm3', 'm4', 'resbgsub_m5', 'resbgsub_m6']


def _sc(t):
    s = t['skycoord']
    return s if isinstance(s, SkyCoord) else SkyCoord(s)


def ref_union():
    """o004 m6 vetted plus each earlier phase's sources > 30 mas from the union."""
    scs = []
    for ph in PH:
        scs.append(_sc(Table.read(f'{CAT}/f200w_merged_o004_indivexp_merged_{ph}_dao_basic_vetted.fits')))
    c0 = scs[-1][0]

    def _plane(s):
        # offsets (mas) on the plane tangent at c0: the field spans a few
        # arcmin, so 30 mas separations are preserved to far below a mas
        dx, dy = c0.spherical_offsets_to(s)
        return np.c_[dx.to_value(u.mas), dy.to_value(u.mas)]

    u_ = scs[-1]
    for s in scs[-2::-1]:
        sep, _ = cKDTree(_plane(u_)).query(_plane(s))
        u_ = SkyCoord([u_, s[sep > 30]])
    return u_


seedpath, outdir = sys.argv[1], sys.argv[2]
t = Table.read(seedpath)
sc = _sc(t)
org = np.asarray(t['seed_origin']).astype(str)
own = org == 'own_m6'
inprod = np.asarray(t['sep_prod_m7_mas'], float) < 60
flux = np.asarray(t['flux'], float)
fl = np.where(own, np.asarray(t['m6_flux'], float), flux)
lf = np.log10(np.clip(flux, 1e-30, None))

# nearest brighter seed source within 3 FWHM: separation (FWHM), flux ratio, PA
i1, i2, d, _ = sc.search_around_sky(sc, 3 * FWHM * u.arcsec)
d = d.to_value(u.arcsec) / FWHM
ok = (i1 != i2) & (fl[i2] > fl[i1])
i1, i2, d = i1[ok], i2[ok], d[ok]
order = np.lexsort((d, i1))
i1, i2, d = i1[order], i2[order], d[order]
first = np.r_[True, i1[1:] != i1[:-1]]
sep_b = np.full(len(t), np.inf)
ratio = np.full(len(t), np.nan)
pa = np.full(len(t), np.nan)
sep_b[i1[first]] = d[first]
ratio[i1[first]] = fl[i1[first]] / fl[i2[first]]
pa[i1[first]] = sc[i2[first]].position_angle(sc[i1[first]]).to_value(u.rad)

refs = {'o004_m7': _sc(Table.read(f'{CAT}/f200w_merged_o004_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')),
        'o004_m6': _sc(Table.read(f'{CAT}/f200w_merged_o004_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits')),
        'o004_m2m6': ref_union()}
fp = in_footprint(sc, refs['o004_m7'])
rest = own & ~inprod & fp
ctrl = own & inprod & fp
print('n restored', rest.sum(), 'control', ctrl.sum(), {k: len(v) for k, v in refs.items()}, flush=True)


def harm(p, m):
    p = p[np.isfinite(p)]
    return float(np.abs(np.mean(np.exp(1j * m * p)))) if len(p) else np.nan


SEPB = [(0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
RATB = [(0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1.0001)]
out = []
caches = {k: {} for k in refs}
for grpname, grp in (('restored', rest), ('in_prod', ctrl)):
    for slo, shi in SEPB:
        for rlo, rhi in RATB + [(0, 1.0001)]:
            g = grp & (sep_b >= slo) & (sep_b < shi) & (ratio >= rlo) & (ratio < rhi)
            row = dict(group=grpname, sep=[slo, shi], ratio=[rlo, rhi], n=int(g.sum()),
                       pa_m2=harm(pa[g], 2), pa_m6=harm(pa[g], 6),
                       pa_iso=float(np.sqrt(np.pi / (4 * max(g.sum(), 1)))))
            for k, ref in refs.items():
                if g.sum() >= 20:
                    m, ch = match_fraction(sc[g], ref)
                    rel, _ = rel_flux_matched(sc, lf, g, ctrl, ref, caches[k])
                    row[k] = dict(match=m, chance=ch, rel=rel)
            out.append(row)
            print(json.dumps(row), flush=True)
# position angles for the figure
np.savez(f'{outdir}/companion_pa.npz', pa=pa, sep_b=sep_b, ratio=ratio,
         rest=rest, ctrl=ctrl, lf=lf)
with open(f'{outdir}/companion_conf.json', 'w') as fh:
    json.dump(out, fh, indent=1)
