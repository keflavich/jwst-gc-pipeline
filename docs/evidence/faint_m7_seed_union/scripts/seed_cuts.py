"""Which own-band m6 seeds that production m7 lacks are real?

Splits the 'own_m6 NOT in production m7' group (seed_realness.py) by m6
properties and reports the flux-matched realness (vs. 'own_m6 in production
m7') in each slice, against Brick 1182/o004 F200W m7 vetted.

usage: python seed_cuts.py [band]
"""
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from realness import match_fraction, in_footprint  # noqa: E402
from seed_realness import REF, EDGES  # noqa: E402


def rel_flux_matched(sc, lf, g, refgrp, ref, cache):
    num = den = 0.0
    n_used = 0
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = g & (lf >= lo) & (lf < hi)
        if k.sum() < 20:
            continue
        if (lo, hi) not in cache:
            kr = np.flatnonzero(refgrp & (lf >= lo) & (lf < hi))
            if kr.size > 3000:
                kr = np.random.default_rng(0).choice(kr, 3000, replace=False)
            cache[(lo, hi)] = match_fraction(sc[kr], ref) if kr.size >= 20 else (np.nan, np.nan)
        mr, cr = cache[(lo, hi)]
        if not (np.isfinite(mr) and mr > cr):
            continue
        mk, ck = match_fraction(sc[k], ref)
        num += k.sum() * (mk - ck)
        den += k.sum() * (mr - cr)
        n_used += int(k.sum())
    return (num / den if den > 0 else np.nan), n_used


def main(band='f182m'):
    t = Table.read(f'{HERE}/brick/seed_{band}.fits')
    sc = t['skycoord'] if isinstance(t['skycoord'], SkyCoord) else SkyCoord(t['skycoord'])
    ref = Table.read(REF)['skycoord']
    ref = ref if isinstance(ref, SkyCoord) else SkyCoord(ref)
    fp = in_footprint(sc, ref)
    org = np.asarray(t['seed_origin']).astype(str)
    inprod = np.asarray(t['sep_prod_m7_mas'], float) < 60
    lf = np.log10(np.clip(np.asarray(t['flux'], float), 1e-30, None))
    f = np.asarray(t['m6_flux'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snrp = f / np.asarray(t['m6_flux_err_prop'], float)
        snrf = f / np.asarray(t['m6_flux_err'], float)
    qf = np.asarray(t['m6_qfit'], float)
    pr = np.asarray(t['m6_prominence'], float)
    nm = np.asarray(t['m6_nmatch'], float)
    skc = np.asarray(t['m6_sky_clean'], bool)
    refgrp = (org == 'own_m6') & inprod & fp
    tgt = (org == 'own_m6') & ~inprod & fp
    cache = {}
    slices = [
        ('all', np.ones(len(t), bool)),
        ('qfit<=0.2', qf <= 0.2), ('qfit 0.2-0.4', (qf > 0.2) & (qf <= 0.4)), ('qfit>0.4', qf > 0.4),
        ('prom<5', pr < 5), ('prom 5-10', (pr >= 5) & (pr < 10)), ('prom 10-20', (pr >= 10) & (pr < 20)),
        ('prom>=20', pr >= 20),
        ('S/N_prop<10', snrp < 10), ('S/N_prop 10-20', (snrp >= 10) & (snrp < 20)),
        ('S/N_prop 20-40', (snrp >= 20) & (snrp < 40)), ('S/N_prop>=40', snrp >= 40),
        ('S/N_frame<3', snrf < 3), ('S/N_frame 3-5', (snrf >= 3) & (snrf < 5)), ('S/N_frame>=5', snrf >= 5),
        ('nmatch<=4', nm <= 4), ('nmatch 5-10', (nm >= 5) & (nm <= 10)), ('nmatch 11-20', (nm >= 11) & (nm <= 20)),
        ('nmatch>20', nm > 20),
        ('sky_clean', skc), ('not sky_clean', ~skc),
        ('qfit<=0.4 & prom>=10', (qf <= 0.4) & (pr >= 10)),
        ('prom>=10 & S/N_frame>=5', (pr >= 10) & (snrf >= 5)),
    ]
    lines = [f'== Brick {band.upper()}: own-band m6 seeds NOT in production m7 (in F200W o004 footprint, '
             f'n={tgt.sum()}); flux-matched realness vs own-band m6 seeds production m7 has (1 = as real)']
    for name, s in slices:
        g = tgt & s
        if g.sum() < 20:
            lines.append(f'  {name:26s} n={g.sum():7d}')
            continue
        m, ch = match_fraction(sc[g], ref)
        rel, nu = rel_flux_matched(sc, lf, g, refgrp, ref, cache)
        gi = refgrp & s
        lines.append(f'  {name:26s} n={g.sum():7d}  match {m:.2f} (chance {ch:.2f})  rel {rel:.2f}   '
                     f'[same slice of the in-prod group: n={gi.sum()}]')
    txt = '\n'.join(lines)
    print(txt, flush=True)
    with open(f'{HERE}/seed_cuts_{band}.txt', 'w') as fh:
        fh.write(txt + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
