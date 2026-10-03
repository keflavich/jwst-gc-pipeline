"""Realness of the m7 seed groups #1015 adds, against an independent visit.

Seed table (seeds_fullfield.py): every F182M m7 seed with its origin
(crossband / own_m6 / i2d) and the separation to the nearest production m7
vetted source.  Groups:
  own_m6, in production m7   (sep < 60 mas): own-band m6 sources production m7 also has
  own_m6, NOT in production m7: the sources #1015 restores to the m7 seed
  i2d,   in / not in production m7
  crossband
Reference: Brick 1182/o004 F200W m7 vetted (independent visit, other
detectors).  Per log-flux bin: n, match fraction within 60 mas, chance
(shifted positions); rel = (m - ch) / (m_ref - ch_ref) against the
'own_m6, in production m7' group of the same flux bin.

usage: python seed_realness.py [band]
"""
import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from realness import match_fraction, in_footprint  # noqa: E402

REF = '/orange/adamginsburg/jwst/brick/catalogs/f200w_merged_o004_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
EDGES = np.arange(0.0, 6.01, 0.5)


def main(band='f182m'):
    t = Table.read(f'{HERE}/brick/seed_{band}.fits')
    sc = t['skycoord'] if isinstance(t['skycoord'], SkyCoord) else SkyCoord(t['skycoord'])
    ref = Table.read(REF)['skycoord']
    ref = ref if isinstance(ref, SkyCoord) else SkyCoord(ref)
    fp = in_footprint(sc, ref)
    org = np.asarray(t['seed_origin']).astype(str)
    inprod = np.asarray(t['sep_prod_m7_mas'], float) < 60
    flux = np.asarray(t['flux'], float)
    lf = np.log10(np.clip(flux, 1e-30, None))
    qf = np.asarray(t['m6_qfit'], float)
    groups = {
        'own_m6 in prod m7': (org == 'own_m6') & inprod,
        'own_m6 NOT in prod m7': (org == 'own_m6') & ~inprod,
        'own_m6 NOT in prod m7, qfit<=0.2': (org == 'own_m6') & ~inprod & (qf <= 0.2),
        'own_m6 NOT in prod m7, qfit>0.2': (org == 'own_m6') & ~inprod & (qf > 0.2),
        'i2d in prod m7': (org == 'i2d') & inprod,
        'i2d NOT in prod m7': (org == 'i2d') & ~inprod,
        'crossband': org == 'crossband',
    }
    out = dict(band=band, ref=REF, frac_in_footprint=float(fp.mean()), groups={})
    refgrp = groups['own_m6 in prod m7'] & fp
    refbins = {}
    lines = [f'== Brick {band.upper()} m7 seed groups vs F200W o004 m7 vetted (independent visit); '
             f'{fp.mean():.2f} of seeds in its footprint']
    for name, g in groups.items():
        g = g & fp
        m, ch = match_fraction(sc[g], ref)
        rows = []
        num, den = 0.0, 0.0
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            k = g & (lf >= lo) & (lf < hi)
            if k.sum() < 20:
                continue
            mk, ck = match_fraction(sc[k], ref)
            if (lo, hi) not in refbins:
                kr = refgrp & (lf >= lo) & (lf < hi)
                if kr.sum() > 3000:
                    kr_idx = np.random.default_rng(0).choice(np.flatnonzero(kr), 3000, replace=False)
                else:
                    kr_idx = np.flatnonzero(kr)
                refbins[(lo, hi)] = match_fraction(sc[kr_idx], ref) if kr_idx.size >= 20 else (np.nan, np.nan)
            mr, cr = refbins[(lo, hi)]
            rel = (mk - ck) / (mr - cr) if np.isfinite(mr) and mr > cr else np.nan
            rows.append(dict(lo=lo, hi=hi, n=int(k.sum()), match=mk, chance=ck, ref_match=mr, rel=rel))
            if np.isfinite(rel):
                num += k.sum() * (mk - ck)
                den += k.sum() * (mr - cr)
        rel_all = num / den if den > 0 else np.nan
        out['groups'][name] = dict(n=int(g.sum()), match=m, chance=ch, rel_flux_matched=rel_all, bins=rows)
        lines.append(f'  {name:36s} n={g.sum():7d}  match {m:.2f} (chance {ch:.2f})  rel(flux-matched) {rel_all:.2f}')
        lines.append('      ' + '  '.join(f'[{r["lo"]:.1f},{r["hi"]:.1f}) n={r["n"]} m={r["match"]:.2f} rel={r["rel"]:.2f}'
                                         for r in rows))
    txt = '\n'.join(lines)
    print(txt, flush=True)
    with open(f'{HERE}/seed_realness_{band}.txt', 'w') as fh:
        fh.write(txt + '\n')
    with open(f'{HERE}/seed_realness_{band}.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
