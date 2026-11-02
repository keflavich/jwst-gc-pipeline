"""#1017: the sources replay A keeps and replay B does not (default: the
current base, base1016, against this branch at k = 5, promq5b: the refused
set), with realness by k_eff = S/N sqrt(qfit^2 - 0.2^2), by per-frame S/N,
and by distance to the nearest saturated star.  Realness as in compare.py,
against the #1015 base's kept stars of the same flux, with the binomial error
on the match fraction.

usage: python refused_slices.py <field> [A-B, default base1016-promq5b]
writes refused_slices_<field>_<A>_minus_<B>.json
"""
import json
import os
import sys

import numpy as np
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import FIELDS, realness, sky  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', '..', 'faint_m7_seed_union', 'scripts'))
from realness import in_footprint  # noqa: E402

KEFF = ((5, 6), (6, 8), (8, 12), (12, 20), (20, np.inf))
SNR = ((0, 10), (10, 20), (20, 50), (50, np.inf))
DSAT = ((0, 1), (1, 2), (2, np.inf))


def rel_err(r):
    if not r.get('n'):
        return np.nan
    m, n = r['match'], r['n']
    return float(np.sqrt(m * (1 - m) / n) / (r['match_exp'] - r['chance_exp']))


def main(field, v='base1016-promq5b'):
    band = FIELDS[field]['band']
    v_a, _, v_b = v.partition('-')
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    ta = Table.read(f'{HERE}/out/{field}_{band}_{v_a}.fits')
    tb = Table.read(f'{HERE}/out/{field}_{band}_{v_b}.fits')
    for t in (ta, tb):
        assert np.array_equal(np.asarray(t['rowid']), np.asarray(base['rowid']))
    kb = np.asarray(base['kept'], bool)
    sel = np.asarray(ta['kept'], bool) & ~np.asarray(tb['kept'], bool)
    ref = sky(Table.read(FIELDS[field]['ref']))
    sc = sky(base)
    infp = in_footprint(sc, ref)
    flux = np.asarray(base['flux'], float)
    snr = flux / np.asarray(tb['flux_err'], float)
    qf = np.asarray(tb['qfit'], float)
    pr = np.asarray(tb['prominence'], float)
    with np.errstate(invalid='ignore'):
        keff = snr * np.sqrt(np.clip(qf ** 2 - 0.2 ** 2, 0, None))
    keff[~(snr > 0)] = np.nan
    dsat = np.load(f'{HERE}/out/{field}_{band}_dsat.npy')
    bsel = kb & infp
    out = {'n': int(sel.sum()), 'n_in_footprint': int((sel & infp).sum()),
           'n_no_keff': int((sel & ~np.isfinite(keff)).sum())}
    print(f'== {field} {band}: {sel.sum()} sources {v_a} keeps and {v_b} does not; '
          f'prominence min {np.nanmin(pr[sel]):.2f}, qfit median {np.nanmedian(qf[sel]):.2f}, '
          f'S/N median {np.nanmedian(snr[sel]):.1f}, k_eff min {np.nanmin(keff[sel]):.2f}, '
          f'no k_eff {out["n_no_keff"]}')
    for name, x, bins in (('k_eff', keff, KEFF), ('snr', snr, SNR), ('dsat_arcsec', dsat, DSAT)):
        rows = []
        for a, b in bins:
            m = sel & (x >= a) & (x < b)
            r = realness(sc[m & infp], flux[m & infp], ref, sc[bsel], flux[bsel])
            rows.append(dict(lo=a, hi=b, n=int(m.sum()), n_in_footprint=int((m & infp).sum()),
                             rel=r.get('rel', np.nan), rel_err=rel_err(r)))
            print(f'   {name} [{a:g},{b:g}): n {int(m.sum()):5d} rel {r.get("rel", np.nan):.2f} '
                  f'+/- {rel_err(r):.2f}')
        out[name] = rows
    with open(f'{HERE}/refused_slices_{field}_{v.replace("-", "_minus_")}.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
