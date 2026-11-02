"""#1017 qfit noise bound: realness of the prominence-keep sources in each
k_eff band (the sources that each step of k admits).

k_eff = S/N * sqrt(qfit^2 - qfit_max^2) is the smallest k whose bound
sqrt(qfit_max^2 + (k/S/N)^2) admits a source; a source in band (a, b] is
admitted when k moves from a to b.  Sources with a non-finite qfit or a
non-finite or non-positive S/N have no k_eff and get their own row (k_lo NaN):
the bound refuses them at every k > 0.

Two ways to pick the sources:
  <variant>        the variant's additions over the #1015 seed base with
                   prominence >= 7 (the #1018-era study, variant prom2 = #1018
                   at k = 0)
  <A>-<B>          sources kept by replay A and not by replay B; with A the k = 0
                   replay and B the k = 1e-6 replay (bound = flat qfit_max) this
                   is exactly the set whose keep depends on the bound
                   (current base: promq0b-promqeb)
Realness as in compare.py, against #1015-base-kept stars of the same flux.

usage: python kband_qsnr.py <field> [variant=prom2 | A-B]
writes kband_qsnr_<field>_<variant>.json ('-' written as '_minus_')
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

BANDS = ((0, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.0),
         (6.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, np.inf), (np.nan, np.nan))
PROM_MIN = 7.0
QFIT_MAX = 0.2


def k_eff(snr, qf):
    """Smallest k whose bound admits each source; NaN when the bound cannot (or
    need not) be expressed by k: non-finite qfit, non-finite or non-positive S/N."""
    ok = np.isfinite(snr) & (snr > 0) & np.isfinite(qf)
    out = np.full(snr.shape, np.nan)
    out[ok] = snr[ok] * np.sqrt(np.clip(qf[ok] ** 2 - QFIT_MAX ** 2, 0, None))
    return out


def main(field, v='prom2'):
    cfg = FIELDS[field]
    band = cfg['band']
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    v_a, _, v_b = v.partition('-')
    t = Table.read(f'{HERE}/out/{field}_{band}_{v_a}.fits')
    assert np.array_equal(np.asarray(t['rowid']), np.asarray(base['rowid']))
    kb = np.asarray(base['kept'], bool)
    if v_b:
        tb = Table.read(f'{HERE}/out/{field}_{band}_{v_b}.fits')
        assert np.array_equal(np.asarray(tb['rowid']), np.asarray(base['rowid']))
        sel = np.asarray(t['kept'], bool) & ~np.asarray(tb['kept'], bool)
        assert not (np.asarray(tb['kept'], bool) & ~np.asarray(t['kept'], bool)).any(), f'{v_b} keeps a source {v_a} does not'
    else:
        sel = np.asarray(t['kept'], bool) & ~kb
    ref = sky(Table.read(cfg['ref']))
    sc = sky(base)
    infp = in_footprint(sc, ref)
    flux = np.asarray(base['flux'], float)
    snr = flux / np.asarray(t['flux_err'], float)
    qf = np.asarray(t['qfit'], float)
    pr = np.asarray(t['prominence'], float)
    keff = k_eff(snr, qf)
    sel = sel & (pr >= PROM_MIN)
    bsel = kb & infp
    res = []
    print(f'== {field} {band} ({v}): {int(sel.sum())} sources with prominence >= {PROM_MIN:g} per k_eff band')
    for a, b in BANDS:
        if np.isnan(a):
            m = sel & ~np.isfinite(keff)
        else:
            m = sel & (keff > a) & (keff <= b + 1e-9)
        r = realness(sc[m & infp], flux[m & infp], ref, sc[bsel], flux[bsel])
        res.append(dict(k_lo=a, k_hi=b, n_added=int(m.sum()), **{f"r_{x}": y for x, y in r.items()}))
        print(f'   k_eff ({a:g},{b:g}]: n {int(m.sum()):6d} in footprint {int((m & infp).sum()):6d} '
              f'rel {r.get("rel", np.nan):.2f}', flush=True)
    with open(f'{HERE}/kband_qsnr_{field}_{v.replace("-", "_minus_")}.json', 'w') as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
