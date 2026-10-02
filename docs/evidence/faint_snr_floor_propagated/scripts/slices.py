"""Realness of the sources a branch adds, sliced by the quantities its rule uses.

Same realness definition as compare.py: matched within 60 mas against the
reference catalog, chance from shifted positions, expectation from the
#1015 base-kept sources of the same flux (all saturated-star distances).

  qsnr (#1017): k_eff = S/N * sqrt(qfit^2 - qfit_max^2), the smallest k that
        admits the source (S/N = flux/flux_err, as the vetting uses); also
        by prominence and S/N.
  prom (#1018): branch (plain prominence >= 5 vs robust-only), prominence,
        robust prominence, core concentration; lost sources by prominence.
  lsky (#1019): by S/N and prominence.

usage: python slices.py <field> <variant> [...]
"""
import json
import os
import sys

import numpy as np
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import FIELDS, realness, sky  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'evid1015'))
from realness import in_footprint  # noqa: E402

QFIT_MAX = 0.2


def table(field, v, ref, base, kb, infp, flux, sc):
    t = Table.read(f'{HERE}/out/{field}_{FIELDS[field]["band"]}_{v}.fits')
    assert np.array_equal(np.asarray(t['rowid']), np.asarray(base['rowid']))
    kv = np.asarray(t['kept'], bool)
    add, lost = kv & ~kb, kb & ~kv
    bsel = kb & infp
    snr = np.asarray(t['flux'], float) / np.asarray(t['flux_err'], float)
    qf = np.asarray(t['qfit'], float)
    pr = np.asarray(t['prominence'], float)

    def rel(mask):
        m = mask & infp
        r = realness(sc[m], flux[m], ref, sc[bsel], flux[bsel])
        return r

    out = {}

    def run(name, base_mask, col, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = base_mask & (col >= lo) & (col < hi)
            r = rel(k)
            rows.append(dict(lo=lo, hi=hi, n_all=int(k.sum()), **r))
        out[name] = rows

    if v.startswith('qsnr'):
        keff = snr * np.sqrt(np.clip(qf ** 2 - QFIT_MAX ** 2, 0, None))
        run('added by k_eff', add, keff, [0, 2, 3, 3.5, 4, 4.5, 5.01])
        run('added by prominence', add, pr, [3, 5, 7, 10, 20, np.inf])
        run('added by S/N', add, snr, [5, 7, 10, 15, 25, np.inf])
        run('added by qfit', add, qf, [0.2, 0.3, 0.4, 0.5, 0.7, 2])
    elif v.startswith('prom2'):
        # #1018 v2: (peakSB AND prominence >= guard) OR prominence >= keep; robust branch off
        ov = json.loads(t.meta['PROVJSON']).get('overrides', {})
        pmin = ov.get('manual_ext_star_prom_min', 7.0)
        out['added: all'] = [dict(n_all=int(add.sum()), **rel(add))]
        out['lost: all'] = [dict(n_all=int(lost.sum()), **rel(lost))]
        run('added by prominence', add, np.nan_to_num(pr, nan=-99), [-100, pmin, 8.5, 10, 20, 50, np.inf])
        run('added by S/N', add, snr, [0, 5, 10, 20, 50, np.inf])
        run('added by qfit', add, qf, [0, 0.2, 0.3, 0.5, 1, np.inf])
        run('lost by prominence', lost, np.nan_to_num(pr, nan=-99), [-100, 0, 1, 2, 3, 4, 5, 6, 7])
        run('lost by S/N', lost, snr, [0, 5, 10, 20, 50, np.inf])
        prb = np.asarray(base['prominence'], float) if 'prominence' in base.colnames else pr
        run('base-kept, all, by prominence', kb, np.nan_to_num(prb, nan=-99), [-100, 0, 1, 2, 3, 4, 5, 6, 7, 8.5, 10, 20, np.inf])
    elif v.startswith('prom'):
        nan = np.full(len(t), np.nan)
        prr = np.asarray(t['prominence_robust'], float) if 'prominence_robust' in t.colnames else nan
        conc = np.asarray(t['core_concentration'], float) if 'core_concentration' in t.colnames else nan
        pmin = json.loads(t.meta['PROVJSON']).get('overrides', {}).get('manual_ext_star_prom_min', 5.0)
        plain = add & (pr >= pmin)
        robust_only = add & ~(pr >= pmin)
        out[f'added: plain prominence >= {pmin:g}'] = [dict(n_all=int(plain.sum()), **rel(plain))]
        out['added: robust only'] = [dict(n_all=int(robust_only.sum()), **rel(robust_only))]
        run('added plain by prominence', plain, pr, [5, 7, 10, 20, 50, np.inf])
        run('added robust-only by robust prominence', robust_only, prr, [8, 10, 15, 25, np.inf])
        run('added by concentration', add, np.nan_to_num(conc, nan=-1), [-1.5, 0, 0.6, 0.8, 1.0, 1.2, np.inf])
        run('added by qfit', add, qf, [0, 0.2, 0.3, 0.5, 1, np.inf])
        run('added by S/N', add, snr, [0, 5, 10, 20, 50, np.inf])
        run('lost by prominence', lost, np.nan_to_num(pr, nan=-99), [-100, 0, 1, 2, 3, 4, 5, 6, 7, 8.5, 10.01])
        prb = np.asarray(base['prominence'], float) if 'prominence' in base.colnames else pr
        run('base-kept, all, by prominence', kb, np.nan_to_num(prb, nan=-99), [-100, 0, 1, 2, 3, 4, 5, 6, 7, 8.5, 10, 20, np.inf])
        run('lost by S/N', lost, snr, [0, 5, 10, 20, 50, np.inf])
    elif v == 'snr':
        # #1016: S/N floor on flux/flux_err_prop.  Keep path of each addition,
        # and what #1018 v2's peak_SB prominence guard (>= 4) would leave.
        snrp = np.asarray(t['flux'], float) / np.asarray(t['flux_err_prop'], float)
        lb = np.asarray(t['local_bkg'], float)
        psb = np.asarray(t['peak_sb'], float)
        fl = np.asarray(t['flags'], int) if 'flags' in t.colnames else np.zeros(len(t), int)
        path_q = qf <= QFIT_MAX
        path_f = ~path_q & (fl == 1)
        path_p = ~path_q & ~path_f & (lb > 0) & (psb > 20 * lb)
        path_o = ~(path_q | path_f | path_p)
        out['added: all'] = [dict(n_all=int(add.sum()), **rel(add))]
        for nm, pm in (('qfit<=0.2', path_q), ('flags==1', path_f), ('peakSB', path_p), ('other (sky-clean etc.)', path_o)):
            out[f'added via {nm}'] = [dict(n_all=int((add & pm).sum()), **rel(add & pm))]
        g = np.isfinite(pr) & (pr < 4)
        out['added via peakSB, prominence < 4 (#1018 v2 guard drops)'] = [dict(n_all=int((add & path_p & g).sum()), **rel(add & path_p & g))]
        out['added via peakSB, prominence >= 4 or unmeasured'] = [dict(n_all=int((add & path_p & ~g).sum()), **rel(add & path_p & ~g))]
        run('added by prominence', add, np.nan_to_num(pr, nan=-99), [-100, 0, 2, 3, 4, 5, 7, 10, np.inf])
        run('added by propagated S/N', add, snrp, [0, 5, 7, 10, 17, np.inf])
        run('added by per-frame S/N', add, snr, [0, 1, 2, 3, 4, 5])
        run('added by qfit', add, qf, [0, 0.2, 0.4, 0.6, 1, np.inf])
    elif v == 'lsky':
        run('added by S/N', add, snr, [0, 4, 5, 7, 10, np.inf])
        run('added by prominence', add, pr, [0, 5, 7, 10, 20, np.inf])
        run('added by qfit', add, qf, [0, 0.2, 0.4, 0.6, 1, np.inf])
    return out


def main(field, *variants):
    cfg = FIELDS[field]
    base = Table.read(f'{HERE}/out/{field}_{cfg["band"]}_seed.fits')
    ref = sky(Table.read(cfg['ref']))
    sc = sky(base)
    kb = np.asarray(base['kept'], bool)
    infp = in_footprint(sc, ref)
    flux = np.asarray(base['flux'], float)
    res = {}
    for v in variants:
        res[v] = table(field, v, ref, base, kb, infp, flux, sc)
        for name, rows in res[v].items():
            print(f'== {field} {v}: {name}')
            for r in rows:
                rng = f'[{r["lo"]:g}, {r["hi"]:g})' if 'lo' in r else ''
                if r.get('n', 0) == 0:
                    print(f'   {rng:>14s} n_all {r["n_all"]:6d}  n=0')
                    continue
                print(f'   {rng:>14s} n_all {r["n_all"]:6d} n_fp {r["n"]:6d} match {r["match"]:.2f} '
                      f'chance {r["chance"]:.2f} exp {r["match_exp"]:.2f} rel {r["rel"]:.2f}', flush=True)
    with open(f'{HERE}/slices_{field}_{"-".join(variants)}.json', 'w') as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main(*sys.argv[1:])
