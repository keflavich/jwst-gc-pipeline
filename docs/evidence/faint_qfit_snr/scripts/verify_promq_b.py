"""#1017 replay identity checks on the post-#1016 base (pr/faint-reference-fields
ef01f404, which carries #1016's flux_err_prop floor and #1019's local sky-clean):

  * promq0b (this branch, qfit_snr_k = 0) keeps exactly the rows base1016 keeps;
  * promq5b (this branch at defaults, k = 5) keeps a subset of base1016;
    base1016 & ~promq5b is the refused set;
  * every refused source has prominence >= 7 and k_eff > 5 or no finite k_eff;
  * with promqeb (k = 1e-6, bound = flat qfit_max; skipped until that replay
    exists): promqeb is a subset of
    promq5b, and the refused set is exactly the bound-dependent set
    (promq0b & ~promqeb) with k_eff > 5 or no finite k_eff;
  * the refused set against the #1018-era refused set (prom2 adds that the
    first #1017, qsnrp7, does not add), and its realness.

Realness as in compare.py, against the #1015 base's kept stars of the same flux
(variant seed), with a binomial error on the match fraction.

usage: python verify_promq_b.py [field ...]
writes verify_promq_b.json
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


def load(field, v):
    t = Table.read(f'{HERE}/out/{field}_{FIELDS[field]["band"]}_{v}.fits')
    return t, np.asarray(t['kept'], bool)


def rel_err(r):
    if not r.get('n'):
        return np.nan
    m, n = r['match'], r['n']
    return float(np.sqrt(m * (1 - m) / n) / (r['match_exp'] - r['chance_exp']))


def main(*fields):
    fields = fields or ('brick', 'sgrb2', 'w51')
    out = {}
    for f in fields:
        seed, ks = load(f, 'seed')
        tb, kb = load(f, 'base1016')
        t0, k0 = load(f, 'promq0b')
        t5, k5 = load(f, 'promq5b')
        _, k2 = load(f, 'prom2')
        _, kq = load(f, 'qsnrp7')
        for t in (tb, t0, t5):
            assert np.array_equal(np.asarray(t['rowid']), np.asarray(seed['rowid']))
        refused = kb & ~k5
        only1018 = k2 & ~ks & ~(kq & ~ks)
        snr = np.asarray(t5['flux'], float) / np.asarray(t5['flux_err'], float)
        qf = np.asarray(t5['qfit'], float)
        pr = np.asarray(t5['prominence'], float)
        with np.errstate(invalid='ignore'):
            keff = snr * np.sqrt(np.clip(qf ** 2 - 0.2 ** 2, 0, None))
        # the bound is the flat qfit_max at S/N <= 0, so k_eff is undefined there
        keff[~(snr > 0)] = np.nan
        fin = np.isfinite(keff)
        ref = sky(Table.read(FIELDS[f]['ref']))
        sc = sky(seed)
        infp = in_footprint(sc, ref)
        flux = np.asarray(seed['flux'], float)
        bsel = ks & infp
        rr = realness(sc[refused & infp], flux[refused & infp], ref, sc[bsel], flux[bsel])
        r = dict(n_seed=int(ks.sum()), n_base1016=int(kb.sum()), n_promq0b=int(k0.sum()),
                 n_promq5b=int(k5.sum()),
                 promq0b_eq_base1016=bool(np.array_equal(k0, kb)),
                 promq5b_not_in_base1016=int((k5 & ~kb).sum()),
                 refused=int(refused.sum()),
                 refused_min_prom=float(np.nanmin(pr[refused])) if refused.any() else None,
                 refused_min_keff=float(np.min(keff[refused & fin])) if (refused & fin).any() else None,
                 refused_keff_le5=int((refused & fin & (keff <= 5.0)).sum()),
                 refused_nonfinite_keff=int((refused & ~fin).sum()),
                 refused_in_seed=int((refused & ks).sum()),
                 only1018=int(only1018.sum()),
                 refused_and_only1018=int((refused & only1018).sum()),
                 refused_not_only1018=int((refused & ~only1018).sum()),
                 only1018_not_refused=int((only1018 & ~refused).sum()),
                 only1018_not_in_base1016=int((only1018 & ~kb).sum()),
                 refused_in_footprint=int(rr.get('n', 0)),
                 refused_rel=rr.get('rel'), refused_rel_err=rel_err(rr))
        if not os.path.exists(f'{HERE}/out/{f}_{FIELDS[f]["band"]}_promqeb.fits'):
            out[f] = r
            print(f, json.dumps(r), '(no promqeb replay)')
            continue
        te, ke = load(f, 'promqeb')
        assert np.array_equal(np.asarray(te['rowid']), np.asarray(seed['rowid']))
        kdep = k0 & ~ke
        r.update(n_promqeb=int(ke.sum()),
                 promqeb_not_in_promq5b=int((ke & ~k5).sum()),
                 bound_dependent=int(kdep.sum()),
                 refused_eq_dependent_keff_gt5=bool(np.array_equal(
                     refused, kdep & (~fin | (keff > 5.0)))))
        out[f] = r
        print(f, json.dumps(r))
    with open(f'{HERE}/verify_promq_b.json', 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
