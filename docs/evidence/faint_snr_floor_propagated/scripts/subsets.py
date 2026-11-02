"""Realness of three low-quality subsets of the sources this branch adds over
the #1018 v2 replay: prominence < 4 (or unmeasured), qfit 0.2-0.6, and the
flags == 1 keep path; their union and the rest; and a split at
qfit * S/N = 3 (S/N = flux/flux_err, as the vetting uses).

Same realness definition as compare.py and slices.py, expectation from the
#1015 base-kept sources of the same flux at all saturated-star distances.

usage: python subsets.py <field>   (reads out/<field>_<band>_{seed,prom2,snrp2}.fits)
"""
import os
import sys

import numpy as np
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import FIELDS, realness, sky  # noqa: E402
# realness.py lives with the #1015 seed-union evidence scripts
sys.path.insert(0, os.path.join(HERE, '..', '..', 'faint_m7_seed_union', 'scripts'))
from realness import in_footprint  # noqa: E402


def main(field):
    cfg = FIELDS[field]
    band = cfg['band']
    base = Table.read(f'{HERE}/out/{field}_{band}_seed.fits')
    ref = sky(Table.read(cfg['ref']))
    sc = sky(base)
    infp = in_footprint(sc, ref)
    flux = np.asarray(base['flux'], float)
    bsel = np.asarray(base['kept'], bool) & infp
    t = Table.read(f'{HERE}/out/{field}_{band}_snrp2.fits')
    t0 = Table.read(f'{HERE}/out/{field}_{band}_prom2.fits')
    add = np.asarray(t['kept'], bool) & ~np.asarray(t0['kept'], bool)
    qf = np.asarray(t['qfit'], float)
    pr = np.asarray(t['prominence'], float)
    # merged flags is the per-frame mean (float); the vetting keeps exactly 1.0
    fl = np.asarray(t['flags'], float)
    snr = np.asarray(t['flux'], float) / np.asarray(t['flux_err'], float)

    def show(mask, name):
        m = mask & add
        r = realness(sc[m & infp], flux[m & infp], ref, sc[bsel], flux[bsel])
        print(f'{field:6s} {name:32s} n {int(m.sum()):6d} n_fp {r.get("n", 0):6d} '
              f'rel {r.get("rel", float("nan")):.2f}')

    lowpr = ~(np.nan_to_num(pr, nan=-99) >= 4)
    lowq = (qf >= 0.2) & (qf < 0.6)
    pf = (qf > 0.2) & (fl == 1)
    low = lowpr | lowq | pf
    show(lowpr, 'prominence < 4 (or unmeasured)')
    show(lowq, 'qfit 0.2-0.6')
    show(pf, 'flags == 1 path')
    show(low, 'union of the three')
    show(~low, 'rest')
    show(qf * snr < 3, 'qfit * S/N < 3')
    show(qf * snr >= 3, 'qfit * S/N >= 3')


if __name__ == '__main__':
    main(sys.argv[1])
