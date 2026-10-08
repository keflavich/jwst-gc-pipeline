"""Does the LW faint-end ours-dolphot offset depend on how our row was made? Split by iter_found and nmatch."""
import numpy as np
from astropy.table import Table

t = Table.read('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_mainfcbg.fits')
for b, lo, hi in (('410M', 18.5, 20.5), ('405N', 17.5, 19.5), ('277W', 19, 21), ('200W', 19, 21)):
    r, o = np.asarray(t[f'ref_{b}'], float), np.asarray(t[f'our_{b}'], float)
    sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
    x = 0.5 * (o + r)
    ok = np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90) & (x >= lo) & (x < hi)
    it = np.asarray(t[f'our_iter_found_{b}'])
    nm = np.asarray(t[f'our_nmatch_{b}'], float)
    print(f'== {b} mean mag [{lo},{hi}) N={ok.sum()} median dm={np.median(o[ok]-r[ok]):+.3f}')
    for v in np.unique(it[ok]):
        s = ok & (it == v)
        if s.sum() >= 20:
            print(f'   iter_found={v!s:>6}: N={s.sum():5d} dm={np.median(o[s]-r[s]):+.3f}')
    for v in np.unique(nm[ok & np.isfinite(nm)]):
        s = ok & (nm == v)
        if s.sum() >= 20:
            print(f'   nmatch={v:4.0f}: N={s.sum():5d} dm={np.median(o[s]-r[s]):+.3f}')
