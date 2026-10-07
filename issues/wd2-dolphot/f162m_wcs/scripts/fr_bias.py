"""dm = (ours - dolphot) by forced_refit_frac per SW band, Q mainfcbg m8.

Forced-refit rows keep the cross-band seed position.  The m7 seed drifts by
0.2-0.35 px radially in F162M/F164N (per-detector frame term, ~10 mas) and
<= 0.15 px in F212N, so if the seed error matters, F162M/F164N forced rows
read fainter than their free rows by more than F212N's do.  ZP per band from
free rows (frac == 0), 17-21 mag, not saturated/replaced."""
import numpy as np
from astropy.table import Table

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
m = Table.read(f'{D}/matched_Q_mainfcbg.fits')
t = Table.read(f'{Q}/tree_mainfcbg/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits')
oi = np.asarray(m['our_idx'])
ok_i = oi >= 0


def col(tab, c, idx=None, fill=np.nan):
    a = tab[c] if idx is None else tab[c][idx]
    return np.asarray(a.filled(fill) if hasattr(a, 'filled') else a)


BANDS = ['115W', '150W', '162M', '164N', '182M', '187N', '200W', '212N']
BINS = [('free (0)', lambda f: f == 0), ('0<f<0.5', lambda f: (f > 0) & (f < 0.5)),
        ('0.5<=f<1', lambda f: (f >= 0.5) & (f < 1)), ('f==1', lambda f: f == 1)]
out = {}
lines = []
for b in BANDS:
    lb = 'f' + b.lower()
    ref = col(m, 'ref_' + b).astype(float)
    our = col(m, 'our_' + b).astype(float)
    frac = np.full(len(m), np.nan)
    frac[ok_i] = col(t, 'forced_refit_frac_' + lb, oi[ok_i])
    bad = np.zeros(len(m), bool)
    for c in ('is_saturated_', 'replaced_saturated_'):
        if c + lb in t.colnames:
            bad[ok_i] |= col(t, c + lb, oi[ok_i], False).astype(bool)
    ok = np.isfinite(ref) & np.isfinite(our) & ok_i & np.isfinite(frac) & ~bad
    dm = our - ref
    zp = np.nanmedian(dm[ok & (frac == 0) & (ref > 17) & (ref < 21)])
    dm = dm - zp
    out[b] = (ref, dm, frac, ok)
    lines.append(f'F{b}: zp {zp:+.3f}  N matched ok {ok.sum()}')
    for lo, hi in ((15, 19), (19, 21), (21, 23)):
        sel = ok & (ref >= lo) & (ref < hi)
        cells = []
        for name, fn in BINS:
            s = sel & fn(frac)
            med = np.nanmedian(dm[s]) if s.sum() >= 5 else np.nan
            mad = 1.4826 * np.nanmedian(np.abs(dm[s] - med)) if s.sum() >= 5 else np.nan
            cells.append(f'{name}: N {s.sum():5d} med {med:+.3f} madstd {mad:.3f}')
        lines.append(f'  {lo}-{hi}  ' + ' | '.join(cells))
txt = '\n'.join(lines)
print(txt)
open('fr_bias.txt', 'w').write(txt + '\n')
np.save('fr_bias_cache.npy', {b: tuple(np.asarray(x) for x in v) for b, v in out.items()}, allow_pickle=True)
