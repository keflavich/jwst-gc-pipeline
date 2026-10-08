"""Round 3 scoring.  usage: python score3.py BAND  -> out3/score3_<band>.pkl, out3/tables3_<band>.md
Merges out/ (round 1 columns) and out2/ (round 2) per frame, builds +cap variants (a = min(a_variant, cap*peak ratio)) and the
wing self-calibration variants (flux / C(r_star), r_star = a: wingcal_rmask, b: pre-ZF DQ SATURATED component radius, c: post-ZF deep core radius)."""
import glob, sys, pickle, os
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
band = sys.argv[1]
BINS = {'150W': [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19)],
        '200W': [(13, 14.5), (14.5, 15), (15, 15.5), (15.5, 16), (16, 18)],
        '250M': [(10, 13), (13, 14), (14, 15), (15, 16), (16, 18)],
        '300M': [(10, 13), (13, 14), (14, 15), (15, 16), (16, 18)]}[band]
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
tgt = A.sky[A.idx[ok]]
ref = A.ref[band][ok]
edge = float(np.percentile(ref, 95))
dm_final = A.dm(band)[ok]
tabs = []
for fn2 in sorted(glob.glob(f'{Q}/satrefit/out2/{band}_*_satrefit.fits')):
    t2 = Table.read(fn2)
    fn1 = fn2.replace('/out2/', '/out/')
    if os.path.exists(fn1):
        t1 = Table.read(fn1)
        assert len(t1) == len(t2) and np.allclose(np.asarray(t1['a_cat'], float), np.asarray(t2['a_cat'], float), equal_nan=True)
        for c in t1.colnames:
            if c.startswith('a_') and c not in t2.colnames:
                t2[c] = t1[c]
    fn3 = fn2.replace('/out2/', '/out3/').replace('_satrefit.fits', '_satrefit3.fits')
    t3 = Table.read(fn3)
    assert len(t3) == len(t2) and np.allclose(np.asarray(t3['a_cat'], float), np.asarray(t2['a_cat'], float), equal_nan=True)
    for c in t3.colnames:
        if (c.startswith('a_') or c.startswith('nrw_')) and c not in t2.colnames:
            t2[c] = t3[c]
    t2['nfit3'] = t3['nfit']
    assert np.allclose(np.asarray(t3['a_base'], float), np.asarray(t2['a_base'], float), rtol=1e-6, equal_nan=True)
    t2.meta = {k: v for k, v in t2.meta.items() if k in ('WC_L', 'WC_CAL', 'WC_NPK')}
    t2['frame'] = os.path.basename(fn2)
    tabs.append(t2)
T = vstack(tabs, metadata_conflicts='silent')
good = np.asarray(T['label']) > 0
sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
vnames = [c[2:] for c in T.colnames if c.startswith('a_') and c not in ('a_cat', 'a_raw')]
a = {v: np.asarray(T['a_' + v], float) for v in vnames}
acat, araw = np.asarray(T['a_cat'], float), np.asarray(T['a_raw'], float)
capped = araw < 0.999 * acat
cap = np.where(capped, araw, np.inf)
n = ok.sum()


def med_per_star(x):
    out = np.full(n, np.nan)
    m = good[j] & np.isfinite(x[j])
    ii, xx = i[m], x[j][m]
    order = np.argsort(ii, kind='stable')
    ii, xx = ii[order], xx[order]
    u_, st = np.unique(ii, return_index=True)
    for k, s, e in zip(u_, st, list(st[1:]) + [len(ii)]):
        out[k] = np.median(xx[s:e])
    return out


s_cap = med_per_star(-2.5 * np.log10(acat / araw))
dm_unc = dm_final + s_cap
have_base = np.isfinite(s_cap)
dm = {'final': dm_final, 'uncapped': dm_unc}
ratio = {}


def add(name, a_eff):
    with np.errstate(invalid='ignore', divide='ignore'):
        sh = med_per_star(-2.5 * np.log10(a_eff / a['base']))
        ratio[name] = med_per_star(a_eff / a['base'])
    dm[name] = dm_unc + sh


RW = [v for v in vnames if v.startswith('rw')]
add('bgfree', a['bgfree'])
add('bgfree+cap', np.minimum(a['bgfree'], cap))
pk = np.asarray(T['pk_bgfree+v7b'], float)
pk = np.where(np.isfinite(pk), pk, 1.0)
add('bgfree+v7b+cap', np.minimum(a['bgfree+v7b'], cap * pk))
add('base+cap', np.minimum(a['base'], cap))
for v in RW:
    add(v, a[v])
    add(v + '+cap', np.minimum(a[v], cap))
names = ['final', 'uncapped', 'base+cap', 'bgfree', 'bgfree+cap', 'bgfree+v7b+cap'] + [k for k in dm if k not in ('final', 'uncapped', 'base+cap', 'bgfree', 'bgfree+cap', 'bgfree+v7b+cap')]
have = have_base.copy()
for k in names:
    have &= np.isfinite(dm[k])
frac = {}
for v in RW:
    if v.endswith('+bgfree'):
        continue
    nm = 'nrw_' + v
    with np.errstate(invalid='ignore', divide='ignore'):
        frac[v] = med_per_star(np.asarray(T[nm], float) / np.asarray(T['nfit3'], float))
zpwin_ok = A.matched & ~A.rep[band] & np.isfinite(A.dm(band)) & np.isfinite(A.ref[band])
unsat = zpwin_ok & (A.ref[band] >= edge) & (A.ref[band] < edge + 1)
unsat_dm = float(np.nanmedian(A.dm(band)[unsat]))
pickle.dump(dict(ref=ref, dm=dm, have=have, have_base=have_base, bins=BINS, unsat_dm=unsat_dm, edge=edge, names=names,
                 ratio=ratio, frac=frac), open(f'{Q}/satrefit/out3/score3_{band}.pkl', 'wb'))


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


L = [f'#### F{band}: {n} satstar-replaced matched stars; {have_base.sum()} with rows in the refit frames; {have.sum()} with every variant finite',
     f'Satstar faint edge {edge:.2f}; unsaturated reference {edge:.2f}-{edge + 1:.2f} mag: median dm {unsat_dm:+.3f} (N={unsat.sum()}).', '']
L.append('| variant | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all med / MAD | trend (faint-bright bin) | step (faint bin - unsat ref) |')
L.append('|---|' + '---|' * (len(BINS) + 3))
ncount = [int((have & (ref >= lo) & (ref < hi)).sum()) for lo, hi in BINS]
for v in names:
    d = dm[v]
    cells, meds = [], []
    for lo, hi in BINS:
        s = have & (ref >= lo) & (ref < hi)
        if s.sum() >= 5:
            meds.append(np.median(d[s]))
            cells.append(f'{meds[-1]:+.3f} ({mad(d[s]):.3f})')
        else:
            meds.append(np.nan)
            cells.append('-')
    s = have
    fin = [m for m in meds if np.isfinite(m)]
    tr = fin[-1] - fin[0] if len(fin) > 1 else np.nan
    st = fin[-1] - unsat_dm if fin else np.nan
    L.append(f'| {v} | ' + ' | '.join(cells) + f' | {np.median(d[s]):+.3f} / {mad(d[s]):.3f} | {tr:+.3f} | {st:+.3f} |')
L.append('')
L.append('N per bin: ' + ', '.join(map(str, ncount)))
L.append('')
L.append('Fraction of fit pixels rewritten (median over stars per bin) and amplitude ratio variant/base (median over stars per bin, uncapped):')
L.append('')
L.append('| variant | quantity | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' |')
L.append('|---|---|' + '---|' * len(BINS))
for v in RW:
    for q, dct, key in (('rewritten frac', frac, v), ('a_v/a_base', ratio, v)):
        if q == 'rewritten frac' and v.endswith('+bgfree'):
            continue
        x = dct[key]
        cells = []
        for lo, hi in BINS:
            s = have & (ref >= lo) & (ref < hi) & np.isfinite(x)
            cells.append(f'{np.median(x[s]):.3f}' if s.sum() >= 5 else '-')
        L.append(f'| {v} | {q} | ' + ' | '.join(cells) + ' |')
L.append('')
L.append(f'Capped rows in the refit frames: {capped.sum()} of {np.isfinite(acat).sum()}.')
open(f'{Q}/satrefit/out3/tables3_{band}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
