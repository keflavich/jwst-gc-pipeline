"""Round 4 scoring.  usage: python score4.py BAND -> out4/score4_<band>.pkl, out4/tables4_<band>.md
Merges out/ (a_bgfree), out2/ (bgfree+v7b, pk), out3/ (rw12 variants) and out4/ (rw12h_e1, rw12h_e0, rw12p_e1) per frame.
Tables for all stars and split by detector (SW bands)."""
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
        '200W': [(13, 14), (14, 15), (15, 16), (16, 17), (17, 18)],
        '250M': [(10, 13), (13, 14), (14, 15), (15, 16), (16, 17)],
        '300M': [(10, 13), (13, 14), (14, 15), (15, 16), (16, 17)]}[band]
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
    for d_, suf in (('out', '_satrefit.fits'), ('out3', '_satrefit3.fits'), ('out4', '_satrefit4.fits')):
        fnx = fn2.replace('/out2/', f'/{d_}/').replace('_satrefit.fits', suf)
        if not os.path.exists(fnx):
            if d_ == 'out':
                continue
            raise FileNotFoundError(fnx)
        tx = Table.read(fnx)
        assert len(tx) == len(t2) and np.allclose(np.asarray(tx['a_cat'], float), np.asarray(t2['a_cat'], float), equal_nan=True)
        for c in tx.colnames:
            if (c.startswith('a_') or c.startswith('nrw_') or c in ('nnew', 'q_new_med', 'q_new_sum', 'q_old_med')) and c not in t2.colnames:
                t2[c] = tx[c]
        if d_ == 'out3':
            t2['nfit3'] = tx['nfit']
        if d_ == 'out4':
            assert np.array_equal(np.asarray(tx['nfit'], float)[np.asarray(t2['label']) > 0], np.asarray(t2['nfit3'], float)[np.asarray(t2['label']) > 0])
            t2['nrw_rw12_o4'] = tx['nrw_rw12']
    t2.meta = {}
    t2['frame'] = os.path.basename(fn2)
    t2['det'] = 'nrcb1' if 'nrcb1' in fn2 else ('nrcb3' if 'nrcb3' in fn2 else 'nrcblong')
    tabs.append(t2)
T = vstack(tabs, metadata_conflicts='silent')
good = np.asarray(T['label']) > 0
assert np.array_equal(np.asarray(T['nrw_rw12'], float)[good], np.asarray(T['nrw_rw12_o4'], float)[good]), 'rw12 pixel counts differ between out3 and out4'
sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
a = {c[2:]: np.asarray(T[c], float) for c in T.colnames if c.startswith('a_') and c not in ('a_cat', 'a_raw')}
acat, araw = np.asarray(T['a_cat'], float), np.asarray(T['a_raw'], float)
capped = araw < 0.999 * acat
cap = np.where(capped, araw, np.inf)
n = ok.sum()
detrow = np.asarray(T['det'])
star_det = np.full(n, '', dtype='<U8')
for ii, jj in zip(i, j):
    if good[jj]:
        star_det[ii] = detrow[jj]


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
        ratio[name] = med_per_star(a_eff / a['base'])
        dm[name] = dm_unc + med_per_star(-2.5 * np.log10(a_eff / a['base']))


add('bgfree', a['bgfree'])
pk = np.asarray(T['pk_bgfree+v7b'], float)
pk = np.where(np.isfinite(pk), pk, 1.0)
add('bgfree+v7b+cap', np.minimum(a['bgfree+v7b'], cap * pk))
NEW = ['rw12h_e1', 'rw12h_e0', 'rw12p_e1']
VAR = ['rw12'] + NEW
for v in VAR:
    add(v, a[v])
    add(v + '+cap', np.minimum(a[v], cap))
    add(v + '+bgfree', a[v + '+bgfree'])
    add(v + '+bgfree+cap', np.minimum(a[v + '+bgfree'], cap))
names = ['final', 'uncapped', 'bgfree', 'bgfree+v7b+cap'] + [x for v in VAR for x in (v, v + '+cap', v + '+bgfree', v + '+bgfree+cap')]
have = have_base.copy()
for k in names:
    have &= np.isfinite(dm[k])
frac = {}
nfit = np.asarray(T['nfit3'], float)
for v in VAR:
    with np.errstate(invalid='ignore', divide='ignore'):
        frac[v] = med_per_star(np.asarray(T['nrw_' + v], float) / nfit)
with np.errstate(invalid='ignore', divide='ignore'):
    frac_new = med_per_star(np.asarray(T['nnew'], float) / nfit)
nnew_s = med_per_star(np.asarray(T['nnew'], float))
qnew = med_per_star(np.asarray(T['q_new_med'], float))
qold = med_per_star(np.asarray(T['q_old_med'], float))
# pooled newly-rewritten pixel counts (all rows of the matched stars' frames)
pool = dict(nnew=float(np.nansum(np.asarray(T['nnew'], float)[good])), nfit=float(np.nansum(nfit[good])),
            qsum=float(np.nansum(np.asarray(T['q_new_sum'], float)[good])))
zpwin_ok = A.matched & ~A.rep[band] & np.isfinite(A.dm(band)) & np.isfinite(A.ref[band])
unsat = zpwin_ok & (A.ref[band] >= edge) & (A.ref[band] < edge + 1)
unsat_dm = float(np.nanmedian(A.dm(band)[unsat]))
pickle.dump(dict(ref=ref, dm=dm, have=have, bins=BINS, unsat_dm=unsat_dm, edge=edge, names=names, ratio=ratio, frac=frac, frac_new=frac_new,
                 nnew=nnew_s, qnew=qnew, qold=qold, star_det=star_det, pool=pool), open(f'{Q}/satrefit/out4/score4_{band}.pkl', 'wb'))


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


L = []
dets = [('all', np.ones(n, bool))] + ([(d_, star_det == d_) for d_ in ('nrcb1', 'nrcb3')] if band in ('150W', '200W') else [])
for dname_, dsel in dets:
    h = have & dsel
    L += [f'#### F{band} {dname_}: {int(h.sum())} stars (unsaturated reference {edge:.2f}-{edge + 1:.2f} mag: dm {unsat_dm:+.3f})', '',
          '| variant | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all med / MAD | trend |', '|---|' + '---|' * (len(BINS) + 2)]
    for v in names:
        d = dm[v]
        cells, meds = [], []
        for lo, hi in BINS:
            s = h & (ref >= lo) & (ref < hi)
            if s.sum() >= 5:
                meds.append(np.median(d[s]))
                cells.append(f'{meds[-1]:+.3f} ({mad(d[s]):.3f})')
            else:
                meds.append(np.nan)
                cells.append('-')
        fin = [m_ for m_ in meds if np.isfinite(m_)]
        tr = fin[-1] - fin[0] if len(fin) > 1 else np.nan
        L.append(f'| {v} | ' + ' | '.join(cells) + f' | {np.median(d[h]):+.3f} / {mad(d[h]):.3f} | {tr:+.3f} |')
    L.append('')
    L.append('N per bin: ' + ', '.join(str(int((h & (ref >= lo) & (ref < hi)).sum())) for lo, hi in BINS))
    L += ['', f'Rewritten fraction of fit pixels (median over stars) and amplitude ratio a_v/a_base (uncapped), F{band} {dname_}:', '',
          '| variant | quantity | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all |', '|---|---|' + '---|' * (len(BINS) + 1)]
    for v in VAR:
        for q_, dct, key in (('rewritten frac', frac, v), ('a_v/a_base', ratio, v), ('a_v/a_base bgfree', ratio, v + '+bgfree')):
            x = dct[key]
            cells = []
            for lo, hi in BINS:
                s = h & (ref >= lo) & (ref < hi) & np.isfinite(x)
                cells.append(f'{np.median(x[s]):.4f}' if s.sum() >= 5 else '-')
            cells.append(f'{np.median(x[h & np.isfinite(x)]):.4f}')
            L.append(f'| {v} | {q_} | ' + ' | '.join(cells) + ' |')
    L += ['', f'Pixels newly rewritten by rw12h (g0 above the top wingmig R bin), F{band} {dname_}: per-star median of (new pixels / fit pixels), median per-star count, q of the new pixels, q of the round-3 rw12 pixels of the same stars:', '',
          '| quantity | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all |', '|---|' + '---|' * (len(BINS) + 1)]
    for q_, x, fmt in (('new/fit', frac_new, '{:.5f}'), ('count', nnew_s, '{:.2f}'), ('q new px (median of star medians; stars with new px)', np.where(nnew_s > 0, qnew, np.nan), '{:.3f}'),
                       ('q rw12 px (same stars)', np.where(nnew_s > 0, qold, np.nan), '{:.3f}')):
        cells = []
        for lo, hi in BINS:
            s = h & (ref >= lo) & (ref < hi) & np.isfinite(x)
            cells.append(fmt.format(np.median(x[s])) if s.sum() >= 3 else '-')
        sa = h & np.isfinite(x)
        cells.append(fmt.format(np.median(x[sa])) if sa.sum() else '-')
        L.append(f'| {q_} | ' + ' | '.join(cells) + ' |')
    L.append('')
with np.errstate(invalid='ignore', divide='ignore'):
    L.append(f"Pooled over all rows of the refit frames (all {dets[0][0]}): newly rewritten pixels {pool['nnew']:.0f} of {pool['nfit']:.0f} fit pixels ({pool['nnew'] / pool['nfit']:.5f}); pixel-weighted mean q of the new pixels {pool['qsum'] / pool['nnew']:.3f}.")
open(f'{Q}/satrefit/out4/tables4_{band}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
