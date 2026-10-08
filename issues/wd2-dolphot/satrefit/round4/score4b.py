"""Round 4 addendum scoring.  usage: python score4b.py BAND (150W|200W) -> out4/score4b_<band>.pkl, out4/tables4b_<band>.md
A. norim variant (rim pixels excluded from the fit): dm tables split by detector, as score4.
B. per-pixel data/model ratio with model = F_dolphot * PSF, F_dolphot = median over the star's rows of flux_fit * 10^(0.4 dm_final)
   (dm = ours - dolphot - ZP, so this converts the dolphot magnitude to the catalog flux scale through the same ZP and zero point)."""
import glob, sys, pickle, os
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
O = Q + '/satrefit/out4'
band = sys.argv[1]
BINS = {'150W': [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19)], '200W': [(13, 14), (14, 15), (15, 16), (16, 17), (17, 18)]}[band]
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
tgt = A.sky[A.idx[ok]]
ref = A.ref[band][ok]
dm_final = A.dm(band)[ok]
tabs, offs, frames = [], [], []
off = 0
for fn2 in sorted(glob.glob(f'{Q}/satrefit/out2/{band}_*_satrefit.fits')):
    t2 = Table.read(fn2)
    fnb = fn2.replace('/out2/', '/out4/').replace('_satrefit.fits', '_satrefit4b.fits')
    tb = Table.read(fnb)
    assert len(tb) == len(t2) and np.allclose(np.asarray(tb['a_cat'], float), np.asarray(t2['a_cat'], float), equal_nan=True)
    f1 = fn2.replace('/out2/', '/out/')
    if os.path.exists(f1):
        t1 = Table.read(f1)
        for c in t1.colnames:
            if c.startswith('a_') and c not in t2.colnames:
                t2[c] = t1[c]
    if 'a_bgfree' not in t2.colnames:
        t2['a_bgfree'] = tb['a_bgfree']
    t3 = Table.read(fn2.replace('/out2/', '/out3/').replace('_satrefit.fits', '_satrefit3.fits'))
    for c in t3.colnames:
        if c.startswith('a_') and c not in t2.colnames:
            t2[c] = t3[c]
    for c in ('a_norim', 'a_norim+bgfree', 'nfit', 'nrim_fit', 'nfit_norim', 'flux_fit'):
        t2[c if c != 'nfit' else 'nfit4b'] = tb[c]
    ok_l = np.asarray(t2['label']) > 0
    assert np.allclose(np.asarray(tb['a_base'], float)[ok_l], np.asarray(t2['a_base'], float)[ok_l], rtol=1e-6)
    t2.meta = {}
    t2['det'] = 'nrcb1' if 'nrcb1' in fn2 else 'nrcb3'
    t2['pixfile'] = os.path.basename(fn2).replace('_satrefit.fits', '.npz')
    tabs.append(t2)
    offs.append(off)
    frames.append(os.path.basename(fn2))
    off += len(t2)
T = vstack(tabs, metadata_conflicts='silent')
good = np.asarray(T['label']) > 0
sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
a = {c[2:]: np.asarray(T[c], float) for c in T.colnames if c.startswith('a_') and c not in ('a_cat', 'a_raw')}
acat, araw = np.asarray(T['a_cat'], float), np.asarray(T['a_raw'], float)
cap = np.where(araw < 0.999 * acat, araw, np.inf)
n = ok.sum()
detrow = np.asarray(T['det'])
star_det = np.full(n, '', dtype='<U8')
star_of_row = np.full(len(T), -1)
for ii, jj in zip(i, j):
    if good[jj]:
        star_det[ii] = detrow[jj]
        star_of_row[jj] = ii


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
have = np.isfinite(s_cap)
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
add('rw12', a['rw12'])
add('rw12+bgfree', a['rw12+bgfree'])
add('rw12+bgfree+cap', np.minimum(a['rw12+bgfree'], cap))
for v in ('norim', 'norim+bgfree'):
    add(v, a[v])
    add(v + '+cap', np.minimum(a[v], cap))
names = ['final', 'uncapped', 'bgfree', 'bgfree+v7b+cap', 'rw12', 'rw12+bgfree', 'rw12+bgfree+cap', 'norim', 'norim+cap', 'norim+bgfree', 'norim+bgfree+cap']
for k in names:
    have &= np.isfinite(dm[k])
with np.errstate(invalid='ignore', divide='ignore'):
    rimfrac = med_per_star(np.asarray(T['nrim_fit'], float) / np.asarray(T['nfit4b'], float))
    left = med_per_star(np.asarray(T['nfit_norim'], float))
# ---- B: per-pixel ratio
Fr = np.asarray(T['flux_fit'], float) * 10 ** (0.4 * np.where(star_of_row >= 0, dm_final[np.clip(star_of_row, 0, None)], np.nan))
Fstar = np.full(n, np.nan)
for k in np.unique(star_of_row[star_of_row >= 0]):
    v = Fr[star_of_row == k]
    v = v[np.isfinite(v) & (v > 0)]
    if len(v):
        Fstar[k] = np.median(v)
pr = {k: [] for k in ('ratio', 'g0', 'r', 'cat', 'peak', 'det', 'ref', 'star')}
for t2, o0, fr in zip(tabs, offs, frames):
    z = np.load(f'{O}/pix_' + fr.replace('_satrefit.fits', '.npz').replace(f'{band}_', f'{band}_', 1))
    rowg = z['row'].astype(int) + o0
    sidx = star_of_row[rowg]
    m = sidx >= 0
    Fs = Fstar[np.clip(sidx, 0, None)]
    m &= np.isfinite(Fs) & (Fs > 0)
    pr['ratio'].append((z['u'][m] / Fs[m]).astype(np.float32))
    for k in ('g0', 'r', 'cat', 'peak'):
        pr[k].append(z[k][m])
    pr['det'].append(np.full(int(m.sum()), 1 if t2['det'][0] == 'nrcb1' else 3, np.int8))
    pr['ref'].append(ref[sidx[m]].astype(np.float32))
    pr['star'].append(sidx[m].astype(np.int32))
pr = {k: np.concatenate(v) for k, v in pr.items()}
pickle.dump(dict(ref=ref, dm=dm, have=have, bins=BINS, names=names, ratio=ratio, rimfrac=rimfrac, left=left, star_det=star_det,
                 pix={k: pr[k] for k in pr}), open(f'{O}/score4b_{band}.pkl', 'wb'))


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


L = []
for dname_, dsel in [('all', np.ones(n, bool)), ('nrcb1', star_det == 'nrcb1'), ('nrcb3', star_det == 'nrcb3')]:
    h = have & dsel
    L += [f'#### F{band} {dname_}: {int(h.sum())} stars', '', '| variant | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all med / MAD | trend |', '|---|' + '---|' * (len(BINS) + 2)]
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
        L.append(f'| {v} | ' + ' | '.join(cells) + f' | {np.median(d[h]):+.3f} / {mad(d[h]):.3f} | {(fin[-1] - fin[0]) if len(fin) > 1 else np.nan:+.3f} |')
    L += ['', f'norim bookkeeping, F{band} {dname_}: median over stars per bin', '', '| quantity | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all |', '|---|' + '---|' * (len(BINS) + 1)]
    for q_, x, fmt in (('rim px / fit px (base fit)', rimfrac, '{:.3f}'), ('fit px left after removing rim', left, '{:.0f}'), ('a_norim / a_base', ratio['norim'], '{:.4f}'), ('a_norim+bgfree / a_base', ratio['norim+bgfree'], '{:.4f}')):
        cells = []
        for lo, hi in BINS:
            s = h & (ref >= lo) & (ref < hi) & np.isfinite(x)
            cells.append(fmt.format(np.median(x[s])) if s.sum() >= 5 else '-')
        cells.append(fmt.format(np.median(x[h & np.isfinite(x)])))
        L.append(f'| {q_} | ' + ' | '.join(cells) + ' |')
    L.append('')
# B tables
gE = [200, 400, 800, 1600, 3200, 6400, 12800, 1e9]
rE = [0, 2, 3, 4, 6, 9, 14, 31]


def cell(sel):
    v = pr['ratio'][sel]
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return np.nan, np.nan, len(v)
    m = np.median(v)
    return m, 1.2533 * mad(v) / np.sqrt(len(v)), len(v)


def table(title, cat_sel, edges, key, labfmt):
    L.append(title)
    L.append('')
    L.append('| bin | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |')
    L.append('|---|---|---|---|')
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (pr[key] >= lo) & (pr[key] < hi) & cat_sel
        c1 = cell(b & (pr['det'] == 1))
        c3 = cell(b & (pr['det'] == 3))
        rr = c3[0] / c1[0] if np.isfinite(c1[0]) and np.isfinite(c3[0]) else np.nan
        f = lambda c: f'{c[0]:.3f}+-{c[1]:.3f} ({c[2]})' if np.isfinite(c[0]) else f'- ({c[2]})'
        L.append(f'| {labfmt(lo, hi)} | {f(c1)} | {f(c3)} | {rr:.3f} |')
    L.append('')


gl = lambda lo, hi: f'{lo:.0f}-{hi:.0f}' if hi < 1e8 else f'>{lo:.0f}'
rl = lambda lo, hi: f'{lo:.0f}-{hi:.0f}'
L.append(f'### F{band} per-pixel data/model, model = F_dolphot x PSF at the fitted position (data = neighbour-subtracted, minus pipeline annulus bkg); pixels within 30 px with PSF >= 1e-3 of peak; se = 1.2533 MAD/sqrt(N)')
L.append('')
tot = {(c, d): int(((pr['cat'] == c) & (pr['det'] == d)).sum()) for c in (0, 1, 2) for d in (1, 3)}
L.append('Pixel counts (cat, det): ' + ', '.join(f'{c}/nrcb{d}: {tot[(c, d)]}' for c in (0, 1, 2) for d in (1, 3)) + ' (cat 0: rim in fit, values after rewrite; 1: never-rewritten crf pixel in fit; 2: rim pixel masked from the fit)')
L.append('')
for c, nm in ((0, '(a) rim pixels in the fit'), (1, '(b) never-rewritten crf pixels in the fit')):
    sel = pr['cat'] == c
    table(f'**{nm}: by g0 (DN)**', sel, gE, 'g0', gl)
    table(f'**{nm}: by r from the fitted position (px)**', sel, rE, 'r', rl)
for c, nm in ((0, 'rim in the fit'), (1, 'never-rewritten')):
    sel = pr['cat'] == c
    med_ref = np.median(ref)
    for lab, ms in (('brighter half of stars', pr['ref'] < med_ref), ('fainter half of stars', pr['ref'] >= med_ref)):
        cs = [cell(sel & ms & (pr['det'] == d)) for d in (1, 3)]
        L.append(f'- {nm}, {lab} (dolphot mag {"<" if "bright" in lab else ">="} {med_ref:.2f}): nrcb1 {cs[0][0]:.3f} (N={cs[0][2]}), nrcb3 {cs[1][0]:.3f} (N={cs[1][2]}), nrcb3/nrcb1 {cs[1][0] / cs[0][0]:.3f}')
L.append('')
L.append('**(a) model peak pixel (psf maximum of the cutout)**')
L.append('')
L.append('| category | nrcb1 median +- se (N) | nrcb3 median +- se (N) | nrcb3/nrcb1 |')
L.append('|---|---|---|---|')
for c, nm in ((0, 'peak pixel is a rim pixel in the fit'), (2, 'peak pixel is a rim pixel masked from the fit'), (1, 'peak pixel is a never-rewritten pixel in the fit')):
    sel = (pr['cat'] == c) & pr['peak']
    c1, c3 = cell(sel & (pr['det'] == 1)), cell(sel & (pr['det'] == 3))
    f = lambda cc: f'{cc[0]:.3f}+-{cc[1]:.3f} ({cc[2]})' if np.isfinite(cc[0]) else f'- ({cc[2]})'
    L.append(f'| {nm} | {f(c1)} | {f(c3)} | {c3[0] / c1[0]:.3f} |')
L.append('')
open(f'{O}/tables4b_{band}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
