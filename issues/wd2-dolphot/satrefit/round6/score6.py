"""Round 5 scoring.  usage: python score5.py BAND NSTAR   (BAND in 150W 200W 250M 300M)
Variants (a = amplitude from the weighted linear solve at the pipeline's fitted (x, y), brighter-first neighbour subtraction):
  RN<N*>, RN5   rim pixels rewritten with the pipeline R curve re-measured with edt >= N (same g0, same pixel set), base / bgfree,
                each uncapped and +cap with the recovered-core cap recomputed from the rewritten cutout (cap_<v>).
  RN<N*>+h0     the same rim plus the round-4 wing rewrite (rw12h, D = 12, crf ERR kept = e0).
  round-4 comparison columns: rw12h_e1, rw12h_e0 (+cap uses the catalogue cap).
Writes out5/score5_<band>.pkl and out5/tables5_<band>.md.
"""
import glob, os, pickle, sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
O5, O4, O6 = Q + '/satrefit/out5', Q + '/satrefit/out4', Q + '/satrefit/out6'
band, NS = sys.argv[1], int(sys.argv[2])
SW = band in ('150W', '200W')
BINS = {'150W': [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19)], '200W': [(13, 14), (14, 15), (15, 16), (16, 17), (17, 18)]}.get(
    band, [(10, 13), (13, 14), (14, 15), (15, 16), (16, 17)])
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
tgt = A.sky[A.idx[ok]]
ref = np.asarray(A.ref[band][ok], float)
dm_final = A.dm(band)[ok]
n = int(ok.sum())
VN = 'RN%d' % NS
tabs, offs, frames = [], [], []
off = 0
for fn2 in sorted(glob.glob(f'{Q}/satrefit/out2/{band}_*_satrefit.fits')):
    t2 = Table.read(fn2)
    f4 = fn2.replace('/out2/', '/out4/').replace('_satrefit.fits', '_satrefit4.fits')
    f5 = fn2.replace('/out2/', '/out5/').replace('_satrefit.fits', '_satrefit5.fits')
    f6 = fn2.replace('/out2/', '/out6/').replace('_satrefit.fits', '_satrefit6.fits')
    t4, t5, t6 = Table.read(f4), Table.read(f5), Table.read(f6)
    assert len(t6) == len(t2)
    assert len(t4) == len(t2) == len(t5)
    good_l = np.asarray(t2['label']) > 0
    assert np.allclose(np.asarray(t5['a_cat'], float), np.asarray(t2['a_cat'], float), equal_nan=True)
    assert np.allclose(np.asarray(t5['a_base'], float)[good_l], np.asarray(t4['a_base'], float)[good_l], rtol=1e-6, equal_nan=True)
    tt = Table()
    for c in ('label', 'a_cat', 'a_raw', 'ra', 'dec'):
        tt[c] = t2[c]
    for c in t4.colnames:
        if c.startswith('a_') and c not in ('a_cat', 'a_raw', 'a_base'):
            tt['r4_' + c[2:]] = t4[c]
    for c in t5.colnames:
        if c.startswith('a_') or c.startswith('cap_') or c in ('nfit', 'nrim_fit', 'nrw_h', 'flux_fit'):
            tt[c] = t5[c]
    for c in t6.colnames:
        if (c.startswith('a_H') or c.startswith('cap_H')):
            tt[c] = t6[c]
    tt['det'] = 'nrcb1' if 'nrcb1' in fn2 else ('nrcb3' if 'nrcb3' in fn2 else 'nrcblong')
    tt['pixfile'] = os.path.basename(fn2).replace(f'{band}_', '').replace('_satrefit.fits', '')
    tabs.append(tt)
    offs.append(off)
    frames.append(os.path.basename(fn2))
    off += len(tt)
T = vstack(tabs, metadata_conflicts='silent')
good = np.asarray(T['label']) > 0
sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
col = lambda c: np.asarray(T[c], float)
acat, araw, abase = col('a_cat'), col('a_raw'), col('a_base')
capped = araw < 0.999 * acat
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
catcap = np.where(capped, araw, np.inf)


def add(name, a_eff):
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio[name] = med_per_star(a_eff / abase)
        dm[name] = dm_unc + med_per_star(-2.5 * np.log10(a_eff / abase))


# validation of the cap reproduction
cb = col('cap_base')
okc = good & capped & np.isfinite(cb)
val = dict(n_capped=int((good & capped).sum()), n_capped_match=int((okc & (np.abs(cb / araw - 1) < 0.01)).sum()),
           n_capped_finite=int(okc.sum()), n_uncapped=int((good & ~capped).sum()),
           n_uncapped_bind=int((good & ~capped & np.isfinite(cb) & (cb < 0.999 * acat)).sum()))
rcor = np.where(capped & np.isfinite(cb) & (cb > 0), araw / cb, 1.0)    # absorbs own_cell / blend differences of the cap region


def cap_of(v):
    c = col('cap_' + v) * rcor
    return np.where(np.isfinite(c), c, np.inf)


def add_family(tag, key, capkey=None, col_a=None):
    """key: column prefix in T for the uncapped amplitude ('a_<key>'), capkey for cap_<capkey> (None = catalogue cap)"""
    a0, a1 = col('a_' + key), col('a_' + key + '+bgfree')
    cp = catcap if capkey is None else cap_of(capkey)
    add(tag, a0)
    add(tag + '+cap', np.minimum(a0, cp))
    add(tag + '+bgfree', a1)
    add(tag + '+bgfree+cap', np.minimum(a1, cp))


for t_, k_ in (('rw12h_e0', 'r4_rw12h_e0'),):
    a0, a1 = col(k_), col(k_ + '+bgfree')
    add(t_, a0)
    add(t_ + '+cap', np.minimum(a0, catcap))
    add(t_ + '+bgfree', a1)
    add(t_ + '+bgfree+cap', np.minimum(a1, catcap))
add('base+cap(recomputed)', np.minimum(abase, cap_of('base')))
FAM = [VN, VN + '+h0', 'H', 'H+h0']
for v in FAM:
    add_family(v, v, v)
fam_names = [x for v in FAM for x in (v, v + '+cap', v + '+bgfree', v + '+bgfree+cap')]
r4n = [x for v in ('rw12h_e0',) for x in (v, v + '+cap', v + '+bgfree', v + '+bgfree+cap')]
names = ['final', 'uncapped', 'base+cap(recomputed)'] + r4n + fam_names
for k in names:
    have &= np.isfinite(dm[k])
with np.errstate(invalid='ignore', divide='ignore'):
    capratio = {v: med_per_star(col('cap_' + v) * rcor / np.where(np.isfinite(cb), cb, np.nan)) for v in (VN, 'H')}
edge = float(np.percentile(ref, 95))
zpwin_ok = A.matched & ~A.rep[band] & np.isfinite(A.dm(band)) & np.isfinite(A.ref[band])
unsat = zpwin_ok & (A.ref[band] >= edge) & (A.ref[band] < edge + 1)
unsat_dm = float(np.nanmedian(A.dm(band)[unsat]))


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


L = []
groups = [('all', np.ones(n, bool))] + ([('nrcb1', star_det == 'nrcb1'), ('nrcb3', star_det == 'nrcb3')] if SW else [])
for dname_, dsel in groups:
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
        L.append(f'| {v} | ' + ' | '.join(cells) + f' | {np.median(d[h]):+.3f} / {mad(d[h]):.3f} | {(fin[-1] - fin[0]) if len(fin) > 1 else np.nan:+.3f} |')
    L += ['', f'amplitude ratio to base (median over stars) and cap ratio, F{band} {dname_}', '', '| quantity | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in BINS) + ' | all |', '|---|' + '---|' * (len(BINS) + 1)]
    items = [(f'a_{v}/a_base', ratio[v]) for v in (VN, 'H', 'H+h0', 'H+bgfree')] + [(f'cap_{v}/cap_base', capratio[v]) for v in (VN, 'H')]
    items.append(('rim px / fit px', med_per_star(col('nrim_fit') / col('nfit'))))
    for q_, x in items:
        cells = []
        for lo, hi in BINS:
            s = h & (ref >= lo) & (ref < hi) & np.isfinite(x)
            cells.append(f'{np.median(x[s]):.4f}' if s.sum() >= 5 else '-')
        cells.append(f'{np.median(x[h & np.isfinite(x)]):.4f}')
        L.append(f'| {q_} | ' + ' | '.join(cells) + ' |')
    L.append('')
L.append(f'Cap reproduction check (base cutout through the pipeline cap functions): capped rows {val["n_capped"]}, finite cap {val["n_capped_finite"]}, '
         f'within 1 % of the catalogue cap {val["n_capped_match"]}; uncapped rows {val["n_uncapped"]}, of which {val["n_uncapped_bind"]} would be capped by the recomputed cap. '
         'The capped-row correction factor a_raw/cap_base is applied to the recomputed caps of all variants (blend own-cell differences).')
L.append('')
# ---- per-pixel ratio
Fr = col('flux_fit') * 10 ** (0.4 * np.where(star_of_row >= 0, dm_final[np.clip(star_of_row, 0, None)], np.nan))
Fstar = np.full(n, np.nan)
for k in np.unique(star_of_row[star_of_row >= 0]):
    v = Fr[star_of_row == k]
    v = v[np.isfinite(v) & (v > 0)]
    if len(v):
        Fstar[k] = np.median(v)
pr = {k: [] for k in ('r0', 'rN', 'rH', 'g0', 'r', 'cat', 'peak', 'det', 'ref')}
for tt, o0, fr in zip(tabs, offs, frames):
    nm = fr.replace('_satrefit.fits', '.npz')
    z = np.load(f'{O6}/pix6_' + nm)
    z5 = np.load(f'{O5}/pix5_' + nm)
    assert len(z['g0']) == len(z5['g0']) and np.allclose(z['g0'], z5['g0'], equal_nan=True) and np.allclose(z['u0'], z5['u0'], equal_nan=True, rtol=1e-4)
    sidx = star_of_row[z['row'].astype(int) + o0]
    m = sidx >= 0
    Fs = Fstar[np.clip(sidx, 0, None)]
    m &= np.isfinite(Fs) & (Fs > 0)
    pr['r0'].append((z['u0'][m] / Fs[m]).astype(np.float32))
    pr['rH'].append((z['uH'][m] / Fs[m]).astype(np.float32))
    pr['rN'].append((z5['uN'][m] / Fs[m]).astype(np.float32))
    for k in ('g0', 'r', 'cat', 'peak'):
        pr[k].append(z[k][m])
    pr['det'].append(np.full(int(m.sum()), 1 if tt['det'][0] == 'nrcb1' else (3 if tt['det'][0] == 'nrcb3' else 0), np.int8))
    pr['ref'].append(ref[sidx[m]].astype(np.float32))
pr = {k: np.concatenate(v) for k, v in pr.items()}


def cell(v):
    v = v[np.isfinite(v)]
    return (float(np.median(v)), len(v)) if len(v) >= 20 else (np.nan, len(v))


L.append(f'### F{band} per-pixel data/model of rim pixels (cat 0) with the rim rewritten by: pipeline N=0, {VN}, R_header (H)')
L.append('')
dets = [(1, 'nrcb1'), (3, 'nrcb3')] if SW else [(0, 'nrcblong')]
L.append('| subset (r < 6 px) | ' + ' | '.join(f'{d} N0 / RN25 / H' for _, d in dets) + (' | nrcb3/nrcb1 N0 / RN25 / H |' if SW else ' |'))
L.append('|---|' + '---|' * (len(dets) + (1 if SW else 0)))
bsel = (pr['cat'] == 0) & (pr['r'] < 6)
for lab, sel in (('all rim', bsel), ('peak pixel', (pr['cat'] == 0) & pr['peak'])):
    cells, med = [], {}
    for dv, dn in dets:
        vals = [cell(pr[k][sel & (pr['det'] == dv)]) for k in ('r0', 'rN', 'rH')]
        med[dn] = [v[0] for v in vals]
        cells.append(' / '.join(f'{v[0]:.3f}' for v in vals) + f' (N {vals[0][1]})')
    if SW:
        cells.append(' / '.join(f'{b3 / b1:.3f}' for b3, b1 in zip(med['nrcb3'], med['nrcb1'])))
    L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
L.append('')
if SW:
    gE = [200, 400, 800, 1600, 3200, 6400, 12800, 1e9]
    L.append('by g0 (all rim, cat 0): nrcb1 / nrcb3 / ratio for N0 | RN25 | H')
    L.append('')
    L.append('| g0 | N0 | RN25 | H |')
    L.append('|---|---|---|---|')
    for lo, hi in zip(gE[:-1], gE[1:]):
        b = (pr['g0'] >= lo) & (pr['g0'] < hi) & (pr['cat'] == 0)
        cs = []
        for k in ('r0', 'rN', 'rH'):
            c1 = cell(pr[k][b & (pr['det'] == 1)])[0]
            c3 = cell(pr[k][b & (pr['det'] == 3)])[0]
            cs.append(f'{c1:.3f} / {c3:.3f} / **{c3 / c1:.3f}**')
        L.append(f'| {lo:.0f}-{hi:.0f} | ' + ' | '.join(cs) + ' |')
    L.append('')
pickle.dump(dict(ref=ref, dm=dm, have=have, bins=BINS, names=names, ratio=ratio, star_det=star_det, pix=pr, val=val, NS=NS, unsat_dm=unsat_dm),
            open(f'{O6}/score6_{band}.pkl', 'wb'))
open(f'{O6}/tables6_{band}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
