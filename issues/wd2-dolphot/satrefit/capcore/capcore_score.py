"""Score cap / core-fit estimators vs dolphot.  usage: python capcore_score.py  -> capcore_tables.md, capcore.png, capcore_diag.md, out/score_<band>.pkl
dm = ours - dolphot - ZP (analyze.Arm 'main2'); an estimator's per-star dm is the final-catalog dm plus the per-star median over rows of
-2.5 log10(a_est / flux_fit_raw), the same construction as score2.py."""
import glob
import os
import pickle
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402

HERE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capcore'
BANDS = ['150W', '200W', '250M', '300M']
LW = ('250M', '300M')
MINPX = 5
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def per_star(i, x, n):
    out = np.full(n, np.nan)
    m = np.isfinite(x)
    ii, xx = i[m], x[m]
    if not len(ii):
        return out
    o = np.argsort(ii, kind='stable')
    ii, xx = ii[o], xx[o]
    uq, st = np.unique(ii, return_index=True)
    for k, s, e in zip(uq, st, list(st[1:]) + [len(ii)]):
        out[k] = np.median(xx[s:e])
    return out


def running(x, y, nwin=40):
    o = np.argsort(x)
    x, y = x[o], y[o]
    if len(x) < nwin:
        return np.array([]), np.array([])
    h = nwin // 2
    xs = np.array([np.median(x[max(0, k - h):k + h]) for k in range(h, len(x) - h, max(1, h // 2))])
    ys = np.array([np.median(y[max(0, k - h):k + h]) for k in range(h, len(x) - h, max(1, h // 2))])
    return xs, ys


EST = ['final', 'precap', 'cap', 'core_c', 'core_f', 'core_c|precap', 'core_f|precap', 'core_c(cov>=0.5)|precap', 'cap|precap']
res = {}
md = []
diag_rows = {}
for band in BANDS:
    ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
    tgt = A.sky[A.idx[ok]]
    ref = A.ref[band][ok]
    dm_final = A.dm(band)[ok]
    n = ok.sum()
    tabs = []
    for fn in sorted(glob.glob(f'{HERE}/out/{band}_*_capcore.fits')):
        t = Table.read(fn)
        t['frame'] = os.path.basename(fn)
        for k, v in t.meta.items():
            t[f'm_{k.lower()}'] = float(v)
        tabs.append(t)
    T = vstack(tabs, metadata_conflicts='silent')
    sk = SkyCoord(np.asarray(T['ra']) * u.deg, np.asarray(T['dec']) * u.deg)
    i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
    g = lambda c: np.asarray(T[c], float)  # noqa: E731
    pre, raw, fin = g('c_flux_fit_precap'), g('c_flux_fit_raw'), g('c_flux_fit')
    lab_ok = g('label') > 0
    cap = np.where(lab_ok, g('cap'), np.nan)
    ncore = g('n_core')
    ac, af = g('a_core_c'), g('a_core_f')
    ac = np.where((ncore >= MINPX) & (ac > 0), ac, np.nan)
    af = np.where((ncore >= MINPX) & (af > 0), af, np.nan)
    ac_cov = np.where(g('cov_core') >= 0.5, ac, np.nan)
    a = {'final': fin, 'precap': pre, 'cap': cap, 'core_c': ac, 'core_f': af,
         'core_c|precap': np.where(np.isfinite(ac), ac, pre), 'core_f|precap': np.where(np.isfinite(af), af, pre),
         'core_c(cov>=0.5)|precap': np.where(np.isfinite(ac_cov), ac_cov, pre),
         'cap|precap': np.where(np.isfinite(cap), cap, pre)}
    # row-level quantities on the stars (rows within 0.1")
    dm = {}
    for k, x in a.items():
        with np.errstate(invalid='ignore', divide='ignore'):
            s = -2.5 * np.log10(x / raw)
        s = np.where(lab_ok, s, np.nan)
        dm[k] = dm_final + per_star(i, s[j], n) if k != 'final' else dm_final.copy()
    havebase = np.isfinite(per_star(i, np.where(lab_ok, 0.0, np.nan)[j], n))
    # rows -> star index (rows can match one star; take first match per row)
    row_star = np.full(len(T), -1)
    row_star[j] = i
    # reproduction check on all rows
    capped = lab_ok & (raw < 0.999 * pre)
    capfin = np.isfinite(cap)
    chk = dict(n_rows=int(lab_ok.sum()), n_capped=int(capped.sum()), capped_capfin=int((capped & capfin).sum()),
               capped_capnan=int((capped & ~capfin).sum()), maxrel=float(np.max(np.abs(cap[capped & capfin] / raw[capped & capfin] - 1))) if (capped & capfin).any() else np.nan,
               n95=float(np.percentile(np.abs(cap[capped & capfin] / raw[capped & capfin] - 1), 99.9)) if (capped & capfin).any() else np.nan,
               notcapped_cap_below_pre=int((~capped & lab_ok & capfin & (cap < 0.999 * pre)).sum()),
               capfin=int((lab_ok & capfin).sum()), wc_ne1=int((np.abs(g('c_wingcal_ratio') - 1) > 1e-6).sum()))
    # availability
    avail_row = {'cap': np.isfinite(cap) & lab_ok, 'core_c': np.isfinite(ac) & lab_ok, 'core_f': np.isfinite(af) & lab_ok,
                 'core_c(cov>=0.5)': np.isfinite(ac_cov) & lab_ok}
    avail_star = {k: np.isfinite(per_star(i, np.where(v, 0.0, np.nan)[j], n)) for k, v in avail_row.items()}
    res[band] = dict(ref=ref, dm=dm, avail_star=avail_star, chk=chk, avail_row=avail_row, rows=T, row_star=row_star, dm_final=dm_final,
                     havebase=havebase, lab_ok=lab_ok, i=i, j=j, n=n, a=a, raw=raw, pre=pre)
    print(band, chk, flush=True)

# ---- tables ----
L = []
for band in BANDS:
    r = res[band]
    ref, dm, hb = r['ref'], r['dm'], r['havebase']
    lo0 = int(np.floor(ref[hb].min()))
    hi0 = int(np.ceil(np.percentile(ref[hb], 99.5)))
    bins = [(b, b + 1) for b in range(lo0, hi0)]
    unsat = A.matched & ~A.rep[band] & np.isfinite(A.dm(band)) & np.isfinite(A.ref[band]) & (A.ref[band] >= np.percentile(ref, 95)) & (A.ref[band] < np.percentile(ref, 95) + 1)
    L.append(f'#### F{band}: {r["n"]} satstar-replaced matched stars, {hb.sum()} with rows in the frames.  Cells: median dm (MAD) [N]; unsaturated reference 1 mag beyond the satstar faint edge: median dm {np.nanmedian(A.dm(band)[unsat]):+.3f}')
    L.append('')
    L.append('| estimator | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in bins) + ' | all: med / MAD [N] | trend: slope dm per mag | faint-bright (bins N>=10) |')
    L.append('|---|' + '---|' * (len(bins) + 3))
    for k in EST:
        d = dm[k]
        cells, meds = [], []
        for lo, hi in bins:
            s = hb & np.isfinite(d) & (ref >= lo) & (ref < hi)
            if s.sum() >= 3:
                cells.append(f'{np.median(d[s]):+.3f} ({mad(d[s]):.3f}) [{s.sum()}]')
                if s.sum() >= 10:
                    meds.append(np.median(d[s]))
            else:
                cells.append(f'- [{s.sum()}]')
        s = hb & np.isfinite(d)
        sl = np.nan
        ss = s & (np.abs(d - np.median(d[s])) < 0.5)
        if ss.sum() > 20:
            sl = np.polyfit(ref[ss], d[ss], 1)[0]
        tr = meds[-1] - meds[0] if len(meds) > 1 else np.nan
        L.append(f'| {k} | ' + ' | '.join(cells) + f' | {np.median(d[s]):+.3f} / {mad(d[s]):.3f} [{s.sum()}] | {sl:+.3f} | {tr:+.3f} |')
    L.append('')
    L.append('Availability (core estimator defined: >= 5 measured px within r <= ' + ('2.0' if band in LW else '2.5') + ' px and amplitude > 0; stars with >= 1 defined row / rows):')
    L.append('')
    L.append('| estimator | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in bins) + ' | all | rows (all frames) |')
    L.append('|---|' + '---|' * (len(bins) + 2))
    rows_lab = r['lab_ok'].sum()
    for k in ('cap', 'core_c', 'core_f', 'core_c(cov>=0.5)'):
        av = r['avail_star'][k]
        cells = []
        for lo, hi in bins:
            s = hb & (ref >= lo) & (ref < hi)
            cells.append(f'{av[s].mean():.2f} [{s.sum()}]' if s.sum() else '-')
        L.append(f'| {k} | ' + ' | '.join(cells) + f' | {av[hb].mean():.2f} | {r["avail_row"][k].sum()}/{rows_lab} = {r["avail_row"][k].sum() / rows_lab:.2f} |')
    L.append('')
    c = r['chk']
    L.append(f'Cap reproduction ({band}): {c["n_rows"]} rows with a label; {c["n_capped"]} rows have flux_fit_raw < 0.999 flux_fit_precap; for {c["capped_capfin"]} of them the reproduced cap is finite and its max |cap/flux_fit_raw - 1| is {c["maxrel"]:.2e} (99.9th pct {c["n95"]:.2e}); {c["capped_capnan"]} capped rows have a NaN reproduced cap; {c["notcapped_cap_below_pre"]} uncapped rows have a reproduced cap below 0.999 precap; cap finite on {c["capfin"]} rows; rows with wingcal_ratio != 1: {c["wc_ne1"]}.')
    L.append('')
open(f'{HERE}/capcore_tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
pickle.dump({b: {k: v for k, v in res[b].items() if k in ('ref', 'dm', 'avail_star', 'chk', 'havebase')} for b in BANDS}, open(f'{HERE}/out/score.pkl', 'wb'))

# ---- plot ----
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
sty = {'final': ('k', '-'), 'precap': ('C0', '-'), 'cap': ('C3', '-'), 'core_c': ('C2', '-'), 'core_f': ('C1', '-'),
       'core_c|precap': ('C2', '--'), 'core_f|precap': ('C1', '--')}
for ax, band in zip(axs.ravel(), BANDS):
    r = res[band]
    for k, (col, ls) in sty.items():
        s = r['havebase'] & np.isfinite(r['dm'][k])
        x, y = running(r['ref'][s], r['dm'][k][s], 40 if s.sum() > 150 else 20)
        if len(x):
            ax.plot(x, y, color=col, ls=ls, label=f'{k} (N={s.sum()})')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'F{band}')
    ax.set_xlabel('dolphot mag')
    ax.set_ylim(-0.3, 0.45)
    ax.legend(fontsize=7)
axs[0, 0].set_ylabel('running median dm (ours - dolphot - ZP)')
axs[1, 0].set_ylabel('running median dm (ours - dolphot - ZP)')
fig.tight_layout()
fig.savefig(f'{HERE}/capcore.png', dpi=110)

# ---- LW cap diagnosis ----
D = []
for band in BANDS:
    r = res[band]
    T, rs = r['rows'], r['row_star']
    ref = np.where(rs >= 0, r['ref'][np.clip(rs, 0, None)], np.nan)
    dmf = np.where(rs >= 0, r['dm_final'][np.clip(rs, 0, None)], np.nan)
    g = lambda c: np.asarray(T[c], float)  # noqa: E731
    cap = g('cap')
    lab = g('label') > 0
    # flux implied by dolphot on the pipeline scale: final flux * 10^(0.4 dm)   (mag_ours - dm = dolphot-scale mag)
    f_dol = g('c_flux_fit') * 10 ** (0.4 * dmf)
    ppk = g('ppk')
    rows = {}
    rows['n rows'] = lab
    base = lab & (rs >= 0)
    D.append(f'#### F{band} per-row diagnostics by dolphot mag bin (rows matched to a replaced star)')
    D.append('')
    bins = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 18)] if band in LW else [(13, 14.5), (14.5, 15), (15, 16), (16, 17), (17, 19)]
    hdr = ['bin', 'rows', 'cap defined', 'median lost frac (cap-def rows)', 'cap/precap', 'cap/f_dolphot', 'precap/f_dolphot', 'core_c/f_dolphot', 'core_f/f_dolphot',
           'peak px / (f_dol*ppk)', 'peak val MJy/sr', 'peak val / L', 'peak g0 DN', 'g0 / g0sat99', 'peak first-frame DN', 'frac peak in rim', 'frac peak g0-sat', 'frac peak = model-peak px', 'median pk_psf_rel', 'median n_rec']
    D.append('| ' + ' | '.join(hdr) + ' |')
    D.append('|' + '---|' * len(hdr))
    for lo, hi in bins:
        s = base & (ref >= lo) & (ref < hi)
        c = s & np.isfinite(cap)
        if s.sum() < 3:
            continue
        def med(x, m=c):
            v = x[m & np.isfinite(x)]
            return f'{np.median(v):.3g}' if len(v) else '-'
        pkr = g('pk_val') / (f_dol * ppk)
        D.append(f'| {lo}-{hi} | {s.sum()} | {c.sum()} ({c.sum() / s.sum():.2f}) | {med(g("lostf"))} | {med(cap / g("c_flux_fit_precap"))} | {med(cap / f_dol)} | {med(g("c_flux_fit_precap") / f_dol, s)} | '
                 f'{med(g("a_core_c") / f_dol, s)} | {med(g("a_core_f") / f_dol, s)} | {med(pkr)} | {med(g("pk_val"))} | {med(g("pk_val") / g("m_lsat"))} | {med(g("pk_g0"))} | {med(g("pk_g0") / g("m_g0sat99"))} | '
                 f'{med(g("pk_ff"))} | {med(g("pk_rim"))} | {med(g("pk_g0sat"))} | {med(g("pk_is_ipk"))} | {med(g("pk_psf_rel"))} | {med(g("nrec"))} |')
    D.append('')
    D.append(f'Frame constants (median over frames): L (99.999 pct of finite unsaturated non-rim data, MJy/sr) {np.median(g("m_lsat")):.1f}; 0.99-pct of ZEROFRAME/g0 at SATURATED px (DN) {np.nanmedian(g("m_g0sat99")):.0f}; max g0 {np.nanmedian(g("m_zmax")):.0f}; max first-frame {np.nanmedian(g("m_ffmax")):.0f}; PHOTMJSR {np.median(g("m_photmjsr")):.3f}; header R {np.median(g("m_rhdr")):.4f}')
    D.append('')
open(f'{HERE}/capcore_diag.md', 'w').write('\n'.join(D) + '\n')
print('\n'.join(D))

# ---- per-row table of matched rows for capcore_lwdiag.py ----
for band in BANDS:
    r = res[band]
    T, rs = r['rows'], r['row_star']
    m = rs >= 0
    out = Table()
    out['ref'] = r['ref'][rs[m]]
    out['dm_final'] = r['dm_final'][rs[m]]
    for c in ('cap', 'c_flux_fit', 'c_flux_fit_precap', 'c_flux_fit_raw', 'a_core_c', 'a_core_f', 'n_core', 'cov_core', 'ppk', 'pk_val', 'pk_g0', 'pk_ff',
              'pk_g0sat', 'pk_rim', 'pk_crf', 'pk_psf_rel', 'pk_is_ipk', 'lostf', 'nrec', 'rim_ratio_med', 'rim_g0_med', 'm_lsat', 'm_g0sat99', 'm_zmax', 'm_ffmax',
              'm_photmjsr', 'm_rhdr', 'ipk_val', 'ipk_model', 'pk_model'):
        out[c] = np.asarray(T[c], float)[m]
    out['f_dol'] = out['c_flux_fit'] * 10 ** (0.4 * out['dm_final'])
    out['frame'] = np.asarray(T['frame'])[m]
    out.write(f'{HERE}/out/rows_{band}.fits', overwrite=True)
