"""Cap-off counterfactual for main2 (KEEP_FINITE off) and main2kf (on).

The recovered-core cap only lowers the satstar flux_fit after the fit
(flux_fit = min(fit, cap)); flux_fit_precap keeps the uncapped value.  The wing
self-calibration is skipped in F250M/F277W/F300M/F323N on wd2 ("no usable peak
window"), so f/precap is the cap alone there.  For each m8 star replaced in
both arms, the per-frame m7 satstar rows within 0.1" give
shift = median_frames 2.5 log10(flux_fit / flux_fit_precap) <= 0, and the
predicted cap-off dm is dm + shift.  The effect of the cap on neighbours is
not modelled.
usage: PYTHONPATH=$Q python kf_lwmad/capoff.py
"""
import glob
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import analyze as an

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
an.ZPWIN.update(an.zp_windows())
ARMS = {'main2': an.Arm('main2'), 'main2kf': an.Arm('main2kf')}
BANDS = ['250M', '277W', '300M', '323N', '410M']
BINS = [(10, 13), (13, 14), (14, 15), (15, 16), (16, 18)]


def frame_rows(arm, band):
    fns = sorted(glob.glob(f'{Q}/tree_{arm}/F{band}/pipeline/*nrcblong_align_o005_crf_resbgsub_m7_satstar_catalog.fits'))
    sk, f, pre, wc = [], [], [], []
    for fn in fns:
        t = Table.read(fn)
        c = t['skycoord_fit']
        sk.append(c if isinstance(c, SkyCoord) else SkyCoord(c))
        f.append(np.ma.filled(t['flux_fit'], np.nan).astype(float))
        pre.append(np.ma.filled(t['flux_fit_precap'], np.nan).astype(float))
        wc.append(np.ma.filled(t['wingcal_ratio'], np.nan).astype(float) if 'wingcal_ratio' in t.colnames
                  else np.full(len(t), np.nan))
    from astropy.coordinates import concatenate
    return len(fns), concatenate(sk), np.concatenate(f), np.concatenate(pre), np.concatenate(wc)


def shift(A, arm, band, sel):
    nfr, sk, f, pre, wc = frame_rows(arm, band)
    pos = A.sky[A.idx[sel]]
    i, j, _, _ = sk.search_around_sky(pos, 0.1 * u.arcsec)
    r = 2.5 * np.log10(f[j] / pre[j])
    out = np.full(sel.sum(), np.nan)
    nmatch = np.zeros(sel.sum(), int)
    for k in np.unique(i):
        v = r[i == k]
        v = v[np.isfinite(v)]
        nmatch[k] = len(v)
        if len(v):
            out[k] = np.median(v)
    return nfr, out, nmatch, np.nanmedian(wc)


def st(x):
    x = x[np.isfinite(x)]
    return f'{np.median(x):+.3f} / {an.mad(x):.3f}' if len(x) else '—'


A, B = ARMS['main2'], ARMS['main2kf']
print('Predicted cap-off dm (dm + median-frame 2.5log10(flux_fit/flux_fit_precap)), stars replaced in both arms.')
print('Values: median / MAD of dm after the band zero point.\n')
print('| band | dolphot mag | N | off | off, cap off | on | on, cap off | median shift off / on | frac capped off / on |')
print('|---|---|---|---|---|---|---|---|---|')
for b in BANDS:
    ok = A.matched & B.matched & A.rep[b] & B.rep[b] & np.isfinite(A.ref[b])
    nfa, sa, na, wca = shift(A, 'main2', b, ok)
    nfb, sb, nb, wcb = shift(B, 'main2kf', b, ok)
    dA, dB = A.dm(b)[ok], B.dm(b)[ok]
    ref = A.ref[b][ok]
    have = (na > 0) & (nb > 0)
    for name, s in [('all', have)] + [(f'{lo}–{hi}', have & (ref >= lo) & (ref < hi)) for lo, hi in BINS]:
        if s.sum() < 3:
            continue
        print(f'| F{b} | {name} | {s.sum()} | {st(dA[s])} | {st(dA[s] + sa[s])} | {st(dB[s])} | {st(dB[s] + sb[s])} | '
              f'{np.median(sa[s]):+.3f} / {np.median(sb[s]):+.3f} | {np.mean(sa[s] < -0.001):.2f} / {np.mean(sb[s] < -0.001):.2f} |')
    print(f'<!-- F{b}: {nfa}/{nfb} frames; {ok.sum()} replaced in both, {have.sum()} with frame rows in both; '
          f'median wingcal_ratio off/on {wca:.3f}/{wcb:.3f} -->')
