"""Task 4: cap rule candidates, all four bands.  usage: python an4.py [BAND ...]  -> an4_<band>.pkl, an4_tables.md"""
import pickle, sys
import numpy as np
from cb_lib import Band, mad
from capfun import cap_arrays

BINS = {'150W': [(14, 15), (15, 16), (16, 17), (17, 18), (18, 19)], '200W': [(13, 14), (14, 15), (15, 16), (16, 17), (17, 18)],
        '250M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17)], '300M': [(12.3, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17)]}
REFV = {'150W': 0.004, '200W': -0.017, '250M': 0.003, '300M': -0.003}
FS = (0.5, 0.6, 0.7, 0.8)
SRCBINS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5]


def bind_info(B, cutname):
    """per row: cap (unscaled), binding pixel src fraction, binding pixel index"""
    cap = np.full(B.nrow, np.nan)
    sf = np.full(B.nrow, np.nan)
    for k, r in enumerate(B.rows):
        if r['label'] <= 0:
            continue
        g = r['reg']
        c, b = cap_arrays(g[cutname], g['psf'], g['ur'], r['pkidx'], r['ppk'], return_bind=True)
        cap[k] = c
        if b >= 0:
            sf[k] = B.src_frac(r, k)[b]
    return cap, sf


def kappa_curve(B):
    """median data / (dolphot-implied model) at the binding pixel of the H+h0 cap, by source-DN fraction bin."""
    cap, sf = bind_info(B, 'cutH0')
    ad = B.a_g10
    kap = np.full(B.nrow, np.nan)
    rowref = np.full(B.nrow, np.nan)
    rowref[B.j] = B.ref[B.i]
    for k, r in enumerate(B.rows):
        if r['label'] <= 0 or not np.isfinite(ad[k]) or ad[k] <= 0:
            continue
        g = r['reg']
        c, b = cap_arrays(g['cutH0'], g['psf'], g['ur'], r['pkidx'], r['ppk'], return_bind=True)
        if b >= 0:
            kap[k] = g['cutH0'][b] / (ad[k] * g['psf'][b])
    out = []
    for lo, hi in zip(SRCBINS[:-1], SRCBINS[1:]):
        s = np.isfinite(kap) & np.isfinite(sf) & (sf >= lo) & (sf < hi) & (rowref >= 12.3) & (rowref < 16)
        out.append((lo, hi, int(s.sum()), float(np.median(kap[s])) if s.sum() >= 8 else np.nan))
    return out


def kappa_fn(curve, kref=None):
    """piecewise-constant relative kappa(x) = kappa_bin / kappa_ref (kappa_ref = lowest-DN bins' median), capped at 1."""
    vals = np.array([c[3] for c in curve])
    ref = np.nanmedian(vals[:2]) if kref is None else kref
    rel = np.minimum(vals / ref, 1.0)
    # fill NaN bins from the nearest lower bin
    for i in range(1, len(rel)):
        if not np.isfinite(rel[i]):
            rel[i] = rel[i - 1]
    centers = np.array([0.5 * (c[0] + c[1]) for c in curve])
    centers[-1] = 1.0
    return lambda x: np.interp(x, centers, rel)


def run_band(band, kcurves):
    B = Band(band)
    fam = {'H+h0+bgfree': (B.a_H_h0_bgfree, 'cutH0'), 'H': (B.a_H, 'cutH')}
    dms = {}
    dms['final'] = B.dm_final
    dms['uncapped (a_cat)'] = B.dm_unc
    for fn_, (a, cutn) in fam.items():
        cap0, sf0 = bind_info(B, cutn)
        dms[f'{fn_} uncapped'] = B.dm_of(a)
        dms[f'{fn_} + cap (round 7)'] = B.dm_capped(a, cap0)
        for f in FS:
            for gate, gn in ((True, 'gm'), (False, 'go')):
                cap = B.cap_rows(lambda r, k: cap_arrays(r['reg'][cutn], r['reg']['psf'], r['reg']['ur'], r['pkidx'], r['ppk'],
                                                      excl=B.src_frac(r, k) > f, excl_gate=gate))
                dms[f'{fn_} + rule i f={f} {gn}'] = B.dm_capped(a, cap)
            cap = B.cap_rows(lambda r, k: cap_arrays(r['reg'][cutn], r['reg']['psf'], r['reg']['ur'], r['pkidx'], r['ppk'],
                                                  excl=r['reg']['repl'] & (B.src_frac(r, k) > f), excl_gate=False))
            dms[f'{fn_} + rule i(b-type, replaced only) f={f} go'] = B.dm_capped(a, cap)
        for f in FS + (0.9,):
            capii = np.where(np.isfinite(sf0) & (sf0 > f), np.nan, cap0)
            dms[f'{fn_} + rule ii f={f}'] = B.dm_capped(a, capii)
        # (iii-b) tolerance floor
        for tau in (0.95, 0.97):
            c = cap0 * B.rcor
            c = np.where(np.isfinite(c), c, np.inf)
            dms[f'{fn_} + rule iii-b floor tau={tau}'] = B.dm_of(np.maximum(np.minimum(a, c), tau * a))
        # (iii-a) kappa-corrected cap; kappa curve from the other LW band (cross) and own band (in-sample)
        for src, kc in kcurves.items():
            kf = kappa_fn(kc)
            capk = cap0 / kf(np.where(np.isfinite(sf0), sf0, 0.0))
            dms[f'{fn_} + rule iii-a kappa({src})'] = B.dm_capped(a, capk)
    res = dict(band=band, dms=dms, ref=B.ref, unsat=B.unsat_dm, have0=B.have0, rowinfo=None)
    pickle.dump(res, open(f'an4_{band}.pkl', 'wb'))
    return B, res


def table(band, res, names=None):
    ref, dms = res['ref'], res['dms']
    names = names or list(dms)
    have = res['have0'].copy()
    for nm in names:
        have &= np.isfinite(dms[nm])
    bins = BINS[band]
    L = [f'#### F{band}: {int(have.sum())} stars; unsaturated-reference dm {REFV[band]:+.3f} (this run {res["unsat"]:+.3f}); median dm (MAD) by dolphot bin', '',
         '| variant | ' + ' | '.join(f'{lo}-{hi} (N={int((have & (ref >= lo) & (ref < hi)).sum())})' for lo, hi in bins) + ' | all med / MAD | max abs(bin - ref) |', '|---|' + '---|' * (len(bins) + 2)]
    for nm in names:
        d = dms[nm]
        cells, dev = [], []
        for lo, hi in bins:
            s = have & (ref >= lo) & (ref < hi)
            if s.sum() >= 5:
                m_ = np.median(d[s])
                cells.append(f'{m_:+.3f} ({mad(d[s]):.3f})')
                dev.append(abs(m_ - REFV[band]))
            else:
                cells.append('-')
        L.append(f'| {nm} | ' + ' | '.join(cells) + f' | {np.median(d[have]):+.3f} / {mad(d[have]):.3f} | {max(dev):.3f} |')
    return L + ['']


if __name__ == '__main__':
    bands = sys.argv[1:] or ['250M', '300M', '150W', '200W']
    kcurves = {}
    Bs = {}
    for b in ('250M', '300M'):
        Bs[b] = Band(b)
        kcurves[b] = kappa_curve(Bs[b])
        print(b, 'kappa curve', kcurves[b])
    kcurves['both'] = [(c1[0], c1[1], c1[2] + c2[2], np.nanmean([c1[3], c2[3]])) for c1, c2 in zip(kcurves['250M'], kcurves['300M'])]
    pickle.dump(kcurves, open('an4_kappa.pkl', 'wb'))
    allL = []
    for band in bands:
        B, res = run_band(band, {'250M': kcurves['250M'], '300M': kcurves['300M'], 'both': kcurves['both']})
        allL += table(band, res)
    open('an4_tables_full.md', 'w').write('\n'.join(allL) + '\n')
    print('\n'.join(allL))
