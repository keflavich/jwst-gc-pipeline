"""Internal flux-scale test: satstar-fit flux vs daophot flux for the same star in different frames.

Usage: nice -19 python -u ovlscale.py [BAND ...]   (default: all bands; reads existing files only)
Writes into this directory: ovlscale_results.json, ovlscale_stars_<band>.npz, ovlscale.png, ovlscale_tables.md

Per band:
  S rows = per-frame m7 satstar catalogs (final flux_fit with the pooled wing calibration applied the way the merge
           applies it on read; flux_fit_precap; flux_fit_raw).
  D rows = per-frame m7 daophot tables (flux_fit), clean (flags == 0, not forced_refit), taken only from frames in which
           no satstar row lies within 0.3 arcsec and the crf DQ has no SATURATED pixel in the 5x5 block around the fit.
  Stars = friends-of-friends clusters (0.1 arcsec) of S rows; D rows are matched to the cluster centre within 0.1 arcsec.
  Overlap stars have >= 1 S row and >= 1 D row.  dm = m_S - m_D from medians over rows (negative = satstar brighter).
"""
import sys
import glob
import os
import re
import json
import numpy as np
from astropy.table import Table
from astropy.io import fits
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TREE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2'
OUT = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/ovlscale'
PIPE = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2'
MERGED = f'{TREE}/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
BANDS = ['F150W', 'F162M', 'F182M', 'F200W', 'F250M', 'F277W', 'F300M']
RADIUS = 0.1     # arcsec, star clustering and D matching
SATEX = 0.3      # arcsec, frame counts as saturated for the star if an S row lies this close
ABMAG_OFFSET = 8.9
SATBIT = 2


def sky_xy(ra, dec, ra0, dec0):
    return np.column_stack([(ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0, (dec - dec0) * 3600.0])


def mag(flux, pixsr):
    with np.errstate(invalid='ignore', divide='ignore'):
        return -2.5 * np.log10(np.asarray(flux, float) * 1e6 * pixsr) + ABMAG_OFFSET


def load_pooled(band):
    sys.path.insert(0, PIPE)
    from jwst_gc_pipeline.photometry.merge_catalogs import apply_pooled_wingcal
    return apply_pooled_wingcal


CACHE = '/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst-wd2/0e557564-8b2a-47a4-a008-5add86ab6df7/scratchpad'


def load_band(band):
    cf = f'{CACHE}/ovl_cache_{band}.npz'
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True)
        S = {k[2:]: z[k] for k in z.files if k.startswith('S_')}
        D = {k[2:]: z[k] for k in z.files if k.startswith('D_')}
        return S, D, float(z['pixsr'])
    S, D, pixsr = _load_band(band)
    np.savez(cf, pixsr=pixsr, **{'S_' + k: v for k, v in S.items()}, **{'D_' + k: v for k, v in D.items()})
    return S, D, pixsr


def _load_band(band):
    apply_pooled = load_pooled(band)
    dfiles = sorted(glob.glob(f'{TREE}/{band}/*_resbgsub_m7_daophot_basic.fits'))
    S, D = [], []
    pixsr = None
    for fi, df in enumerate(dfiles):
        m = re.search(r'_(nrc[ab](?:[1-4]|long))_visit\d+_vgroup\d+_exp(\d+)_', os.path.basename(df))
        det, exp = m.group(1), int(m.group(2))
        crf = glob.glob(f'{TREE}/{band}/pipeline/*_{exp:05d}_{det}_align_o005_crf.fits')
        scat = glob.glob(f'{TREE}/{band}/pipeline/*_{exp:05d}_{det}_align_o005_crf_resbgsub_m7_satstar_catalog.fits')
        if len(crf) != 1:
            print('skip frame', df, crf, flush=True)
            continue
        d = Table.read(df)
        pixsr = (d.meta['PIXSCALE'] / 206264.806) ** 2
        with fits.open(crf[0], memmap=False) as h:
            dq = np.asarray(h['DQ'].data).astype(np.int64)
        sat = ((dq & SATBIT) > 0).astype(np.uint8)
        sat5 = maximum_filter(sat, size=5) > 0
        sat1 = sat > 0
        sc = d['skycoord_centroid']
        xx = np.asarray(d['x_fit'], float)
        yy = np.asarray(d['y_fit'], float)
        ix = np.clip(np.rint(xx).astype(int), 0, sat.shape[1] - 1)
        iy = np.clip(np.rint(yy).astype(int), 0, sat.shape[0] - 1)
        D.append(dict(frame=np.full(len(d), fi), det=np.full(len(d), det), ra=sc.ra.deg, dec=sc.dec.deg,
                      flux=np.asarray(d['flux_fit'], float), x=xx, y=yy, flags=np.asarray(d['flags']),
                      forced=np.asarray(d['forced_refit'], bool), qfit=np.asarray(d['qfit'], float),
                      sat5=sat5[iy, ix], sat1=sat1[iy, ix]))
        if scat:
            s = Table.read(scat[0])
            if len(s):
                s = apply_pooled(s, band, basepath=TREE, phase='m7')
                sc = s['skycoord_fit']
                xs = np.asarray(s['xcentroid'], float)
                ys = np.asarray(s['ycentroid'], float)
                S.append(dict(frame=np.full(len(s), fi), det=np.full(len(s), det), ra=sc.ra.deg, dec=sc.dec.deg,
                              flux=np.asarray(s['flux_fit'], float), precap=np.asarray(s['flux_fit_precap'], float),
                              raw=np.asarray(s['flux_fit_raw'], float), area=np.asarray(s['sat_area'], float),
                              x=xs, y=ys, wing=np.asarray(s['wingcal_ratio'], float),
                              seed=np.asarray(s['seed_kind']).astype(str)))
        print(band, fi, det, exp, len(d), len(s) if scat else 0, flush=True)

    def cat(L):
        return {k: np.concatenate([r[k] for r in L]) for k in L[0]}
    return cat(S), cat(D), pixsr


def fof(xy, rad):
    tree = cKDTree(xy)
    pairs = tree.query_pairs(rad, output_type='ndarray')
    n = len(xy)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def phase_r(x, y):
    fx = x - np.rint(x)
    fy = y - np.rint(y)
    return np.hypot(fx, fy)


def mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan


def process(band):
    S, D, pixsr = load_band(band)
    ra0, dec0 = np.median(D['ra']), np.median(D['dec'])
    sxy = sky_xy(S['ra'], S['dec'], ra0, dec0)
    dxy = sky_xy(D['ra'], D['dec'], ra0, dec0)
    lab = fof(sxy, RADIUS)
    nst = lab.max() + 1
    cen = np.array([np.median(sxy[lab == k], axis=0) for k in range(nst)])
    dtree = cKDTree(dxy)
    stree = cKDTree(sxy)
    alltree = cKDTree(np.vstack([sxy, dxy]))
    allflux = np.concatenate([S['flux'], D['flux']])
    dok = (D['flags'] == 0) & ~D['forced'] & np.isfinite(D['flux']) & (D['flux'] > 0) & ~D['sat5']
    dcore = (D['flags'] == 0) & ~D['forced'] & np.isfinite(D['flux']) & (D['flux'] > 0) & ~D['sat1']
    drel = (D['flags'] == 0) & ~D['forced'] & np.isfinite(D['flux']) & (D['flux'] > 0)
    dok1 = (D['flags'] <= 1) & ~D['forced'] & np.isfinite(D['flux']) & (D['flux'] > 0) & ~D['sat5']
    rows = []
    why = dict(Dcand_rows=0, sat5=0, sat1=0, flags=0, forced=0, clean=0)
    for k in range(nst):
        si = np.where(lab == k)[0]
        satframes = set(S['frame'][si].tolist())
        # frames where a satstar row lies near the star centre at all (looser radius)
        near_s = stree.query_ball_point(cen[k], SATEX)
        satframes |= set(S['frame'][near_s].tolist())
        di = [j for j in dtree.query_ball_point(cen[k], RADIUS) if D['frame'][j] not in satframes]
        # one row per frame: nearest
        best = {}
        for j in di:
            dd = np.hypot(*(dxy[j] - cen[k]))
            if D['frame'][j] not in best or dd < best[D['frame'][j]][0]:
                best[D['frame'][j]] = (dd, j)
        dj = np.array([v[1] for v in best.values()], int)
        if len(dj) == 0:
            continue
        fs = np.nanmedian(S['flux'][si])
        # isolation: other rows within 0.5" (beyond 0.15" of centre) brighter than 0.2 x star flux
        nb = alltree.query_ball_point(cen[k], 0.5)
        iso = True
        for q in nb:
            pos = np.vstack([sxy, dxy])[q]
            if np.hypot(*(pos - cen[k])) > 0.15 and allflux[q] > 0.2 * fs:
                iso = False
                break
        g = dj[dok[dj]]
        why['Dcand_rows'] += len(dj)
        why['sat5'] += int(D['sat5'][dj].sum())
        why['sat1'] += int(D['sat1'][dj].sum())
        why['flags'] += int((D['flags'][dj] != 0).sum())
        why['forced'] += int(D['forced'][dj].sum())
        why['clean'] += len(g)
        g1 = dj[dok1[dj]]
        gc = dj[dcore[dj]]
        gr = dj[drel[dj]]
        rows.append(dict(k=k, nS=len(si), nD=len(g), nD1=len(g1), nDraw=len(dj), iso=iso,
                         fS=fs, fP=np.nanmedian(S['precap'][si]), fR=np.nanmedian(S['raw'][si]),
                         fD=np.nanmedian(D['flux'][g]) if len(g) else np.nan,
                         fD1=np.nanmedian(D['flux'][g1]) if len(g1) else np.nan,
                         area=np.nanmedian(S['area'][si]), ra=np.median(S['ra'][si]), dec=np.median(S['dec'][si]),
                         nD_strict=len(g), fD_strict=np.nanmedian(D['flux'][g]) if len(g) else np.nan, didx_strict=g,
                         nD_core=len(gc), fD_core=np.nanmedian(D['flux'][gc]) if len(gc) else np.nan, didx_core=gc,
                         nD_relaxed=len(gr), fD_relaxed=np.nanmedian(D['flux'][gr]) if len(gr) else np.nan, didx_relaxed=gr,
                         sidx=si, didx=g, didx1=g1))
    print(band, 'S rows', len(S['ra']), 'stars', nst, 'with D candidates', len(rows), flush=True)
    print(band, 'D candidate row accounting', why, flush=True)
    return S, D, pixsr, rows, (dxy, dok, ra0, dec0, why, drel)


def summarize(S, D, pixsr, rows, extra):
    dxy, dok, ra0, dec0, why = extra[:5]
    res = {}
    ov = [r for r in rows if r['nD'] >= 1]
    arr = lambda key: np.array([r[key] for r in ov], float)
    mD = mag(arr('fD'), pixsr)
    mS = mag(arr('fS'), pixsr)
    dmF = mag(arr('fS'), pixsr) - mD
    dmP = mag(arr('fP'), pixsr) - mD
    dmR = mag(arr('fR'), pixsr) - mD
    iso = np.array([r['iso'] for r in ov], dtype=bool)
    nS = arr('nS')
    nD = arr('nD')
    area = arr('area')
    res['arrays'] = dict(mD=mD, mS=mS, dmF=dmF, dmP=dmP, dmR=dmR, iso=iso, nS=nS, nD=nD, area=area,
                         ra=arr('ra'), dec=arr('dec'))
    return res, ov


def binstats(m, dm, edges, minn=5):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (m >= lo) & (m < hi) & np.isfinite(dm)
        if s.sum() >= minn:
            out.append((lo, hi, int(s.sum()), float(np.median(dm[s])), float(mad(dm[s]) / np.sqrt(s.sum()) * 1.253),
                        float(mad(dm[s]))))
    return out


def phase_analysis(S, D, ov, pixsr, extra, band):
    """(a) overlap stars: row-level m_D,i - m_S(median) vs daophot pixel phase; S-frame vs D-frame phase.
    (b) control: D-only stars (no satstar within 0.3 arcsec in any frame, >= 3 clean rows) with a magnitude in the
        overlap range: m_i - median(m_star) vs row phase."""
    dxy, dok, ra0, dec0, why = extra[:5]
    out = {}
    pr_D, dm_D, pr_S = [], [], []
    for r in ov:
        mS0 = mag(r['fS'], pixsr)
        for j in r['didx']:
            pr_D.append(phase_r(D['x'][j], D['y'][j]))
            dm_D.append(mS0 - mag(D['flux'][j], pixsr))     # m_S - m_D,i for this single D row
        for i in r['sidx']:
            pr_S.append(phase_r(S['x'][i], S['y'][i]))
    pr_D, dm_D, pr_S = map(np.array, (pr_D, dm_D, pr_S))
    pb = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
    out['overlap_rowD'] = []
    for lo, hi in zip(pb[:-1], pb[1:]):
        s = (pr_D >= lo) & (pr_D < hi)
        out['overlap_rowD'].append((lo, hi, int(s.sum()), float(np.median(dm_D[s])) if s.sum() else np.nan))
    out['phase_S_median'] = float(np.median(pr_S)) if len(pr_S) else np.nan
    out['phase_D_median'] = float(np.median(pr_D)) if len(pr_D) else np.nan
    # control sample
    mlo, mhi = (np.nanpercentile([mag(r['fD'], pixsr) for r in ov], [5, 95]) if ov else (0, 99))
    sxy = sky_xy(S['ra'], S['dec'], ra0, dec0)
    stree = cKDTree(sxy)
    cand = np.where(dok)[0]
    lab = fof(dxy[cand], RADIUS)
    # exclude clusters with any satstar row within SATEX
    near = np.array([len(x) > 0 for x in stree.query_ball_point(dxy[cand], SATEX)])
    ncl = lab.max() + 1
    bad = np.zeros(ncl, bool)
    bad[np.unique(lab[near])] = True
    cm, cph, cdet = [], [], []
    order = np.argsort(lab, kind='stable')
    bounds = np.searchsorted(lab[order], np.arange(ncl + 1))
    for c in range(ncl):
        if bad[c]:
            continue
        idx = cand[order[bounds[c]:bounds[c + 1]]]
        if len(idx) < 3 or len(set(D['frame'][idx].tolist())) != len(idx):
            continue
        mm = mag(D['flux'][idx], pixsr)
        m0 = np.median(mm)
        if not (mlo <= m0 <= mhi):
            continue
        # leave-one-out reference
        for a, j in enumerate(idx):
            ref = np.median(np.delete(mm, a))
            cm.append(mm[a] - ref)
            cph.append(phase_r(D['x'][j], D['y'][j]))
            cdet.append(str(D['det'][j]))
    cm, cph, cdet = np.array(cm), np.array(cph), np.array(cdet)
    out['control_det'] = {d: (int((cdet == d).sum()), float(np.median(cm[cdet == d]))) for d in sorted(set(cdet.tolist()))}
    out['control_range'] = (float(mlo), float(mhi))
    out['control'] = []
    for lo, hi in zip(pb[:-1], pb[1:]):
        s = (cph >= lo) & (cph < hi)
        out['control'].append((lo, hi, int(s.sum()), float(np.median(cm[s])) if s.sum() else np.nan,
                               float(mad(cm[s])) if s.sum() else np.nan))
    return out


def merge_check(res, band):
    mt = Table.read(MERGED)
    b = band.lower()
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    a = res['arrays']
    sc = SkyCoord(a['ra'] * u.deg, a['dec'] * u.deg)
    ra_m = np.asarray(mt[f'skycoord_{b}'].ra.deg, float)
    dec_m = np.asarray(mt[f'skycoord_{b}'].dec.deg, float)
    fin = np.where(np.isfinite(ra_m) & np.isfinite(dec_m))[0]
    mc = SkyCoord(ra_m[fin] * u.deg, dec_m[fin] * u.deg)
    idx, sep, _ = sc.match_to_catalog_sky(mc)
    idx = fin[idx]
    ok = sep.arcsec < 0.1
    rep = np.asarray(mt[f'replaced_saturated_{b}'], bool)[idx]
    magm = np.asarray(mt[f'mag_ab_{b}'], float)[idx]
    return dict(n=int(len(ok)), matched=int(ok.sum()), replaced=int((rep & ok).sum()),
                med_mmerge_minus_mS=float(np.nanmedian((magm - a['mS'])[ok & rep])) if (ok & rep).any() else np.nan,
                med_mmerge_minus_mD_unreplaced=float(np.nanmedian((magm - a['mD'])[ok & ~rep])) if (ok & ~rep).any() else np.nan,
                n_unrep=int((ok & ~rep).sum()),
                med_mmerge_minus_mD_replaced=float(np.nanmedian((magm - a['mD'])[ok & rep])) if (ok & rep).any() else np.nan)


def analyze_variant(S, D, pixsr, rows, extra, band, variant='strict'):
    if variant == 'relaxed':
        extra = (extra[0], extra[5]) + tuple(extra[2:5])
    res, ov = summarize(S, D, pixsr, rows, extra)
    a = res['arrays']
    r = {}
    r['n_overlap_all'] = int(len(ov))
    r['n_overlap_iso'] = int(a['iso'].sum())
    if not len(ov):
        return r, a
    edges = np.arange(np.floor(np.nanmin(a['mD']) * 2) / 2, np.nanmax(a['mD']) + 0.5, 0.5)
    for lab_, sel in (('all', np.ones(len(ov), bool)), ('iso', a['iso'])):
        for kk in ('dmF', 'dmP', 'dmR'):
            v = a[kk][sel]
            r[f'{lab_}_{kk}_median'] = float(np.nanmedian(v)) if len(v) else np.nan
            r[f'{lab_}_{kk}_mad'] = float(mad(v)) if len(v) else np.nan
            r[f'{lab_}_{kk}_bins'] = binstats(a['mD'][sel], v, edges)
    dmF = a['dmF']
    for nm, key, ed in (('area', 'area', [0, 12, 20, 40, 80, 1e5]), ('nS', 'nS', [1, 2, 3, 5, 100]),
                        ('nD', 'nD', [1, 2, 3, 5, 100])):
        r[f'dep_{nm}'] = binstats(a[key], dmF, np.array(ed, float), minn=5)
    det_rows, fr_rows = {}, {}
    for q in ov:
        for i in q['sidx']:
            dmi = float(mag(S['flux'][i], pixsr) - mag(q['fD'], pixsr))
            det_rows.setdefault(str(S['det'][i]), []).append(dmi)
            fr_rows.setdefault(int(S['frame'][i]), []).append(dmi)
    r['dep_det'] = {k: (len(v), float(np.median(v))) for k, v in sorted(det_rows.items())}
    det_iso = {}
    for q in ov:
        if q['iso']:
            for i in q['sidx']:
                det_iso.setdefault(str(S['det'][i]), []).append(float(mag(S['flux'][i], pixsr) - mag(q['fD'], pixsr)))
    r['dep_det_iso'] = {k: (len(v), float(np.median(v))) for k, v in sorted(det_iso.items())}
    r['dep_frame_median_spread'] = [float(np.percentile([np.median(v) for v in fr_rows.values() if len(v) >= 5], p))
                                    for p in (0, 50, 100)] if any(len(v) >= 5 for v in fr_rows.values()) else []
    r['phase'] = phase_analysis(S, D, ov, pixsr, extra, band)
    r['merge'] = merge_check(res, band)
    return r, a


def main():
    bands = sys.argv[1:] or BANDS
    for band in bands:
        S, D, pixsr, rows, extra = process(band)
        r = {'n_S_rows': int(len(S['ra'])), 'n_stars_with_Dcand': len(rows), 'Dcand_accounting': extra[4],
             'pixsr': pixsr}
        for v in ('strict', 'core', 'relaxed'):
            rows_v = [dict(q, nD=q['nD_' + v], fD=q['fD_' + v], didx=q['didx_' + v]) for q in rows]
            rv, a = analyze_variant(S, D, pixsr, rows_v, extra, band, v)
            r[v] = rv
            if len(a):
                np.savez_compressed(f'{OUT}/ovlscale_stars_{band}_{v}.npz', **a)
            print(band, v, {k: x for k, x in rv.items() if not isinstance(x, (list, dict))}, flush=True)
        with open(f'{OUT}/ovlscale_results_{band}.json', 'w') as f:
            json.dump(r, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))




def report():
    """Read ovlscale_results_<band>.json and write ovlscale_tables.md and ovlscale.png."""
    res = {}
    for b in BANDS:
        fn = f'{OUT}/ovlscale_results_{b}.json'
        if os.path.exists(fn):
            res[b] = json.load(open(fn))
    f = lambda x: 'nan' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:+.3f}'
    L = []
    for v in ('strict', 'core', 'relaxed'):
        L.append(f'\n### Summary, sample `{v}`\n')
        L.append('| band | S rows | stars with D cand. | N overlap (all/iso) | final med (MAD) | precap med (MAD) | raw med (MAD) | iso final med |')
        L.append('|---|---|---|---|---|---|---|---|')
        for b, r in res.items():
            q = r[v]
            if not q.get('all_dmF_median') == q.get('all_dmF_median') or 'all_dmF_median' not in q:
                L.append(f"| {b} | {r['n_S_rows']} | {r['n_stars_with_Dcand']} | 0 / 0 | - | - | - | - |")
                continue
            L.append(f"| {b} | {r['n_S_rows']} | {r['n_stars_with_Dcand']} | {q['n_overlap_all']} / {q['n_overlap_iso']} | "
                     f"{f(q['all_dmF_median'])} ({q['all_dmF_mad']:.3f}) | {f(q['all_dmP_median'])} ({q['all_dmP_mad']:.3f}) | "
                     f"{f(q['all_dmR_median'])} ({q['all_dmR_mad']:.3f}) | {f(q['iso_dmF_median'])} |")
    L.append('\n### Binned dm (isolated stars), bins in daophot AB magnitude; median (error of median) [N]\n')
    for v in ('strict', 'relaxed'):
        L.append(f'\nSample `{v}`\n')
        L.append('| band | bin (m_D, AB) | N | final | precap | raw |')
        L.append('|---|---|---|---|---|---|')
        for b, r in res.items():
            q = r[v]
            if 'iso_dmF_bins' not in q:
                continue
            pb = {tuple(x[:2]): x for x in q['iso_dmP_bins']}
            rb = {tuple(x[:2]): x for x in q['iso_dmR_bins']}
            for x in q['iso_dmF_bins']:
                key = tuple(x[:2])
                L.append(f"| {b} | {x[0]:.1f}-{x[1]:.1f} | {x[2]} | {f(x[3])} ({x[4]:.3f}) | "
                         f"{f(pb[key][3]) if key in pb else '-'} | {f(rb[key][3]) if key in rb else '-'} |")
    L.append('\n### Dependence on detector (S-row level dm, median [N]), sample `strict` or `relaxed` if strict is empty\n')
    for b, r in res.items():
        q = r['strict'] if 'dep_det' in r['strict'] else r['relaxed']
        if 'dep_det' not in q:
            continue
        L.append(f"- {b}: " + ', '.join(f"{k} {f(v[1])} [{v[0]}]" for k, v in q['dep_det'].items()))
        L.append(f"  - isolated stars only: " + ', '.join(f"{k} {f(v[1])} [{v[0]}]" for k, v in q['dep_det_iso'].items()))
    L.append('\n### Dependence on sat_area and number of frames (median dm, final), sample `strict` or `relaxed`\n')
    for b, r in res.items():
        q = r['strict'] if 'dep_area' in r['strict'] else r['relaxed']
        if 'dep_area' not in q:
            continue
        L.append(f"- {b} sat_area bins: " + '; '.join(f"{x[0]:.0f}-{x[1]:.0f}: {f(x[3])} [{x[2]}]" for x in q['dep_area']))
        L.append(f"  - nS bins: " + '; '.join(f"{x[0]:.0f}-{x[1]:.0f}: {f(x[3])} [{x[2]}]" for x in q['dep_nS']))
    L.append('\n### Pixel phase (r = distance of fitted position from the nearest pixel centre, pixels)\n')
    L.append('Overlap: rows are single D rows, dm_i = m_S(star median) - m_D,i. Control: D-only stars of the same magnitude range, '
             'm_i - median(other frames).\n')
    for b, r in res.items():
        q = r['strict'] if 'phase' in r['strict'] else r['relaxed']
        if 'phase' not in q:
            continue
        ph = q['phase']
        L.append(f"- {b} (sample {'strict' if 'phase' in r['strict'] else 'relaxed'}): median r of S fits {ph['phase_S_median']:.2f}, of D rows {ph['phase_D_median']:.2f}")
        L.append('  - overlap D rows: ' + '; '.join(f"r {x[0]}-{x[1]}: {f(x[3])} [{x[2]}]" for x in ph['overlap_rowD']))
        L.append('  - control: ' + '; '.join(f"r {x[0]}-{x[1]}: {f(x[3])} [{x[2]}]" for x in ph['control'])
                 + f"; mag range {ph['control_range'][0]:.1f}-{ph['control_range'][1]:.1f}")
        L.append('  - control by detector: ' + ', '.join(f"{k} {f(v[1])} [{v[0]}]" for k, v in ph['control_det'].items()))
    L.append('\n### Merge treatment of the overlap stars (sample `strict` or `relaxed`)\n')
    for b, r in res.items():
        q = r['strict'] if 'merge' in r['strict'] else r['relaxed']
        if 'merge' not in q:
            continue
        m = q['merge']
        L.append(f"- {b}: {m['n']} overlap stars, {m['matched']} matched in m8_dedup (0.1 arcsec), {m['replaced']} with replaced_saturated; "
                 f"median(m_merged - m_S) = {f(m['med_mmerge_minus_mS'])} for replaced; "
                 f"median(m_merged - m_D) = {f(m['med_mmerge_minus_mD_replaced'])} replaced, {f(m['med_mmerge_minus_mD_unreplaced'])} unreplaced [{m['n_unrep']}]")
    open(f'{OUT}/ovlscale_tables.md', 'w').write('\n'.join(L) + '\n')

    fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    axs = axs.ravel()
    cols = {'dmF': ('C0', 'final'), 'dmP': ('C1', 'precap'), 'dmR': ('C2', 'raw')}
    for ax, b in zip(axs, BANDS):
        if b not in res:
            continue
        xs = []
        for v, ls_, mk in (('relaxed', '--', 's'), ('strict', '-', 'o')):
            fn = f'{OUT}/ovlscale_stars_{b}_{v}.npz'
            if not os.path.exists(fn):
                continue
            z = np.load(fn)
            if not len(z['mD']):
                continue
            xs.append(z['mD'])
            edges = np.arange(np.floor(np.nanmin(z['mD']) * 2) / 2, np.nanmax(z['mD']) + 0.5, 0.5)
            for kk, (c, nm) in cols.items():
                if v == 'strict':
                    ax.plot(z['mD'], z[kk], mk, ms=2.5, color=c, alpha=0.25)
                elif kk == 'dmF':
                    ax.plot(z['mD'], z[kk], mk, ms=2, color='gray', alpha=0.2)
                bs = binstats(z['mD'][z['iso']], z[kk][z['iso']], edges, minn=5)
                if bs:
                    ax.errorbar([(x[0] + x[1]) / 2 for x in bs], [x[3] for x in bs], [x[4] for x in bs],
                                fmt=ls_ + ('D' if v == 'strict' else 'X'), color=c, lw=1.5, ms=6 if v == 'strict' else 5,
                                label=f'{nm} ({v})')
        ax.axhline(0, color='gray', lw=0.7)
        ax.set_ylim(-0.3, 0.3)
        if xs:
            allx = np.concatenate(xs)
            ax.set_xlim(np.nanpercentile(allx, 1) - 0.3, np.nanpercentile(allx, 99) + 0.3)
        n1 = res[b]['strict'].get('n_overlap_all', 0)
        n2 = res[b]['relaxed'].get('n_overlap_all', 0)
        ax.set_title(f'{b}: N strict {n1}, relaxed {n2}')
        ax.set_xlabel('daophot AB mag of the star')
        if b == BANDS[0]:
            ax.set_ylabel('m_satstar - m_daophot (mag)')
        ax.legend(fontsize=6, ncol=2, loc='lower left')
    axs[-1].axis('off')
    axs[-1].text(0.0, 0.9, 'Points and solid lines: strict sample (daophot rows with no SATURATED DQ\nin 5x5, flags 0).\n'
                 'Dashed lines, gray squares: relaxed sample (daophot row from a frame\nwith no satstar row, any DQ; includes frames with DQ-saturated cores).\n'
                 'Lines: median over isolated stars in 0.5 mag bins (error of median).\nBlue final, orange precap, green raw.', fontsize=9, va='top')
    fig.tight_layout()
    fig.savefig(f'{OUT}/ovlscale.png', dpi=110)


if __name__ == '__main__':
    if sys.argv[1:] == ['--report']:
        report()
    else:
        main()
