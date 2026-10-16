"""Cutout + radial-profile figures for saturated stars, arm main2kfpr (jicama vs dolphot).
usage: nice -19 python make_cutouts.py   (writes into this directory only; inputs read-only)"""
import sys
import glob
import os
import re
import warnings
import pickle
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
for a in ('main2kfh', 'main2kfpr', 'main2gt'):
    an.PATH.setdefault(a, (f'{an.Q}/tree_{a}/{an.M8}', f'{an.D}/matched_Q_{a}.fits'))
an.ZPWIN.update(an.zp_windows())
ARM = 'main2kfpr'
A = an.Arm(ARM)
T = f'{an.Q}/tree_{ARM}'
OUT = f'{an.Q}/satbias/cutouts'
BOXAS = 2.5          # cutout box, arcsec
ISO_PX = 30          # no other satstar within this many px (same frame)
EDGE = 45            # keep stars this far from frame edges
RNG = np.random.default_rng(1)
NSAMPLE = 300
SW_PS = {'SW': 0.031, 'LW': 0.063}
NSAT_LABEL = {}


def band_info(b):
    filt = 'f' + b.lower()
    return filt, ('SW' if int(b[:3]) < 240 else 'LW')


def frame_paths(cat_fn, filt):
    d = os.path.dirname(cat_fn)
    base = os.path.basename(cat_fn).replace('_resbgsub_m7_satstar_catalog.fits', '')
    m = re.match(r'(jw\d+)_(\d+)_(\d+)_(nrc\w+?)_align_o005_crf', base)
    vg, exp, det = m.group(2), m.group(3), m.group(4)
    dao = glob.glob(f'{d}/jw03523-o005_t001_nircam_clear-{filt}-{det}_visit*_vgroup{vg}_exp{exp}_resbgsub_m7_daophot_basic_mergedcat_model.fits')
    return dict(crf=f'{d}/{base}.fits', sat=f'{d}/{base}_resbgsub_m7_satstar_model.fits', dao=dao[0] if dao else None, name=base)


def gather(b):
    """all saturated dolphot-matched rows with per-frame matches"""
    filt, kind = band_info(b)
    dm = A.dm(b)
    ref = A.ref[b]
    s = A.matched & (A.sat[b] | A.rep[b]) & np.isfinite(dm) & np.isfinite(ref)
    rows = np.where(s)[0]
    sk = A.sky[A.idx[rows]] if len(A.cat) != A.n else A.sky[rows]
    fr = []
    for fn in sorted(glob.glob(f'{T}/F{b}/pipeline/*m7_satstar_catalog.fits')):
        t = Table.read(fn)
        if len(t) == 0:
            continue
        # isolation: nearest other satstar in the same frame
        xy = np.c_[np.asarray(t['x_0'], float), np.asarray(t['y_0'], float)]
        dmin = np.full(len(t), np.inf)
        for i in range(len(t)):
            dd = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
            dd[i] = np.inf
            dmin[i] = dd.min()
        fr.append(dict(path=fn, sky=SkyCoord(t['skycoord_fit']), x=xy[:, 0], y=xy[:, 1], flux=np.asarray(t['flux_fit'], float),
                       rmask=np.asarray(t['wingcal_rmask'], float), iso=dmin, **{'files': frame_paths(fn, filt)}))
    sc = SkyCoord(np.concatenate([f['sky'].ra.deg for f in fr]) * u.deg, np.concatenate([f['sky'].dec.deg for f in fr]) * u.deg)
    fid = np.concatenate([np.full(len(f['x']), i) for i, f in enumerate(fr)])
    cat = {k: np.concatenate([f[k] for f in fr]) for k in ('x', 'y', 'flux', 'rmask', 'iso')}
    ia, ib, _, _ = sc.search_around_sky(sk, 0.1 * u.arcsec)
    stars = []
    for k in np.unique(ia):
        sel = ib[ia == k]
        rmask_med = np.median(cat['rmask'][sel])
        ok = sel[(cat['x'][sel] > EDGE) & (cat['x'][sel] < 2048 - EDGE) & (cat['y'][sel] > EDGE) & (cat['y'][sel] < 2048 - EDGE)
                 & (cat['iso'][sel] > ISO_PX)]
        if len(ok) == 0:
            continue
        medf = np.median(cat['flux'][sel])
        j = ok[np.argmin(np.abs(cat['flux'][ok] - medf))]
        r = rows[k]
        stars.append(dict(row=r, band=b, dm=dm[r], ref=ref[r], our=A.our[b][r], zp=A.zp[b], ra=A.m['RA'][r], dec=A.m['DEC'][r],
                          rmask=rmask_med, x=cat['x'][j], y=cat['y'][j], flux=cat['flux'][j], fid=fid[j],
                          files=fr[fid[j]]['files'], sky=sc[j]))
    print(b, 'saturated rows', len(rows), 'with isolated frame match', len(stars), flush=True)
    return stars


_cache = {}


def load_frame(files):
    key = files['crf']
    if key not in _cache:
        if len(_cache) > 6:
            _cache.pop(next(iter(_cache)))
        d = fits.getdata(files['crf'], 1).astype(float)
        dq = fits.getdata(files['crf'], 3)
        sat = fits.getdata(files['sat']).astype(float)
        dao = fits.getdata(files['dao'], 1).astype(float) if files['dao'] else np.zeros_like(d)
        _cache[key] = (d, dq, sat, np.nan_to_num(dao))
    return _cache[key]


def radial_grid(shape, x0, y0):
    yy, xx = np.indices(shape)
    return np.hypot(xx - x0, yy - y0)


def star_arrays(st, half=None):
    """full-frame sub-arrays around star; returns dict with data(bg-subtracted), sat model, dao model, dq, local bg"""
    d, dq, sat, dao = load_frame(st['files'])
    h = 40 if half is None else half
    h = max(h, 36)
    x0, y0 = st['x'], st['y']
    xi, yi = int(round(x0)), int(round(y0))
    sl = (slice(yi - h, yi + h + 1), slice(xi - h, xi + h + 1))
    D, Q, S, O = d[sl], dq[sl], sat[sl], dao[sl]
    cx, cy = x0 - (xi - h), y0 - (yi - h)
    r = radial_grid(D.shape, cx, cy)
    resid = D - O
    ann = (r >= 25) & (r < 35) & np.isfinite(resid) & ((Q & 3) == 0)
    bg = np.nanmedian(resid[ann]) if ann.sum() > 20 else 0.0
    return dict(D=D - bg, Q=Q, S=S, O=O, cx=cx, cy=cy, r=r, bg=bg, xi=xi, yi=yi, h=h)


def profile(a, rmax=30):
    bad = ((a['Q'] & 3) != 0) | ~np.isfinite(a['D'])
    r = a['r']
    edges = np.arange(0, rmax + 1)
    rc, pd, pm, po = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (r >= lo) & (r < hi) & ~bad
        if k.sum() < 3:
            rc.append(0.5 * (lo + hi)); pd.append(np.nan); pm.append(np.nan); po.append(np.nan)
            continue
        rc.append(r[k].mean()); pd.append(np.median(a['D'][k])); pm.append(np.median(a['S'][k])); po.append(np.median(a['O'][k]))
    return np.array(rc), np.array(pd), np.array(pm), np.array(po)


def ratio_3_8(a):
    bad = ((a['Q'] & 3) != 0) | ~np.isfinite(a['D'])
    rc, pd, pm, po = profile(a, 8)
    k = (rc >= 3) & np.isfinite(pd) & (pm > 0)
    if k.sum() < 2:
        return np.nan, np.nan
    return float(np.median(pd[k] / pm[k])), float(np.median(pd[k] / po[k]))


def sat_component(a):
    m = (a['Q'] & 2) > 0
    lab, n = ndimage.label(m)
    if n == 0:
        return m, 0
    cy, cx = int(round(a['cy'])), int(round(a['cx']))
    if lab[cy, cx] > 0:
        l = lab[cy, cx]
    else:
        dist = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        iy, ix = dist[1][0][cy, cx], dist[1][1][cy, cx]
        l = lab[iy, ix]
    comp = lab == l
    return comp, int(comp.sum())


def choose(stars, b):
    dmall = np.array([s['dm'] for s in stars])
    rm = np.array([s['rmask'] for s in stars])
    med = np.median(dmall)
    chosen, used = [], set()

    def take(order, n, label):
        c = 0
        for i in order:
            if i in used:
                continue
            used.add(i)
            stars[i]['group'] = label
            chosen.append(stars[i])
            c += 1
            if c == n:
                break
    # median: closest to band median among rmask<1
    cand = np.where(rm < 1)[0]
    # spread over dolphot magnitude: take closest, then closest with different ref mag (>0.7 mag apart)
    o = cand[np.argsort(np.abs(dmall[cand] - med))]
    first = o[0]
    sec = [i for i in o if abs(stars[i]['ref'] - stars[first]['ref']) > 0.7]
    take([first] + sec[:1], 2, 'median')
    cand = np.where((rm >= 1) & (rm < 5))[0]
    take(cand[np.argsort(dmall[cand])], 2, 'rmask 1-5 px, most negative dm')
    cand = np.where(dmall > 0.03)[0]
    if len(cand) >= 2:
        o = cand[np.argsort(dmall[cand])]
        take([o[len(o) // 3], o[(2 * len(o)) // 3]], 2, 'dm > +0.03')
    else:
        cand = np.argsort(-dmall)
        take(cand, 2, 'largest dm (fewer than 2 with dm>+0.03)')
    return chosen


def mosaic_cutout(b, sky, size_as):
    filt, kind = band_info(b)
    fn = f'{T}/F{b}/pipeline/jw03523-o005_t001_nircam_clear-{filt}-merged_resbgsub_m7_daophot_basic_mergedcat_residual_i2d.fits'
    with fits.open(fn, memmap=True) as h:
        w = WCS(h[1].header)
        co = Cutout2D(h[1].data, sky, size_as * u.arcsec, wcs=w, mode='partial', fill_value=np.nan)
    return co.data.astype(float)


def draw_row(fig, gs_row_title, gs_row, st, b, kind):
    ps = SW_PS[kind]
    half = int(round(BOXAS / ps / 2))
    a = star_arrays(st, half)
    h, c = a['h'], int(round(a['h']))
    # crop to half
    s0 = slice(h - half, h + half + 1)
    sub = lambda z: z[s0, s0]
    D, S, O, Q = sub(a['D']), sub(a['S']), sub(a['O']), sub(a['Q'])
    scale = 10 ** (0.4 * st['dm'])
    peak_model = np.nanmax(S)
    m_ours = O                      # per-frame daophot 'mergedcat' model already contains the satstar model
    m_dol = O + (scale - 1.0) * S
    res_ours = (D - m_ours) / peak_model
    res_dol = (D - m_dol) / peak_model
    comp, nsat = sat_component(a)
    comp = sub(comp)
    allsat = (Q & 2) > 0
    ext = [-half - 0.5, half + 0.5, -half - 0.5, half + 0.5]
    vmax = np.nanpercentile(D[~allsat], 99.9) if (~allsat).sum() > 10 else np.nanmax(D)
    vmax = max(vmax, 1e-6)
    vmin = np.nanpercentile(D, 1)
    from astropy.visualization import ImageNormalize, AsinhStretch
    nrm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.02))
    # mosaic residual on same surface-brightness scale
    mos = mosaic_cutout(b, st['sky'], BOXAS) / peak_model
    lim = 0.1
    rn = SymLogNorm(linthresh=1e-3, vmin=-lim, vmax=lim, base=10)
    panels = [(D, nrm, 'gray_r', 'data (crf, bg-sub)'), (m_ours, nrm, 'gray_r', 'our model (frame total)'),
              (res_ours, rn, 'RdBu_r', 'our residual'), (mos, rn, 'RdBu_r', 'mosaic residual (i2d, N up)'),
              (res_dol, rn, 'RdBu_r', 'residual if flux = dolphot')]
    axt = fig.add_subplot(gs_row_title)
    axt.axis('off')
    ttl = (f"{st['group']}:  RA/Dec {st['ra']:.5f} {st['dec']:+.5f}   dolphot {st['ref']:.3f}   ours-ZP {st['our'] - st['zp']:.3f}   "
           f"dm {st['dm']:+.3f}   N_sat {nsat} px   rmask {st['rmask']:.1f}   {st['files']['name'][14:]}")
    axt.text(0.0, 0.2, ttl, fontsize=8.5, va='bottom', ha='left', transform=axt.transAxes)
    axes = []
    for j, (img, norm, cm, name) in enumerate(panels):
        ax = fig.add_subplot(gs_row[j])
        axes.append(ax)
        if j == 3:
            n = img.shape[0]
            e = [-n / 2, n / 2, -n / 2, n / 2]
            ax.imshow(img, origin='lower', cmap=cm, norm=norm, extent=e, interpolation='nearest')
        else:
            ax.imshow(img, origin='lower', cmap=cm, norm=norm, extent=ext, interpolation='nearest')
            if j in (0, 1, 2, 4):
                ax.contour(np.arange(-half, half + 1), np.arange(-half, half + 1), allsat.astype(float), levels=[0.5],
                           colors='lime', linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        if st['_first']:
            ax.set_title(name, fontsize=9)
    return axes, rn


FIGS = {}


def sample_ratios(stars, b):
    """median data/model at 3-8 px over a random sample of isolated saturated stars"""
    idx = RNG.permutation(len(stars))[:NSAMPLE]
    byfile = {}
    for i in idx:
        byfile.setdefault(stars[i]['files']['crf'], []).append(i)
    out, profs = [], []
    for fn, lst in byfile.items():
        for i in lst:
            a = star_arrays(stars[i], 36)
            r = ratio_3_8(a)
            rc_, pd_, pm_, po_ = profile(a, 30)
            profs.append(np.c_[pd_ / pm_, pd_ / po_])
            out.append((stars[i]['rmask'], r[0], stars[i]['dm'], stars[i]['ref'], comp_n(a), r[1]))
    return np.array(out), np.array(profs)


def comp_n(a):
    return sat_component(a)[1]


def main():
    bands = {'SW': ['150W', '200W'], 'LW': ['277W', '405N']}
    allstars = {}
    chosen = {}
    for b in ['150W', '200W', '277W', '405N', '323N']:
        allstars[b] = gather(b)
    for b in ['150W', '200W', '277W', '405N']:
        chosen[b] = choose(allstars[b], b)
        for s in chosen[b]:
            print(b, s['group'], 'ref %.2f dm %+.3f rmask %.1f' % (s['ref'], s['dm'], s['rmask']))
    # --- cutout figures
    for kind, fname in (('SW', 'cutouts_sw.png'), ('LW', 'cutouts_lw.png')):
        blist = bands[kind]
        nrow = 6 * len(blist)
        fig = plt.figure(figsize=(12.5, 2.55 * nrow + 0.4), dpi=110)
        gs = GridSpec(2 * nrow, 5, figure=fig, height_ratios=[0.22, 1] * nrow, hspace=0.08, wspace=0.03,
                      left=0.01, right=0.92, top=0.985, bottom=0.005)
        rn = None
        r = 0
        for b in blist:
            fig.text(0.005, 1 - (r / nrow) * 0.985 - 0.003, f'F{b[:-1]}{b[-1]}', fontsize=13, fontweight='bold', va='top')
            for k, st in enumerate(chosen[b]):
                st['_first'] = (r == 0)
                axes, rn = draw_row(fig, gs[2 * r, :], [gs[2 * r + 1, j] for j in range(5)], st, b, kind)
                r += 1
        cax = fig.add_axes([0.935, 0.35, 0.012, 0.3])
        sm = plt.cm.ScalarMappable(norm=rn, cmap='RdBu_r')
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('residual / (peak pixel of satstar model)', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.savefig(f'{OUT}/{fname}', dpi=110)
        plt.close(fig)
        print('wrote', fname, flush=True)
    # --- radial profiles
    stats = {}
    samples = {}
    for b in ['150W', '200W', '277W', '405N', '323N']:
        samples[b] = sample_ratios(allstars[b], b)
    colors = {'median': '#1b7837', 'rmask 1-5 px, most negative dm': '#b2182b', 'dm > +0.03': '#2166ac'}
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.2), dpi=110, sharex=True)
    for j, b in enumerate(['150W', '200W', '277W', '405N']):
        ax0, ax1 = axs[0, j], axs[1, j]
        for st in chosen[b]:
            a = star_arrays(st, 36)
            rc, pd_, pm_, po_ = profile(a)
            col = colors.get(st['group'], '#762a83')
            ax0.plot(rc, pd_ / st['flux'], '-', color=col, lw=1.2)
            ax0.plot(rc, pm_ / st['flux'], '--', color=col, lw=1.2)
            ax1.plot(rc, pd_ / pm_, '-', color=col, lw=1.2)
            ax1.plot(rc, pd_ / po_, ':', color=col, lw=1.0)
            _, ns = sat_component(a)
            rsat = np.sqrt(ns / np.pi)
            ax1.plot([rsat], [1.0], marker='v', color=col, ms=6, clip_on=False, zorder=5)
        ax0.set_yscale('log'); ax0.set_xscale('log')
        ax0.set_title(f'F{b[:-1]}{b[-1]}', fontsize=11)
        ax1.axhline(1, color='k', lw=0.6)
        ax1.axvspan(3, 8, color='0.85', zorder=0)
        ax0.axvspan(3, 8, color='0.85', zorder=0)
        ax1.set_ylim(0.5, 2.0); ax1.set_xlabel('radius from star centre (frame px)')
        ax1.set_xscale('log'); ax1.set_xlim(0.7, 30)
        if j == 0:
            ax0.set_ylabel('annulus median / flux_fit\n(solid: data, dashed: satstar model)')
            ax1.set_ylabel('data / model   (solid: satstar only,\ndotted: total frame model)')
        pr = samples[b][1]
        xs = np.arange(30) + 0.5
        ax1.plot(xs, np.nanmedian(pr[:, :, 0], axis=0), 'k-', lw=2.5, zorder=6)
        ax1.plot(xs, np.nanmedian(pr[:, :, 1], axis=0), 'k:', lw=2.0, zorder=6)
        # sample median ratio curve
    leg = [Line2D([], [], color=c, label=g) for g, c in colors.items()] + [
        Line2D([], [], color='k', lw=2.5, label='sample median over N isolated sat. stars (solid: satstar, dotted: total)'), Line2D([], [], color='k', ls='-', label='data'), Line2D([], [], color='k', ls='--', label='our satstar model'),
        Line2D([], [], color='k', marker='v', ls='', label='equiv. saturated radius sqrt(Nsat/pi)')]
    fig.legend(handles=leg, fontsize=8, loc='lower center', ncol=3)
    axs[1, 0].text(0.03, 0.95, 'grey band: 3-8 px', transform=axs[1, 0].transAxes, fontsize=8, va='top')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f'{OUT}/radial_profiles.png', dpi=110)
    plt.close(fig)
    # --- table
    L = ['| band | N stars | median data/model, 3-8 px (all) | rmask<1 px: N, median | rmask>=1 px: N, median | 16-84% range (all) | median data/(total frame model) |', '|---|---|---|---|---|---|---|']
    for b in ['150W', '200W', '277W', '405N', '323N']:
        sm = samples[b][0]
        ok = np.isfinite(sm[:, 1])
        rm, ra = sm[:, 0], sm[:, 1]

        def mm(k):
            k = k & ok
            if k.sum() < 3:
                return f'{k.sum()}, -'
            x = ra[k]
            return f'{k.sum()}, {np.median(x):.3f} ± {1.2533 * 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(len(x)):.3f}'
        p16, p84 = np.percentile(ra[ok], [16, 84])
        L.append(f'| F{b[:-1]}{b[-1]} | {ok.sum()} | {mm(np.ones(len(ra), bool)).split(", ")[1]} | {mm(rm < 1)} | {mm(rm >= 1)} | {p16:.3f} - {p84:.3f} | {np.nanmedian(sm[:, 5]):.3f} |')
    open(f'{OUT}/radial_ratio_table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    with open(f'{OUT}/samples.pkl', 'wb') as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    main()
