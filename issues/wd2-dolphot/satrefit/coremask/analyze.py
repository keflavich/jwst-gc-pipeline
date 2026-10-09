"""Tables and figures for the core-mask experiment.  Reads unsat_*.pkl and footprint.pkl; writes tables.md, fig_*.png, results.pkl"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = ['150W', '200W', '250M', '300M']
PB = [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 9)]
PBL = ['<0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '>0.9']
CIRC = ['c1.0', 'c1.5', 'c2.0', 'c3.0', 'c4.0', 'c5.0', 'c6.0', 'c8.0', 'c10.0']
THR = ['t0.500', 't0.200', 't0.100', 't0.050', 't0.020', 't0.010', 't0.005', 't0.002']
MASKS = CIRC + THR
rng = np.random.default_rng(5)
L = []


def boot(x, n=300):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    med = np.median(x)
    bs = np.median(rng.choice(x, (n, len(x))), axis=1)
    return med, bs.std(), len(x)


D = {}
for b in BANDS:
    rows, stamps, metas = [], [], []
    for fn in sorted(glob.glob(f'unsat_{b}_*.pkl')):
        d = pickle.load(open(fn, 'rb'))
        off = len(rows)
        for s in d['stamps']:
            s['idx'] += off
            stamps.append(s)
        rows += d['rows']
        metas.append(d['meta'])
    A = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k not in ('frame',)}
    A['det'] = np.array([r['frame'][1] for r in rows])
    A['exp'] = np.array([r['frame'][2] for r in rows])
    # quality: positive full amplitude both modes
    A['ok'] = (A['a_full'] > 0) & (A['b_full'] > 0)
    D[b] = dict(A=A, stamps=stamps, metas=metas)

# ---- counts
L.append('## Star counts (selected, isolated, unsaturated; fit succeeded)\n')
L.append('| band | frames | N total | ' + ' | '.join(PBL) + ' |')
L.append('|---|---|---|' + '---|' * len(PBL))
for b in BANDS:
    A = D[b]['A']
    n = [int(((A['frac'] >= lo) & (A['frac'] < hi) & A['ok']).sum()) for lo, hi in PB]
    L.append(f"| F{b} | {len(D[b]['metas'])} | {int(A['ok'].sum())} | " + ' | '.join(map(str, n)) + ' |')
L.append('')
L.append('Frame saturation-level proxies (peak units MJy/sr): L = p99.999 of unflagged pixels (used); alt = p99 of pixels adjacent to SATURATED pixels.\n')
L.append('| band | L(p99.999) range over frames | alt range |')
L.append('|---|---|---|')
for b in BANDS:
    m = D[b]['metas']
    L.append(f"| F{b} | {min(x['Lsat'] for x in m):.1f}-{max(x['Lsat'] for x in m):.1f} | {min(x['Lnb'] for x in m):.1f}-{max(x['Lnb'] for x in m):.1f} |")
L.append('')

# ---- R tables
RES = {}


def Rtab(b, kind):
    A = D[b]['A']
    pre = 'a_' if kind == 'ann' else 'b_'
    R = {}
    for mk in MASKS:
        with np.errstate(invalid='ignore', divide='ignore'):
            r = A[pre + mk] / A[pre + 'full']
        r[~A['ok']] = np.nan
        R[mk] = r
    return R


for kind, kl in (('ann', 'annulus background (subtracted, fixed)'), ('free', 'free constant background (joint fit)')):
    L.append(f'## R = a_mask / a_full, {kl}; median (bootstrap SE) [N] per peak/saturation bin\n')
    for b in BANDS:
        A = D[b]['A']
        R = Rtab(b, kind)
        L.append(f'### F{b}\n')
        L.append('| mask | median masked px | ' + ' | '.join(PBL) + ' |')
        L.append('|---|---|' + '---|' * len(PBL))
        for mk in MASKS:
            cells = []
            for (lo, hi), lab in zip(PB, PBL):
                s = (A['frac'] >= lo) & (A['frac'] < hi)
                med, se, n = boot(R[mk][s])
                RES[(b, kind, mk, lab)] = (med, se, n)
                cells.append(f'{med:.3f} ({se:.3f}) [{n}]' if np.isfinite(med) else 'n/a')
            L.append(f"| {mk} | {np.median(A['n_' + mk]):.0f} | " + ' | '.join(cells) + ' |')
        L.append('')

# ---- R vs dolphot magnitude
L.append('## R (annulus bg) by dolphot magnitude bin, for masks c2.0 and t0.10 and c4.0; median (SE) [N]\n')
for b in BANDS:
    A = D[b]['A']
    R = Rtab(b, 'ann')
    mg = A['mag']
    edges = np.arange(np.floor(np.nanpercentile(mg, 1)), np.ceil(np.nanpercentile(mg, 99)) + 0.5, 1.0)
    L.append(f'### F{b}\n')
    L.append('| mag bin | N | median peak/sat | c1.0 | c2.0 | c4.0 | t0.10 | t0.02 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (mg >= lo) & (mg < hi) & A['ok']
        if s.sum() < 5:
            continue
        cells = []
        for mk in ('c1.0', 'c2.0', 'c4.0', 't0.100', 't0.020'):
            med, se, n = boot(R[mk][s])
            cells.append(f'{med:.3f} ({se:.3f})')
        L.append(f'| {lo:.0f}-{hi:.0f} | {int(s.sum())} | {np.median(A["frac"][s]):.2f} | ' + ' | '.join(cells) + ' |')
    L.append('')

# ---- dm of full fit
L.append('## dm of the full-PSF fit vs dolphot as a function of peak/saturation\n')
L.append('dm = -2.5 log10(a_full) - MAG(dolphot) - ZP; ZP = median over stars with 0.05 <= peak/sat < 0.3, separately per (band, detector). Bin medians with bootstrap SE [N]. Annulus-bg fit; free-bg fit in second block.\n')
DMR = {}
for kind, pre in (('ann', 'a_'), ('free', 'b_')):
    L.append(f'### {kind}\n')
    L.append('| band | ' + ' | '.join(PBL) + ' |')
    L.append('|---|' + '---|' * len(PBL))
    for b in BANDS:
        A = D[b]['A']
        with np.errstate(invalid='ignore', divide='ignore'):
            inst = -2.5 * np.log10(A[pre + 'full']) - A['mag']
        zp = np.zeros(len(inst))
        for dt in np.unique(A['det']):
            sd = (A['det'] == dt)
            zs = sd & (A['frac'] >= .05) & (A['frac'] < .3) & A['ok']
            zp[sd] = np.nanmedian(inst[zs])
        dm = inst - zp
        dm[~A['ok']] = np.nan
        DMR[(b, kind)] = dm
        cells = []
        for (lo, hi) in PB:
            s = (A['frac'] >= lo) & (A['frac'] < hi)
            med, se, n = boot(dm[s])
            cells.append(f'{med:+.3f} ({se:.3f}) [{n}]')
        L.append(f'| F{b} | ' + ' | '.join(cells) + ' |')
    L.append('')

# ---- stamp core ratio
L.append('## Core ratio of bright unsaturated stars (stamps with peak/sat >= 0.5)\n')
L.append('central-pixel (data - bkg) / (model - bkg) at the full-fit amplitude, and mean residual (data-model)/(model peak - bkg) in radius bins; median over stars.\n')
L.append('| band | peak/sat bin | N | centre data/model | r<1 | 1-2 | 2-3 | 3-5 | 5-8 |')
L.append('|---|---|---|---|---|---|---|---|---|')
CORE = {}
for b in BANDS:
    st = D[b]['stamps']
    for (lo, hi), lab in list(zip(PB, PBL))[3:]:
        ss = [s for s in st if lo <= s['frac'] < hi]
        if len(ss) < 3:
            continue
        cr, prof = [], []
        edges = [0, 1, 2, 3, 5, 8]
        for s in ss:
            d, m = s['data'], s['model']
            yy, xx = np.indices(d.shape)
            rr = np.hypot(xx - s['xf'], yy - s['yf'])
            bk = np.median(m[rr > 10])
            pk = m.max() - bk
            ic = np.unravel_index(np.argmax(m), m.shape)
            cr.append((d[ic] - bk) / pk)
            prof.append([np.mean((d - m)[(rr >= a) & (rr < c)]) / pk for a, c in zip(edges[:-1], edges[1:])])
        prof = np.array(prof)
        CORE[(b, lab)] = (np.median(cr), np.median(prof, axis=0), len(ss))
        L.append(f'| F{b} | {lab} | {len(ss)} | {np.median(cr):.3f} | ' + ' | '.join(f'{x:+.4f}' for x in np.median(prof, axis=0)) + ' |')
L.append('')

# ---- figures
cols = plt.cm.viridis(np.linspace(0, 0.95, len(PB)))
# (a) R vs mask size
for fam, names, xl in (('circ', CIRC, 'circular mask radius (px)'), ('thr', THR, 'threshold f (mask: unit PSF > f * peak)')):
    fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for ci, b in enumerate(BANDS):
        A = D[b]['A']
        for ri, kind in enumerate(('ann', 'free')):
            ax = axs[ri, ci]
            for (lo, hi), lab, c in zip(PB, PBL, cols):
                y, e = [], []
                for mk in names:
                    med, se, n = RES[(b, kind, mk, lab)]
                    y.append(med); e.append(se)
                x = np.arange(len(names))
                ax.errorbar(x, y, e, color=c, marker='o', ms=3, label=lab)
            ax.axhline(1, color='k', lw=0.5)
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels([n[1:] for n in names])
            ax.set_ylim(0.8, 1.2) if kind == 'ann' else ax.set_ylim(0.5, 1.5)
            ax.set_title(f'F{b} {kind} bg')
            if ci == 0:
                ax.set_ylabel('R = a_mask / a_full')
            if ri == 1:
                ax.set_xlabel(xl)
    axs[0, 0].legend(title='peak/sat', fontsize=7)
    fig.tight_layout()
    fig.savefig(f'fig_a_R_vs_mask_{fam}.png', dpi=110)
    plt.close(fig)

# (b) R vs peak/sat
fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for ci, b in enumerate(BANDS):
    A = D[b]['A']
    for ri, kind in enumerate(('ann', 'free')):
        R = Rtab(b, kind)
        ax = axs[ri, ci]
        be = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.05, 3])
        for mk, c in (('t0.100', 'C0'), ('c2.0', 'C3')):
            xc, y, e = [], [], []
            for lo, hi in zip(be[:-1], be[1:]):
                s = (A['frac'] >= lo) & (A['frac'] < hi)
                med, se, n = boot(R[mk][s])
                if n >= 5:
                    xc.append(np.median(A['frac'][s])); y.append(med); e.append(se)
            ax.errorbar(xc, y, e, color=c, marker='o', ms=3, label={'t0.100': 'f=0.1 threshold', 'c2.0': 'r<2 px'}[mk])
        ax.axhline(1, color='k', lw=0.5)
        ax.set_ylim(0.9, 1.1) if kind == 'ann' else ax.set_ylim(0.6, 1.4)
        ax.set_title(f'F{b} {kind} bg')
        ax.set_xlabel('peak / saturation proxy')
        if ci == 0:
            ax.set_ylabel('R = a_mask / a_full')
axs[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig('fig_b_R_vs_peak.png', dpi=110)
plt.close(fig)

# (b2) dm of full fit vs peak
fig, axs = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
for ci, b in enumerate(BANDS):
    A = D[b]['A']
    ax = axs[ci]
    be = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.05, 3])
    for kind, c in (('ann', 'C0'), ('free', 'C1')):
        dm = DMR[(b, kind)]
        xc, y, e = [], [], []
        for lo, hi in zip(be[:-1], be[1:]):
            s = (A['frac'] >= lo) & (A['frac'] < hi)
            med, se, n = boot(dm[s])
            if n >= 5:
                xc.append(np.median(A['frac'][s])); y.append(med); e.append(se)
        ax.errorbar(xc, y, e, color=c, marker='o', ms=3, label=kind + ' bg')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-0.15, 0.15)
    ax.set_title(f'F{b}')
    ax.set_xlabel('peak / saturation proxy')
axs[0].set_ylabel('dm full fit (relative to ZP at 0.05-0.3)')
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig('fig_b2_dm_vs_peak.png', dpi=110)
plt.close(fig)

# (c) cutouts
fig, axs = plt.subplots(len(BANDS) * 3, 6, figsize=(14, 3 * 12 * 0.55 + 2))
for bi, b in enumerate(BANDS):
    st = [s for s in D[b]['stamps'] if 0.7 <= s['frac'] <= 0.95]
    st = sorted(st, key=lambda s: -s['frac'])
    if len(st) > 6:
        st = [st[i] for i in np.linspace(0, len(st) - 1, 6).astype(int)]
    for k in range(6):
        for ri in range(3):
            ax = axs[bi * 3 + ri, k]
            ax.set_xticks([]); ax.set_yticks([])
            if k >= len(st):
                ax.axis('off')
                continue
            s = st[k]
            d, m = s['data'], s['model']
            bk = np.median(m[np.hypot(*(np.indices(m.shape) - np.array([[[s['yf']]], [[s['xf']]]]))) > 10])
            pk = m.max() - bk
            if ri == 0:
                ax.imshow(np.arcsinh((d - bk) / pk * 20), origin='lower', cmap='gray')
                ax.set_title(f"F{b} m={s['mag']:.2f} f={s['frac']:.2f}", fontsize=7)
            elif ri == 1:
                ax.imshow(np.arcsinh((m - bk) / pk * 20), origin='lower', cmap='gray')
            else:
                ax.imshow((d - m) / pk, origin='lower', cmap='RdBu_r', vmin=-0.05, vmax=0.05)
                th = np.hypot(*(np.indices(m.shape) - np.array([[[s['yf']]], [[s['xf']]]])))
                ax.contour(th, levels=[2.0], colors='k', linewidths=0.6)
            if k == 0:
                ax.set_ylabel(['data', 'model', 'resid/peak'][ri], fontsize=7)
fig.tight_layout()
fig.savefig('fig_c_cutouts.png', dpi=100)
plt.close(fig)

# (d) radial residual profile
fig, axs = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
for ci, b in enumerate(BANDS):
    ax = axs[ci]
    for (lo, hi), lab, c in list(zip(PB, PBL, cols))[3:]:
        ss = [s for s in D[b]['stamps'] if lo <= s['frac'] < hi]
        if len(ss) < 3:
            continue
        eds = np.arange(0, 11, 1.0)
        prof = []
        for s in ss:
            d, m = s['data'], s['model']
            yy, xx = np.indices(d.shape)
            rr = np.hypot(xx - s['xf'], yy - s['yf'])
            bk = np.median(m[rr > 10])
            pk = m.max() - bk
            prof.append([np.mean((d - m)[(rr >= a) & (rr < a + 1)]) / pk for a in eds[:-1]])
        ax.plot(eds[:-1] + 0.5, np.median(prof, axis=0), color=c, marker='o', ms=3, label=f'{lab} (N={len(ss)})')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'F{b}')
    ax.set_xlabel('r (px)')
    ax.legend(fontsize=7)
axs[0].set_ylabel('median (data - model) / model peak')
fig.tight_layout()
fig.savefig('fig_d_resid_profile.png', dpi=110)
plt.close(fig)

# ---- satstar-regime prediction
fp = pickle.load(open('footprint.pkl', 'rb'))
L.append('## Satstar footprints and predicted wing-fit bias\n')
L.append('Footprint columns: median sat_area (DQ SATURATED pixels of the star), mask buffer, estimated masked area pi(sqrt(A_sat/pi)+buf)^2, median nfit (pixels used in the 81x81 fit) and 6561-nfit, nrim_fit (rim pixels recovered into the fit). '
         'Predicted bias = -2.5 log10(R) with R interpolated in log masked-pixel count from the measured masks of one family (circular or threshold), for the stated peak bin. Two areas are used: A_deficit = 6561-nfit (pixels the production fit actually leaves out, including any neighbour masks, so an upper bound for the star own core mask) and A_est = pi(sqrt(sat_area/pi)+buf)^2 (the mask if the saturated footprint were masked and no rim recovered). Negative = wing fit reads bright.\n')
PRED = {}
obs = {'150W': {(14, 15): -0.040, (15, 16): -0.041, (16, 17): -0.027}}


def predict(b, kind, lab, fam, area):
    A = D[b]['A']
    xs = np.array([np.median(A['n_' + mk]) for mk in fam])
    ys = np.array([RES[(b, kind, mk, lab)][0] for mk in fam])
    ok = np.isfinite(ys)
    if ok.sum() < 2:
        return None
    xl = np.log(np.maximum(xs[ok], 1.0))
    xq = np.log(max(area, 1.0))
    val = np.interp(xq, xl, ys[ok])
    return -2.5 * np.log10(val), bool((xq > xl.max()) or (xq < xl.min()))


fmt = lambda t: 'n/a' if t is None else f'{t[0]:+.3f}' + ('*' if t[1] else '')
for kind in ('ann', 'free'):
    L.append(f'### {kind} bg\n')
    L.append('| band | mag bin | N sat | sat_area | A_est (sat+buf) | A_deficit (6561-nfit) | nrim_fit | bias at A_deficit: circ / thr (>0.9) | bias at A_est: circ / thr (>0.9) | bias at A_est: circ / thr (0.7-0.9) | observed (F150W only) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for b in BANDS:
        for (lo, hi), f in fp[b].items():
            c = []
            for lab, area in (('>0.9', f['deficit']), ('>0.9', f['area_est']), ('0.7-0.9', f['area_est'])):
                c.append(fmt(predict(b, kind, lab, CIRC, area)) + ' / ' + fmt(predict(b, kind, lab, THR, area)))
                PRED[(b, kind, lo, hi, lab, round(area))] = (predict(b, kind, lab, CIRC, area), predict(b, kind, lab, THR, area))
            ob = obs.get(b, {}).get((lo, hi), '')
            L.append(f"| F{b} | {lo}-{hi} | {f['n']} | {f['sat_area']:.0f} | {f['area_est']:.0f} | {f['deficit']:.0f} | {f['nrim']:.0f} | " + ' | '.join(c) + f' | {ob} |')
    L.append('')
L.append('`*` marks a footprint outside the range of masked-pixel counts covered by the 5 masks of that family (nearest value used).\n')
open('tables.md', 'w').write('\n'.join(L))
pickle.dump(dict(RES=RES, PRED=PRED, CORE=CORE), open('results.pkl', 'wb'))
print('done')
