"""Evidence figures for issues #993 (per-dither halo change) and #996 (far-field floor).

Run from the working directory that holds tgt_e*.npz, trn_*_e*.npz, q3_e*.npz and ../data:
    python make_evidence_figures.py <outdir>
"""
import sys, os, glob, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exposure import Exposure
warnings.simplefilter('ignore')
import stdatamodels.jwst.datamodels as dm

out = sys.argv[1]
RA, DEC = 266.5090306896523, -28.95658817641266
CAL = '../data/jw10678061001_02101_{i:05d}_{det}_{suf}.fits'
J1 = np.load('tgt_e1.npz')['J']


def ratio_map(Mref, Mj, H):
    d1 = Mref.data.astype(float); bad = ~np.isfinite(d1) | ((Mref.dq & 1) > 0)
    co = ndimage.spline_filter(np.where(bad, 0, d1), order=3)
    xt, yt = Mj.meta.wcs.invert(RA, DEC)
    yy, xx = np.mgrid[int(yt)-H:int(yt)+H, int(xt)-H:int(xt)+H]
    inb = (yy >= 0) & (yy < 2048) & (xx >= 0) & (xx < 2048)
    yy = yy.clip(0, 2047); xx = xx.clip(0, 2047)
    ra, dec = Mj.meta.wcs(xx.astype(float), yy.astype(float)); x1, y1 = Mref.meta.wcs.invert(ra, dec)
    v = ndimage.map_coordinates(co, [y1, x1], order=3, prefilter=False, mode='constant', cval=np.nan)
    b = ndimage.map_coordinates(bad.astype(float), [y1, x1], order=1, mode='constant', cval=1) > 0.01
    dj = Mj.data[yy, xx].astype(float)
    ok = inb & ~b & ((Mj.dq[yy, xx] & 1) == 0) & np.isfinite(dj) & np.isfinite(v)
    return np.where(ok, dj/v, np.nan)


# ---------------- fig5: LW vs SW, same dither pairs ----------------
def ratio_profile(Mref, Mj, H, pxscale):
    """median ratio in annuli of fixed ANGULAR radius (arcsec), plus a smoothed ratio map."""
    rat = ratio_map(Mref, Mj, H)
    yy_, xx_ = np.mgrid[-H:H, -H:H]; rr_ = np.hypot(xx_, yy_)*pxscale
    edges = np.array([2.0, 3.0, 4.5, 6.5, 9.0, 13.0, 18.0])
    prof = [np.nanmedian(rat[(rr_ >= lo) & (rr_ < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
    nn = [np.sum(np.isfinite(rat[(rr_ >= lo) & (rr_ < hi)])) for lo, hi in zip(edges[:-1], edges[1:])]
    g = np.isfinite(rat).astype(float)
    smap = ndimage.gaussian_filter(np.nan_to_num(rat), 4)/np.maximum(ndimage.gaussian_filter(g, 4), 1e-3)
    smap[ndimage.gaussian_filter(g, 4) < 0.3] = np.nan
    return 0.5*(edges[1:]+edges[:-1]), np.array(prof), np.array(nn), smap
LW = {i: dm.open(CAL.format(i=i, det='nrcblong', suf='cal')) for i in [3, 4, 5, 6]}
SW = {i: dm.open(CAL.format(i=i, det='nrcb1', suf='cal')) for i in [3, 4, 5, 6]}
fig = plt.figure(figsize=(26, 6.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])
ap = fig.add_subplot(gs[0, 3])
for c, j in enumerate([4, 5, 6]):
    rl, pl, nl, ml = ratio_profile(LW[3], LW[j], 300, 0.0629)
    rs, ps, ns, ms = ratio_profile(SW[3], SW[j], 600, 0.0311)
    a0 = fig.add_subplot(gs[0, c]); im = a0.imshow(ml, origin='lower', vmin=0.7, vmax=1.3, cmap='RdBu_r')
    a0.set_title(f'F480M: dither {j} / dither 3 (+-19")'); a0.set_xticks([]); a0.set_yticks([])
    ap.plot(rl, pl, 'o-', color=f'C{c}', lw=2.5, label=f'F480M (LW)  d{j}/d3')
    ap.plot(rs, ps, 's--', color=f'C{c}', lw=1, label=f'F212N (SW)  d{j}/d3')
ap.axhline(1, color='k', lw=0.5); ap.set_xscale('log'); ap.set_xlabel('distance from star [arcsec]'); ap.set_ylabel('azimuthal median ratio, same sky')
ap.set_title('F480M d5/d3: +26% within 4" of the star, falling outward.\nF212N: within +-6% at all radii, rising outward (a pedestal/background\ndifference), with no star-centred excess'); ap.legend(fontsize=9, ncol=2)
fig.colorbar(im, ax=fig.axes[1:4], location='bottom', shrink=0.5, pad=0.04, label='ratio (GWCS-resampled, 4-px smoothed)')
fig.suptitle('Same star, same pointings, same exposures: the halo changes in F480M but not in the simultaneous F212N', fontsize=14, y=1.12)
plt.savefig(f'{out}/fig5_lw_vs_sw_ratio.png', dpi=60, bbox_inches='tight'); plt.close()
print('fig5')

if len(sys.argv) > 2 and sys.argv[2] == 'fig5':
    sys.exit()
# ---------------- fig6: raw ramps per dither + integrations + persistence ----------------
M = {i: dm.open(CAL.format(i=i, det='nrcblong', suf='cal')) for i in range(1, 7)}
pos = {i: M[i].meta.wcs.invert(RA, DEC) for i in range(1, 7)}
fig, ax = plt.subplots(1, 3, figsize=(21, 5.6))
yy, xx = np.mgrid[:2048, :2048]
lev = np.zeros((6, 2, 4))
pers = []
for i in range(1, 7):
    u = fits.getdata(CAL.format(i=i, det='nrcblong', suf='uncal'), 'SCI').astype(float)
    r = np.hypot(xx-pos[i][0], yy-pos[i][1])
    for it in range(2):
        for c, (a, b) in enumerate([(45, 70), (70, 110), (110, 170), (400, 600)]):
            lev[i-1, it, c] = np.median(((u[it, -1]-u[it, 0])/(u.shape[1]-1))[(r >= a) & (r < b)])
    if i > 1:
        dqp = M[i-1].dq; dqc = M[i].dq
        px, py = pos[i-1]; cx, cy = pos[i]
        rp = np.hypot(xx-px, yy-py)
        sel = ((dqp & 2) > 0) & (rp < 90) & ((dqc & 3) == 0)
        ys, xs = np.nonzero(sel)
        xm = np.round(2*cx-xs).astype(int); ym = np.round(2*cy-ys).astype(int)
        ok = (xm >= 0) & (xm < 2048) & (ym >= 0) & (ym < 2048)
        ys, xs, xm, ym = ys[ok], xs[ok], xm[ok], ym[ok]
        ok = (dqc[ym, xm] & 3) == 0
        ys, xs, xm, ym = ys[ok], xs[ok], xm[ok], ym[ok]
        if len(ys) > 50:
            g = u[0, 1]-u[0, 0]
            pers.append((i, 100*np.median((g[ys, xs]-g[ym, xm])/g[ym, xm]),
                         {(0, 1): 'prev below', (0, -1): 'prev above', (1, 0): 'prev left', (-1, 0): 'prev right'}.get(
                             (int(np.sign(round((cx-px)/100))), int(np.sign(round((cy-py)/100)))), '')))
for c, lab in enumerate(['r 45-70 px', 'r 70-110', 'r 110-170', 'r 400-600 (scene)']):
    ax[0].plot(range(1, 7), lev[:, 0, c]/lev[0, 0, c], 'o-', label=lab)
ax[0].set_xlabel('dither (= exposure order, 4.7 min apart)'); ax[0].set_ylabel('raw _uncal (last-first group)/3, rel. to dither 1')
ax[0].set_title('Raw ramps: star wing flux jumps between exposures\n(r 400-600 mixes different scene as the annulus leaves the detector)'); ax[0].legend()
for c, (lab, (a, b)) in enumerate(zip(['r 70-110', 'r 110-170', 'r 170-250'], [(70, 110), (110, 170), (170, 250)])):
    q = []
    for i in range(1, 7):
        h = fits.open(CAL.format(i=i, det='nrcblong', suf='rateints')); sci = h['SCI'].data.astype(float); dqi = h['DQ'].data
        r = np.hypot(xx-pos[i][0], yy-pos[i][1])
        mm = (r >= a) & (r < b) & ((dqi[0] & 1) == 0) & ((dqi[1] & 1) == 0) & np.isfinite(sci[0]) & np.isfinite(sci[1])
        q.append(np.median(sci[1][mm]/sci[0][mm]))
    ax[1].plot(range(1, 7), q, 'o-', label=lab)
ax[1].set_ylim(0.9, 1.1); ax[1].axhline(1, color='k', lw=0.5)
ax[1].set_xlabel('dither'); ax[1].set_ylabel('_rateints slope: integration 2 / integration 1')
ax[1].set_title('...but not between the two integrations of an exposure'); ax[1].legend()
ax[2].bar([f'e{i}\n{t}' for i, _, t in pers], [p for _, p, _ in pers], color=['C3' if p > 0 else 'C0' for _, p, _ in pers])
ax[2].axhline(0, color='k', lw=0.5)
ax[2].set_ylabel('excess at previous dither\'s saturated core\nvs point-mirrored pixels [%]')
ax[2].set_title('Persistence test: sign follows geometry (PSF asymmetry),\nmean ~0; afterglow would be positive everywhere')
plt.tight_layout(); plt.savefig(f'{out}/fig6_raw_ramps_ints_persistence.png', dpi=75); plt.close()
print('fig6')

# ---------------- fig7: halo index for all bright stars ----------------
def load(f):
    z = np.load(f); d = z['d']; yy2, xx2 = np.mgrid[:d.shape[0], :d.shape[1]]
    return d, np.hypot(xx2-z['xt'], yy2-z['yt']), np.degrees(np.arctan2(yy2-z['yt'], xx2-z['xt'])) % 360, z
d, r, th, z = load('tgt_e1.npz')
m = (r > 70) & (r < 130) & np.isfinite(d)
bins = np.arange(0, 360.5, 0.5); idx = np.digitize(th[m], bins)
ang = np.array([np.nanmedian(d[m][idx == k]) for k in range(1, len(bins))])
sm = ndimage.maximum_filter1d(ang, 21, mode='wrap')
sp = bins[np.nonzero((ang == sm) & (ang > np.percentile(ang, 90)))[0]]+0.25


def hindex(f):
    d, r, th, z = load(f)
    dth = np.min(np.abs(((th[..., None]-sp[None, None, :])+180) % 360-180), -1)
    ok = np.isfinite(d) & ((z['dq'] & 6) == 0)
    bk = np.nanmedian(d[ok & (r > 350) & (r < 450) & (dth > 10)])
    on = np.nanmedian(d[ok & (r > 60) & (r < 120) & (dth < 1.0)])-bk
    off = np.nanmedian(d[ok & (r > 60) & (r < 120) & (dth > 8)])-bk
    return off/on
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
stars = [('tgt', 'target (obs 061)')]+[(f'trn_{o}', f'obs {o}') for o in ['116', '069', '126', '063', '078']]
for pre, lab in stars:
    fs = sorted(glob.glob(f'{pre}_e*.npz'))
    ks = [int(f.split('_e')[-1][:-4]) for f in fs]
    h = np.array([hindex(f) for f in fs])
    ax[0].plot(ks, h/np.mean(h), 'o-', lw=2.5 if pre == 'tgt' else 1, label=lab)
ax[0].axhline(1, color='k', lw=0.5)
ax[0].set_xlabel('dither index'); ax[0].set_ylabel('(between-spike halo)/(spike), normalised per star')
ax[0].set_title('Halo-to-spike ratio changes by dither for every bright NRCB5 star\n(different visits, different detector positions)')
ax[0].legend(fontsize=9)
# angular profile with spikes marked
ax[1].plot(bins[:-1]+0.25, ang, lw=0.8); [ax[1].axvline(s, color='r', lw=0.6, ls=':') for s in sp]
ax[1].set_xlabel('position angle [deg]'); ax[1].set_ylabel('median target flux, r=70-130 px [MJy/sr]')
ax[1].set_yscale('log'); ax[1].set_title('Spike angles used for the index (target, dither 1)')
plt.tight_layout(); plt.savefig(f'{out}/fig7_halo_index.png', dpi=75); plt.close()
print('fig7')

# ---------------- fig8: residual structure near the star ----------------
ex = Exposure('tgt_e1.npz', J1); zq = np.load('q3_e1.npz')
X0, Y0 = ex.X0, ex.Y0; sl = np.s_[Y0:Y0+1024, X0:X0+1024]
cal = fits.open(CAL.format(i=1, det='nrcblong', suf='cal')); rate = fits.open(CAL.format(i=1, det='nrcblong', suf='rate'))
ri = fits.open(CAL.format(i=1, det='nrcblong', suf='rateints'))
conv = (cal['SCI'].data/rate['SCI'].data)[sl]
i1 = ri['SCI'].data[0][sl]*conv; i2 = ri['SCI'].data[1][sl]*conv
v1 = (ri['VAR_POISSON'].data[0]+ri['VAR_RNOISE'].data[0])[sl]*conv**2
v2 = (ri['VAR_POISSON'].data[1]+ri['VAR_RNOISE'].data[1])[sl]*conv**2
w = np.s_[412:612, 412:612]
fig, ax = plt.subplots(1, 4, figsize=(26, 6.2))
for a, img, t in [(ax[0], (i1-zq['pred'])/np.sqrt(v1), 'integration 1: (data - LOO model)/sigma'),
                  (ax[1], (i2-zq['pred'])/np.sqrt(v2), 'integration 2 (same exposure, 96 s later)'),
                  (ax[2], (i1-i2)/np.sqrt(v1+v2), 'integration 1 - integration 2 (pure noise)')]:
    im = a.imshow(np.where(ex.good, img, np.nan)[w], origin='lower', cmap='RdBu_r', vmin=-6, vmax=6); a.set_title(t)
    a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=ax[:3], fraction=0.015, label=r'$\chi$')
chi = (ex.d-zq['pred'])/np.sqrt(ex.var+zq['varQ'])
yy2, xx2 = np.mgrid[:1024, :1024]; rr = np.hypot(xx2-ex.xt0, yy2-ex.yt0)
okk = ex.good & np.isfinite(chi) & ((ex.dq & 6) == 0)
lags = [1, 2, 3, 4, 5, 7, 10]
for lo, hi, lab in [(60, 150, 'r 60-150 px (star halo)'), (300, 500, 'r 300-500 px (scene)')]:
    C = np.where(okk & (rr >= lo) & (rr < hi), chi, np.nan); cc = []
    for L in lags:
        a_, b_ = C[:, :-L], C[:, L:]; g = np.isfinite(a_) & np.isfinite(b_)
        cc.append(np.corrcoef(np.clip(a_[g], -10, 10), np.clip(b_[g], -10, 10))[0, 1])
    ax[3].plot(lags, cc, 'o-', label=lab)
ax[3].axhline(0, color='k', lw=0.5); ax[3].set_xlabel('lag [px]'); ax[3].set_ylabel('residual autocorrelation')
ax[3].set_title('Residual is PSF-scale (lambda/D ~ 2.4 px), not white'); ax[3].legend()
plt.savefig(f'{out}/fig8_residual_structure.png', dpi=60, bbox_inches='tight'); plt.close()
print('fig8')

# ---------------- fig9: every model variant vs the inner halo ----------------
variants = [('static Q, translation only\n(unconverged CG)', 2.18, None),
            ('+ smooth star halo\n(converged)', 1.48, 3.5), ('+ cubic distortion,\nPSF-width, 2nd halos', 1.42, 3.47),
            ('+ rank-2 learned\ntemplate', 1.43, 3.44), ('+ 1/f row/col\noffsets', 1.41, 3.45),
            ('band 0.5 c/px\n(crop)', None, 3.49), ('super-resolved\n0.5 px, band 0.75', None, 3.67)]
post = [('star-centred warp\n364 par (d1, r60-100)', 5.00/5.06), ('multiplicative halo\nm<=12, 500 par (d3, r60-100)', 4.04/4.21),
        ('additive halo\nm<=12, 500 par (d3, r60-100)', 4.03/4.21), ('brighter-fatter\n(d3, r80-150)', 2.51/2.51),
        ('needed for Poisson\n(d3, r60-100)', 1/4.21)]
fig, ax = plt.subplots(1, 2, figsize=(19, 5.5))
x = np.arange(len(variants))
ax[0].bar(x-0.2, [v[1] or 0 for v in variants], 0.4, label='all held-out pixels')
ax[0].bar(x+0.2, [v[2] or 0 for v in variants], 0.4, label='r 80-120 px')
ax[0].axhline(1, color='k', ls='--'); ax[0].set_xticks(x); ax[0].set_xticklabels([v[0] for v in variants], fontsize=8)
ax[0].set_ylabel(r'robust $\sigma(\chi)$, held-out dither 1'); ax[0].set_title('Model iterations (leave-one-dither-out)'); ax[0].legend()
ax[1].bar(range(len(post)), [p[1] for p in post], color=['C2']*4+['C3']); ax[1].axhline(1, color='k', ls='--')
ax[1].set_xticks(range(len(post))); ax[1].set_xticklabels([p[0] for p in post], fontsize=8)
ax[1].set_ylabel(r'robust $\sigma(\chi)$ after / before'); ax[1].set_title('Extra per-exposure terms fitted post hoc to the held-out residual:\nat most 4% reduction; Poisson needs ~76%')
plt.tight_layout(); plt.savefig(f'{out}/fig9_model_variants.png', dpi=75); plt.close()
print('fig9')

# ---------------- fig10: far field (issue 996) ----------------
fig, ax = plt.subplots(1, 3, figsize=(21, 5.8))
pk = (zq['pred'] == ndimage.maximum_filter(np.nan_to_num(zq['pred']), 7)) & (zq['pred'] > 60)
dist = ndimage.distance_transform_edt(~pk)
far = okk & (rr > 300)
de = np.arange(0, 16)
ax[0].plot(de[:-1]+0.5, [1.4826*np.median(np.abs(chi[far & (dist >= a) & (dist < a+1)])) for a in de[:-1]], 'o-')
ax[0].axhline(1, color='k', ls='--'); ax[0].set_xlabel('distance from nearest field star (model peak > 60 MJy/sr) [px]')
ax[0].set_ylabel(r'robust $\sigma(\chi)$ (r > 300 px from target)'); ax[0].set_title('Far-field excess sits on field-star cores')
best = None
for y0 in range(0, 824, 50):
    for x0 in range(0, 824, 50):
        if np.hypot(x0+100-ex.xt0, y0+100-ex.yt0) < 350: continue
        win = np.s_[y0:y0+200, x0:x0+200]
        nsat = ((ex.dq[win] & 2) > 0).sum(); fin = okk[win].mean()
        if fin > 0.8 and (best is None or nsat < best[0]): best = (nsat, win)
s = best[1]
im = ax[1].imshow(np.where(okk, chi, np.nan)[s], origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
ax[1].contour(zq['pred'][s], levels=[80, 300], colors='k', linewidths=0.4); ax[1].set_title('held-out chi, far field, no saturated stars\n(contours: field stars, model 80 and 300 MJy/sr)')
ax[1].set_xticks([]); ax[1].set_yticks([]); fig.colorbar(im, ax=ax[1], fraction=0.046)
R = {}
for k in [1, 2, 3, 5]:
    exk = Exposure(f'tgt_e{k}.npz', J1); zk = np.load(f'q3_e{k}.npz')
    ck = (exk.d-zk['pred'])/np.sqrt(exk.var+zk['varQ'])
    y3, x3 = np.mgrid[:1024, :1024]; r3 = np.hypot(x3-exk.xt0, y3-exk.yt0)
    ok3 = exk.good & np.isfinite(ck) & ((exk.dq & 6) == 0) & (r3 > 250) & (np.abs(ck) < 8)
    full = np.full((2048, 2048), np.nan); sub = np.where(ok3, (exk.d-zk['pred'])/np.maximum(zk['pred'], 1), np.nan)
    ys, xs = np.nonzero(np.isfinite(sub)); full[ys+exk.Y0, xs+exk.X0] = sub[ys, xs]; R[k] = full
a_, b_ = R[1], R[2]; g = np.isfinite(a_) & np.isfinite(b_)
H2, xe, ye = np.histogram2d(np.clip(a_[g], -0.1, 0.1), np.clip(b_[g], -0.1, 0.1), bins=80)
ax[2].imshow(np.log10(H2.T+1), origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect='auto', cmap='viridis')
ax[2].set_xlabel('relative residual, dither 1'); ax[2].set_ylabel('relative residual, dither 2 (same detector pixel)')
ax[2].set_title(f'Detector-fixed component is weak: corr = {np.corrcoef(a_[g], b_[g])[0, 1]:+.3f}')
plt.tight_layout(); plt.savefig(f'{out}/fig10_farfield.png', dpi=75); plt.close()
print('fig10 done')
