"""Analyse stamps_<band>.npz: normalise, remove azimuthal median, stack, measure at expected ghost positions.
usage: nice -19 python analyze_stack.py <band> <dx> <dy>   (expected ghost offset in arcsec, East/North)"""
import sys, glob, json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

band = sys.argv[1]; EX = (float(sys.argv[2]), float(sys.argv[3]))
NORTH = (0.05, 0.69)
D = '/orange/adamginsburg/jwst/wd2'; OUT = f'{D}/dolphot_benchmark/Q_integ/ghost_test'
z = np.load(f'{OUT}/stamps_{band}.npz')
S = z['sky']; g = z['g']; mag = z['mag']; det = z['detector']; exp = z['exp']; pa = z['pa']
X, Y = np.meshgrid(g, g); R = np.hypot(X, Y)
ann = (R > 0.25) & (R < 0.6); bk = (R > 1.0) & (R < 1.2)

def prep(S, R, ann, bk, rbin):
    """normalise by annulus flux, subtract bkg and azimuthal median. returns residual stamps, normalised stamps, flux."""
    out = np.full(S.shape, np.nan, np.float32); nrm = out.copy(); F = np.full(len(S), np.nan)
    edges = np.arange(0, R.max() + rbin, rbin); ib = np.digitize(R, edges)
    for i, s in enumerate(S):
        if np.isfinite(s[ann]).mean() < 0.9 or np.isfinite(s[bk]).mean() < 0.5: continue
        b = np.nanmedian(s[bk]); t = s - b
        f = np.nansum(t[ann])
        if not np.isfinite(f) or f <= 0: continue
        t = t / f; F[i] = f
        prof = np.array([np.nanmedian(t[ib == k]) if np.isfinite(t[ib == k]).any() else np.nan for k in range(len(edges) + 1)])
        out[i] = t - prof[ib]; nrm[i] = t
    return out, nrm, F
res, nrm, F = prep(S, R, ann, bk, 0.03)
good = np.isfinite(F)
print(band, 'stamps', len(S), 'used', good.sum(), 'stars', len(np.unique(z['row'][good])))
# PSF annulus fraction (nrca1 model, oversample 2, 0.0311"/px detector)
ph = fits.open(f'{D}/psfs/nircam_nrca1_f{band.lower()}_fovp512_samp2_npsf16.fits')[0].data[0]
ps = 0.0311 / 2; c = (ph.shape[0] - 1) / 2
yy, xx = np.indices(ph.shape); rr = np.hypot(xx - c, yy - c) * ps
annfrac = ph[(rr > 0.25) & (rr < 0.6)].sum() / ph.sum()
print('PSF annulus(0.25-0.6) fraction of total', annfrac)

def aper(img, p, r=0.08):
    m = np.hypot(X - p[0], Y - p[1]) < r
    return np.nanmean(img[m]), np.nansum(img[m])
def ringstats(img, p0, r=0.08, excl=25, step=10):
    rad = np.hypot(*p0); a0 = np.degrees(np.arctan2(p0[1], p0[0]))
    vals = []
    for a in np.arange(0, 360, step):
        d = (a - a0 + 180) % 360 - 180
        if abs(d) < excl: continue
        p = (rad * np.cos(np.radians(a)), rad * np.sin(np.radians(a)))
        vals.append(aper(img, p, r)[1])
    return np.array(vals)
def measure(stack, p0):
    v = aper(stack, p0)[1]; ring = ringstats(stack, p0)
    mir = aper(stack, (-p0[0], -p0[1]))[1]
    return dict(sum_at=v, ring_mean=ring.mean(), ring_std=ring.std(), z=(v - ring.mean()) / ring.std(), mirror=mir,
                excess_frac_total=(v - ring.mean()) * annfrac, n_ring=len(ring))
results = {}
stack_med = np.nanmedian(res[good], axis=0); stack_mean = np.nanmean(res[good], axis=0)
for name, p in [('E/W expected', EX), ('north', NORTH)]:
    results[name] = {'median': measure(stack_med, p), 'mean': measure(stack_mean, p)}
# by parent magnitude
bins = [15.5, 16.5, 17.25, 18.0]
results['by_mag'] = {}
for lo, hi in zip(bins[:-1], bins[1:]):
    m = good & (mag >= lo) & (mag < hi)
    st = np.nanmedian(res[m], axis=0)
    results['by_mag'][f'{lo}-{hi}'] = dict(n=int(m.sum()), **measure(st, EX))
# by detector and by exposure
results['by_detector'] = {}
for d in sorted(set(det)):
    m = good & (det == d)
    results['by_detector'][d] = dict(n=int(m.sum()), **measure(np.nanmedian(res[m], axis=0), EX))
results['by_exposure'] = {}
for e in sorted(set(exp)):
    m = good & (exp == e)
    results['by_exposure'][e] = dict(n=int(m.sum()), **measure(np.nanmedian(res[m], axis=0), EX))
results['PA_V3'] = sorted(set(np.round(pa.astype(float), 2).tolist()))
results['psf_annulus_fraction'] = annfrac
# peak search: max of smoothed median residual in r 0.3-0.9
from scipy.ndimage import uniform_filter
sm = uniform_filter(np.nan_to_num(stack_med), 3)
w = (R > 0.3) & (R < 0.9)
k = np.argmax(np.where(w, sm, -9)); results['peak_in_0.3-0.9'] = (float(X.flat[k]), float(Y.flat[k]), float(sm.flat[k]))
kmin = np.argmin(np.where(w, sm, 9)); results['trough_in_0.3-0.9'] = (float(X.flat[kmin]), float(Y.flat[kmin]), float(sm.flat[kmin]))
json.dump(results, open(f'{OUT}/stack_results_{band}.json', 'w'), indent=1, default=float)
np.save(f'{OUT}/stack_med_{band}.npy', stack_med)

# ---- detector-aligned stacks (per detector) and PSF model
zz = z; gp = z['gp']; Dt = z['det']
PX, PY = np.meshgrid(gp, gp); RP = np.hypot(PX, PY) * 0.0311
# expected detector pixel offset per detector from a frame WCS
def exp_pix(detname, p):
    f = sorted(glob.glob(f'{D}/F{band}/pipeline/jw03523*_{detname.lower()}_align_o005_crf.fits'))[0]
    w = WCS(fits.getheader(f, 'SCI'))
    ra0, dec0 = 266.5, -57.7
    h = fits.getheader(f, 'SCI'); ra0, dec0 = h['CRVAL1'], h['CRVAL2']
    x0, y0 = w.wcs_world2pix(ra0, dec0, 0)
    x1, y1 = w.wcs_world2pix(ra0 + p[0] / 3600 / np.cos(np.radians(dec0)), dec0 + p[1] / 3600, 0)
    return float(x1 - x0), float(y1 - y0)
ep = {d: exp_pix(d, EX) for d in sorted(set(det))}
print('expected detector-pixel offsets', ep)
resd = np.full(Dt.shape, np.nan, np.float32)
annp = (RP > 0.25) & (RP < 0.6); bkp = (RP > 0.9) & (RP < 1.2)
resd, nrmd, Fd = prep(Dt, RP, annp, bkp, 0.0311 / 2)
gd = np.isfinite(Fd)
detres = {}
for d in sorted(set(det)):
    m = gd & (det == d)
    st = np.nanmedian(resd[m], axis=0)
    p = ep[d]
    mm = np.hypot(PX - p[0], PY - p[1]) < 2.5
    # ring at same pixel radius
    rad = np.hypot(*p); a0 = np.degrees(np.arctan2(p[1], p[0])); vals = []
    for a in np.arange(0, 360, 10):
        if abs((a - a0 + 180) % 360 - 180) < 25: continue
        q = (rad * np.cos(np.radians(a)), rad * np.sin(np.radians(a)))
        vals.append(np.nansum(st[np.hypot(PX - q[0], PY - q[1]) < 2.5]))
    vals = np.array(vals); v = np.nansum(st[mm])
    detres[d] = dict(n=int(m.sum()), exp_pix=p, sum_at=float(v), ring_mean=float(vals.mean()), ring_std=float(vals.std()), z=float((v - vals.mean()) / vals.std()))
json.dump(detres, open(f'{OUT}/stack_detaligned_{band}.json', 'w'), indent=1)
print(json.dumps(results, indent=1, default=float)[:6000]); print(json.dumps(detres, indent=1))

# ---- figure
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
ext = [g[0] - 0.015, g[-1] + 0.015, g[0] - 0.015, g[-1] + 0.015]
mstack = np.nanmedian(nrm[good], axis=0)
a = ax[0, 0]; im = a.imshow(np.arcsinh(mstack / 1e-3), origin='lower', extent=ext, cmap='gray_r')
a.set_title(f'F{band}: median normalised stack (sky-aligned), N={good.sum()} stamps'); 
for a_, tit in [(ax[0, 1], 'residual after azimuthal-median subtraction (median stack)'), (ax[0, 2], 'residual, mean stack')]:
    st = stack_med if 'median' in tit else stack_mean
    v = np.nanpercentile(np.abs(st[(R > 0.2) & (R < 1.0)]), 99.5)
    a_.imshow(st, origin='lower', extent=ext, cmap='RdBu_r', vmin=-v, vmax=v); a_.set_title(tit, fontsize=9)
for a_ in [ax[0, 0], ax[0, 1], ax[0, 2]]:
    for p, col, lab in [(EX, 'lime', 'E/W expected'), ((-EX[0], -EX[1]), 'orange', 'mirror'), (NORTH, 'magenta', 'north')]:
        a_.add_patch(plt.Circle(p, 0.1, fill=False, ec=col, lw=1.3))
    a_.set_xlim(1.1, -1.1); a_.set_ylim(-1.1, 1.1); a_.set_xlabel('dRA* (arcsec, East left)'); a_.set_ylabel('dDec')
# azimuthal profile at r = |EX| and |NORTH|
a = ax[1, 0]
for p, col, lab in [(EX, 'k', f'r={np.hypot(*EX):.2f}"'), (NORTH, 'm', f'r={np.hypot(*NORTH):.2f}"')]:
    rad = np.hypot(*p); ang = np.arange(0, 360, 5)
    vals = [aper(stack_med, (rad * np.cos(np.radians(t)), rad * np.sin(np.radians(t))))[1] for t in ang]
    a.plot(ang, vals, col + '-', label=lab)
    a.axvline(np.degrees(np.arctan2(p[1], p[0])) % 360, color=col, ls=':')
a.set_xlabel('position angle from +East (deg, toward North)'); a.set_ylabel('aperture sum r<0.08" (annulus-normalised)')
a.set_title('azimuthal profile of residual stack'); a.legend()
# by-mag
a = ax[1, 1]
xs = []; ys = []; es = []
for k_, v in results['by_mag'].items():
    xs.append(np.mean([float(t) for t in k_.split('-')])); ys.append(v['sum_at'] - v['ring_mean']); es.append(v['ring_std'])
a.errorbar(xs, ys, es, fmt='o'); a.axhline(0, c='k', lw=0.5); a.set_xlabel('parent mag'); a.set_ylabel('ghost excess (annulus-norm. flux)')
a.set_title('excess at expected position vs parent mag')
# detector aligned example
d0 = 'NRCA1' if 'NRCA1' in detres else sorted(detres)[0]
a = ax[1, 2]
m = gd & (det == d0); st = np.nanmedian(resd[m], axis=0)
v = np.nanpercentile(np.abs(st[(RP > 0.2) & (RP < 1.0)]), 99.5)
a.imshow(st, origin='lower', extent=[gp[0] - .5, gp[-1] + .5, gp[0] - .5, gp[-1] + .5], cmap='RdBu_r', vmin=-v, vmax=v)
p = ep[d0]; a.add_patch(plt.Circle(p, 2.5, fill=False, ec='lime')); a.add_patch(plt.Circle((-p[0], -p[1]), 2.5, fill=False, ec='orange'))
a.set_title(f'{d0} detector-aligned residual (pixels), N={m.sum()}', fontsize=9)
fig.suptitle(f'F{band} saturated-star stack; green circle = dolphot-unmatched offset {EX}, orange = point mirror, magenta = north')
fig.tight_layout(); fig.savefig(f'{OUT}/fig/ghost_stack_{band}.png', dpi=110)
