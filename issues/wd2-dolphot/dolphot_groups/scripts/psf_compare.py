"""Detector-aligned stack minus PSF-model (stpsf grid PSFs on disk) both processed identically.
usage: nice -19 python psf_compare.py <band> <dx> <dy>"""
import sys, glob, json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import uniform_filter, map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
band = sys.argv[1]; EX = (float(sys.argv[2]), float(sys.argv[3])); NORTH = (0.05, 0.69)
D = '/orange/adamginsburg/jwst/wd2'; OUT = f'{D}/dolphot_benchmark/Q_integ/ghost_test'
z = np.load(f'{OUT}/stamps_{band}.npz'); Dt = z['det']; det = z['detector']; gp = z['gp']; mag = z['mag']
PX, PY = np.meshgrid(gp, gp); PS = 0.0311; RP = np.hypot(PX, PY) * PS
annp = (RP > 0.25) & (RP < 0.6); bkp = (RP > 0.9) & (RP < 1.2)
ib = np.digitize(RP, np.arange(0, RP.max() + PS / 2, PS / 2))
def process(s):
    if np.isfinite(s[annp]).mean() < 0.9 or np.isfinite(s[bkp]).mean() < 0.5: return None
    t = s - np.nanmedian(s[bkp]); f = np.nansum(t[annp])
    if not np.isfinite(f) or f <= 0: return None
    t = t / f
    prof = {k: np.nanmedian(t[ib == k]) for k in np.unique(ib) if np.isfinite(t[ib == k]).any()}
    pm = np.array([prof.get(k, np.nan) for k in ib.ravel()]).reshape(ib.shape)
    return t - pm
# detector expected pixel offset (mean over detectors)
def exp_pix(detname, p):
    f = sorted(glob.glob(f'{D}/F{band}/pipeline/jw03523*_{detname.lower()}_align_o005_crf.fits'))[0]
    h = fits.getheader(f, 'SCI'); w = WCS(h); ra0, dec0 = h['CRVAL1'], h['CRVAL2']
    x0, y0 = w.wcs_world2pix(ra0, dec0, 0)
    x1, y1 = w.wcs_world2pix(ra0 + p[0] / 3600 / np.cos(np.radians(dec0)), dec0 + p[1] / 3600, 0)
    return np.array([x1 - x0, y1 - y0])
dets = sorted(set(det))
ep = {d: exp_pix(d, EX) for d in dets}; epN = {d: exp_pix(d, NORTH) for d in dets}
pe = np.mean(list(ep.values()), axis=0); pn = np.mean(list(epN.values()), axis=0)
print('mean expected pix', pe, 'north', pn)
res_data, res_mod = [], []
for d in dets:
    pf = f'{D}/psfs/nircam_{d.lower()}_f{band.lower()}_fovp512_samp2_npsf16.fits'
    ph = np.median(fits.getdata(pf), axis=0)
    ph = uniform_filter(ph, 2); c = (ph.shape[0] - 1) / 2
    mod = map_coordinates(ph, [c + 2 * PY, c + 2 * PX], order=1)
    rm = process(mod.astype(float))
    m = np.where(det == d)[0]
    for i in m:
        r = process(Dt[i].astype(float))
        if r is not None:
            res_data.append(r); res_mod.append(rm)
res_data = np.array(res_data); res_mod = np.array(res_mod)
sd = np.nanmedian(res_data, axis=0); sm = res_mod[0] if False else np.nanmean(res_mod, axis=0)
# also compare shape: scale model residual by LSQ over annulus 0.2-0.9 excluding aperture region? use fit amplitude
def aps(img, p, r=2.5):
    return np.nansum(img[np.hypot(PX - p[0], PY - p[1]) < r])
def ring(img, p, excl=25):
    rad = np.hypot(*p); a0 = np.degrees(np.arctan2(p[1], p[0])); v = []
    for a in np.arange(0, 360, 10):
        if abs((a - a0 + 180) % 360 - 180) < excl: continue
        v.append(aps(img, (rad * np.cos(np.radians(a)), rad * np.sin(np.radians(a)))))
    return np.array(v)
out = {}
annfrac = json.load(open(f'{OUT}/stack_results_{band}.json'))['psf_annulus_fraction']
diff = sd - sm
for nm, img in [('data', sd), ('psf_model', sm), ('data_minus_model', diff)]:
    for pn_, p in [('E/W', pe), ('north', pn)]:
        v = aps(img, p); r = ring(img, p)
        out[f'{nm}@{pn_}'] = dict(sum_at=float(v), ring_mean=float(r.mean()), ring_std=float(r.std()), z=float((v - r.mean()) / r.std()),
                                  excess_frac_total=float((v - r.mean()) * annfrac))
# also model-scaled: robust amplitude of model structure in data (LSQ over 0.2-0.9")
w = (RP > 0.2) & (RP < 0.9) & np.isfinite(sd)
amp = float(np.nansum(sd[w] * sm[w]) / np.nansum(sm[w] ** 2)); out['model_amp_in_data'] = amp
out['n_stamps'] = int(len(res_data)); out['mean_exp_pix'] = pe.tolist()
# mag-bin
out['by_mag'] = {}
magg = np.array([mag[i] for d in dets for i in np.where(det == d)[0] if process(Dt[i].astype(float)) is not None])
for lo, hi in [(15.5, 16.5), (16.5, 17.25), (17.25, 18.0)]:
    m = (magg >= lo) & (magg < hi); st = np.nanmedian(res_data[m], axis=0) - sm
    v = aps(st, pe); r = ring(st, pe)
    out['by_mag'][f'{lo}-{hi}'] = dict(n=int(m.sum()), sum_at=float(v), ring_mean=float(r.mean()), ring_std=float(r.std()), z=float((v - r.mean()) / r.std()))
json.dump(out, open(f'{OUT}/psf_compare_{band}.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ext = [gp[0] - .5, gp[-1] + .5, gp[0] - .5, gp[-1] + .5]
v = np.nanpercentile(np.abs(sd[(RP > 0.2) & (RP < 1)]), 99.5)
for a, img, t in zip(ax, [sd, sm, diff], ['data det-aligned residual (all detectors)', 'stpsf model residual', 'data - model']):
    a.imshow(img, origin='lower', extent=ext, cmap='RdBu_r', vmin=-v, vmax=v); a.set_title(t)
    a.add_patch(plt.Circle(pe, 2.5, fill=False, ec='lime', lw=1.3)); a.add_patch(plt.Circle(-pe, 2.5, fill=False, ec='orange', lw=1.3))
    a.add_patch(plt.Circle(pn, 2.5, fill=False, ec='magenta', lw=1.3))
    a.set_xlim(-30, 30); a.set_ylim(-30, 30)
fig.suptitle(f'F{band}: detector-pixel frame; green=expected E/W ghost, orange=mirror, magenta=north')
fig.tight_layout(); fig.savefig(f'{OUT}/fig/ghost_psfcompare_{band}.png', dpi=100)
