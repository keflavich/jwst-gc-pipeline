"""fig_flat.png: the pixel flat in the ZEROFRAME rim rewrite (wd2 F200W/F150W).
(a,b) F200W nrcb1/nrcb3 R_FLAT maps with the satstar positions of exposure 1;
(c) unsaturated isolated stars: PSF-fit (R_header g0) / cal against the PSF^2-weighted flat (no flat in the rewrite);
(d,e) satstars, per-star dm - bin median against 2.5 log10(star flat), H (R_header g0) and Hf (R_header g0 / flat), SW;
(f) per-detector median satstar dm by dolphot bin, H and Hf, with the unsaturated-star detector medians."""
import sys, re, glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
from cb_lib import Band, mad
from an4 import REFV
from score7f import load, caps, BINS
import run_frames5 as R5

CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam/'
D7 = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/rstar/out7f'
C1, C3 = '#2a6fbb', '#d9622b'
fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 3)

# (a,b) flat maps
for k, det in enumerate(('nrcb1', 'nrcb3')):
    ax = fig.add_subplot(gs[0, k])
    fn = R5.path_of('200W', det, 1)
    h0 = fits.getheader(fn)
    fl = fits.getdata(CRDS + h0['R_FLAT'].split('//')[1], 'SCI').astype(float)
    im = ax.imshow(fl, origin='lower', cmap='RdBu_r', vmin=0.94, vmax=1.06, interpolation='nearest')
    t = Table.read(f"{D7}/200W_{R5.stem_of('200W', det, 1)}_satrefit7f.fits")
    t = t[t['label'] > 0]
    w = WCS(fits.getheader(fn, 'SCI'))
    x, y = w.world_to_pixel_values(np.asarray(t['ra']), np.asarray(t['dec']))
    ax.scatter(x, y, s=6, facecolor='none', edgecolor='k', lw=0.5)
    ax.set_title(f'({"ab"[k]}) F200W {det} R_FLAT ({h0["R_FLAT"].split("_")[-1][:-5]}), satstars of exp. 1', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('flat', fontsize=8)

# (c) unsaturated isolated stars
ax = fig.add_subplot(gs[0, 2])
for det, c in (('nrcb1', C1), ('nrcb3', C3)):
    R = []
    for band in ('150W', '200W'):
        for fn in sorted(glob.glob(f'rstar_{band}_{det}_*.pkl')):
            R += pickle.load(open(fn, 'rb'))['rows']
    g = lambda k: np.array([r[k] for r in R], float)
    ok = (g('a_cal') > 0) & (g('a_gh') / g('ae_gh') > 30)
    fl, rr = g('flat_w')[ok], (g('a_gh') / g('a_cal'))[ok]
    ax.scatter(fl, rr, s=2, alpha=0.25, color=c, rasterized=True)
    e = np.linspace(0.95, 1.05, 11)
    xm = [np.median(fl[(fl >= a) & (fl < b)]) for a, b in zip(e[:-1], e[1:]) if ((fl >= a) & (fl < b)).sum() > 20]
    ym = [np.median(rr[(fl >= a) & (fl < b)]) for a, b in zip(e[:-1], e[1:]) if ((fl >= a) & (fl < b)).sum() > 20]
    ax.plot(xm, ym, 'o-', color=c, ms=5, mec='w', label=f'{det}: binned median')
xx = np.linspace(0.94, 1.06, 2)
ax.plot(xx, 0.977 * xx, 'k:', lw=1, label='ratio proportional to the flat')
ax.set_xlim(0.94, 1.06); ax.set_ylim(0.9, 1.06)
ax.set_xlabel('PSF$^2$-weighted flat over the fit pixels')
ax.set_ylabel('PSF fit of R_header g0 / PSF fit of cal')
ax.set_title('(c) unsaturated isolated stars, F150W + F200W (no flat in the rewrite)', fontsize=9)
ax.legend(fontsize=7.5, loc='upper left')

# (d,e,f) satstars
res = {}
for band in ('150W', '200W'):
    B = Band(band)
    col, G = load(B)
    det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
    isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
    sfl = B.med_per_star(col('flat_rimw'))
    for nm in ('H', 'Hf'):
        c = caps(B, G, nm) * B.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        d = B.dm_of(np.minimum(col('a_' + nm), c)) - REFV[band]
        res[(band, nm)] = (d, isb3, sfl, B.ref, B.have0)
for k, nm in enumerate(('H', 'Hf')):
    ax = fig.add_subplot(gs[1, k])
    for band, mk in (('150W', 'o'), ('200W', 's')):
        d, isb3, sfl, ref, have = res[(band, nm)]
        for v, c, det in ((0, C1, 'nrcb1'), (1, C3, 'nrcb3')):
            s = have & (isb3 == v) & np.isfinite(d) & np.isfinite(sfl) & (ref < 17) & (ref >= BINS[band][0][0])
            y = d[s].copy()
            for lo, hi in BINS[band]:
                b = (ref[s] >= lo) & (ref[s] < hi)
                if b.any():
                    y[b] -= np.median(y[b])
            x = 2.5 * np.log10(sfl[s])
            ax.scatter(x, y, s=5, marker=mk, alpha=0.35, color=c, rasterized=True,
                       label=f'F{band} {det}')
    xx = np.linspace(-0.05, 0.04, 2)
    ax.plot(xx, -xx, 'k:', lw=1, label='slope -1 (flat missing)')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlim(-0.05, 0.04); ax.set_ylim(-0.12, 0.12)
    ax.set_xlabel('2.5 log$_{10}$(PSF$^2$-weighted flat at the star rim)')
    ax.set_ylabel('satstar dm - bin median (mag)')
    ax.set_title(f'({"de"[k]}) satstars < 17 mag, rim = R_header g0' + (' / flat' if nm == 'Hf' else ''), fontsize=9)
    ax.legend(fontsize=7, loc='upper right', ncol=2, markerscale=2)
# (f)
ax = fig.add_subplot(gs[1, 2])
# unsaturated main2 stars on one detector only (unsat_det.txt): F150W pooled 18-21 mag, F200W pooled 16-21 mag
unsat = {('150W', 'nrcb1'): +0.0007, ('150W', 'nrcb3'): -0.0119, ('200W', 'nrcb1'): +0.0007, ('200W', 'nrcb3'): -0.0247}
xt = []
for j, band in enumerate(('150W', '200W')):
    for i, (v, det, c) in enumerate(((0, 'nrcb1', C1), (1, 'nrcb3', C3))):
        for m, (nm, mk) in enumerate((('H', 'o'), ('Hf', 'D'))):
            d, isb3, sfl, ref, have = res[(band, nm)]
            s = have & (isb3 == v) & np.isfinite(d) & (ref >= BINS[band][0][0]) & (ref < 17)
            x0 = j * 3 + i + (m - 0.5) * 0.3
            ax.errorbar(x0, np.median(d[s]), 1.2533 * mad(d[s]) / np.sqrt(s.sum()), fmt=mk, color=c,
                        mfc=c if nm == 'H' else 'w', ms=7, label=f'{det}, rim {nm}' if j == 0 else None)
        ax.plot([j * 3 + i - 0.35, j * 3 + i + 0.35], [unsat[(band, det)] - REFV[band]] * 2,
                color=c, lw=2.5, alpha=0.5, label=f'{det}, unsaturated stars' if j == 0 else None)
ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(['F150W', 'F200W'])
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('median dm - unsaturated reference (mag)')
ax.set_title('(f) per-detector median, satstars < 17 mag', fontsize=9)
ax.set_ylim(-0.075, 0.05)
ax.legend(fontsize=7, loc='upper center', ncol=3)
fig.tight_layout()
fig.savefig('fig_flat.png', dpi=100)
print('saved')
