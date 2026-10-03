"""Figure for #1020: where the loose-roundness seeds land at full frame.

Reads the seedspike.py outputs (A_tight / B_loose / C_guard i2dseed tables)
and draws, for one field:
  row 1: position angle (mod 60 deg) about the nearest top-0.5% star for the
         seeds within 2", per population, normalised to the median bin;
         rank of the m6 smoothed background at each seed (CDF).
  row 2: 4" cutouts of the detection image (m6 residual i2d - smoothed bg)
         around the four bright stars with the most loose seeds within 2".
  row 3: 4" cutouts at the four 4" cells with the most loose seeds among
         cells in the top 10% of smoothed background.
Markers: white dots = previous (m6 vetted) seeds, cyan circles = new tight
seeds, magenta x = loose-only seeds, red squares = loose seeds the spike
guard drops.

usage: python spikefig.py <field> <filt> <module> <out.png>
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy import wcs  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.visualization import simple_norm  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

FIELD, FILT, MOD, OUT = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
R = '/orange/adamginsburg/jwst'
CFG = {
    'brick': dict(pipe=f'{R}/brick/{FILT.upper()}/pipeline',
                  stem=f'jw02221-o001_t001_nircam_clear-{FILT}-{MOD}_resbgsub_m6_daophot_basic_mergedcat_residual',
                  label='Brick NRCB F182M'),
    'sgrb2': dict(pipe=f'{R}/sgrb2/NB/{FILT.upper()}/pipeline',
                  stem=f'jw05365-o001_t001_nircam_clear-{FILT}-{MOD}_resbgsub_m6_daophot_basic_mergedcat_residual',
                  label='Sgr B2 F187N'),
}[FIELD]
det = f"{CFG['pipe']}/{CFG['stem']}_i2d.fits"
bgp = f"{CFG['pipe']}/{CFG['stem']}_smoothed_bg_i2d.fits"
run = f'{HERE}/run_{FIELD}_{FILT}_{MOD}'
summ = json.load(open(f'{HERE}/seedspike_{FIELD}_{FILT}_{MOD}.json'))


def seedtab(name):
    fn = [f for f in os.listdir(f'{run}/{name}') if f.endswith('_i2dseed.fits')][0]
    return Table.read(f'{run}/{name}/{fn}')


A, B, Cg = seedtab('A_tight'), seedtab('B_loose'), seedtab('C_guard')
with fits.open(det) as h:
    ww = wcs.WCS(h['SCI'].header)
    img = np.asarray(h['SCI'].data, float)
with fits.open(bgp) as h:
    bgim = np.asarray(h['SCI'].data, float)
img = img - np.nan_to_num(bgim)
pixscale = float(np.sqrt(abs(np.linalg.det(ww.pixel_scale_matrix))) * 3600)


def xy(t, sel=None):
    sc = t['skycoord'] if sel is None else t['skycoord'][sel]
    x, y = ww.world_to_pixel(sc)
    return np.asarray(x), np.asarray(y)


newA = np.asarray(A['seed_origin']).astype(str) == 'i2d'
looseB = np.asarray(B['seed_round_loose'], bool)
looseC = np.asarray(Cg['seed_round_loose'], bool)
xp, yp = xy(A, ~newA)
xt, yt = xy(A, newA)
xl, yl = xy(B, looseB)
xc, yc = xy(Cg, looseC)
dc, _ = cKDTree(np.c_[xc, yc]).query(np.c_[xl, yl])
drop = dc * pixscale > 0.03
# bright stars: top 0.5% of the previous (m6 vetted) seed by flux
pf = np.asarray(A['flux'], float)[~newA]
g = np.isfinite(pf) & (pf > 0)
thr = np.percentile(pf[g], 99.5)
bx, by = xp[g & (pf >= thr)], yp[g & (pf >= thr)]

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1, 1], hspace=0.28, wspace=0.12)
ax = fig.add_subplot(gs[0, :2])
cen = np.arange(2.5, 60, 5)
for key, lab, col in (('tight_new', 'new tight seeds (±0.5)', 'tab:cyan'),
                      ('loose_no_guard', 'loose-only seeds (0.5–0.8)', 'tab:purple'),
                      ('loose_guard', 'loose-only, after spike guard', 'tab:orange'),
                      ('dropped_by_guard', 'dropped by spike guard', 'tab:red')):
    h = np.asarray(summ[key]['hist'], float)
    ax.step(cen, h / np.median(h), where='mid', color=col, lw=2,
            label=f"{lab}: {summ[key]['n_near']:,} within 2″ (max/median {summ[key]['max_over_median']:.2f})")
ax.axhline(1, color='k', lw=0.5)
ax.set_xlabel('position angle about nearest top-0.5% star, mod 60° (pixel frame)')
ax.set_ylabel('count / median bin')
ax.set_title(f"{CFG['label']}: seeds within 2″ of a bright star")
ax.legend(fontsize=8, loc='upper left')

ax = fig.add_subplot(gs[0, 2:])
fin = np.sort(bgim[np.isfinite(bgim)].ravel())


def rank(x, y):
    ix = np.clip(np.round(x).astype(int), 0, bgim.shape[1] - 1)
    iy = np.clip(np.round(y).astype(int), 0, bgim.shape[0] - 1)
    v = bgim[iy, ix]
    v = v[np.isfinite(v)]
    return np.searchsorted(fin, v) / len(fin)


br = summ['bg_rank']
for (x, y), key, lab, col in (((xp, yp), 'base', 'previous (m6 vetted) seeds', 'k'),
                              ((xt, yt), 'tight_new', 'new tight seeds', 'tab:cyan'),
                              ((xl, yl), 'loose_no_guard', 'loose-only seeds', 'tab:purple')):
    r = np.sort(rank(x, y))
    ax.plot(r, np.linspace(0, 1, len(r)), color=col, lw=2,
            label=f"{lab}: {br[key]['frac_top10']:.0%} in top 10% of bg")
ax.plot([0, 1], [0, 1], 'k:', lw=0.5)
ax.set_xlabel('rank of m6 smoothed background at the seed (all finite pixels)')
ax.set_ylabel('cumulative fraction of seeds')
ax.set_title('Background at the seed position')
ax.legend(fontsize=8, loc='upper left')

half = int(round(2.0 / pixscale))


def cutout(axc, cx, cy, title):
    cx, cy = int(round(cx)), int(round(cy))
    sub = img[cy - half:cy + half + 1, cx - half:cx + half + 1]
    fsub = sub[np.isfinite(sub)]
    # noise-scaled stretch: bright-star residuals saturate, faint peaks stay visible
    med = np.median(fsub)
    sig = 1.4826 * np.median(np.abs(fsub - med))
    norm = simple_norm(fsub, 'linear', vmin=med - 4 * sig, vmax=med + 12 * sig)
    axc.imshow(sub, origin='lower', cmap='gray_r', norm=norm,
               extent=(cx - half - 0.5, cx + half + 0.5, cy - half - 0.5, cy + half + 0.5))
    win = lambda x, y: (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)  # noqa: E731
    w = win(xp, yp)
    axc.plot(xp[w], yp[w], '.', color='w', ms=3, mec='k', mew=0.3)
    w = win(xt, yt)
    axc.plot(xt[w], yt[w], 'o', mfc='none', mec='tab:cyan', ms=7, mew=1.2)
    w = win(xl, yl)
    axc.plot(xl[w], yl[w], 'x', color='m', ms=7, mew=1.5)
    w = win(xl, yl) & drop
    axc.plot(xl[w], yl[w], 's', mfc='none', mec='r', ms=11, mew=1.2)
    nl = int(win(xl, yl).sum())
    axc.set_title(f'{title}\n{nl} loose, {int((win(xl, yl) & drop).sum())} guard-dropped', fontsize=9)
    axc.set_xlim(cx - half - 0.5, cx + half + 0.5)
    axc.set_ylim(cy - half - 0.5, cy + half + 0.5)
    axc.set_xticks([])
    axc.set_yticks([])


# row 2: bright stars with the most loose seeds within 2"
cnt = np.asarray([len(v) for v in cKDTree(np.c_[xl, yl]).query_ball_point(np.c_[bx, by], 2.0 / pixscale)])
ok = (bx > half) & (bx < img.shape[1] - half) & (by > half) & (by < img.shape[0] - half)
order = np.argsort(-np.where(ok, cnt, -1))
picked = []
for i in order:
    if all(np.hypot(bx[i] - bx[j], by[i] - by[j]) > 2 * half for j in picked):
        picked.append(i)
    if len(picked) == 4:
        break
for k, i in enumerate(picked):
    cutout(fig.add_subplot(gs[1, k]), bx[i], by[i], f'bright star ({bx[i]:.0f}, {by[i]:.0f})')

# row 3: top-10%-background 4" cells with the most loose seeds
cell = 2 * half + 1
top10 = fin[int(0.9 * len(fin))]
ny, nx = img.shape[0] // cell, img.shape[1] // cell
H, _, _ = np.histogram2d(yl, xl, bins=[ny, nx], range=[[0, ny * cell], [0, nx * cell]])
bgc = np.nanmedian(bgim[:ny * cell, :nx * cell].reshape(ny, cell, nx, cell), axis=(1, 3))
score = np.where(np.isfinite(bgc) & (bgc >= top10), H, -1)
picked = []
for flat in np.argsort(-score.ravel()):
    j, i = np.unravel_index(flat, score.shape)
    cx, cy = (i + 0.5) * cell, (j + 0.5) * cell
    if all(np.hypot(cx - px, cy - py) > 3 * cell for px, py in picked):
        picked.append((cx, cy))
    if len(picked) == 4:
        break
for k, (cx, cy) in enumerate(picked):
    cutout(fig.add_subplot(gs[2, k]), cx, cy, f'top-10% background cell ({cx:.0f}, {cy:.0f})')

c = summ['counts']
fig.suptitle(f"{CFG['label']}, m6 residual i2d − smoothed bg (detection image).  "
             f"New tight seeds {c['A_new']:,}; loose-only seeds {c['B_loose']:,}; spike guard (3″) keeps {c['C_loose']:,}.\n"
             '4″ cutouts (linear, median −4σ to +12σ): white dot = previous seed, cyan circle = new tight seed, '
             'magenta x = loose-only seed, red square = dropped by spike guard', fontsize=11)
fig.savefig(OUT, dpi=110, bbox_inches='tight')
print('wrote', OUT)
