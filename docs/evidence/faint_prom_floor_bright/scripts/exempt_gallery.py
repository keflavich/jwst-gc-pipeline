"""Stamp gallery of W51 sources in the prominence-floor exemption region.

Rows: random qfit <= 0.2, prominence 2-3 sources at S/N_prop >= 40, split by
the local-peak requirement: on a data_i2d peak (exempted by the branch) and
off any peak (rejected by the peak requirement).
Columns: band data i2d, production m6 residual (the source was vetted out by
the floor, so a real star is left unsubtracted there), continuum data i2d.
Title colour: continuum counterpart within 60 mas (green) or not (red).

usage: python exempt_gallery.py <field> <band> <cont> <pl_table> <out_prefix> [nrow]
writes <out_prefix>_peak.png and <out_prefix>_nopeak.png
"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from astropy.io import fits                          # noqa: E402
from astropy.table import Table                      # noqa: E402
from astropy.coordinates import SkyCoord             # noqa: E402
from astropy.nddata import Cutout2D                  # noqa: E402
from astropy.visualization import simple_norm        # noqa: E402
from astropy import wcs                              # noqa: E402

FIELD, BAND, CONT, TAB, OUT = sys.argv[1:6]
NROW = int(sys.argv[6]) if len(sys.argv) > 6 else 6
P = '/orange/adamginsburg/jwst/{f}/{B}/pipeline/jw*-o00?_t001_nircam_clear-{b}-merged_{s}.fits'


def load(B, s):
    fn = sorted(glob.glob(P.format(f=FIELD, B=B.upper(), b=B.lower(), s=s)))
    assert fn, P.format(f=FIELD, B=B.upper(), b=B.lower(), s=s)
    with fits.open(fn[0]) as h:
        return h['SCI'].data.astype(float), wcs.WCS(h['SCI'].header)


def local_peak(d, w, sc):
    x, y = w.world_to_pixel(sc)
    out = np.zeros(len(sc), bool)
    for j, (xi, yi) in enumerate(zip(x, y)):
        ix, iy = int(round(xi)), int(round(yi))
        if 3 <= ix < d.shape[1] - 3 and 3 <= iy < d.shape[0] - 3:
            box = d[iy - 3:iy + 4, ix - 3:ix + 4]
            if np.isfinite(box[2:5, 2:5]).any():
                out[j] = np.nanmax(box[2:5, 2:5]) >= np.nanmax(box)
    return out


img = {'data': load(BAND, 'data_i2d'),
       'm6 residual': load(BAND, 'resbgsub_m6_daophot_basic_mergedcat_residual_i2d'),
       f'{CONT} data': load(CONT, 'data_i2d')}
t = Table.read(TAB)
snrp = np.asarray(t['flux'] / t['flux_err'], float) * np.sqrt(np.clip(np.asarray(t['nmatch'], float), 1, None))
qf = np.asarray(t['qfit'], float)
pr = np.asarray(t['prominence'], float)
m = np.asarray(t['cont_match'], bool)
reg = (qf <= 0.2) & (pr >= 2) & (pr < 3) & (snrp >= 40)
pk = np.zeros(len(t), bool)
pk[reg] = local_peak(*img['data'], SkyCoord(t['skycoord'][reg]))
rng = np.random.default_rng(7)
groups = [('peak', 'S/N_prop >= 40, on a local data peak (exempted by the branch)', np.flatnonzero(reg & pk)),
          ('nopeak', 'S/N_prop >= 40, off any data peak (rejected by the peak requirement)',
           np.flatnonzero(reg & ~pk))]
half = 1.0  # arcsec
for gtag, glab, idx in groups:
    if idx.size == 0:
        continue
    pick = rng.choice(idx, min(2 * NROW, idx.size), replace=False)
    nr = (len(pick) + 1) // 2
    fig, axes = plt.subplots(nr, 6, figsize=(12.5, 2.25 * nr + 0.9), squeeze=False)
    for a in axes.ravel():
        a.set_axis_off()
    nm = int(m[idx].sum())
    for k, i in enumerate(pick):
        r, c0 = k // 2, 3 * (k % 2)
        sc = SkyCoord(t['skycoord'][i])
        for j, (lab, (d, w)) in enumerate(img.items()):
            ax = axes[r, c0 + j]
            ax.set_axis_on()
            ax.set_xticks([]); ax.set_yticks([])
            px = half / (wcs.utils.proj_plane_pixel_scales(w)[0] * 3600)
            try:
                cut = Cutout2D(d, sc, (2 * px, 2 * px), wcs=w, mode='partial', fill_value=np.nan)
            except (ValueError, wcs.NoConvergence):
                continue
            ref = cut.data if j != 1 else Cutout2D(img['data'][0], sc, (2 * px, 2 * px), wcs=img['data'][1],
                                                   mode='partial', fill_value=np.nan).data
            # scale: floor = 10th percentile of the stamp, top = peak of the
            # central 0.5" of the DATA stamp (the residual uses the data scale)
            n = ref.shape[0]
            q = max(2, int(round(0.25 / half * n / 2)))
            ctr = ref[n // 2 - q:n // 2 + q + 1, n // 2 - q:n // 2 + q + 1]
            if np.isfinite(ctr).any():
                lo, hi = np.nanpercentile(ref, 10), np.nanmax(ctr)
                ax.imshow(cut.data, origin='lower', cmap='gray_r', vmin=lo, vmax=hi)
            x, y = cut.wcs.world_to_pixel(sc)
            ax.plot([x - 6, x - 3], [y, y], color='C1', lw=1)
            ax.plot([x, x], [y + 3, y + 6], color='C1', lw=1)
            if k < 2:
                ax.set_title(lab, fontsize=8)
        axes[r, c0].set_ylabel(f'S/N {snrp[i]:.0f} prom {pr[i]:.2f}\nqfit {qf[i]:.2f}',
                               fontsize=8, color='green' if m[i] else 'red')
    fig.suptitle(f'{FIELD} {BAND}, qfit <= 0.2, prominence 2-3, {glab}:\n'
                 f'{nm}/{idx.size} have a {CONT} counterpart within 60 mas (label green = yes, red = no); '
                 f'2" stamps, {len(pick)} drawn at random', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if nr > 2 else 0.88), w_pad=0.3, h_pad=0.6)
    fig.savefig(f'{OUT}_{gtag}.png', dpi=110)
    plt.close(fig)
    print(f'{OUT}_{gtag}.png')
