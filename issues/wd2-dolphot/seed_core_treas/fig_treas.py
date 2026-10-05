"""Cutouts of gc-treasury F480M stars whose seed differs between arms (#1101).
Left: crf SCI (asinh), with NaN-VAR_POISSON pixels in the SATURATED component marked
(red squares = OUTLIER / bad-pixel bits, cyan squares = DO_NOT_USE | SATURATED only).
Markers: seed per arm (x), fit per arm (o).  usage: fig_treas.py armA armB [armC] -> fig/seed_treas_<arms>.png"""
import os, sys, warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
S = os.path.dirname(os.path.abspath(__file__))
FR = 'jw10678040001_02101_0000{}_nrcalong_destreak_o040_crf'
BAD = (16 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 262144 | 524288 | 1048576 | 2097152
       | 4194304 | 2 ** 30 | 2 ** 31)
# (ra, dec, exposure, note)
CASES = [(266.394943, -29.170064, 4, 'improved'), (266.390931, -29.178742, 6, 'improved'),
         (266.394321, -29.165140, 6, 'improved'), (266.393643, -29.184144, 2, 'worse in s1'),
         (266.370961, -29.191244, 5, 'worse in s1 (frame edge)')]
COL = {'s0': 'magenta', 's1': 'lime', 's2': 'orange'}


def row(arm, e, sc):
    t = Table.read(f'{S}/tree_{arm}/F480M/pipeline/{FR.format(e)}_rctest_satstar_catalog.fits')
    c = SkyCoord(t['skycoord_fit'])
    i = int(np.argmin(c.separation(sc)))
    r = t[i]
    return (float(r['x_init'] + r['x_0'] - r['x_fit']), float(r['y_init'] + r['y_0'] - r['y_fit']),
            float(r['x_0']), float(r['y_0']), int(r['flags']), c[i].separation(sc).to(u.arcsec).value)


def main(arms):
    fig, axes = plt.subplots(1, len(CASES), figsize=(3.4 * len(CASES), 3.9))
    for ax, (ra, dec, e, note) in zip(axes, CASES):
        sc = SkyCoord(ra * u.deg, dec * u.deg)
        with fits.open(f'{S}/tree_{arms[0]}/F480M/pipeline/{FR.format(e)}.fits') as h:
            sci, dq, vp = h['SCI'].data, h['DQ'].data.astype(np.int64), h['VAR_POISSON'].data
        R = {a: row(a, e, sc) for a in arms}
        x0, y0 = R[arms[0]][2], R[arms[0]][3]
        xc, yc = int(round(np.median([R[a][0] for a in arms]))), int(round(np.median([R[a][1] for a in arms])))
        h = 9
        ys, xs = slice(max(yc - h, 0), yc + h + 1), slice(max(xc - h, 0), xc + h + 1)
        im = sci[ys, xs]
        v = np.nanpercentile(im, [5, 99.5])
        ax.imshow(np.arcsinh((im - v[0]) / (0.02 * (v[1] - v[0]) + 1e-9)), origin='lower', cmap='gray',
                  extent=(xs.start - 0.5, xs.start + im.shape[1] - 0.5, ys.start - 0.5, ys.start + im.shape[0] - 0.5))
        nan = ~np.isfinite(vp[ys, xs]) & ((dq[ys, xs] & 2) > 0)
        for (j, i) in np.argwhere(nan):
            bad = (dq[ys, xs][j, i] & BAD) > 0
            ax.add_patch(plt.Rectangle((xs.start + i - 0.5, ys.start + j - 0.5), 1, 1, fill=False,
                                       ec='red' if bad else 'cyan', lw=1.2))
        for a in arms:
            sx, sy, fx, fy, fl, _ = R[a]
            ax.plot(sx, sy, 'x', color=COL.get(a, 'yellow'), ms=10, mew=2, label=f'{a} seed')
            ax.plot(fx, fy, 'o', mfc='none', mec=COL.get(a, 'yellow'), ms=9, mew=1.5, label=f'{a} fit fl{fl}')
        ax.set_title(f'e{e} ({xc},{yc})\n{note}', fontsize=9)
        ax.legend(fontsize=6, loc='upper right', framealpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('gc-treasury F480M nrcalong: satstar seed (x) and fit (o) per arm; '
                 'red = NaN-var px with OUTLIER/bad-pixel bits, cyan = SATURATED-only NaN-var px', fontsize=9)
    fig.tight_layout()
    os.makedirs(f'{S}/fig', exist_ok=True)
    out = f'{S}/fig/seed_treas_{"_".join(arms)}.png'
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main(sys.argv[1:] or ['s0', 's1'])
