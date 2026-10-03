"""Shape of the NRCBLONG saturated-star deficit on the detector (#1013 follow-up).

    python perexp_column_map.py pxh.npz <out prefix>

From the per-exposure table of ``perexp_measure.py`` (``area`` = saturated-core
area / its star-visit mean, ``a`` = halo change), for NRCBLONG:

1. the 32-px column profile of both, against the readout-amplifier
   boundaries (x = 512, 1024, 1536);
2. a 256 x 256 px map of the core-area change over the detector;
3. the in-band (x = 256-512) change by date quartile.

Writes ``<prefix>.png`` and ``<prefix>.json``.
"""
import json
import os
import sys

import numpy as np
from astropy.time import Time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AMP_EDGES = (512, 1024, 1536)


def binmed(x, v, edges, nmin):
    k = np.digitize(x, edges) - 1
    med = np.full(len(edges) - 1, np.nan); n = np.zeros(len(edges) - 1, int)
    for i in range(len(edges) - 1):
        s = (k == i) & np.isfinite(v)
        n[i] = s.sum()
        if n[i] >= nmin:
            med[i] = np.median(v[s])
    return med, n


def main():
    d = np.load(sys.argv[1], allow_pickle=True)
    outp = sys.argv[2]
    m = np.char.lower(d['det'].astype(str)) == 'nrcblong'
    x, y, t = d['xc'][m], d['yc'][m], d['expstart'][m]
    area, a = d['area'][m] - 1, d['a'][m]
    e32 = np.arange(0, 2049, 32)
    pa, na = binmed(x, area, e32, 15)
    ph, nh = binmed(x, a, e32, 15)
    e256 = np.arange(0, 2049, 256)
    grid = np.full((8, 8), np.nan); ngrid = np.zeros((8, 8), int)
    for j in range(8):
        for i in range(8):
            s = (x >= e256[i]) & (x < e256[i + 1]) & (y >= e256[j]) & (y < e256[j + 1])
            ngrid[j, i] = s.sum()
            if s.sum() >= 8:
                grid[j, i] = np.median(area[s])
    inb, out = (x > 256) & (x < 512), (x < 150) | (x > 750)
    q = np.quantile(t, [0, .25, .5, .75, 1])
    dates = []
    for j in range(4):
        s = (t >= q[j]) & (t <= q[j + 1])
        dates.append(dict(start=Time(q[j], format='mjd').iso[:10], end=Time(q[j + 1], format='mjd').iso[:10],
                          band_area=float(np.median(area[s & inb])), n_band=int((s & inb).sum()),
                          out_area=float(np.median(area[s & out]))))
    res = dict(edges32=e32.tolist(), area_prof32=[None if not np.isfinite(v) else float(v) for v in pa],
               n_area32=na.tolist(), a_prof32=[None if not np.isfinite(v) else float(v) for v in ph],
               edges256=e256.tolist(), area_map256=[[None if not np.isfinite(v) else float(v) for v in r] for r in grid],
               n_map256=ngrid.tolist(), by_date=dates)
    json.dump(res, open(outp + '.json', 'w'), indent=1)
    for r in dates:
        print(r)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.4, 1, 0.9]))
    xc = 0.5 * (e32[1:] + e32[:-1])
    ax[0].plot(xc, pa, 'o-', ms=3, label='saturated-core area − 1')
    ax[0].plot(xc, ph, 's-', ms=3, label='halo change a (15–80 px)')
    for xa in AMP_EDGES:
        ax[0].axvline(xa, color='0.5', ls=':', lw=1)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('NRCBLONG detector x [px] (dotted: amplifier boundaries)')
    ax[0].set_ylabel('median, relative to the star-visit mean')
    ax[0].legend(fontsize=8); ax[0].set_title('32-px column profile', fontsize=9)
    im = ax[1].imshow(grid, origin='lower', extent=[0, 2048, 0, 2048], cmap='RdBu', vmin=-0.3, vmax=0.3)
    for xa in AMP_EDGES:
        ax[1].axvline(xa, color='0.3', ls=':', lw=1)
    fig.colorbar(im, ax=ax[1], label='median core area − 1')
    ax[1].set_xlabel('x [px]'); ax[1].set_ylabel('y [px]'); ax[1].set_title('256-px map', fontsize=9)
    ax[2].errorbar(range(4), [r['band_area'] for r in dates], fmt='o', label='x = 256–512')
    ax[2].errorbar(range(4), [r['out_area'] for r in dates], fmt='s', label='x < 150 or > 750')
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels([r['start'][5:] for r in dates], fontsize=8)
    ax[2].axhline(0, color='k', lw=0.5); ax[2].legend(fontsize=8)
    ax[2].set_xlabel('date-quartile start (2026)'); ax[2].set_ylabel('median core area − 1')
    ax[2].set_title('by date (10678)', fontsize=9)
    fig.suptitle('NRCBLONG F480M saturated stars (10678): where the deficit sits', fontsize=10)
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 130)))


if __name__ == '__main__':
    main()
