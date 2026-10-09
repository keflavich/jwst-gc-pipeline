import glob, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
band = '250M'
cuts = {}
for f in sorted(glob.glob(f'cut_{band}_*.pkl')):
    cuts.update({(f, k): v for k, v in pickle.load(open(f, 'rb')).items()})
s2 = vstack([Table.read(f) for f in sorted(glob.glob(f's2_{band}_*.fits'))])
st = Table.read(f'star_{band}.fits')
groups = [[1014, 1019, 1016, 519], [1018, 195, 298, 234]]
titles = ['largest |dm|', 'typical']
for gi, grp in enumerate(groups):
    fig, axs = plt.subplots(len(grp), 3, figsize=(11, 3.4 * len(grp)))
    for ri, istar in enumerate(grp):
        r = [x for x in s2 if x['istar'] == istar][0]
        c = cuts[(f"cut_{band}_{r['pixfile']}.pkl", f"{istar}_{r['row']}")]
        s = st[st['istar'] == istar][0]
        cx, cy = r['x_fit'], r['y_fit']
        h = 20
        x0, x1, y0, y1 = int(cx) - h, int(cx) + h + 1, int(cy) - h, int(cy) + h + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        sl = (slice(y0, y1), slice(x0, x1))
        d = c['data'][sl]; m = c['model'][sl]
        ext = (x0 - .5, x1 - .5, y0 - .5, y1 - .5)
        pos = d[d > 0]
        vmax = np.nanpercentile(pos, 99.9); vmin = max(np.nanpercentile(pos, 5), vmax * 1e-4)
        from matplotlib.colors import LogNorm
        ax = axs[ri, 0]
        ax.imshow(np.where(d > 0, d, vmin), origin='lower', extent=ext, norm=LogNorm(vmin, vmax), cmap='gray_r')
        ax.contour(np.arange(x0, x0 + d.shape[1]), np.arange(y0, y0 + d.shape[0]), c['sat'][sl].astype(float), levels=[0.5], colors='tab:red', linewidths=1.2)
        yy, xx = np.nonzero(c['rim'][sl])
        ax.plot(xx + x0, yy + y0, 's', ms=2.5, mfc='none', mec='tab:blue', mew=0.5)
        ax.plot(r['ex'], r['ey'], 'x', c='tab:green', ms=9, mew=1.5, label='dolphot')
        ax.plot(cx, cy, '+', c='tab:orange', ms=11, mew=1.5, label='fit')
        ax.set_title(f"F{band} star {istar} ({['1st'][0]} frame {r['pixfile'][18:23]})\nref {s['ref']:.2f}  dm(final) {s['dm_final']:+.2f}  dm(H+cap) {s['dm_H+cap']:+.2f}", fontsize=8)
        if ri == 0:
            ax.legend(fontsize=6, loc='upper right')
        ax = axs[ri, 1]
        ax.imshow(np.where(m > 0, m, vmin), origin='lower', extent=ext, norm=LogNorm(vmin, vmax), cmap='gray_r')
        ax.contour(np.arange(x0, x0 + d.shape[1]), np.arange(y0, y0 + d.shape[0]), c['sat'][sl].astype(float), levels=[0.5], colors='tab:red', linewidths=1.0)
        ax.set_title(f"model (amp {r['amp']:.3g}; a_raw/a_cat {r['a_raw']/r['a_cat']:.2f})", fontsize=8)
        ax = axs[ri, 2]
        res = np.where(d > 0, d - m, np.nan)
        lim = np.nanpercentile(np.abs(res), 95)
        im = ax.imshow(res, origin='lower', extent=ext, vmin=-lim, vmax=lim, cmap='RdBu_r')
        ax.contour(np.arange(x0, x0 + d.shape[1]), np.arange(y0, y0 + d.shape[0]), c['sat'][sl].astype(float), levels=[0.5], colors='k', linewidths=1.0)
        ax.set_title(f'data - model (linear, +-{lim:.3g}; white = no data); rim px: {int(c["rim"].sum())}', fontsize=8)
        for a in axs[ri]:
            a.tick_params(labelsize=6)
    fig.suptitle(f'F250M 10-13 mag stars, {titles[gi]}: crf data (log), SATURATED outline (red), recovered rim px (blue squares)', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'cutouts_F250M_{gi + 1}.png', dpi=110)
    plt.close(fig)
print('ok')
