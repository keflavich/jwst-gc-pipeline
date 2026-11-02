"""Figure: are the companions the m7 companion cut leaves out real?

Reads companion_conf.json (scripts/companion_conf.py) and draws, for the
restored own-band companions in Brick 2221/o001 F182M, binned by separation
from the nearest brighter seed source (FWHM) and flux ratio to it:

* left: flux-matched confirmation rate in Brick 1182/o004 F200W m7 vetted
  (what the original cut's evidence compared against);
* middle: the same against o004's m6 vetted catalog;
* third: the amplitude of the m=2 position-angle harmonic around the
  brighter neighbour, |<exp(2i PA)>| (isotropic positions: ~sqrt(pi/4n),
  below 0.02 in every bin with n > 2000);
* right: position-angle histograms at 1.5-2 FWHM per flux-ratio bin
  (companion_pa.npz from companion_conf.py).

usage: python companion_fig.py companion_conf.json companion_pa.npz out.png
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SEPB = [(0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
RATB = [(0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1.0001)]


def grid(rows, key):
    g = np.full((len(RATB), len(SEPB)), np.nan)
    n = np.zeros_like(g, dtype=int)
    for r in rows:
        if r['group'] != 'restored' or tuple(r['ratio']) not in RATB:
            continue
        i, j = RATB.index(tuple(r['ratio'])), SEPB.index(tuple(r['sep']))
        n[i, j] = r['n']
        if key == 'pa':
            g[i, j] = r['pa_m2'] if r['n'] >= 20 else np.nan
        elif key in r and r[key]['rel'] is not None:
            g[i, j] = r[key]['rel']
    return g, n


def main(inp, npz, out):
    rows = json.load(open(inp))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    panels = [('o004_m7', 'confirmed by o004 m7 vetted\n(the original evidence)', 'rel', (0, 1)),
              ('o004_m6', 'confirmed by o004 m6 vetted', 'rel', (0, 1)),
              ('pa', 'PA anisotropy |<exp(2i PA)>|', 'amplitude', (0, 0.3))]
    for ax, (key, title, unit, lim) in zip(axes[:3], panels):
        g, n = grid(rows, key)
        im = ax.imshow(g, origin='lower', cmap='viridis' if key != 'pa' else 'magma',
                       vmin=lim[0], vmax=lim[1], aspect='auto')
        for i in range(len(RATB)):
            for j in range(len(SEPB)):
                if n[i, j] >= 20 and np.isfinite(g[i, j]):
                    txt = f'{g[i, j]:.2f}'
                    ax.text(j, i, f'{txt}\nn={n[i, j]}', ha='center', va='center', fontsize=8,
                            color='w' if g[i, j] < 0.6 * lim[1] else 'k')
        ax.set_xticks(range(len(SEPB)), [f'{a:g}-{b:g}' for a, b in SEPB])
        ax.set_yticks(range(len(RATB)), ['<0.03', '0.03-0.1', '0.1-0.3', '0.3-1'])
        ax.set_xlabel('separation from brighter seed source (PSF FWHM)')
        ax.set_ylabel('flux ratio to it')
        ax.axhline(1.5, color='r', lw=1.5, ls='--')
        ax.axvline(3.5, color='r', lw=1.5, ls=':')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label=unit, shrink=0.85)
    d = np.load(npz)
    ax = axes[3]
    sel = d['rest'] & (d['sep_b'] >= 1.5) & (d['sep_b'] < 2.0)
    bins = np.linspace(-np.pi, np.pi, 37)
    for (lo, hi), c in zip(RATB[1:], ('C3', 'C1', 'C0')):
        k = sel & (d['ratio'] >= lo) & (d['ratio'] < hi)
        pa = np.angle(np.exp(1j * d['pa'][k]))
        h, _ = np.histogram(pa, bins)
        ax.step(np.degrees(bins[:-1]), h / h.mean(), where='post', color=c,
                label=f'ratio {lo:g}-{min(hi, 1):g} (n={k.sum()})')
    ax.axhline(1, color='k', lw=0.8)
    ax.set_xlabel('PA of companion around brighter source (deg E of N)')
    ax.set_ylabel('count / mean')
    ax.set_title('restored companions at 1.5-2 FWHM', fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle('Brick 2221/o001 F182M own-band m6 sources left out of production m7, '
                 'within 3 FWHM of a brighter seed source.  Dashed: proposed ratio gate 0.1; '
                 'dotted: companion radius 2.5 FWHM', fontsize=10)
    fig.savefig(out, dpi=110)


if __name__ == '__main__':
    main(*sys.argv[1:])
