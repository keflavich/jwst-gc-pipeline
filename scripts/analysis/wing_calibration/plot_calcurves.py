"""Figure for the masked-core flux-bias calibration product.

Top: bias vs equivalent mask radius, all filters, one axis (log-x).
Bottom: per-filter small multiples, bias vs N masked brightest pixels,
with the circular-mask flavor, the null machinery control, and the
independent masked-core star-fit experiment points where available.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table

OUT = ('/blue/adamginsburg/adamginsburg/tmp/claude-3663/'
       '-blue-adamginsburg-adamginsburg-repos-jwst-gc-pipeline/'
       'd046d21c-902e-4ed6-b230-4b20ac93f54c/scratchpad/agent_calcurves')

# validated categorical palette (light mode), fixed wavelength order
FILTERS = ['f090w', 'f150w', 'f182m', 'f187n', 'f212n', 'f405n', 'f410m',
           'f466n']
PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948',
           '#e87ba4', '#eb6834']
FIELD = {'f090w': 'M92', 'f150w': 'M92'}

# independent masked-core star-fit experiment medians (circular mask,
# PSFPhotometry, per-star fits): {filter: {r_mask: bias}}
STARFIT = {
    'f090w': {2: 1.000, 3: 1.032, 4: 1.095, 6: 1.159, 8: 0.938, 10: 0.390,
              12: 0.243, 16: 0.130, 20: 0.057},
    'f150w': {2: 0.943, 3: 0.875, 4: 0.840, 6: 1.033, 8: 0.798, 10: 0.459,
              12: 0.358, 16: 0.194, 20: 0.132},
    'f182m': {3: 1.02, 4: 1.11, 8: 1.19, 12: 1.26},
    'f187n': {3: 1.00, 8: 0.94},
}

INK = '#333330'
MUTED = '#71716c'
GRID = '#e5e5e0'

plt.rcParams.update({
    'font.size': 9.5, 'text.color': INK, 'axes.edgecolor': GRID,
    'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.titlecolor': INK, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

tabs = {f: Table.read(f'{OUT}/calcurve_{f}.ecsv') for f in FILTERS}

fig = plt.figure(figsize=(12.5, 10.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.55, 1, 1], hspace=0.42,
                      wspace=0.28, left=0.065, right=0.975, top=0.92,
                      bottom=0.07)

# ------------------------------------------------------ top: all filters
ax = fig.add_subplot(gs[0, :])
YLO, YHI = -0.12, 1.45
ends = []
for f, c in zip(FILTERS, PALETTE):
    t = tabs[f][tabs[f]['N_masked_px'] > 0]
    ax.plot(t['r_mask_equiv'], t['bias_model_to_data'], color=c, lw=1.8,
            marker='o', ms=3.2, mec='white', mew=0.5, zorder=3)
    lbl = f.upper() + (' (M92)' if f in FIELD else '')
    ends.append([float(t['r_mask_equiv'][-1]),
                 float(np.clip(t['bias_model_to_data'][-1], YLO + 0.03,
                               YHI - 0.03)), lbl, c])
# spread colliding end labels vertically (min 0.065 separation)
ends.sort(key=lambda e: e[1])
for j in range(1, len(ends)):
    if ends[j][1] - ends[j - 1][1] < 0.065:
        ends[j][1] = ends[j - 1][1] + 0.065
for x, y, lbl, c in ends:
    ax.annotate(lbl, (x, y), xytext=(7, 0), textcoords='offset points',
                color=c, fontsize=9, fontweight='bold', va='center')
ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.set_xscale('log')
ax.set_xlim(1.1, 42)
ax.set_ylim(YLO, YHI)
ax.set_xticks([1.5, 2, 3, 4, 6, 8, 12, 20, 25])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('equivalent mask radius  $r_{mask} = \\sqrt{N/\\pi}$  [px]')
ax.set_ylabel('fitted flux / unmasked fitted flux')
ax.set_title('Self-fit flux bias of the STPSF model on the stacked empirical '
             'PSF, inner $N$ brightest px masked\n(uniform-weight LSQ, '
             'fit region $r \\leq 50$ px; 1.0 = unbiased)', fontsize=11,
             loc='left')
ax.grid(True, color=GRID, lw=0.6, zorder=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# --------------------------------------------- small multiples: vs N
first = True
for i, (f, c) in enumerate(zip(FILTERS, PALETTE)):
    axs = fig.add_subplot(gs[1 + i // 4, i % 4])
    t = tabs[f][tabs[f]['N_masked_px'] > 0]
    N = np.asarray(t['N_masked_px'], float)
    axs.plot(N, t['bias_circmask'], color=c, lw=1.2, alpha=0.45,
             ls=(0, (5, 2)), label='circular mask, same area')
    axs.plot(N, t['bias_model_to_data'], color=c, lw=1.8, marker='o',
             ms=2.8, mec='white', mew=0.4, label='N brightest px masked')
    if np.isfinite(t['null_machinery']).any():
        axs.plot(N, t['null_machinery'], color=MUTED, lw=1.0, ls=':',
                 label='null control (model+noise)')
    if f in STARFIT:
        rs = np.array(sorted(STARFIT[f]))
        axs.plot(np.pi * rs ** 2, [STARFIT[f][r] for r in rs], ls='none',
                 marker='D', ms=4.5, mfc='white', mec=INK, mew=0.9,
                 label='star-fit experiment (circ.)')
    axs.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    axs.set_xscale('log')
    axs.set_xlim(4, 2600)
    axs.set_ylim(-0.05, 1.45)
    axs.set_title(f.upper() + (' (M92)' if f in FIELD else ''), fontsize=10,
                  color=c, fontweight='bold')
    axs.grid(True, color=GRID, lw=0.5)
    for s in ('top', 'right'):
        axs.spines[s].set_visible(False)
    if i % 4 == 0:
        axs.set_ylabel('flux bias')
    if i >= 4:
        axs.set_xlabel('$N$ masked brightest px')
    # secondary top axis: equivalent radius
    sec = axs.secondary_xaxis(
        'top', functions=(lambda n: np.sqrt(n / np.pi),
                          lambda r: np.pi * r ** 2))
    sec.set_xticks([2, 4, 8, 16])
    sec.tick_params(labelsize=7.5, colors=MUTED)
    if first:
        axs.legend(fontsize=7, loc='lower left', frameon=False,
                   handlelength=2.2)
        first = False

fig.savefig(f'{OUT}/calcurves_bias.png', dpi=170)
print(f'wrote {OUT}/calcurves_bias.png')
