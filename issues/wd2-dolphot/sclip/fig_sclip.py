"""Figure for the all-clipped issue: (a-c) per-frame flux / RA / Dec of one F277W star with the
per-axis sigma_clip masks; (d) nmatch_good for nmatch=4 rows in F277W / F250M / F300M."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
ra = np.array([155.894726688, 155.894726603, 155.894726516, 155.894726218])
dec = np.array([-57.773432621, -57.773433066, -57.773432523, -57.773432539])
fl = np.array([46.67, 46.55, 45.88, 44.22])
cosd = np.cos(np.deg2rad(dec.mean()))
vals = {'flux': fl, 'RA offset (mas)': (ra - ra.mean()) * cosd * 3.6e6, 'Dec offset (mas)': (dec - dec.mean()) * 3.6e6}
masks = {'flux': sigma_clip(fl[None].astype('float32'), stdfunc='mad_std', axis=1).mask[0],
         'RA offset (mas)': sigma_clip(ra[None], stdfunc='mad_std', axis=1).mask[0],
         'Dec offset (mas)': sigma_clip(dec[None], stdfunc='mad_std', axis=1).mask[0]}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={'width_ratios': [1, 1, 1, 1.6]})
for ax, (k, v) in zip(axs[:3], vals.items()):
    m = masks[k]
    ax.plot(np.arange(4)[~m], v[~m], 'o', color='#1f77b4', ms=9, label='kept')
    ax.plot(np.arange(4)[m], v[m], 'x', color='#d62728', ms=11, mew=2.5, label='clipped')
    ax.axhline(np.median(v), color='0.6', lw=1, ls='--')
    ax.set_xticks(range(4)); ax.set_xlabel('frame'); ax.set_title(k, fontsize=10)
axs[0].legend(fontsize=8, loc='lower left')
fig.text(0.02, 0.97, 'wd2 F277W star (dolphot 19.97 mag): union of the three masks = all 4 frames -> NaN position, row dropped', fontsize=10, va='top')
cols = {'277W': '#1f77b4', '250M': '#ff7f0e', '300M': '#2ca02c'}
w = 0.27
for i, b in enumerate(['277W', '250M', '300M']):
    d = np.load('/blue/adamginsburg/adamginsburg/tmp/claude-3663/overclip_%s.npz' % b)
    nm = d['nm']; ra2 = d['ra']; de2 = d['dec']; f2 = d['fl']
    s = nm == 4
    cf = sigma_clip(f2[s], stdfunc='mad_std', axis=1).mask
    cr = sigma_clip(ra2[s], stdfunc='mad_std', axis=1).mask
    cd = sigma_clip(de2[s], stdfunc='mad_std', axis=1).mask
    ng = (np.isfinite(f2[s]) & ~(cf | cr | cd)).sum(1)
    h = np.bincount(ng, minlength=5) / s.sum()
    axs[3].bar(np.arange(5) + (i - 1) * w, h, width=w - 0.02, color=cols[b], label=f'F{b} ({s.sum()} rows)')
    for x, y in zip(np.arange(5), h):
        if x == 0:
            axs[3].text(x + (i - 1) * w, y + 0.01, f'{int(round(y * s.sum()))}', ha='center', fontsize=7)
axs[3].set_xlabel('nmatch_good (frames kept by the union clip)'); axs[3].set_ylabel('fraction of nmatch=4 rows')
axs[3].set_title('nmatch=4 rows, wd2 m7 (main2kfpk)', fontsize=10); axs[3].legend(fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig('/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst-wd2/0e557564-8b2a-47a4-a008-5add86ab6df7/scratchpad/sclip/sclip_allclipped.png', dpi=110)
