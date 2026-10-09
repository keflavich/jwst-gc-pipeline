"""Second-pass replace_saturated pairs in main2kfpk (0.5" radius): separation vs old-row minus satstar mag,
by what dolphot shows at the old and satstar positions; vertical line = 1.5 FWHM radius of #1121."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/lwrest_b2'
fw = {'277W': 0.091, '250M': 0.084, '300M': 0.1}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, b in zip(axs, ['277W', '250M', '300M']):
    t = Table.read(f'{D}/second_pass_{b}.ecsv'); p = t[t['pass'] == 2]
    co = np.asarray(p['dold'] < 0.08); cs = np.asarray(p['dsat'] < 0.08); sd = np.asarray(p['same_dol'])
    dm = np.asarray(p['oldmag'] - p['satmag']); sep = np.asarray(p['satsep'])
    cats = [('old row = another dolphot star (overwritten)', co & cs & ~sd, '#d62728', 'o'),
            ('old row = no dolphot star (fragment)', ~co, '#7f7f7f', '.'),
            ('other', ~(co & cs & ~sd) & co, '#1f77b4', 's')]
    for lab, m, c, mk in cats:
        ax.scatter(sep[m], dm[m], s=14 if mk != '.' else 10, c=c, marker=mk, alpha=0.75, label=f'{lab}: {m.sum()}')
    ax.axvline(1.5 * fw[b], color='k', lw=1.2, ls='--')
    ax.text(1.5 * fw[b] + 0.005, 8.3, '1.5 FWHM\n(#1121)', fontsize=8)
    ax.set_xlabel('satstar - old row separation (arcsec)'); ax.set_title(f'F{b} second-pass pairs ({len(p)})', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
axs[0].set_ylabel('old-row mag - satstar mag')
fig.tight_layout()
fig.savefig('/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst-wd2/0e557564-8b2a-47a4-a008-5add86ab6df7/scratchpad/sclip/secondpass_sep_dmag.png', dpi=110)
