"""Per-detector median offsets against r = 5 px AREA-corrected apertures (det_split.txt).

Left: our PSF photometry with the grid fix applied (ours - (pred - predT)) minus the aperture.
Right: dolphot with its area double count removed (dolphot + pred) minus the aperture.
Each value is relative to the band median; points are SW bands, x is the detector.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix'
rows = [ln.split('|')[1:-1] for ln in open(f'{D}/det_split.txt') if ln.startswith('| F')]
rows = [[c.strip() for c in r] for r in rows]
dets = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
bands = sorted({r[0] for r in rows}, key=lambda b: int(b[1:4]))
colors = ['#2a6fdb', '#e8590c', '#2b8a3e', '#c2255c', '#5f3dc4', '#868e96']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
for ax, col, title in ((axes[0], 3, 'ours (grid fix) - aperture'), (axes[1], 4, 'dolphot + pred - aperture')):
    for k, b in enumerate(bands):
        sel = [r for r in rows if r[0] == b]
        x = np.array([dets.index(r[1]) for r in sel]) + (k - (len(bands) - 1) / 2) * 0.1
        y = np.array([float(r[col]) for r in sel])
        ax.plot(x, y, 'o', ms=5, color=colors[k % len(colors)], label=b)
    allv = np.array([float(r[col]) for r in rows])
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xticks(range(len(dets)), dets, fontsize=8)
    ax.set_title(f'{title}\nrms over detectors and bands {np.std(allv):.4f} mag', fontsize=9)
    ax.grid(axis='y', color='0.9')
axes[0].set_ylabel('median offset rel. band median [mag]')
axes[1].legend(fontsize=7, ncol=2, loc='upper left')
fig.savefig(f'{D}/det_split.png', dpi=110)
print('wrote det_split.png')
