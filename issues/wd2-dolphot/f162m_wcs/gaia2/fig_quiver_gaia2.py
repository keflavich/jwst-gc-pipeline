"""Quiver of per-star Gaia-residual vectors (mean over exposures, per-frame c0 removed) vs detector position; red = fitted linear term."""
import sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.argv = [sys.argv[0]]
import datascale_gaia2 as D
P = pickle.load(open('gaia_fit_wd2_q03s10.pkl', 'rb'))
data, G = P['data'], P['G']
pm = np.hypot(np.asarray(G['pmra'], float), np.asarray(G['pmdec'], float))
epm = np.hypot(np.asarray(G['pmra_error'], float), np.asarray(G['pmdec_error'], float))
cols = [('nrcb3', 'NRCB3'), ('nrcb4', 'NRCB4'), ('nrca2', 'NRCA2')]
bands = ['f212n', 'f164n']
fig, axs = plt.subplots(len(bands), len(cols), figsize=(15, 9), sharex=True, sharey=True)
for i, b in enumerate(bands):
    for j, (det, nm) in enumerate(cols):
        ax = axs[i, j]
        d = data[(b, det)]
        arr = (d['fr'], d['gid'], d['x'], d['y'], d['rx'], d['ry'])
        mask, s = D.clipped(arr)
        J, A, cx, cy = D.fit(*arr, np.ones(len(mask)), mask)
        nf = d['fr'].max() + 1
        # remove per-frame constant only
        rx = d['rx'] - A[:, :nf] @ cx[:nf]
        ry = d['ry'] - A[:, :nf] @ cy[:nf]
        gid = np.unique(d['gid'][mask])
        X, Y, U, V, PM, EP = [], [], [], [], [], []
        for g in gid:
            k = (d['gid'] == g) & mask
            X.append(d['x'][k].mean() + 1023.5); Y.append(d['y'][k].mean() + 1023.5)
            U.append(rx[k].mean()); V.append(ry[k].mean()); PM.append(pm[g]); EP.append(epm[g])
        X, Y, U, V, EP = map(np.array, (X, Y, U, V, EP))
        q = ax.quiver(X, Y, U, V, EP, angles='xy', scale_units='xy', scale=0.04, cmap='viridis', clim=(0, 1.5), width=0.004)
        gx, gy = np.meshgrid(np.linspace(100, 1950, 5), np.linspace(100, 1950, 5))
        ax.quiver(gx, gy, J[0, 0] * (gx - 1023.5) + J[0, 1] * (gy - 1023.5), J[1, 0] * (gx - 1023.5) + J[1, 1] * (gy - 1023.5),
                  angles='xy', scale_units='xy', scale=0.04, color='r', width=0.003, alpha=0.7)
        m = D.inv(J)[2]
        ax.set_title(f'{b.upper()} {nm}: {len(gid)} stars, rms {s:.1f} mas, m = {m:.1f}"', fontsize=10)
        ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
        if i == len(bands) - 1:
            ax.set_xlabel('detector x [pix]')
        if j == 0:
            ax.set_ylabel('detector y [pix]')
        if j == len(cols) - 1:
            fig.colorbar(q, ax=ax, label='Gaia pm error [mas/yr]', fraction=0.046)
fig.suptitle('Gaia-frame residuals (JWST - Gaia, mean over 4 exposures, per-frame constant removed); black: stars, red: fitted linear term. Arrow length: 25 pix = 1 mas. Components are (RA, Dec) on sky, positions in detector pixels.', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig('fig_gaia2_quiver.png', dpi=110)
