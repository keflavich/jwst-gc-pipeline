"""fig15: the target's halo change is in the raw ramps from the first group on, is constant
through the ramp, and moves with the saturated core's area; the detector-wide ramp
acceleration next to saturated clusters (ramp_satedge.py) for comparison.

    python ramp_fig.py <ramp_satedge.npz> <out.png>      (run in the directory with the target _uncal files)
"""
import sys
import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = [(1022, 1075), (1019, 1247), (1406, 1247), (1794, 1245), (1796, 1073), (1409, 1075)]
FN = 'jw10678061001_02101_0000{}_nrcblong_uncal.fits'
H = 160
RB = np.arange(50, 161, 5)
PLATEAU = 59200.


def profiles(k):
    x0, y0 = P[k-1]
    with fits.open(FN.format(k)) as f:
        full = f['SCI'].data.astype(np.float32)
    # group-level bias drift: subtract each group's reference-pixel median per amplifier
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            for a in range(4):
                sl = slice(512*a, 512*(a+1))
                full[i, j, :, sl] -= np.median(np.concatenate([full[i, j, :4, sl].ravel(), full[i, j, -4:, sl].ravel()]))
    d = full[:, :, y0-H:y0+H+1, x0-H:x0+H+1]
    yy, xx = np.mgrid[-H:H+1, -H:H+1]; r = np.hypot(xx, yy)
    out = []
    for i in range(d.shape[0]):
        g = d[i]; dg = np.diff(g, axis=0)
        sat1 = (dg[0] < 100) & (g[0] > 0.8*PLATEAU-12000)
        ok = (g[3] < 0.9*PLATEAU-12000) & (dg[0] > 100)
        prof = np.array([[np.median(dg[j][ok & (r >= a) & (r < b)]) for a, b in zip(RB[:-1], RB[1:])] for j in range(3)])
        out.append((prof, [int((g[j] > 0.9*PLATEAU).sum()) for j in range(4)], int(sat1.sum())))
    return out


def main():
    satfn, outfn = sys.argv[1:3]
    rc = 0.5*(RB[1:]+RB[:-1])
    res = {k: profiles(k) for k in range(1, 7)}
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
    p1 = np.mean([res[1][i][0][0] for i in range(2)], 0)
    for k in range(1, 7):
        pr = np.mean([res[k][i][0] for i in range(2)], 0)
        ax[0].plot(rc, pr[0]/p1, color=cols[k-1], lw=2, label=f'e{k} (x,y)={P[k-1]}')
        ax[1].plot(rc, pr[2]/pr[0], color=cols[k-1], lw=2, label=f'e{k}')
    ax[0].set_ylim(0.8, 1.12); ax[0].axhline(1, color='k', lw=.5); ax[0].set_xlabel('r [px]'); ax[0].set_ylabel('first group difference (g2-g1) / dither 1')
    ax[0].set_title('(a) raw ramps: the halo change is there\nfrom the first group difference'); ax[0].legend(fontsize=8)
    ax[1].axhline(1, color='k', lw=.5); ax[1].set_ylim(0.85, 1.15); ax[1].set_xlabel('r [px]'); ax[1].set_ylabel('(g4-g3) / (g2-g1)')
    ax[1].set_title('(b) ...and stays constant through the ramp\n(reference-pixel corrected; no accumulating term)'); ax[1].legend(fontsize=8)
    # (c) saturated area vs halo
    m = (rc >= 60) & (rc < 100)
    hal = [np.mean([res[k][i][0][0][m].mean() for i in range(2)])/np.mean(p1[m]) for k in range(1, 7)]
    ns = [np.mean([res[k][i][2] for i in range(2)]) for k in range(1, 7)]
    ns = np.array(ns)/ns[0]
    for k in range(6):
        ax[2].scatter(ns[k], hal[k], color=cols[k], s=80, zorder=3); ax[2].annotate(f' e{k+1}', (ns[k], hal[k]))
    ax[2].set_xlabel('group-1 saturated area / dither 1'); ax[2].set_ylabel('halo (r=60-100 px, g2-g1) / dither 1')
    ax[2].set_title('(c) the saturated core shrinks when the halo dims:\nthe inner PSF (r~25 px) changes too')
    # (d) detector-wide ramp ratio table
    z = np.load(satfn)
    im = ax[3].imshow(z['med'], origin='lower', cmap='magma', vmin=0.9, vmax=3, aspect='auto')
    D = z['DISTB']; F = z['FILLB']
    ax[3].set_xticks(range(len(D)-1)); ax[3].set_xticklabels([f'{D[j]:.0f}-' for j in range(len(D)-1)])
    ax[3].set_yticks(range(len(F)-1)); ax[3].set_yticklabels([f'{F[i]/1e3:.0f}-{F[i+1]/1e3:.0f}k' for i in range(len(F)-1)])
    ax[3].set_xlabel('distance to nearest group-1-saturated pixel [px]'); ax[3].set_ylabel('pixel fill at group 4 [DN]')
    plt.colorbar(im, ax=ax[3], label='median (g4-g3)/(g2-g1)')
    ax[3].set_title('(d) whole detector, 6 exposures: ramps accelerate\n3-8 px from saturated clusters (charge overflow)')
    fig.suptitle('Target (obs 061, NRCB5), raw _uncal ramps, both integrations', fontsize=12)
    fig.tight_layout(); fig.savefig(outfn, dpi=80)
    print('halo', np.round(hal, 3), 'nsat', np.round(ns, 3))


if __name__ == '__main__':
    main()
