"""fig17: the same-position test with and without masking the field stars (NBRMASK),
plus one star's exposure-difference map with its neighbour mask.

    python halocal_samepos_nbr_fig.py <samepos.json> <samepos_nbr.json> <meas.npz> <patfile> <out.png>
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import halocal_samepos as hs

a = json.load(open(sys.argv[1]))['NRCBLONG']['test1']
b = json.load(open(sys.argv[2]))['NRCBLONG']['test1']
xs = [np.sqrt(max(s0, 1)*s1) for s0, s1 in hs.SEPB]
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
for p, (part, lo, hi) in enumerate([('halo', 50, 80), ('halo', 80, 124), ('spike', 80, 124)]):
    for res, lab, st in [(a, 'all pixels', 'o--'), (b, 'field stars masked', 'o-')]:
        y = [res.get(f'diffvisit_noanchor_{part}_{lo}-{hi}_{s0}-{s1}', {}).get('ratio', np.nan) for s0, s1 in hs.SEPB]
        c = [res.get(f'diffvisit_noanchor_{part}_{lo}-{hi}_{s0}-{s1}', {}).get('ceil', np.nan) for s0, s1 in hs.SEPB]
        ax[p].plot(xs, y, st, lw=2, label=f'{lab} (noise ceiling ~{np.nanmedian(c):.2f})')
    ax[p].set_xscale('log'); ax[p].set_ylim(-0.05, 0.7); ax[p].axhline(0, color='k', lw=.5)
    ax[p].set_xticks(xs); ax[p].set_xticklabels([f'{s0}-{s1}' for s0, s1 in hs.SEPB])
    ax[p].set_xlabel("distance between the two stars' detector positions [px]")
    ax[p].set_ylabel('corr(D_A, D_B) / noise ceiling')
    ax[p].set_title(f'r = {lo}-{hi} px, {"between spikes" if part == "halo" else "on spikes"}')
    ax[p].legend(fontsize=9)
# example map
hs.NBRMASK = 5
z = np.load(sys.argv[3]); pz = np.load(sys.argv[4])
st = hs.load_star(sys.argv[4], z, 1.0, np.zeros((3, len(hs.RBINS)-1, len(hs.RHOB)-1, 4)))
nm = hs.neighbour_mask(pz['Q'].astype(float), int(pz['N']))
D = st['m'][2]-st['m'][1]
I = st['I']/np.nanmean(st['a'])
f = np.where((hs.RR >= 12) & (I > 0), D/I, np.nan)
sm = ndimage.gaussian_filter(np.nan_to_num(f), 1.5)/np.maximum(ndimage.gaussian_filter(np.isfinite(f)*1., 1.5), 1e-3)
sm[~np.isfinite(f)] = np.nan
vm = np.nanpercentile(np.abs(sm), 97)
ax[3].imshow(sm, origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm, extent=[-hs.RC-.5, hs.RC+.5]*2)
ax[3].contour(np.arange(-hs.RC, hs.RC+1), np.arange(-hs.RC, hs.RC+1), nm, levels=[0.5], colors='k', linewidths=0.5)
ax[3].set_title(f'{sys.argv[4].split("/")[-1][9:-4]}: exposure difference, field stars outlined\n(5 px masks, {100*nm[hs.RR >= 12].mean():.0f}% of pixels)')
ax[3].set_xticks([]); ax[3].set_yticks([])
fig.suptitle('NRCB5: are the outer-halo differences between stars at the same position caused by crowding?', fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[5], dpi=80)
