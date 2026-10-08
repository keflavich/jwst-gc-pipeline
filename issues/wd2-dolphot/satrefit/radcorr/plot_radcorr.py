import numpy as np, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(fn, ncol):
    out = []
    for l in open(fn):
        p = l.split()
        if len(p) == ncol and re.match(r'^\d+\.\d+$', p[0]):
            out.append([float(x) for x in p])
    return np.array(out)


fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
# radprof offset-corrected (F150W main2, iso, neighbours removed, precap) at r"
rp_r = np.array([0.070, 0.085, 0.101, 0.116, 0.140, 0.170, 0.217, 0.279, 0.341])
rp_sat = np.array([0.038, 0.065, 0.058, 0.095, 0.223, 0.135, 0.164, 0.127, -0.021])
rp_uns = np.array([-0.007, 0.021, 0.053, 0.056, 0.014, -0.037, 0.119, -0.010, -0.066])
rp_uns_e = np.array([0.006, 0.006, 0.007, 0.010, 0.017, 0.019, 0.025, 0.057, 0.068])
for ax, band in zip(axs, ['150W', '200W']):
    t = rows(f't_{band}_b1_1.log', 5)   # r_px r_as inframe n offline
    p = rows(f'p_{band}_nrcb1_1.log', 9)  # r_px r_as rawmed S_all dbar n Sb Sm Sf
    ax.axhline(0, color='0.6', lw=1)
    ax.plot(t[:, 1], t[:, 4], 'o-', color='k', label='v7b input: faint unsat, raw (offline)')
    ax.plot(p[:, 1], p[:, 2], 's--', color='tab:gray', ms=4, label='in-frame bright unsat, raw (median)')
    ax.plot(p[:, 1], p[:, 3], 's-', color='tab:blue', label='in-frame bright unsat, offset fit')
    if band == '150W':
        ax.errorbar(rp_r, rp_uns, rp_uns_e, fmt='^-', color='tab:green', capsize=2,
                    label='faint unsat, offset fit (radprof)')
        ax.plot(rp_r, rp_sat, 'D-', color='tab:red', label='satstars, offset fit (radprof, precap)')
    ax.set_xlim(0, 0.35)
    ax.set_ylim(-0.7, 0.35)
    ax.set_xlabel('r (arcsec)')
    ax.set_title(f'F{band} nrcb1')
axs[0].set_ylabel(r'$\Delta(r)$ = data / (flux $\times$ PSF) $-$ 1')
axs[0].legend(fontsize=7.5, loc='lower left')
fig.tight_layout()
fig.savefig('radcorr_inframe.png', dpi=110)
