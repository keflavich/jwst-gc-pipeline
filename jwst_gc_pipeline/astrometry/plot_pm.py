"""Diagnostic PM figures for a multi-epoch PM catalog (build_multiepoch_pm output).

Side-by-side: (left) vector-point diagram of significant PM detections
(low pm error); (right) spatial map of all stars with the PM-detected stars
overplotted.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table


def pm_significance(t):
    err = np.hypot(t['pm_ra_err'], t['pm_dec_err'])
    return np.asarray(t['pm_tot'] / err), np.asarray(err)


def plot_pm_figure(pm_path, out_png, sig_thresh=5.0, max_err=2.0, label=None,
                   m7_path=None):
    t = Table.read(pm_path)
    finite = np.isfinite(t['pm_ra']) & np.isfinite(t['pm_dec']) & np.isfinite(t['pm_ra_err'])
    sig, err = pm_significance(t)
    # guard against pathological fits leaking in as "detections"
    sane = finite & (np.abs(t['pm_tot']) < 200) & (np.abs(t['ra0'] - np.median(t['ra0'][finite])) < 0.5)
    # significant PM = high S/N AND small formal error (a real, low-error detection)
    detected = sane & (sig > sig_thresh) & (err < max_err)
    label = label or pm_path.split('/')[-1]
    # full stellar field for the spatial-map background
    bg_ra = bg_dec = None
    if m7_path:
        from astropy.coordinates import SkyCoord
        m7 = Table.read(m7_path)
        bsc = SkyCoord(m7['skycoord_ref'])
        bg_ra, bg_dec = bsc.ra.deg, bsc.dec.deg

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))

    # ---- LEFT: vector-point diagram (pmRA vs pmDec), low-pm-error detections ----
    pr, pd = np.asarray(t['pm_ra']), np.asarray(t['pm_dec'])
    axL.scatter(pr[finite], pd[finite], s=2, c='0.8', alpha=0.4, label=f'all PM ({finite.sum()})')
    sc = axL.scatter(pr[detected], pd[detected], s=10, c=err[detected],
                     cmap='viridis_r', vmin=0, vmax=max_err, alpha=0.9,
                     label=f'detected (S/N>{sig_thresh:g}, err<{max_err:g}): {detected.sum()}')
    lim = np.nanpercentile(np.abs(np.concatenate([pr[detected], pd[detected]])), 99) if detected.sum() else 30
    lim = max(lim, 10)
    axL.set_xlim(-lim, lim); axL.set_ylim(-lim, lim)
    axL.axhline(0, color='k', lw=0.5); axL.axvline(0, color='k', lw=0.5)
    axL.set_xlabel(r'$\mu_\alpha\cos\delta$ (mas/yr)'); axL.set_ylabel(r'$\mu_\delta$ (mas/yr)')
    axL.set_title(f'Vector-point diagram\n{label}')
    axL.set_aspect('equal'); axL.legend(loc='upper right', fontsize=8)
    cb = fig.colorbar(sc, ax=axL, fraction=0.046, pad=0.04); cb.set_label('PM error (mas/yr)')

    # ---- RIGHT: spatial map, full stellar field + PM-detected overplotted ----
    ra, dec = np.asarray(t['ra0']), np.asarray(t['dec0'])
    if bg_ra is not None:
        axR.scatter(bg_ra, bg_dec, s=0.5, c='0.8', alpha=0.3,
                    label=f'all m7 stars ({len(bg_ra)})', rasterized=True)
    else:
        axR.scatter(ra[finite], dec[finite], s=1, c='0.8', alpha=0.4,
                    label=f'PM stars ({finite.sum()})', rasterized=True)
    sc2 = axR.scatter(ra[detected], dec[detected], s=8, c=err[detected],
                      cmap='plasma_r', vmin=0, vmax=max_err, alpha=0.9,
                      label=f'PM-detected ({detected.sum()})', zorder=5)
    axR.invert_xaxis()
    axR.set_xlabel('RA (deg)'); axR.set_ylabel('Dec (deg)')
    axR.set_title('Spatial map: all stars + PM detections')
    axR.legend(loc='upper right', fontsize=8); axR.set_aspect('equal', adjustable='datalim')
    cb2 = fig.colorbar(sc2, ax=axR, fraction=0.046, pad=0.04); cb2.set_label('PM error (mas/yr)')

    fig.suptitle(f'Proper motions: {label}  '
                 f'({finite.sum()} PM, {detected.sum()} significant)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return dict(n_pm=int(finite.sum()), n_detected=int(detected.sum()),
                med_err=float(np.median(err[finite])))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pm_catalog')
    ap.add_argument('out_png')
    ap.add_argument('--m7', default=None, help='m7 catalog for the full-field background')
    ap.add_argument('--sig', type=float, default=5.0)
    ap.add_argument('--max-err', type=float, default=2.0)
    a = ap.parse_args()
    print(plot_pm_figure(a.pm_catalog, a.out_png, sig_thresh=a.sig,
                         max_err=a.max_err, m7_path=a.m7))
