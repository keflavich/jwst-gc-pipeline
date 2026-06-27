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


def plot_pm_l_vs_extinction(pm_path, gns_path, out_png, sig_thresh=3.0, max_err=2.0,
                            ehk_intrinsic=0.10, aks_per_ehk=1.328, match_radius=0.3):
    """pm in Galactic longitude (mu_l*cosb) vs near-IR extinction A_Ks.

    A_Ks from GNS (H-Ks) color: A_Ks = aks_per_ehk * ((H-Ks) - ehk_intrinsic).
    In the GC, low-extinction foreground disk stars and high-extinction bulge /
    nuclear-disk stars occupy different mu_l -- this plot separates them.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    t = Table.read(pm_path)
    fin = np.isfinite(t['pm_ra']) & np.isfinite(t['pm_dec']) & np.isfinite(t['pm_ra_err'])
    sig, err = pm_significance(t)
    good = fin & (sig > sig_thresh) & (err < max_err) & (np.abs(t['pm_tot']) < 200)

    # ICRS pm -> Galactic (mu_l*cosb, mu_b)
    c = SkyCoord(ra=np.asarray(t['ra0'])[good] * u.deg, dec=np.asarray(t['dec0'])[good] * u.deg,
                 pm_ra_cosdec=np.asarray(t['pm_ra'])[good] * u.mas / u.yr,
                 pm_dec=np.asarray(t['pm_dec'])[good] * u.mas / u.yr, frame='icrs')
    g = c.galactic
    pm_l = g.pm_l_cosb.to(u.mas / u.yr).value
    pm_b = g.pm_b.to(u.mas / u.yr).value

    # extinction from GNS H-Ks
    gns = Table.read(gns_path)
    gsc = SkyCoord(np.asarray(gns['RAJ2000'], float) * u.deg, np.asarray(gns['DEJ2000'], float) * u.deg)
    idx, sep, _ = c.match_to_catalog_sky(gsc)
    H = np.asarray(gns['Hmag'], float)[idx]; K = np.asarray(gns['Ksmag'], float)[idx]
    hk = H - K
    aks = aks_per_ehk * (hk - ehk_intrinsic)
    m = (sep < match_radius * u.arcsec) & np.isfinite(aks) & np.isfinite(pm_l)
    aks, pm_l, pm_b = aks[m], pm_l[m], pm_b[m]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), sharey=False)
    # scatter: pm_l vs A_Ks
    ax1.scatter(aks, pm_l, s=4, c='k', alpha=0.25, rasterized=True)
    # running median + 16/84
    bins = np.linspace(np.nanpercentile(aks, 1), np.nanpercentile(aks, 99), 16)
    bc = 0.5 * (bins[1:] + bins[:-1])
    med = np.array([np.nanmedian(pm_l[(aks >= bins[i]) & (aks < bins[i + 1])]) for i in range(len(bc))])
    lo = np.array([np.nanpercentile(pm_l[(aks >= bins[i]) & (aks < bins[i + 1])], 16) if ((aks >= bins[i]) & (aks < bins[i + 1])).sum() > 5 else np.nan for i in range(len(bc))])
    hi = np.array([np.nanpercentile(pm_l[(aks >= bins[i]) & (aks < bins[i + 1])], 84) if ((aks >= bins[i]) & (aks < bins[i + 1])).sum() > 5 else np.nan for i in range(len(bc))])
    ax1.plot(bc, med, 'r-', lw=2, label='median')
    ax1.fill_between(bc, lo, hi, color='r', alpha=0.2, label='16-84%')
    ax1.axhline(0, color='b', lw=0.5, ls='--')
    ax1.set_xlabel(r'$A_{Ks}$ (mag)'); ax1.set_ylabel(r'$\mu_\ell\cos b$ (mas/yr)')
    ax1.set_ylim(np.nanpercentile(pm_l, [1, 99]) * 1.3)
    ax1.set_title('PM in Galactic longitude vs extinction')
    ax1.legend(fontsize=9)
    # 2D density
    hb = ax2.hexbin(aks, pm_l, gridsize=45, bins='log', cmap='inferno',
                    extent=(np.nanpercentile(aks, 0.5), np.nanpercentile(aks, 99.5),
                            *np.nanpercentile(pm_l, [1, 99])))
    ax2.axhline(0, color='c', lw=0.6, ls='--')
    ax2.set_xlabel(r'$A_{Ks}$ (mag)'); ax2.set_ylabel(r'$\mu_\ell\cos b$ (mas/yr)')
    ax2.set_title(f'density ({m.sum()} stars, S/N>{sig_thresh:g})')
    fig.colorbar(hb, ax=ax2, label='log N')
    fig.suptitle(f'{pm_path.split("/")[-1]}: $\\mu_\\ell$ vs $A_{{Ks}}$', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return dict(n=int(m.sum()), aks_range=[float(np.nanmin(aks)), float(np.nanmax(aks))],
                pm_l_med=float(np.nanmedian(pm_l)))
