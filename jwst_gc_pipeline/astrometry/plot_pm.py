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


# A_filt / A_Ks for the GC NIR extinction law (A_lambda ~ lambda^-2; lambda_Ks=2.15um)
_AK_COEFF = {'f150w': 2.054, 'f200w': 1.167, 'f277w': 0.607, 'f212n': 1.043, 'f323n': 0.448}


def _aks_from_jwst(c, m7_path, blue, red, color_intrinsic, match_radius):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    m7 = Table.read(m7_path)
    filt = [cn[len('mag_ab_'):] for cn in m7.colnames
            if cn.startswith('mag_ab_') and cn[len('mag_ab_'):] in _AK_COEFF]
    filt = sorted(filt, key=lambda f: -_AK_COEFF[f])
    blue = blue or filt[0]; red = red or filt[-1]
    msc = SkyCoord(m7['skycoord_ref'])
    idx, sep, _ = c.match_to_catalog_sky(msc)
    mb = np.asarray(m7[f'mag_ab_{blue}'], float)[idx]
    mr = np.asarray(m7[f'mag_ab_{red}'], float)[idx]
    color = mb - mr
    # bright PM stars saturate in JWST -> reject saturated / bad photometry
    base = (sep < match_radius * u.arcsec) & np.isfinite(color)
    base &= (mb > 13) & (mb < 24) & (mr > 13) & (mr < 24)
    for f in (blue, red):
        sat = f'is_saturated_{f}'
        if sat in m7.colnames:
            base &= ~np.asarray(m7[sat], bool)[idx]
    clo, chi = np.nanpercentile(color[base], [2, 98]); base &= (color > clo) & (color < chi)
    ci = float(np.nanpercentile(color[base], 5)) if color_intrinsic is None else color_intrinsic
    aks = (color - ci) / (_AK_COEFF[blue] - _AK_COEFF[red])
    return aks, base, f'JWST {blue.upper()}-{red.upper()}'


def _aks_from_gns(c, gns_path, match_radius, ehk_intrinsic=0.10, aks_per_ehk=1.328):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    gns = Table.read(gns_path)
    gsc = SkyCoord(np.asarray(gns['RAJ2000'], float) * u.deg, np.asarray(gns['DEJ2000'], float) * u.deg)
    idx, sep, _ = c.match_to_catalog_sky(gsc)
    hk = np.asarray(gns['Hmag'], float)[idx] - np.asarray(gns['Ksmag'], float)[idx]
    aks = aks_per_ehk * (hk - ehk_intrinsic)
    base = (sep < match_radius * u.arcsec) & np.isfinite(aks)
    return aks, base, 'GNS H-Ks'


def plot_pm_l_vs_extinction(pm_path, out_png, ext_source='jwst', m7_path=None,
                            gns_path=None, max_err=2.5, blue=None, red=None,
                            color_intrinsic=None, match_radius=0.15):
    """pm in Galactic longitude (mu_l*cosb) vs near-IR extinction A_Ks.

    ext_source='jwst': A_Ks from the JWST m7 color (needs m7_path); clean,
    non-saturated photometry only.  ext_source='gns': A_Ks from GNS H-Ks
    (needs gns_path).  Only PM uncertainty < max_err kept.  In the GC, low- vs
    high-extinction (foreground vs distant) populations stream differently in mu_l.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    t = Table.read(pm_path)
    fin = np.isfinite(t['pm_ra']) & np.isfinite(t['pm_dec']) & np.isfinite(t['pm_ra_err'])
    sig, err = pm_significance(t)
    good = fin & (err < max_err) & (np.abs(t['pm_tot']) < 200)
    c = SkyCoord(ra=np.asarray(t['ra0'])[good] * u.deg, dec=np.asarray(t['dec0'])[good] * u.deg,
                 pm_ra_cosdec=np.asarray(t['pm_ra'])[good] * u.mas / u.yr,
                 pm_dec=np.asarray(t['pm_dec'])[good] * u.mas / u.yr, frame='icrs')
    pm_l = c.galactic.pm_l_cosb.to(u.mas / u.yr).value
    if ext_source == 'gns':
        aks, base, color_label = _aks_from_gns(c, gns_path, match_radius)
    else:
        aks, base, color_label = _aks_from_jwst(c, m7_path, blue, red, color_intrinsic, match_radius)
    m = base & np.isfinite(pm_l)
    aks, pm_l = aks[m], pm_l[m]

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
    ax2.set_title(f'density ({m.sum()} stars, $\\sigma_{{pm}}$<{max_err:g} mas/yr)')
    fig.colorbar(hb, ax=ax2, label='log N')
    fig.suptitle(f'{pm_path.split("/")[-1]}: $\\mu_\\ell$ vs $A_{{Ks}}$ '
                 f'($A_{{Ks}}$ from {color_label}; $\\sigma_{{pm}}$<{max_err:g} mas/yr)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return dict(n=int(m.sum()), aks_range=[float(np.nanmin(aks)), float(np.nanmax(aks))],
                pm_l_med=float(np.nanmedian(pm_l)))


def make_all_pm_plots(pm_path, m7_path, gns_path, outdir, tag):
    """Produce all PM figures for one field: VPD+spatial map, plus mu_l-vs-A_Ks
    from BOTH GNS H-Ks and the JWST color."""
    import os
    res = {}
    res['vpd'] = plot_pm_figure(pm_path, os.path.join(outdir, f'pm_{tag}_figure.png'),
                                sig_thresh=5.0, max_err=1.0, m7_path=m7_path)
    res['ext_gns'] = plot_pm_l_vs_extinction(
        pm_path, os.path.join(outdir, f'pm_{tag}_pml_vs_Aks_gns.png'),
        ext_source='gns', gns_path=gns_path, max_err=2.5)
    res['ext_jwst'] = plot_pm_l_vs_extinction(
        pm_path, os.path.join(outdir, f'pm_{tag}_pml_vs_Aks_jwst.png'),
        ext_source='jwst', m7_path=m7_path, max_err=2.5)
    return res
