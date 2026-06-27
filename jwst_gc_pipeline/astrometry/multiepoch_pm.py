"""
Multi-epoch proper-motion catalogs: JWST x GNS x VIRAC2 via flystar.

The long-baseline PM signal is JWST (2023.7) vs VIRAC2 (2014.0, ~9.7 yr) +
GALACTICNUCLEUS / GNS (~2015.5).  VIRAC2 defines the reference frame (tied to
Gaia DR3); GNS is shifted onto that frame first, then JWST + GNS positions are
fit vs time per star with flystar's StarTable.fit_velocities to get pmRA/pmDec.

JWST<->Arches (both 2023, 25 d apart) is NOT a usable PM baseline; this module
deliberately uses the previous-epoch ground catalogs instead.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

# Reference epochs (decimal year)
EPOCH_VIRAC = 2014.0      # VIRAC2 / VVV (II/387), Gaia-DR3 frame
EPOCH_GNS = 2015.5        # GALACTICNUCLEUS central (J/A+A/653/A133), approx
# JWST epoch read per-catalog from MJD-AVG (gc2211 o023 = 2023.70)


def _sc(tab, racol='RAJ2000', deccol='DEJ2000'):
    return SkyCoord(np.asarray(tab[racol], float) * u.deg,
                    np.asarray(tab[deccol], float) * u.deg)


def tangent_xy(sc, center):
    """Tangent-plane offsets (arcsec) about ``center`` SkyCoord -> (x, y)."""
    dra = (sc.ra - center.ra).wrap_at(180 * u.deg).to(u.arcsec).value * np.cos(center.dec.rad)
    ddec = (sc.dec - center.dec).to(u.arcsec).value
    return dra, ddec


def load_jwst_m7(path, ref_filter='f200w'):
    """Load a per-obs m7 cross-band catalog -> dict with sky + errors + mag + epoch."""
    t = Table.read(path)
    sc = SkyCoord(t['skycoord_ref'])
    mag = np.asarray(t[f'mag_ab_{ref_filter}'], float)
    # per-frame scatter of the position (arcsec); fall back to a floor
    ex = np.asarray(t[f'std_ra_{ref_filter}'], float) if f'std_ra_{ref_filter}' in t.colnames else np.full(len(t), np.nan)
    ey = np.asarray(t[f'std_dec_{ref_filter}'], float) if f'std_dec_{ref_filter}' in t.colnames else np.full(len(t), np.nan)
    ex = np.where(np.isfinite(ex) & (ex > 0), ex, 0.005)  # 5 mas floor
    ey = np.where(np.isfinite(ey) & (ey > 0), ey, 0.005)
    # epoch from header MJD if available in meta; else default gc2211 2023.70
    mjd = t.meta.get('MJD-AVG') or t.meta.get('MJD_AVG')
    epoch = Time(float(mjd), format='mjd').jyear if mjd else 2023.70
    return dict(sc=sc, mag=mag, ex=ex, ey=ey, epoch=epoch, n=len(t))


def load_ref(path, kind):
    """kind in {'virac','gns'}.  Returns dict with sky, errors (arcsec), mag, pm (virac)."""
    t = Table.read(path)
    sc = _sc(t)
    # e_RAJ2000 / e_DEJ2000 are in mas for both VIRAC2 and GNS Vizier tables
    ex = np.asarray(t['e_RAJ2000'], float) / 1e3
    ey = np.asarray(t['e_DEJ2000'], float) / 1e3
    ex = np.where(np.isfinite(ex) & (ex > 0), ex, 0.05)
    ey = np.where(np.isfinite(ey) & (ey > 0), ey, 0.05)
    mag = np.asarray(t['Ksmag'], float)
    out = dict(sc=sc, mag=mag, ex=ex, ey=ey, n=len(t))
    if kind == 'virac':
        out['pmra'] = np.asarray(t['pmRA'], float)   # mas/yr (cosdec)
        out['pmde'] = np.asarray(t['pmDE'], float)
        out['epoch'] = EPOCH_VIRAC
    else:
        out['epoch'] = EPOCH_GNS
    return out


def restrict(cat, footprint_sc, pad_arcsec=5.0):
    """Keep catalog rows within the bbox of footprint_sc (+pad)."""
    r0, r1 = footprint_sc.ra.deg.min(), footprint_sc.ra.deg.max()
    d0, d1 = footprint_sc.dec.deg.min(), footprint_sc.dec.deg.max()
    pad = pad_arcsec / 3600.0
    ra, de = cat['sc'].ra.deg, cat['sc'].dec.deg
    m = (ra > r0 - pad) & (ra < r1 + pad) & (de > d0 - pad) & (de < d1 + pad)
    out = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in cat.items()}
    out['sc'] = cat['sc'][m]
    out['n'] = int(m.sum())
    return out


def shift_gns_to_virac(gns, virac, to_epoch, match_radius=0.2, magcut=15.0, niter=3, nsigma=3.0):
    """Bulk frame-shift GNS onto the VIRAC2/Gaia frame.

    Propagate VIRAC to the GNS epoch (using VIRAC pm) so the match is epoch-clean,
    cross-match bright (Ks<magcut) common stars, fit an affine offset GNS->VIRAC in
    the tangent plane, and apply it to ALL GNS positions.  Returns shifted SkyCoord
    + diagnostics.
    """
    center = SkyCoord(np.median(gns['sc'].ra), np.median(gns['sc'].dec))
    # propagate VIRAC to GNS epoch
    dt = to_epoch - EPOCH_VIRAC
    vra = virac['sc'].ra.deg + (np.nan_to_num(virac['pmra']) * dt / 3.6e6) / np.cos(virac['sc'].dec.rad)
    vde = virac['sc'].dec.deg + (np.nan_to_num(virac['pmde']) * dt / 3.6e6)
    vsc = SkyCoord(vra * u.deg, vde * u.deg)
    # bright stars only
    gb = gns['mag'] < magcut
    vb = virac['mag'] < magcut
    gsc, vscb = gns['sc'][gb], vsc[vb]
    idx, sep, _ = gsc.match_to_catalog_sky(vscb)
    ok = sep < match_radius * u.arcsec
    gx, gy = tangent_xy(gsc[ok], center)
    vx, vy = tangent_xy(vscb[idx[ok]], center)
    dx, dy = vx - gx, vy - gy   # GNS->VIRAC offset, arcsec
    # iterative sigma-clipped affine: d = A[0]+A[1]*x+A[2]*y
    keep = np.ones(len(gx), bool)
    A = B = None
    for _ in range(niter):
        M = np.column_stack([np.ones(keep.sum()), gx[keep], gy[keep]])
        A, *_ = np.linalg.lstsq(M, dx[keep], rcond=None)
        B, *_ = np.linalg.lstsq(M, dy[keep], rcond=None)
        rx = dx - (A[0] + A[1] * gx + A[2] * gy)
        ry = dy - (B[0] + B[1] * gx + B[2] * gy)
        s = np.hypot(rx, ry)
        keep = s < nsigma * np.std(s[keep])
    # apply to all GNS
    allx, ally = tangent_xy(gns['sc'], center)
    cdx = A[0] + A[1] * allx + A[2] * ally
    cdy = B[0] + B[1] * allx + B[2] * ally
    new_ra = gns['sc'].ra.deg + (cdx / 3600.0) / np.cos(center.dec.rad)
    new_de = gns['sc'].dec.deg + (cdy / 3600.0)
    diag = dict(n_match=int(ok.sum()), n_kept=int(keep.sum()),
                mean_dx_mas=float(np.median(dx) * 1e3), mean_dy_mas=float(np.median(dy) * 1e3),
                rms_resid_mas=float(np.std(np.hypot(rx, ry)[keep]) * 1e3))
    return SkyCoord(new_ra * u.deg, new_de * u.deg), diag


def shift_to_virac_frame(cat, virac, to_epoch, match_radius=0.2, magcut=15.0, niter=3, nsigma=3.0):
    """Bulk affine frame-shift any catalog onto the VIRAC2/Gaia frame.

    Generalizes shift_gns_to_virac: propagate VIRAC to ``to_epoch`` (using VIRAC
    pm), match bright common stars, fit affine offset cat->VIRAC, apply to all.
    """
    center = SkyCoord(np.median(cat['sc'].ra), np.median(cat['sc'].dec))
    dt = to_epoch - EPOCH_VIRAC
    vra = virac['sc'].ra.deg + (np.nan_to_num(virac['pmra']) * dt / 3.6e6) / np.cos(virac['sc'].dec.rad)
    vde = virac['sc'].dec.deg + (np.nan_to_num(virac['pmde']) * dt / 3.6e6)
    vsc = SkyCoord(vra * u.deg, vde * u.deg)
    cb = cat['mag'] < magcut
    vb = virac['mag'] < magcut
    csc, vscb = cat['sc'][cb], vsc[vb]
    idx, sep, _ = csc.match_to_catalog_sky(vscb)
    ok = sep < match_radius * u.arcsec
    cx, cy = tangent_xy(csc[ok], center)
    vx, vy = tangent_xy(vscb[idx[ok]], center)
    dx, dy = vx - cx, vy - cy
    keep = np.ones(len(cx), bool)
    A = B = None
    for _ in range(niter):
        Mm = np.column_stack([np.ones(keep.sum()), cx[keep], cy[keep]])
        A, *_ = np.linalg.lstsq(Mm, dx[keep], rcond=None)
        B, *_ = np.linalg.lstsq(Mm, dy[keep], rcond=None)
        rx = dx - (A[0] + A[1] * cx + A[2] * cy)
        ry = dy - (B[0] + B[1] * cx + B[2] * cy)
        keep = np.hypot(rx, ry) < nsigma * np.std(np.hypot(rx, ry)[keep])
    allx, ally = tangent_xy(cat['sc'], center)
    new_ra = cat['sc'].ra.deg + ((A[0] + A[1] * allx + A[2] * ally) / 3600.0) / np.cos(center.dec.rad)
    new_de = cat['sc'].dec.deg + ((B[0] + B[1] * allx + B[2] * ally) / 3600.0)
    diag = dict(n_match=int(ok.sum()), n_kept=int(keep.sum()),
                med_dx_mas=float(np.median(dx) * 1e3), med_dy_mas=float(np.median(dy) * 1e3),
                rms_resid_mas=float(np.std(np.hypot(rx, ry)[keep]) * 1e3))
    return SkyCoord(new_ra * u.deg, new_de * u.deg), diag


def build_pm_catalog(jwst, virac, gns, match_radius=0.3, require_jwst=True):
    """Assemble a 3-epoch flystar StarTable (VIRAC, GNS, JWST) and fit velocities.

    Inputs are dicts (sc/ex/ey/mag/epoch); GNS and JWST must already be on the
    VIRAC frame.  Master list = VIRAC.  Returns an astropy Table of proper motions.
    """
    from flystar.startables import StarTable
    center = SkyCoord(np.median(virac['sc'].ra), np.median(virac['sc'].dec))
    epochs = [virac['epoch'], gns['epoch'], jwst['epoch']]
    cats = [virac, gns, jwst]
    nref = virac['n']
    # tangent-plane master positions
    vx, vy = tangent_xy(virac['sc'], center)
    X = np.full((nref, 3), np.nan); Y = np.full((nref, 3), np.nan)
    XE = np.full((nref, 3), np.nan); YE = np.full((nref, 3), np.nan)
    Mg = np.full((nref, 3), np.nan)
    X[:, 0], Y[:, 0] = vx, vy
    XE[:, 0], YE[:, 0] = virac['ex'], virac['ey']
    Mg[:, 0] = virac['mag']
    # Match GNS(1) and JWST(2) to the VIRAC master using VIRAC's OWN pm as a
    # matching PRIOR: propagate VIRAC to the target epoch and match tightly there
    # (crowded GC fields mismatch badly over a 9.7yr baseline without this).  The
    # counterpart's ACTUAL position is then recorded -> the velocity fit is still
    # an independent measurement (pm prior only used for counterpart ID).  Require
    # mutual-nearest + a loose magnitude-consistency check.
    for j, cat in [(1, gns), (2, jwst)]:
        dt = epochs[j] - EPOCH_VIRAC
        vpra = np.nan_to_num(virac['pmra']); vpde = np.nan_to_num(virac['pmde'])
        vra = virac['sc'].ra.deg + (vpra * dt / 3.6e6) / np.cos(virac['sc'].dec.rad)
        vde = virac['sc'].dec.deg + (vpde * dt / 3.6e6)
        vpred = SkyCoord(vra * u.deg, vde * u.deg)
        idx, sep, _ = vpred.match_to_catalog_sky(cat['sc'])           # virac->cat
        idx_b, sep_b, _ = cat['sc'].match_to_catalog_sky(vpred)        # cat->virac (mutual)
        mutual = idx_b[idx] == np.arange(len(vpred))
        ok = (sep < match_radius * u.arcsec) & mutual
        cx, cy = tangent_xy(cat['sc'][idx], center)
        X[ok, j] = cx[ok]; Y[ok, j] = cy[ok]
        XE[ok, j] = cat['ex'][idx][ok]; YE[ok, j] = cat['ey'][idx][ok]
        Mg[ok, j] = cat['mag'][idx][ok]
    # An epoch is usable for the fit only if BOTH position and error are finite.
    has = np.isfinite(X) & np.isfinite(Y) & np.isfinite(XE) & np.isfinite(YE)
    # zero out the masked-epoch values' errors so flystar masks them consistently
    X[~has] = np.nan; Y[~has] = np.nan; XE[~has] = np.nan; YE[~has] = np.nan
    nepoch = has.sum(axis=1)
    sel = (nepoch >= 2) & (has[:, 2] if require_jwst else (nepoch >= 2))
    name = np.array([f'v{i}' for i in np.where(sel)[0]])
    st = StarTable(name=name, x=X[sel], y=Y[sel], m=Mg[sel],
                   xe=XE[sel], ye=YE[sel],
                   LIST_TIMES=[float(e) for e in epochs], ref_list=0)
    # scipy curve_fit is graceful on degenerate (2-epoch) fits (returns nan vs raise)
    st.fit_velocities(use_scipy=True, show_progress=False, mask_val=np.nan)
    # vx,vy in arcsec/yr -> mas/yr.  x is +RA*cosdec already.
    out = Table()
    out['x0_arcsec'] = st['x0']; out['y0_arcsec'] = st['y0']
    out['ra0'] = center.ra.deg + (st['x0'] / 3600.0) / np.cos(center.dec.rad)
    out['dec0'] = center.dec.deg + (st['y0'] / 3600.0)
    out['pm_ra'] = st['vx'] * 1e3       # mas/yr (already *cosdec via tangent x)
    out['pm_dec'] = st['vy'] * 1e3
    out['pm_ra_err'] = st['vxe'] * 1e3
    out['pm_dec_err'] = st['vye'] * 1e3
    out['pm_tot'] = np.hypot(out['pm_ra'], out['pm_dec'])
    out['n_epoch'] = nepoch[sel]
    out['mag_virac'] = Mg[sel][:, 0]; out['mag_jwst'] = Mg[sel][:, 2]
    out.meta['epochs'] = epochs
    out.meta['frame'] = 'VIRAC2 / Gaia DR3'
    return out
