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


def affine_tie(src_sc, src_mag, ref_sc, ref_mag, magcut=15.0, match_radius=0.2,
               niter=3, nsigma=3.0):
    """Iterative sigma-clipped affine tie: fit a full 6-parameter linear map
    (dx = A0 + A1*x + A2*y, dy = B0 + B1*x + B2*y) from src onto ref's frame.

    Generalizes shift_to_virac_frame/shift_gns_to_virac to two plain catalogs
    with no external pm prior -- e.g. two independent JWST epochs of the same
    field, where neither side has a trustworthy proper-motion catalog to
    propagate first.  A translation-only tie leaves any relative rotation,
    plate-scale, or SHEAR difference between the two pipelines' astrometric
    solutions in the residual, which then reads as a spurious COHERENT
    proper motion across the whole field (see brick-1182-astrometry-bug: ~20
    mas inter-module residuals alone are ~3 mas/yr of spurious PM over a ~7
    yr JWST baseline, worse over a ~2 yr one). The full linear fit (not just
    shift+rotate+scale) removes all of that.

    CAVEAT -- this makes the output frame RELATIVE, not absolute: a genuine
    bulk/rotation/shear velocity field is *also* linear to first order across
    a field of a few arcmin (Galactic rotation, bulge streaming, NSC
    rotation), and this fit cannot distinguish that from a plate-scale/shear
    error in the astrometric solution -- it removes both by construction.
    Downstream proper motions from a tie built this way are relative to the
    mean motion + mean shear of whichever stars were used for the tie (the
    ``magcut`` bright/compact sample here), not an absolute frame. A rotation
    curve measured from PMs tied this way will read closer to zero than it
    should, with no warning baked into the numbers alone -- callers that need
    an absolute measurement must look at the returned coefficients (A, B in
    the diagnostics dict) to see how much was removed, and add it back if the
    physical signal of interest is on that same linear scale.
    """
    center = SkyCoord(np.median(src_sc.ra), np.median(src_sc.dec))
    sb = src_mag < magcut
    rb = ref_mag < magcut
    ssc, rsc = src_sc[sb], ref_sc[rb]
    idx, sep, _ = ssc.match_to_catalog_sky(rsc)
    ok = sep < match_radius * u.arcsec
    sx, sy = tangent_xy(ssc[ok], center)
    rx, ry = tangent_xy(rsc[idx[ok]], center)
    dx, dy = rx - sx, ry - sy
    keep = np.ones(len(sx), bool)
    A = B = None
    for _ in range(niter):
        Mm = np.column_stack([np.ones(keep.sum()), sx[keep], sy[keep]])
        A, *_ = np.linalg.lstsq(Mm, dx[keep], rcond=None)
        B, *_ = np.linalg.lstsq(Mm, dy[keep], rcond=None)
        resx = dx - (A[0] + A[1] * sx + A[2] * sy)
        resy = dy - (B[0] + B[1] * sx + B[2] * sy)
        s = np.hypot(resx, resy)
        keep = s < nsigma * np.std(s[keep])
    allx, ally = tangent_xy(src_sc, center)
    cdx = A[0] + A[1] * allx + A[2] * ally
    cdy = B[0] + B[1] * allx + B[2] * ally
    new_ra = src_sc.ra.deg + (cdx / 3600.0) / np.cos(center.dec.rad)
    new_de = src_sc.dec.deg + (cdy / 3600.0)
    diag = dict(n_match=int(ok.sum()), n_kept=int(keep.sum()),
                med_dx_mas=float(np.median(dx) * 1e3), med_dy_mas=float(np.median(dy) * 1e3),
                rms_resid_mas=float(np.std(np.hypot(resx, resy)[keep]) * 1e3),
                # dx = A0 + A1*x + A2*y (arcsec, arcsec/arcsec); dy likewise
                # with B. Keep these numeric (not just the summary stats
                # above) so a later reader can reconstruct exactly how much
                # linear motion/shear this tie removed -- see the "makes the
                # frame relative" caveat in this function's docstring.
                A=[float(v) for v in A], B=[float(v) for v in B],
                center_ra_deg=float(center.ra.deg), center_dec_deg=float(center.dec.deg))
    return SkyCoord(new_ra * u.deg, new_de * u.deg), diag


def build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=3.0,
                            flux_ratio_cap=0.10, isolation_radius=None):
    """2-epoch flystar PM fit between two plain catalogs (e.g. two independent
    JWST epochs), after ``ref``'s coords have already been affine-tied onto
    ``src`` (or vice versa) with :func:`affine_tie`.

    ``src``/``ref`` are dicts with sc/ex/ey/mag/flux/epoch (flux optional --
    needed only for the isolation/"trustworthy" cut).  Master list = src.
    Mirrors build_pm_catalog's mutual-NN matching + isolation filter, minus
    the pm-prior propagation step (neither epoch has one here).
    """
    from flystar.startables import StarTable
    from astropy.coordinates import search_around_sky
    center = SkyCoord(np.median(src['sc'].ra), np.median(src['sc'].dec))
    idx, sep, _ = src['sc'].match_to_catalog_sky(ref['sc'])
    idx_b, _, _ = ref['sc'].match_to_catalog_sky(src['sc'])
    mutual = idx_b[idx] == np.arange(src['n'])
    matched = (sep < match_radius * u.arcsec) & mutual

    sx, sy = tangent_xy(src['sc'], center)
    rx, ry = tangent_xy(ref['sc'][idx], center)
    n = src['n']
    X = np.full((n, 2), np.nan); Y = np.full((n, 2), np.nan)
    XE = np.full((n, 2), np.nan); YE = np.full((n, 2), np.nan); Mg = np.full((n, 2), np.nan)
    X[:, 0], Y[:, 0], XE[:, 0], YE[:, 0], Mg[:, 0] = sx, sy, src['ex'], src['ey'], src['mag']
    X[matched, 1] = rx[matched]; Y[matched, 1] = ry[matched]
    XE[matched, 1] = ref['ex'][idx][matched]; YE[matched, 1] = ref['ey'][idx][matched]
    Mg[matched, 1] = ref['mag'][idx][matched]
    sel = matched.copy()
    name = np.array([f's{i}' for i in np.where(sel)[0]])
    st = StarTable(name=name, x=X[sel], y=Y[sel], m=Mg[sel], xe=XE[sel], ye=YE[sel],
                   LIST_TIMES=[float(src['epoch']), float(ref['epoch'])], ref_list=0)
    st.fit_velocities(use_scipy=True, show_progress=False, mask_val=np.nan)

    dt = ref['epoch'] - src['epoch']
    out = Table()
    out['ra0'] = center.ra.deg + (st['x0'] / 3600.0) / np.cos(center.dec.rad)
    out['dec0'] = center.dec.deg + (st['y0'] / 3600.0)
    out['pm_ra'] = st['vx'] * 1e3
    out['pm_dec'] = st['vy'] * 1e3
    out['pm_tot'] = np.hypot(out['pm_ra'], out['pm_dec'])
    selidx = np.where(sel)[0]
    # Formal position errors (src/ref per-frame scatter), NOT the flystar fit
    # covariance: with exactly 2 epochs the fit is an exact line through 2
    # points, so its own formal error collapses toward zero regardless of how
    # noisy the input positions were.  This is the quantity fit_velocities'
    # own vxe/vye badly underestimate in the 2-epoch case.
    out['pm_ra_err'] = np.hypot(src['ex'][selidx], ref['ex'][idx][selidx]) * 1e3 / abs(dt)
    out['pm_dec_err'] = np.hypot(src['ey'][selidx], ref['ey'][idx][selidx]) * 1e3 / abs(dt)
    out['mag_src'] = src['mag'][selidx]

    good_err = (np.isfinite(out['pm_ra_err']) & np.isfinite(out['pm_dec_err']) &
                (out['pm_ra_err'] < err_cap_mas) & (out['pm_dec_err'] < err_cap_mas))
    trust = good_err.copy()
    if 'flux' in src and 'flux' in ref:
        beam = isolation_radius if isolation_radius is not None else match_radius
        prim_ridx = idx[selidx]
        prim_flux = ref['flux'][prim_ridx]
        si, ri, _, _ = search_around_sky(src['sc'][sel], ref['sc'], beam * 3 * u.arcsec)
        comp_flux = np.zeros(int(sel.sum()))
        for gg, bb in zip(si, ri):
            if bb == prim_ridx[gg]:
                continue
            if ref['flux'][bb] > comp_flux[gg]:
                comp_flux[gg] = ref['flux'][bb]
        frac = np.where(prim_flux > 0, comp_flux / prim_flux, np.nan)
        ss_i, _, _, _ = search_around_sky(src['sc'][sel], src['sc'], beam * 3 * u.arcsec)
        n_src_near = np.bincount(ss_i, minlength=int(sel.sum()))
        dominant = (frac < flux_ratio_cap) & (n_src_near == 1)
        out['comp_flux_ratio'] = frac
        out['n_src_within_beam'] = n_src_near
        out['dominant'] = dominant
        trust = trust & dominant
    out['trustworthy'] = trust
    out.meta['epochs'] = [float(src['epoch']), float(ref['epoch'])]
    out.meta['baseline_yr'] = float(dt)
    return out
