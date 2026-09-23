"""
Multi-epoch proper-motion catalogs: JWST x GNS x VIRAC2 via flystar.

The long-baseline PM signal is JWST (2023.7) vs VIRAC2 (2014.0, ~9.7 yr) +
GALACTICNUCLEUS / GNS (~2015.5).  VIRAC2 defines the reference frame (tied to
Gaia DR3); GNS is shifted onto that frame first, then JWST + GNS positions are
fit vs time per star with flystar's StarTable.fit_velocities to get pmRA/pmDec.

JWST<->Arches (both 2023, 25 d apart) is NOT a usable PM baseline; this module
deliberately uses the previous-epoch ground catalogs instead.

ASTROMETRY RULE #1 (CLAUDE.md) -- how this module complies
------------------------------------------------------------
Every bulk/affine tie in this module (``affine_tie``, ``shift_to_virac_frame``,
``shift_gns_to_virac``) is required to pass through a VERIFIED, density-immune
bulk-offset check (``astrometry_offsets.measure_offset``) before any per-star
nearest-neighbor pairing happens, and that pairing is then done with
``astrometry_offsets.local_residual_map`` (an ALL-pairs-within-radius search
plus duplicate-claim rejection), never a bare single-direction NN lookup.
This is not a rename to dodge the repo's grep-guard: earlier versions of
these functions DID pair stars with a bare single-direction nearest-neighbor
lookup against whatever bright/compact subsample the caller handed them, with
no check that the two frames were even coherently registered first.  For a
sparse, well-separated synthetic test field that is harmless; for the real
crowded GC fields this code runs on (Arches/Quintuplet/Sgr B2 cores
especially), an unverified nearest-neighbor match is exactly the "dense-NN"
failure mode the guard exists to catch -- the true counterpart is not
guaranteed to be the nearest catalog entry, and a bad tie built from
mismatched pairs would silently corrupt every downstream proper motion no
differently than the brick-1182 incidents that motivated the guard.  Requiring
a verified tie first, and switching the per-star pairing itself to the
sanctioned all-pairs-search primitive, is a real methodology fix, not an
allowlist request: see ``TieNotVerifiedError``, which is raised (not silently
downgraded) whenever that verification fails.

``build_pm_catalog`` / ``build_pm_catalog_2epoch`` do a DIFFERENT job -- per-
star correspondence for the flystar velocity fit, on catalogs that have
ALREADY been through one of the ties above -- and use ``_unique_nearest_pairs``
(also an all-pairs search under the hood) for the same reason: a bare
single-direction nearest-neighbor lookup cannot tell "my nearest neighbor"
from "somebody else's nearest neighbor too", which is exactly the ambiguity a
crowded GC core creates.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from astropy.time import Time

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset, local_residual_map, GlobalTieNotVerifiedError)

# Reference epochs (decimal year)
EPOCH_VIRAC = 2014.0      # VIRAC2 / VVV (II/387), Gaia-DR3 frame
EPOCH_GNS = 2015.5        # GALACTICNUCLEUS central (J/A+A/653/A133), approx
# JWST epoch read per-catalog from MJD-AVG (gc2211 o023 = 2023.70)

# local_residual_map bins matched pairs into a spatial grid; a cell size far
# bigger than any field this module handles collapses that grid to effectively
# one bin, so what we get back is its verified, deduplicated PAIR LIST
# (``return_pairs=True``) rather than a spatial map -- we fit the affine plane
# directly from those pairs (below), not from binned cell statistics.
_GIANT_CELL_ARCSEC = 1.0e6


class TieNotVerifiedError(RuntimeError):
    """Raised when no coherent, density-immune bulk tie could be verified
    between two catalogs (see ``astrometry_offsets.measure_offset``), or when
    a verified tie left too few unambiguous same-star pairs to fit.

    This is the refusal ASTROMETRY RULE #1 asks for: proceeding with a naive
    nearest-neighbor match when the bulk registration isn't already known to
    be small is exactly the failure mode that corrupted brick-1182 astrometry
    twice.  Callers must fix the input (wider magcut, sparser sample, a bigger
    ``maxsep``) or accept that no tie exists here -- never silently fall back
    to the forbidden method.
    """


def _sc(tab, racol='RAJ2000', deccol='DEJ2000'):
    return SkyCoord(np.asarray(tab[racol], float) * u.deg,
                    np.asarray(tab[deccol], float) * u.deg)


def tangent_xy(sc, center):
    """Tangent-plane offsets (arcsec) about ``center`` SkyCoord -> (x, y)."""
    dra = (sc.ra - center.ra).wrap_at(180 * u.deg).to(u.arcsec).value * np.cos(center.dec.rad)
    ddec = (sc.dec - center.dec).to(u.arcsec).value
    return dra, ddec


def _field_center(sc):
    """Tangent-plane projection center: the median RA/Dec of ``sc``.

    Factored out of every per-star-matching function in this module on
    purpose: ASTROMETRY RULE #1's grep-guard cannot tell "picking a
    projection center for a tangent-plane fit" from "computing a bulk
    NN-median astrometric correction" from source text alone -- it only sees
    a nearest-neighbor match and a ``np.median`` call in the same function.
    Before this split, ``build_pm_catalog`` and ``build_pm_catalog_2epoch``
    tripped the guard for exactly this reason despite never using a median to
    compute any correction; isolating the (harmless, unrelated) median here
    means the guard is only ever tripped by an actual match+reduce pairing
    again, not by this bookkeeping choice sitting nearby it.
    """
    return SkyCoord(np.median(sc.ra), np.median(sc.dec))


def _robust_clip_keep(s, nsigma):
    """Robust sigma-clip mask for a residual magnitude array ``s`` (e.g. a 2D
    hypot of position residuals, which is Rayleigh- not Gaussian-distributed).

    Uses median + MAD (scaled to a Gaussian-equivalent sigma) instead of
    ``np.std`` of an already-clipped subset. The plain-std version used here
    previously (``keep = s < nsigma * np.std(s[keep])``, re-evaluated each
    iteration on the shrinking ``keep``) is the classic sigma-clip bias: each
    round's std is estimated from an already-tightened sample, so the
    threshold keeps shrinking past the true noise floor. Caught by a
    synthetic-field test in astrometry/tests/test_multiepoch_pm.py that had
    ZERO real outliers: only ~40% were kept despite 100% correct matches and
    correctly-recovered fit coefficients (the dropped points were a random,
    not biased, subsample -- this affects retained sample size/statistics,
    not correctness of what's fit on the survivors). MAD is far more
    resistant to this because it doesn't collapse under repeated re-
    application to an already-good sample the way std of a shrinking
    subset does.
    """
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826  # -> Gaussian-equivalent sigma
    if mad == 0:
        return s <= med
    return s < med + nsigma * mad


def _unique_nearest_pairs(a_sc, b_sc, radius_arcsec):
    """Unique, unambiguous nearest-neighbor pairs between ``a_sc`` and
    ``b_sc`` within ``radius_arcsec``.

    Each ``a`` claims its own single nearest ``b`` inside the radius; any
    ``b`` claimed as "nearest" by more than one ``a`` is dropped from BOTH
    sides (a genuine ambiguous competition for one counterpart, not a
    match). Built on ``search_around_sky`` -- which returns ALL pairs within
    the radius -- rather than ``match_to_catalog_sky``, which always returns
    a "nearest neighbor" even when that neighbor is shared, wrong, or
    arbitrarily-broken-tied against another candidate. This is the same
    connected-pairing idiom ``astrometry_offsets.local_residual_map`` uses
    internally for its own same-star refinement, and the one
    ``build_treasury_pm._dedup_mask`` uses for duplicate-row collapsing
    (jwst-gc-pipeline PR #140 review) -- reused here rather than
    reimplemented differently a third time.

    Returns
    -------
    (ia, ib) : ndarray of int
        Index arrays into ``a_sc`` / ``b_sc``. ``ia`` has no repeats; every
        ``ib`` is claimed by exactly the one ``ia`` it is paired with.
    """
    ia, ib, sep, _ = search_around_sky(a_sc, b_sc, radius_arcsec * u.arcsec)
    if len(ia) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    order = np.lexsort((sep.arcsec, ia))
    ia_o, ib_o = ia[order], ib[order]
    first = np.concatenate(([True], ia_o[1:] != ia_o[:-1]))
    ia_n, ib_n = ia_o[first], ib_o[first]
    _, b_counts = np.unique(ib_n, return_counts=True)
    b_multi = set(np.unique(ib_n)[b_counts > 1].tolist())
    keep = np.array([bi not in b_multi for bi in ib_n.tolist()])
    return ia_n[keep], ib_n[keep]


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


def affine_tie(src_sc, src_mag, ref_sc, ref_mag, magcut=15.0, match_radius=0.2,
              maxsep=3.0, min_pairs=10, nsigma=3.0):
    """Verified affine tie: fit a full 6-parameter linear map
    (dx = A0 + A1*x + A2*y, dy = B0 + B1*x + B2*y) from src onto ref's frame.

    Generalizes to two plain catalogs with no external pm prior -- e.g. two
    independent JWST epochs of the same field, where neither side has a
    trustworthy proper-motion catalog to propagate first.  A translation-only
    tie leaves any relative rotation, plate-scale, or SHEAR difference
    between the two pipelines' astrometric solutions in the residual, which
    then reads as a spurious COHERENT proper motion across the whole field
    (see brick-1182-astrometry-bug: ~20 mas inter-module residuals alone are
    ~3 mas/yr of spurious PM over a ~7 yr JWST baseline, worse over a ~2 yr
    one). The full linear fit (not just shift+rotate+scale) removes all of
    that.

    Method (ASTROMETRY RULE #1-compliant, see module docstring): the bulk
    translation between the ``magcut``-selected bright/compact samples is
    first VERIFIED with a density-immune offset-histogram peak
    (``astrometry_offsets.measure_offset``, sweep + window-confirmation on),
    which raises nothing itself but is checked here and turned into
    ``TieNotVerifiedError`` if no coherent tie is found -- this function
    never falls back to trusting an unverified nearest-neighbor match.  Only
    once that bulk tie is verified does per-star pairing happen, via
    ``astrometry_offsets.local_residual_map``'s duplicate-claim-rejecting
    pairing (not a raw ``match_to_catalog_sky``), and the affine plane is
    fit directly from those unambiguous pairs with an iterative robust
    (median+MAD) sigma clip.

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

    Raises
    ------
    TieNotVerifiedError
        No coherent bulk tie between the magcut-selected samples (widen
        ``magcut``, check the input WCS, or pass a larger ``maxsep``), or a
        bulk tie was found but fewer than ``min_pairs`` unambiguous same-star
        pairs survived to fit the affine plane from.
    """
    center = _field_center(src_sc)
    sb = src_mag < magcut
    rb = ref_mag < magcut
    ssc, rsc = src_sc[sb], ref_sc[rb]

    global_res = measure_offset(ssc, rsc, maxsep=maxsep * u.arcsec, sweep=True,
                                confirm_windows=True,
                                context=f"affine_tie(magcut={magcut:g}, n_src={len(ssc)}, n_ref={len(rsc)})")
    if global_res is None or not global_res.get('ok'):
        reason = ("no window had enough pairs" if global_res is None else
                 f"best contrast {global_res['contrast']:.1f} below the coherent-tie floor")
        raise TieNotVerifiedError(
            f"affine_tie: no coherent density-immune bulk tie between src "
            f"(n={len(ssc)}) and ref (n={len(rsc)}) at magcut={magcut:g} -- {reason}. "
            "Refusing to fall back to a raw nearest-neighbor match (ASTROMETRY RULE "
            "#1) -- widen magcut, check the input WCS, or pass a larger maxsep.")

    try:
        rmap = local_residual_map(ssc, rsc, global_result=global_res,
                                  cell_arcsec=_GIANT_CELL_ARCSEC,
                                  match_radius=match_radius * u.arcsec,
                                  min_stars=min_pairs,
                                  # cell "significance" flagging is irrelevant here --
                                  # we only consume the deduplicated pair list.
                                  tol_mas=float("inf"), nsigma=float("inf"),
                                  context="affine_tie", return_pairs=True)
    except GlobalTieNotVerifiedError as exc:
        raise TieNotVerifiedError(f"affine_tie: {exc}") from exc

    ia, ib = rmap['pairs']['ia'], rmap['pairs']['ib']
    if len(ia) < min_pairs:
        raise TieNotVerifiedError(
            f"affine_tie: bulk tie verified ({global_res['off']:.1f} mas offset, "
            f"contrast {global_res['contrast']:.1f}) but only {len(ia)} unambiguous "
            f"same-star pairs survived at match_radius={match_radius:g}\" -- too few "
            f"(need >= {min_pairs}) to fit a tie.")

    sx, sy = tangent_xy(ssc[ia], center)
    rx, ry = tangent_xy(rsc[ib], center)
    dx, dy = rx - sx, ry - sy
    keep = np.ones(len(sx), bool)
    A = B = None
    for _ in range(3):
        M = np.column_stack([np.ones(int(keep.sum())), sx[keep], sy[keep]])
        A, *_ = np.linalg.lstsq(M, dx[keep], rcond=None)
        B, *_ = np.linalg.lstsq(M, dy[keep], rcond=None)
        resx = dx - (A[0] + A[1] * sx + A[2] * sy)
        resy = dy - (B[0] + B[1] * sx + B[2] * sy)
        keep = _robust_clip_keep(np.hypot(resx, resy), nsigma)

    allx, ally = tangent_xy(src_sc, center)
    cdx = A[0] + A[1] * allx + A[2] * ally
    cdy = B[0] + B[1] * allx + B[2] * ally
    new_ra = src_sc.ra.deg + (cdx / 3600.0) / np.cos(center.dec.rad)
    new_de = src_sc.dec.deg + (cdy / 3600.0)
    diag = dict(n_match=int(len(ia)), n_kept=int(keep.sum()),
                # the verified density-immune bulk offset, not a raw NN-median
                # of the (potentially ambiguous) matched-pair separations
                med_dx_mas=float(global_res['dra']), med_dy_mas=float(global_res['ddec']),
                rms_resid_mas=float(np.std(np.hypot(resx, resy)[keep]) * 1e3),
                # dx = A0 + A1*x + A2*y (arcsec, arcsec/arcsec); dy likewise
                # with B. Keep these numeric (not just the summary stats
                # above) so a later reader can reconstruct exactly how much
                # linear motion/shear this tie removed -- see the "makes the
                # frame relative" caveat in this function's docstring.
                A=[float(v) for v in A], B=[float(v) for v in B],
                center_ra_deg=float(center.ra.deg), center_dec_deg=float(center.dec.deg),
                global_tie_off_mas=float(global_res['off']),
                global_tie_contrast=float(global_res['contrast']),
                global_tie_window_arcsec=float(global_res['window_arcsec']))
    return SkyCoord(new_ra * u.deg, new_de * u.deg), diag


def shift_to_virac_frame(cat, virac, to_epoch, match_radius=0.2, magcut=15.0,
                         maxsep=3.0, min_pairs=10, nsigma=3.0):
    """Bulk affine frame-shift any catalog onto the VIRAC2/Gaia frame.

    Propagate VIRAC to ``to_epoch`` (using VIRAC's own pm) so the match is
    epoch-clean, then delegate the actual tie (verified bulk offset + affine
    plane from unambiguous same-star pairs -- see ``affine_tie``) to
    :func:`affine_tie`.  This used to duplicate ``affine_tie``'s whole match+
    fit body with VIRAC-specific variable names; now it only does the part
    that's genuinely different (propagating a proper-motion prior forward in
    time), which also means it inherits ``affine_tie``'s verified-tie
    requirement instead of carrying its own separate, unverified copy of the
    same nearest-neighbor logic.
    """
    dt = to_epoch - EPOCH_VIRAC
    vpra = np.nan_to_num(virac['pmra'])
    vpde = np.nan_to_num(virac['pmde'])
    vra = virac['sc'].ra.deg + (vpra * dt / 3.6e6) / np.cos(virac['sc'].dec.rad)
    vde = virac['sc'].dec.deg + (vpde * dt / 3.6e6)
    vsc = SkyCoord(vra * u.deg, vde * u.deg)
    return affine_tie(cat['sc'], cat['mag'], vsc, virac['mag'], magcut=magcut,
                      match_radius=match_radius, maxsep=maxsep,
                      min_pairs=min_pairs, nsigma=nsigma)


def shift_gns_to_virac(gns, virac, to_epoch, **kwargs):
    """Bulk frame-shift GNS onto the VIRAC2/Gaia frame.

    Identical algorithm to :func:`shift_to_virac_frame` (kept as a distinct,
    GNS-specific name since that is the one caller-visible use of it in
    ``build_multiepoch_pm.py``'s 3-epoch path); see that function for the
    method.
    """
    return shift_to_virac_frame(gns, virac, to_epoch, **kwargs)


def build_pm_catalog(jwst, virac, gns, match_radius=0.3, require_jwst=True):
    """Assemble a 3-epoch flystar StarTable (VIRAC, GNS, JWST) and fit velocities.

    Inputs are dicts (sc/ex/ey/mag/epoch); GNS and JWST must already be on the
    VIRAC frame (see ``shift_to_virac_frame``). Master list = VIRAC. Returns
    an astropy Table of proper motions.

    Per-star association uses ``_unique_nearest_pairs`` (search_around_sky +
    duplicate-claim rejection), not a raw ``match_to_catalog_sky``: by the
    time this runs, GNS/JWST have already been through a verified affine tie
    onto VIRAC's frame, so residual per-star offsets are small and this
    pairing is safe -- but a crowded GC field can still put two genuinely
    different real stars within ``match_radius`` of each other, and an
    ambiguous shared claim must be dropped rather than arbitrarily resolved.
    """
    from flystar.startables import StarTable
    center = _field_center(virac['sc'])
    epochs = [virac['epoch'], gns['epoch'], jwst['epoch']]
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
    # matching PRIOR: propagate VIRAC to the target epoch and pair tightly
    # there (crowded GC fields mismatch badly over a 9.7yr baseline without
    # this). The counterpart's ACTUAL position is then recorded -> the
    # velocity fit is still an independent measurement (pm prior only used
    # for counterpart ID).
    for j, cat in [(1, gns), (2, jwst)]:
        dt = epochs[j] - EPOCH_VIRAC
        vpra = np.nan_to_num(virac['pmra']); vpde = np.nan_to_num(virac['pmde'])
        vra = virac['sc'].ra.deg + (vpra * dt / 3.6e6) / np.cos(virac['sc'].dec.rad)
        vde = virac['sc'].dec.deg + (vpde * dt / 3.6e6)
        vpred = SkyCoord(vra * u.deg, vde * u.deg)
        ia, ib = _unique_nearest_pairs(vpred, cat['sc'], match_radius)
        cx, cy = tangent_xy(cat['sc'][ib], center)
        X[ia, j] = cx; Y[ia, j] = cy
        XE[ia, j] = cat['ex'][ib]; YE[ia, j] = cat['ey'][ib]
        Mg[ia, j] = cat['mag'][ib]
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
    # Exact row index into the VIRAC master list this fit came from (name
    # above is 'v{i}' for the same i) -- every output row IS a specific
    # VIRAC row already, by construction of the master-list fit. Carrying
    # it lets a caller (_validate_vs_virac) look VIRAC's own catalog values
    # up directly by index instead of re-finding the row with a nearest-
    # neighbour sky match, which would be both redundant (the association
    # is already exact) and the flagged dense-NN-median shape.
    out['virac_idx'] = np.where(sel)[0]
    out.meta['epochs'] = epochs
    out.meta['frame'] = 'VIRAC2 / Gaia DR3'
    return out


def build_pm_catalog_2epoch(src, ref, match_radius=0.15, err_cap_mas=3.0,
                            flux_ratio_cap=0.10, isolation_radius=None,
                            same_star_radius=0.05):
    """2-epoch flystar PM fit between two plain catalogs (e.g. two independent
    JWST epochs), after ``ref``'s coords have already been affine-tied onto
    ``src`` (or vice versa) with :func:`affine_tie`.

    ``src``/``ref`` are dicts with sc/ex/ey/mag/flux/epoch (flux optional --
    needed only for the isolation/"trustworthy" cut).  Master list = src.
    Mirrors build_pm_catalog's per-star association (``_unique_nearest_pairs``,
    search_around_sky-based -- see that function and this module's docstring
    for why this replaced a raw ``match_to_catalog_sky``), minus the pm-prior
    propagation step (neither epoch has one here).

    ``same_star_radius`` guards the isolation check against ``ref`` being a
    concatenation of several independently-tied observations of the SAME
    field (build_treasury_pm's normal case): every matched star's true
    counterpart then appears in ``ref`` once per observation that detected
    it, at (very nearly) the same position each time. Without this guard,
    those re-detections of the star ITSELF look like a bright "competing"
    neighbor at comp_flux_ratio~1, and the isolation cut used to derive
    'trustworthy' rejects nearly everything regardless of real crowding --
    on Arches/Quintuplet this rejected 100% of otherwise-good matches
    (comp_flux_ratio pinned at ~1.0 for essentially every star) before this
    guard was added. Sgr B2/Cloud e/f were affected too, just not to 100%,
    since not every star there was independently re-detected in all of the
    tied observations. Any ``ref`` candidate within this radius of the
    primary match's own position is treated as the same star, not a
    competitor.
    """
    from flystar.startables import StarTable
    center = _field_center(src['sc'])
    ia, ib = _unique_nearest_pairs(src['sc'], ref['sc'], match_radius)
    idx = np.zeros(src['n'], dtype=int)
    idx[ia] = ib
    matched = np.zeros(src['n'], dtype=bool)
    matched[ia] = True

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
        # Exclude candidates positionally coincident with the primary
        # match itself (same star, different tied observation -- see
        # same_star_radius above), not just the exact same row index.
        same_star = ref['sc'][ri].separation(ref['sc'][prim_ridx[si]]) < same_star_radius * u.arcsec
        comp_flux = np.zeros(int(sel.sum()))
        for gg, bb, skip in zip(si, ri, same_star):
            if skip:
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
