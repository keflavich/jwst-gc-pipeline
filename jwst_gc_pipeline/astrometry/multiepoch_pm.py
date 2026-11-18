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
import warnings

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from astropy.time import Time

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset, local_residual_map, GlobalTieNotVerifiedError,
    refusal_reason, DEFAULT_MIN_CONTRAST)

# Reference epochs (decimal year)
EPOCH_VIRAC = 2014.0      # VIRAC2 / VVV (II/387), Gaia-DR3 frame
EPOCH_GNS = 2015.5        # GALACTICNUCLEUS central (J/A+A/653/A133), approx
# JWST epoch read per-catalog from MJD-AVG (gc2211 o023 = 2023.70)

# A matched pair this close AND this flux-identical after a verified bulk
# tie is not "excellent agreement" -- two genuinely independent epochs of a
# real star essentially never coincide to native-pixel precision AND exact
# flux. It is the literal duplicate-row contamination jwst-gc-pipeline#958
# found across nominally-independent GC Treasury observations (54-62% of
# rows bit-identical in RA/Dec and flux). A few percent is well below that
# and well above the coincidence rate expected of real data (pr-reviewer
# PR #140 round 5 review); measure_offset's own coherence check cannot catch
# this on its own, since duplicated rows pile up at the SAME true offset and
# satisfy translational coherence trivially.
_DUP_ROW_FRACTION_LIMIT = 0.02
# Deliberately far tighter than any real (or synthetic-test) per-star
# centroid noise floor (this pipeline's own formal errors run ~1-2 mas; a
# literal duplicate row lands here at floating-point precision, effectively
# 0). A genuinely noisy real pair landing this close by chance is rare
# enough not to force the false-positive rate anywhere near
# _DUP_ROW_FRACTION_LIMIT even for a whole matched sample.
_DUP_ROW_SEP_MAS = 0.1
_DUP_ROW_DMAG = 1.0e-4

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

    This is bookkeeping for the tangent-plane projection, not an astrometric
    correction -- picking WHERE to center the (x, y) coordinate system a fit
    is expressed in has no bearing on what that fit computes. The five
    functions in this module that need a projection center (``build_pm_
    catalog``, ``build_pm_catalog_2epoch``, ``affine_tie``, ``shift_to_virac_
    frame``/``shift_gns_to_virac`` via ``affine_tie``) used to each compute
    it inline, identically; pulling the one-line calculation out here once
    is the ordinary DRY fix for five copies of the same thing.
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

    This is a DIFFERENT criterion from mutual-nearest-neighbor (each side's
    own nearest neighbor is the other), not a strictly better one: it drops
    ALL claimants of a contested ``b`` -- including a clean, unambiguous pair
    that merely happens to share a competitor -- and it accepts some
    genuinely non-mutual pairs mutual-NN would reject. Measured on a
    synthetic field at 3 stars/arcsec^2 (pr-reviewer, PR #140 round 5): this
    recovers 93.8% of true pairs at a 0.75% false-pair rate, vs mutual-NN's
    98.8% / 1.12%. Lower completeness, lower contamination -- an acceptable
    trade for a tie-fitting sample (bad pairs corrupt the fit; a somewhat
    smaller clean sample does not), but not a free upgrade, and not the
    right choice everywhere.

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


#: ``assign='mincost'`` defaults (see ``_gated_mincost_pairs``).
ASSIGN_SEARCH_ARCSEC = 0.06
#: 99.7% point of a 2-dof chi2: the position gate on (dx, dy).
ASSIGN_POS_GATE_CHI2 = 11.8
#: Added in quadrature to the two epochs' per-star position errors:
#: sqrt(3.5**2 + 5**2) mas = per-observation frame-tie residual plus the
#: displacement a 2 yr baseline gives a ~2.5 mas/yr star.
ASSIGN_FLOOR_MAS = 6.1
#: Systematic ln-flux scatter between the epochs (bright stars are limited by
#: the PSF model and crowding, not by the formal flux error).
ASSIGN_FLUX_FLOOR = 0.10
#: 99.7% point of a 1-dof chi2: the gate on ln(flux_ref / flux_src).
ASSIGN_FLUX_GATE_CHI2 = 9.0


def _pos_sigma_mas(cat):
    """Per-star 1-axis position error in mas from a catalog dict's ``ex``/``ey`` (arcsec)."""
    return np.hypot(np.asarray(cat['ex'], float), np.asarray(cat['ey'], float)) / np.sqrt(2.0) * 1e3


def _flux_noise(cat):
    """Per-row local flux noise of a catalog dict; 0 where unknown, so a missing
    estimate leaves only the systematic floor in the flux gate rather than
    failing it."""
    if 'flux_noise' not in cat:
        return np.zeros(len(cat['sc']))
    return np.nan_to_num(np.asarray(cat['flux_noise'], float))


def _gated_mincost_pairs(a, b, match_radius=0.15, search_arcsec=ASSIGN_SEARCH_ARCSEC,
                         pos_gate_chi2=ASSIGN_POS_GATE_CHI2, floor_mas=ASSIGN_FLOOR_MAS,
                         flux_floor=ASSIGN_FLUX_FLOOR, flux_gate_chi2=ASSIGN_FLUX_GATE_CHI2):
    """One-to-one pairs between catalog dicts ``a`` and ``b`` (the ``src`` /
    ``ref`` dicts of :func:`build_pm_catalog_2epoch`), in two tiers.

    Tier 1: every (a, b) candidate within ``search_arcsec`` whose position chi2
    (``(dx**2 + dy**2) / (sig_a**2 + sig_b**2 + floor_mas**2)``) is below
    ``pos_gate_chi2`` and whose flux ratio is consistent
    (``ln(F_b/F_a)**2 / (flux_floor**2 + (noise_a/F_a)**2 + (noise_b/F_b)**2)``
    below ``flux_gate_chi2``; skipped when either catalog has no ``flux``) is an
    edge of a bipartite graph.  Each connected component is solved exactly with
    ``scipy.optimize.linear_sum_assignment`` on the summed chi2, preferring
    more pairs over a lower total (a non-edge costs 1e6).  Where
    :func:`_unique_nearest_pairs` drops ALL claimants of a ``b`` that two ``a``
    both call nearest, this keeps the one that fits (position AND flux) and
    leaves the other unmatched.

    Tier 2: the :func:`_unique_nearest_pairs` pairs (``match_radius``) that
    tier 1 did not touch -- both ends unused -- are kept as they are, so a star
    whose true displacement is outside the tier-1 gate (a ~10-70 mas/yr mover)
    is matched exactly as before.  Tier 2 never ADDS a pair
    :func:`_unique_nearest_pairs` would not have made.  (Re-running the
    nearest rule on tier 1's leftovers instead does add pairs, but 80% of those
    lie at 100-150 mas with flux ratios off by factors of 2-20: chance
    coincidences.)

    Returns ``(ia, ib, tier)``: index arrays into ``a['sc']`` / ``b['sc']``
    (``ia`` has no repeats, ``ib`` has no repeats) and the tier (1 or 2) of each pair.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    na, nb = len(a['sc']), len(b['sc'])
    ia, ib, _, _ = search_around_sky(a['sc'], b['sc'], search_arcsec * u.arcsec)
    t1a = t1b = np.zeros(0, dtype=int)
    if len(ia):
        dlon, dlat = a['sc'][ia].spherical_offsets_to(b['sc'][ib])
        dx, dy = dlon.to_value(u.mas), dlat.to_value(u.mas)
        sig2 = _pos_sigma_mas(a)[ia] ** 2 + _pos_sigma_mas(b)[ib] ** 2 + floor_mas ** 2
        cost = (dx ** 2 + dy ** 2) / sig2
        keep = cost < pos_gate_chi2
        if 'flux' in a and 'flux' in b:
            fa, fb = np.asarray(a['flux'], float)[ia], np.asarray(b['flux'], float)[ib]
            noise_a, noise_b = _flux_noise(a)[ia], _flux_noise(b)[ib]
            with np.errstate(divide='ignore', invalid='ignore'):
                chi_f = np.log(fb / fa) ** 2 / (flux_floor ** 2 + (noise_a / fa) ** 2 + (noise_b / fb) ** 2)
            keep &= chi_f < flux_gate_chi2
            cost = cost + np.where(keep, chi_f, 0.0)
        ia, ib, cost = ia[keep], ib[keep], cost[keep]

    if len(ia):
        graph = coo_matrix((np.ones(len(ia)), (ia, na + ib)), shape=(na + nb, na + nb))
        ncomp, label = connected_components(graph, directed=False)
        elab = label[ia]
        single = np.bincount(elab, minlength=ncomp)[elab] == 1
        out_a, out_b = [ia[single]], [ib[single]]
        multi = np.flatnonzero(~single)
        if len(multi):
            multi = multi[np.argsort(elab[multi], kind='stable')]
            for grp in np.split(multi, np.flatnonzero(np.diff(elab[multi])) + 1):
                ua, ca = np.unique(ia[grp], return_inverse=True)
                ub, cb = np.unique(ib[grp], return_inverse=True)
                cmat = np.full((len(ua), len(ub)), 1e6)
                np.minimum.at(cmat, (ca, cb), cost[grp])
                ri, ci = linear_sum_assignment(cmat)
                ok = cmat[ri, ci] < 1e5
                out_a.append(ua[ri[ok]])
                out_b.append(ub[ci[ok]])
        t1a, t1b = np.concatenate(out_a), np.concatenate(out_b)

    n_a, n_b = _unique_nearest_pairs(a['sc'], b['sc'], match_radius)
    free = ~np.isin(n_a, t1a) & ~np.isin(n_b, t1b)
    ia_all = np.concatenate([t1a, n_a[free]]).astype(int)
    ib_all = np.concatenate([t1b, n_b[free]]).astype(int)
    tier = np.concatenate([np.ones(len(t1a), np.int16), np.full(int(free.sum()), 2, np.int16)])
    return ia_all, ib_all, tier


def local_flux_noise(ra_deg, dec_deg, flux, flux_err, cell_arcsec=8.0, min_rows=8,
                     sn_range=(5.0, 20.0)):
    """Local 1-sigma flux noise at every row: the median ``flux_err`` of the
    ``sn_range`` S/N rows in the row's ``cell_arcsec`` cell, smoothed with the
    median of the 3x3 neighbouring cells (cells with fewer than ``min_rows``
    contribute nothing; rows outside every populated neighbourhood get the
    global median).  All-NaN when too few rows are in ``sn_range``.

    The median over S/N 5-20 rows reads off the noise a faint star would have
    HERE (crowding, background) and, unlike a row's own ``flux_err``, is not
    inflated by a group fit that exhausted its LM budget and lost its
    covariance (photutils flag bit 16): such a bright star has
    ``flux / flux_err`` < 5 only because ``flux_err`` is wrong.
    """
    from scipy.ndimage import generic_filter
    from scipy.stats import binned_statistic_2d
    ra, dec = np.asarray(ra_deg, float), np.asarray(dec_deg, float)
    flux, ferr = np.asarray(flux, float), np.asarray(flux_err, float)
    out = np.full(len(ra), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        sn = flux / ferr
    use = np.isfinite(sn) & (ferr > 0) & (sn > sn_range[0]) & (sn < sn_range[1])
    fin = np.isfinite(ra) & np.isfinite(dec)
    if use.sum() < min_rows or not fin.any():
        return out
    ra0, dec0 = np.mean(ra[fin]), np.mean(dec[fin])
    x = (((ra - ra0 + 180.0) % 360.0) - 180.0) * np.cos(np.deg2rad(dec0)) * 3600.0
    y = (dec - dec0) * 3600.0
    xe = np.arange(x[fin].min() - cell_arcsec, x[fin].max() + 2 * cell_arcsec, cell_arcsec)
    ye = np.arange(y[fin].min() - cell_arcsec, y[fin].max() + 2 * cell_arcsec, cell_arcsec)
    u_ = use & fin
    med = binned_statistic_2d(x[u_], y[u_], ferr[u_], 'median', bins=[xe, ye])[0]
    cnt = binned_statistic_2d(x[u_], y[u_], ferr[u_], 'count', bins=[xe, ye])[0]
    med = np.where(cnt >= min_rows, med, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN neighbourhoods
        smooth = generic_filter(med, np.nanmedian, size=3, mode='nearest')
    glob = np.median(ferr[u_])
    ix = np.clip(((x[fin] - xe[0]) // cell_arcsec).astype(int), 0, len(xe) - 2)
    iy = np.clip(((y[fin] - ye[0]) // cell_arcsec).astype(int), 0, len(ye) - 2)
    v = smooth[ix, iy]
    out[fin] = np.where(np.isfinite(v), v, glob)
    return out


# merge_catalogs.py writes std_ra/std_dec as the scatter of avg_ra/avg_dec,
# built as SkyCoord(..., unit=(u.deg, u.deg)): the columns are in DEGREES and
# carry no FITS unit.  std_ra is coordinate-RA scatter (no cos dec).
STD_DEG_TO_ARCSEC = 3600.0
# Whether the RA axis is scaled by cos(dec) to an on-sky error.  The PM code
# works in on-sky tangent offsets, so the scaling is the consistent choice
# (13% at the GC), but it changes the RA errors that build_treasury_pm has
# produced so far.  Off until that policy question (jwst-gc-pipeline#922) is
# settled; both call sites read this one switch.
STD_RA_APPLY_COSDEC = False


def std_columns_to_arcsec(t, filt, dec_deg=None, apply_cosdec=False):
    """Per-axis position scatter ``(ex, ey)`` in ARCSEC from ``std_ra``/``std_dec``.

    The bare ``std_ra``/``std_dec`` column is used when present, else
    ``std_ra_{filt}``/``std_dec_{filt}`` (so a table carrying both is read from
    the bare pair).  Columns with no unit are degrees, see
    ``STD_DEG_TO_ARCSEC``; a column that carries an angular unit is converted
    from that unit, and one with a non-angular unit raises
    ``astropy.units.UnitConversionError`` rather than being multiplied by 3600
    as if it were degrees.  With ``apply_cosdec`` the RA axis is multiplied by
    cos(``dec_deg``), converting the coordinate scatter to an on-sky one;
    ``dec_deg`` (per-row array or scalar, degrees) is then required.

    Entries that are absent, non-finite or not positive come back as NaN, so
    the caller chooses its own floor.  That covers the exact 0.0 that older
    catalogues carry for the degenerate-position guard in merge_catalogs.py.
    """
    def _get(base):
        for name in (base, f'{base}_{filt}'):
            if name in t.colnames:
                col = t[name]
                unit = getattr(col, 'unit', None)
                if isinstance(col, u.Quantity):
                    vals = np.asarray(col.value, float)
                else:
                    vals = np.asarray(np.ma.filled(col.astype(float), np.nan), float)
                if unit is None or unit == u.dimensionless_unscaled:
                    return vals * STD_DEG_TO_ARCSEC
                return (vals * unit).to_value(u.arcsec)
        return np.full(len(t), np.nan)

    ex = _get('std_ra')
    ey = _get('std_dec')
    if apply_cosdec:
        if dec_deg is None:
            raise ValueError('apply_cosdec=True needs dec_deg')
        ex = ex * np.cos(np.deg2rad(np.asarray(dec_deg, float)))
    ex = np.where(np.isfinite(ex) & (ex > 0), ex, np.nan)
    ey = np.where(np.isfinite(ey) & (ey > 0), ey, np.nan)
    return ex, ey


def load_jwst_m7(path, ref_filter='f200w'):
    """Load a per-obs m7 cross-band catalog -> dict with sky + errors + mag + epoch.

    ``ex``/``ey`` are in arcsec, converted from the degree-valued
    ``std_ra_/std_dec_{ref_filter}`` columns (see ``std_columns_to_arcsec``,
    which reads a bare ``std_ra``/``std_dec`` first if the table has one).
    Rows with no measured scatter (absent, NaN, non-positive) take a 5 mas
    floor.  That floor is not small next to the measured scatter: in the brick
    o001 m7 catalogue the median measured scatter is 1.7 mas (RA) / 1.3 mas
    (Dec) in F182M but 5.3 / 4.0 mas in F212N, so floor rows are penalised
    against measured rows in F182M and not in F212N.  With a floor on both epochs and a 1.2 yr baseline
    pm_err = hypot(5, 5) mas / 1.2 yr = 5.9 mas/yr, above the default ``err_cap_mas=3``
    cut, so floor rows never pass the ``trustworthy`` error cut.
    ``std_ra`` is the scatter of the per-exposure positions, not the error of
    their mean (that would be smaller by about sqrt(nmatch)); the error model
    is unchanged from before the unit fix.
    """
    t = Table.read(path)
    sc = SkyCoord(t['skycoord_ref'])
    mag = np.asarray(t[f'mag_ab_{ref_filter}'], float)
    # per-frame scatter of the position (arcsec); fall back to a 5 mas floor
    ex, ey = std_columns_to_arcsec(t, ref_filter, dec_deg=sc.dec.deg,
                                   apply_cosdec=STD_RA_APPLY_COSDEC)
    ex = np.where(np.isfinite(ex), ex, 0.005)
    ey = np.where(np.isfinite(ey), ey, 0.005)
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
              maxsep=3.0, min_pairs=30, nsigma=3.0):
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

    KNOWN LIMITATION (pr-reviewer PR #140 round 5): pairs are found ONCE,
    before the affine fit, and never re-paired against the fitted plane. A
    star near the field edge under a large enough rotation/scale term could
    have its true counterpart pushed outside ``match_radius`` at the time of
    pairing even though the FITTED solution would place it correctly --
    dropping it from the fit rather than mis-fitting it. Unlikely to matter
    for JWST<->VIRAC/JWST<->JWST ties (the rotation/scale terms measured on
    real GC data so far are small), but a field with a genuinely large
    relative rotation could lose edge stars this way.

    Raises
    ------
    TieNotVerifiedError
        No coherent bulk tie between the magcut-selected samples (widen
        ``magcut``, check the input WCS, or pass a larger ``maxsep``); a
        bulk tie was found but fewer than ``min_pairs`` unambiguous same-star
        pairs survived to fit the affine plane from; or more than
        ``_DUP_ROW_FRACTION_LIMIT`` of the matched pairs are BOTH
        positionally coincident and flux-identical after the verified bulk
        tie -- the signature of literal duplicate-row contamination between
        src and ref (jwst-gc-pipeline#958), which a translation-only
        coherence check cannot detect on its own (duplicated rows pile up at
        the same true offset and satisfy it trivially).
    """
    center = _field_center(src_sc)
    sb = src_mag < magcut
    rb = ref_mag < magcut
    ssc, rsc = src_sc[sb], ref_sc[rb]

    global_res = measure_offset(ssc, rsc, maxsep=maxsep * u.arcsec, sweep=True,
                                confirm_windows=True,
                                context=f"affine_tie(magcut={magcut:g}, n_src={len(ssc)}, n_ref={len(rsc)})")
    if global_res is None or not global_res.get('ok'):
        if global_res is None:
            reason = "no window had enough pairs"
        elif global_res.get('alias_rejected'):
            reason = ("the peak did not reproduce at an independent window "
                      "(footprint geometry, not a tie)")
        else:
            reason = refusal_reason(global_res, DEFAULT_MIN_CONTRAST)
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

    # A verified translational tie (measure_offset) cannot by itself rule out
    # literal duplicate-row contamination between src and ref: duplicated
    # rows pile up at the SAME true offset and satisfy translational
    # coherence trivially, often with an inflated contrast (see
    # jwst-gc-pipeline#958 -- pr-reviewer PR #140 round 5). Check directly:
    # a matched pair this close AND this flux-identical is the fingerprint of
    # a literal duplicate detection, not two independent measurements of the
    # same star.
    #
    # Two separation measures, either one enough (pr-reviewer PR #140 round
    # 7): ``resid_mas`` is separation AFTER subtracting the GLOBAL bulk
    # offset estimate, which only reads near-zero for a duplicate pair when
    # duplicates are common enough to dominate that estimate -- a MINORITY of
    # duplicates under a real few-mas tie sits at resid_mas ~= -(the real
    # offset), not near zero, and was missed entirely (measured on real
    # o105-vs-o108 data the majority-contamination case DID raise, but a
    # synthetic 5-30% duplicate fraction under a small real offset did not,
    # frac reported 0.000 against a true raw-frame fraction of 0.05-0.32).
    # ``raw_sep_mas`` -- separation BEFORE any bulk-offset correction -- is
    # what actually reads near-zero for a literal duplicate regardless of the
    # real tie's size, since a copied row carries no offset from its source
    # at all. Checking both catches the majority- and minority-contamination
    # cases alike.
    smag_b, rmag_b = src_mag[sb], ref_mag[rb]
    resid_mas = rmap['pairs']['resid_mas']
    raw_sep_mas = ssc[ia].separation(rsc[ib]).to(u.mas).value
    dmag = np.abs(smag_b[ia] - rmag_b[ib])
    dup_suspect = (dmag < _DUP_ROW_DMAG) & (
        (resid_mas < _DUP_ROW_SEP_MAS) | (raw_sep_mas < _DUP_ROW_SEP_MAS))
    frac_dup = float(dup_suspect.sum()) / len(ia)
    if frac_dup > _DUP_ROW_FRACTION_LIMIT:
        raise TieNotVerifiedError(
            f"affine_tie: {100 * frac_dup:.1f}% of {len(ia)} matched pairs are "
            f"positionally coincident (<{_DUP_ROW_SEP_MAS:g} mas, raw OR post-tie "
            f"residual) AND flux-identical (|dmag|<{_DUP_ROW_DMAG:g}) -- the signature "
            "of literal duplicate-row contamination between src and ref (see "
            "jwst-gc-pipeline#958: nominally-independent observations sharing "
            "bit-identical detections), not real astrometric agreement. Investigate "
            "the input catalogs (row-level RA/Dec/flux comparison) before trusting "
            "this tie.")

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
                global_tie_window_arcsec=float(global_res['window_arcsec']),
                # monitoring for the jwst-gc-pipeline#958-class failure even
                # below the raise threshold -- see _DUP_ROW_FRACTION_LIMIT
                dup_row_suspect_fraction=frac_dup)
    return SkyCoord(new_ra * u.deg, new_de * u.deg), diag


def shift_to_virac_frame(cat, virac, to_epoch, match_radius=0.2, magcut=15.0,
                         maxsep=3.0, min_pairs=30, nsigma=3.0):
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
                            same_star_radius=0.05, src_isolation='flux',
                            assign='nearest', rescued_trustworthy=False):
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

    ``src_isolation`` sets the source-epoch half of the isolation cut, within
    3 x the isolation beam. 'flux' (default) rejects a star only when some
    other src row is brighter than ``flux_ratio_cap`` x its own src flux;
    neighbours at contrast >= 1/flux_ratio_cap (10 by default) do not count.
    'any' is the previous rule: any other src row in the beam rejects the
    star. Output columns: ``src_comp_flux_ratio`` (brightest other src row /
    own src flux; 0 when there is none), ``n_src_within_beam`` (src rows in
    the beam including the star itself, any brightness) and
    ``meta['src_isolation']``.

    ``assign`` picks the src-ref pairing.  'nearest' (default) is
    :func:`_unique_nearest_pairs`, which drops every claimant of a ref row two
    src rows both call nearest (4.4% of the covered Sgr B2 src stars).
    'mincost' is :func:`_gated_mincost_pairs`: a position + flux gated
    one-to-one assignment that keeps the claimant that fits, then the same
    nearest pairs for whatever it left alone; adds ``assign_tier`` (1 = gated
    assignment, 2 = nearest pair outside the gate) and ``meta['assign']``.

    Rows loaded with ``rescue_k`` (``build_treasury_pm._load_one``) carry
    ``err_rescued``; the output then has ``src_err_rescued`` /
    ``ref_err_rescued``, and a pair with either set is not ``trustworthy``
    unless ``rescued_trustworthy`` is True.
    """
    if src_isolation not in ('flux', 'any'):
        raise ValueError(f"src_isolation must be 'flux' or 'any', got {src_isolation!r}")
    if assign not in ('nearest', 'mincost'):
        raise ValueError(f"assign must be 'nearest' or 'mincost', got {assign!r}")
    from flystar.startables import StarTable
    center = _field_center(src['sc'])
    if assign == 'mincost':
        ia, ib, tier = _gated_mincost_pairs(src, ref, match_radius)
    else:
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
    if 'sc_raw' in ref:
        # Untied ref-frame position (the frame the Treasury HiPS/catalogs live in);
        # ra0/dec0 are in the src frame after the per-obs affine tie.
        raw = ref['sc_raw'][idx][selidx]
        out['ra_ref'] = raw.ra.deg
        out['dec_ref'] = raw.dec.deg
        if 'obs_index' in ref:
            out['ref_obs_index'] = np.asarray(ref['obs_index'])[idx][selidx]
    # Saturated-star-replacement flags (catalog ``replaced_saturated``), when
    # the caller loaded them: a saturated star's position comes from a
    # separate fit (saturated_star_finding), so a bias in that fit shows up
    # as a PM offset of the saturated population alone. build_treasury_pm's
    # saturation_pm_offset reads these columns.
    if 'sat' in src:
        out['src_saturated'] = np.asarray(src['sat'], bool)[selidx]
    if 'sat' in ref:
        out['ref_saturated'] = np.asarray(ref['sat'], bool)[idx][selidx]
    if assign == 'mincost':
        # int16, not int8: astropy writes an int8 column to FITS as a logical (L)
        # column, which reads back as all-True and loses the tier.
        tier_all = np.zeros(src['n'], np.int16)
        tier_all[ia] = tier
        out['assign_tier'] = tier_all[selidx]
    rescued = np.zeros(len(selidx), bool)
    if 'err_rescued' in src:
        out['src_err_rescued'] = np.asarray(src['err_rescued'], bool)[selidx]
        rescued |= out['src_err_rescued']
    if 'err_rescued' in ref:
        out['ref_err_rescued'] = np.asarray(ref['err_rescued'], bool)[idx][selidx]
        rescued |= out['ref_err_rescued']

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
        # Source-epoch side: a src neighbour disqualifies the star only if it
        # is brighter than flux_ratio_cap x the star's own src flux (contrast
        # < 1/flux_ratio_cap), the same test the ref side applies. Counting
        # every src neighbour regardless of brightness (src_isolation='any',
        # the pre-2026-10-20 rule) let a deeper src catalogue reject most of a
        # crowded field: the Arches v6fix src, built on the faint-star stack,
        # kept 1,339 trustworthy stars that way against 6,025 in v6.
        # Non-finite neighbour flux counts as competing (unknown brightness).
        ss_i, ss_j, _, _ = search_around_sky(src['sc'][sel], src['sc'], beam * 3 * u.arcsec)
        n_src_near = np.bincount(ss_i, minlength=int(sel.sum()))
        other = ss_j != selidx[ss_i]
        src_flux = np.asarray(src['flux'], dtype=float)
        nb_flux = src_flux[ss_j[other]]
        nb_flux = np.where(np.isfinite(nb_flux), nb_flux, np.inf)
        src_comp = np.zeros(int(sel.sum()))
        np.maximum.at(src_comp, ss_i[other], nb_flux)
        own = src_flux[selidx]
        own_ok = np.isfinite(own) & (own > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            src_frac = np.where(src_comp == 0, 0.0,
                                np.where(own_ok, src_comp / np.where(own_ok, own, 1.0), np.inf))
        src_isolated = (src_frac <= flux_ratio_cap) if src_isolation == 'flux' else (n_src_near == 1)
        dominant = (frac < flux_ratio_cap) & src_isolated
        out['comp_flux_ratio'] = frac
        out['src_comp_flux_ratio'] = src_frac
        out['n_src_within_beam'] = n_src_near
        out['dominant'] = dominant
        out.meta['src_isolation'] = src_isolation
        trust = trust & dominant
    if not rescued_trustworthy:
        trust = trust & ~rescued
    out['trustworthy'] = trust
    out.meta['assign'] = assign
    out.meta['epochs'] = [float(src['epoch']), float(ref['epoch'])]
    out.meta['baseline_yr'] = float(dt)
    return out
