"""Visit-consensus astrometry — per-exposure verification against the visit.

The recurring astrometric corruption mode is a SINGLE exposure (or visit) sitting
off from its neighbours while every field-average check reads ~0 (brick-1182
v001 ~20", the 2221 offsets-table collapse, the CRDS module swap...).  The
defense implemented here: after the first per-frame photometry pass (m1), build
a MERGED consensus catalog per (visit, filter) from reliable stars across all of
that visit's exposures, then re-measure each exposure's bulk offset against the
visit consensus.  A per-exposure disagreement > ``EXPOSURE_CONSENSUS_TOL_MAS``
means the im0 alignment of that exposure is wrong and must be replaced (see
``astrometry_checkpoint``).

The consensus is then tied to the absolute reference (VIRAC2/Gaia) with MULTIPLE
independent checks (dense + sparse reference, cross-reference agreement,
per-tile map) — never a single number, never NN-median (see CLAUDE.md rule #1).

All offsets here are measured with the sanctioned, density-immune
``astrometry_offsets.measure_offset`` (offset-histogram stacking with window
sweep).  Source ASSOCIATION (building consensus positions) uses
``search_around_sky`` nearest-pair matching, which is safe because it happens
only AFTER each exposure's relative offset has been measured and removed.
"""
from collections import Counter

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from scipy.spatial import cKDTree

from .astrometry_offsets import (
    measure_offset, measure_offset_grid, agree_across_references,
    local_residual_map, GlobalTieNotVerifiedError, KDTreeReference,
    _unit_xyz, _chord,
)

# An exposure whose bulk offset from the visit consensus exceeds this is
# MISALIGNED: its im0 (first-pass) alignment must be replaced.
EXPOSURE_CONSENSUS_TOL_MAS = 2.0

# A cross-module exposure (different NIRCam module -> adjacent, ~disjoint sky
# tile) may only join a component through a GENUINE small-offset overlap, never a
# wide sweep across the ~56" module gap (the issue-158 window-edge alias).  This
# bounds the cross-module growth tie: dither/guide-star scale, no sweep.
_CROSS_MODULE_TIE_MAXSEP = 3.0 * u.arcsec

# Sweep windows for the PER-EXPOSURE tie to the visit consensus (issue #158).
# That tie is mas-scale BY CONSTRUCTION -- it removes guide-star jitter between
# exposures of one visit, not a pointing error -- so offering the histogram a
# 30" or 60" window is offering it a chance to find FIELD GEOMETRY (adjacent
# module footprints, mosaic steps) instead of jitter.  Bounded at 10", well
# below both the NIRCam module separation (~174") and the W51 module-adjacency
# ridge (~56") that produced the issue-158 alias.
#
# This does NOT weaken gross-frame detection.  When the bounded measurement finds
# no trustworthy tie the exposure is re-measured over the FULL sweep, and that
# result is adopted IF its peak reproduces at an independent window: a real 20"
# misalignment (brick-1182 v001 class) is therefore still found and flagged
# exactly as before, for the correction ceiling to rule on.  What the bound
# removes is the OTHER case -- a wide peak that does not reproduce, which stays a
# recorded ``gross_diagnostic`` and a loud could-not-verify instead of becoming a
# per-exposure correction.
PER_EXPOSURE_SWEEP_WINDOWS = (3.0, 10.0)

# --- module antisymmetry (issue #158 backstop) ------------------------------
# Real per-exposure jitter is COMMON-MODE across the detectors of one exposure:
# every detector rides the same guide-star pointing error.  An ALIAS that puts
# each module onto the other module's stars (or a consensus frame corrupted by
# one such tie, then re-centred on the MEDIAN of the per-exposure offsets) reads
# ANTI-symmetric: module A at +d, module B at -d.  W51 F335M/F360M/F405N read
# exactly that at ~28" per module, with the two modules ~56" apart -- the
# adjacency ridge of their footprints, not any pointing error.
#
# The magnitude floor is what makes this specific.  A genuine per-module
# astrometric split (SIAF/distortion residual) is TENS OF MAS -- W51 F480M and
# F410M both carry a real ~35 mas module split in exactly this antisymmetric
# shape, and must NOT be flagged.  The floor is set at the per-exposure
# correction ceiling (astrometry_checkpoint.MAX_CORRECTION_ARCSEC = 0.5"), so
# the guard can only ever fire on a correction that is ALREADY refused as
# unappliable: it takes nothing away, it only replaces an opaque hard stop with
# a specific, recorded diagnosis.
MODULE_ANTISYMMETRY_MIN_MAS = 500.0
# cos of the angle between the two module vectors; -1 = exactly opposed.
MODULE_ANTISYMMETRY_COS_MAX = -0.9
# fractional magnitude agreement required between the two module vectors
MODULE_ANTISYMMETRY_MAG_TOL = 0.3

# Same-star bulk refinement (memory: histogram-vs-samestar-offset-bias): the
# all-pairs offset HISTOGRAM (check A) is biased by several mas against a DENSE
# reference -- two catalogs tracing the same clustered field make a non-uniform
# wrong-pair background that pulls the peak (brick F212N: histogram 9.3 mas / dra
# -6.2 vs same-star 5.7 / dra ~0; the bias sign even flips between filters). Once
# check A has VERIFIED the tie is small, the reported bulk is refined with a
# single-cell same-star (matched-pair) residual -- the sanctioned local_residual_map
# pattern (nearest pair is the RIGHT star only after a verified small global tie),
# NOT ad-hoc dense NN-median. Falls back to the histogram when the tie is large
# (gross-misalignment detection must stay density-immune).
SAMESTAR_MATCH_RADIUS = 0.3 * u.arcsec
SAMESTAR_MIN_PAIRS = 30

# GC reference-frame policy (memory: gc-gaia-frame-not-catalog, user directive
# 2026-07-16): in the Galactic Center, Gaia DR3 defines the absolute FRAME but its
# CATALOG is NEVER the reference catalog -- it is far too sparse (Brick footprint:
# ~1.8k Gaia vs ~113k VIRAC2). VIRAC2 (density-immune histogram tie) is the correct
# GC reference catalog. A direct Gaia<->VIRAC2 crossmatch over the whole Brick
# footprint (same physical stars, no JWST) shows the two frames AGREE to ~2.3 mas
# globally, with only ~5-10 mas spatially-varying local wander and ~40 mas per-star
# VIRAC precision scatter. Therefore the sparse-Gaia cross-check must NOT block a
# VIRAC-coherent tie at the ~5-10 mas level -- that "disagreement" is JWST-side
# (population/crowding-dependent peak against few bright Gaia stars), not a catalog
# conflict.
#
# REFERENCE_AGREE_TOL_MAS: the FINE cross-reference tolerance, kept as a recorded
# DIAGNOSTIC only (reported, never gates apply_ok).
REFERENCE_AGREE_TOL_MAS = 5.0
# REFERENCE_CROSSCHECK_GROSS_MAS: the GROSS cross-reference tolerance that DOES gate
# apply_ok. It exists solely to catch a spurious/window-limited VIRAC peak (a real
# rigid tie is reference-independent; a window-limited artifact is not -- brick-1182
# v001 read VIRAC 2.7" vs Gaia 2.0", ~700 mas apart, the tell the true offset was
# ~20" outside the window). It must sit far above the ~5-10 mas sparse-Gaia noise
# regime so Gaia sparseness can never block a good VIRAC tie.
REFERENCE_CROSSCHECK_GROSS_MAS = 100.0

# VIRAC2 is a Ks-selected survey; Ks pivot wavelength (um).  The JWST filter
# closest to this is the most reliable absolute anchor for cross-filter checks.
VIRAC2_KS_UM = 2.149

# JWST filters whose bandpass overlaps the VIRAC2 J/H/Ks coverage — for these a
# magnitude-windowed (flux-cut) matched check against VIRAC2 is meaningful.
VIRAC2_OVERLAP_UM = (1.0, 2.5)


class DuplicateExposureError(RuntimeError):
    """Two per-frame catalogs claim the SAME exposure identity.

    Distinct from a plain ``ConsensusBuildError``: that means "this visit is too
    sparse / too broken to verify", which a caller may reasonably record as
    unverified and move on.  This means the INPUT SET is malformed -- the
    checkpoint would be measuring a blend of two catalogs of one exposure -- and
    must not be downgraded to "could not verify", or the gate quietly stops
    existing (the same failure shape as the `_group_` glob leak itself).
    """


class ConsensusBuildError(RuntimeError):
    """Raised when a visit consensus cannot be built (too few exposures/stars,
    or an exposure has no measurable tie to the anchor)."""


def filter_wavelength_um(filtername):
    """Approximate pivot wavelength (um) from a JWST filter name (F212N -> 2.12,
    F410M -> 4.10, F770W -> 7.70)."""
    name = str(filtername).strip().upper()
    if not name.startswith("F") or len(name) < 4 or not name[1:4].isdigit():
        raise ValueError(f"cannot parse wavelength from filter name {filtername!r}")
    return int(name[1:4]) / 100.0


def pick_reference_anchor_filter(filternames):
    """The filter closest in wavelength to VIRAC2 Ks — the most reliable
    absolute anchor for the cross-filter astrometry check."""
    return min(filternames, key=lambda f: abs(filter_wavelength_um(f) - VIRAC2_KS_UM))


def catalog_coords(tbl):
    """SkyCoord from a per-frame or merged catalog (skycoord / skycoord_centroid)."""
    colname = "skycoord" if "skycoord" in tbl.colnames else "skycoord_centroid"
    return SkyCoord(tbl[colname]).icrs


def select_reliable_stars(tbl, snr_min=10.0, qfit_max=0.1, require_unsaturated=True):
    """Boolean mask of astrometrically reliable stars in a daophot-basic catalog.

    Reliable = well-fit (qfit), well-detected (S/N), and not a replaced/forced
    saturated fit (satstar centroids ride on spike/core morphology, not the PSF
    peak).  Missing columns degrade gracefully (that cut is skipped) so the
    selector works on early-iteration catalogs.
    """
    n = len(tbl)
    keep = np.ones(n, dtype=bool)
    if "qfit" in tbl.colnames:
        qf = np.asarray(tbl["qfit"], dtype=float)
        keep &= np.isfinite(qf) & (qf <= qfit_max)
    if "flux_fit" in tbl.colnames and "flux_err" in tbl.colnames:
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.asarray(tbl["flux_fit"], dtype=float) / np.asarray(tbl["flux_err"], dtype=float)
        keep &= np.isfinite(snr) & (snr >= snr_min)
    if require_unsaturated and "replaced_saturated" in tbl.colnames:
        rs = np.asarray(tbl["replaced_saturated"])
        keep &= ~(rs.astype(bool))
    return keep


def _meta_lookup(tbl, *names, default=None):
    for name in names:
        for key in (name, name.upper(), name.lower(), name.capitalize()):
            if key in tbl.meta:
                return tbl.meta[key]
    return default


def exposure_key(tbl):
    """(visit, exposure, module, filter, vgroup) identity of a per-frame catalog.

    VGROUP is part of the identity: a visit can dither across several visit
    groups (different sky tiles), and the exposure number restarts within each.
    Without it, cloudc F182M/nrcb1 collapses 16 catalogs onto 8 keys -- two
    physically disjoint pointings merged into one "exposure", then summed onto a
    single offsets-table row.  gc2211 (6 vgroups), sgrb2 F187N and sickle are
    affected the same way.

    Appended LAST so the positional unpacking of the first three elements
    (``key[0..2]`` -> visit/exposure/module, astrometry_checkpoint) is unchanged.
    Note this changes the tuple's shape, so per-exposure baselines recorded by an
    OLDER m2 run will not match and the frozen-stage movement check simply finds
    no baseline (degrades to "unverified", never to a false pass).
    """
    visit = _meta_lookup(tbl, "VISIT", "Visit", "visit")
    exposure = _meta_lookup(tbl, "EXPOSURE", "exposure")
    module = _meta_lookup(tbl, "MODULE", "module")
    filtername = _meta_lookup(tbl, "FILTER", "filter")
    vgroup = _meta_lookup(tbl, "VGROUP", "Vgroup", "vgroup")
    if exposure is not None:
        exposure = int(str(exposure)[-5:])
    return (str(visit), exposure, str(module), str(filtername), str(vgroup))


# A pooled reference (component union / parity half / final consensus) spans
# the whole visit mosaic -- multi-million stars for a 192-exposure filter --
# but an exposure can only pair with reference stars inside its own footprint
# plus the largest sweep window.  Cropping the reference to that bounding box
# before measure_offset is geometrically LOSSLESS for the offset histogram
# (pairs outside it cannot exist at any sweep window) and keeps every KD tree
# O(local density) instead of O(mosaic): without it each measure rebuilt
# multi-million-point trees 8x (4 windows x probe+main) and F182M m2 spent
# ~15 min per exposure (2026-07-13).
_CROP_PAD_ARCSEC = 70.0   # > max(DEFAULT_SWEEP_WINDOWS) = 60"


def _crop_to_footprint(ref, target, pad_arcsec=_CROP_PAD_ARCSEC):
    pad = pad_arcsec / 3600.0
    tdec = target.dec.deg
    dec_lo, dec_hi = float(tdec.min()) - pad, float(tdec.max()) + pad
    cosd = max(np.cos(np.radians(np.clip((dec_lo + dec_hi) / 2.0, -89.9, 89.9))),
               1e-3)
    tra = target.ra.deg
    ra_lo, ra_hi = float(tra.min()) - pad / cosd, float(tra.max()) + pad / cosd
    if ra_hi - ra_lo > 180.0:
        # footprint straddles the RA wrap: the box test is invalid there
        return ref
    sel = ((ref.dec.deg >= dec_lo) & (ref.dec.deg <= dec_hi)
           & (ref.ra.deg >= ra_lo) & (ref.ra.deg <= ra_hi))
    n = int(sel.sum())
    if n == len(ref) or n < 100:
        # no/negligible boxed overlap: keep the full reference so the caller's
        # too-few-pairs / unverified semantics are exactly as before
        return ref
    return ref[sel]


def _cap_stars(sc, n_max=500_000, seed=1182):
    """Deterministic uniform subsample.  The offset histogram peak is a
    density estimate: a uniform subsample preserves its location and its
    contrast statistics (same argument as MAX_PAIRS_PER_WINDOW)."""
    if len(sc) <= n_max:
        return sc
    rng = np.random.default_rng(seed)
    return sc[np.sort(rng.choice(len(sc), n_max, replace=False))]


def build_visit_consensus(exposure_tables, snr_min=10.0, qfit_max=0.1,
                          match_radius=0.2 * u.arcsec, min_exposures=2,
                          min_stars=50, context=""):
    """Build the per-(visit,filter) consensus catalog and measure every
    exposure's bulk offset against it.

    Flow (all bulk offsets via the density-immune ``measure_offset`` + sweep).

    MOSAIC-SAFE (2026-07-12): a real visit is usually a MOSAIC — its exposures
    span several non-overlapping pointing tiles (vgroups) and two modules, so
    "tie everything to one anchor" is impossible by GEOMETRY, not by
    misalignment.  The consensus is therefore built per overlap COMPONENT:

    1. cut each exposure catalog to reliable stars;
    2. component seed = the largest not-yet-tied exposure; grow the component
       by tying exposures to the UNION of the component's already-tied,
       offset-removed stars (sweep on: a grossly shifted exposure is still
       FOUND no matter how large the shift); repeat until no exposure ties,
       then start the next component from the leftovers;
    3. within each component: associate stars across its exposures (nearest
       pair within ``match_radius`` — unambiguous because the relative offsets
       were removed), consensus position = per-star mean over >=
       ``min_exposures`` exposures, re-centred by the MEDIAN of the
       component's per-exposure offsets (one bad exposure cannot drag it);
    4. the visit consensus = the union of all component consensi.  Components
       cannot be tied to EACH OTHER internally (no shared stars) — their
       mutual consistency is exactly what the per-tile REFERENCE map in
       ``measure_reference_tie`` checks;
    5. re-measure every exposure against the consensus.  An exposure whose
       footprint has no measurable consensus overlap is UNVERIFIED (a
       could-not-verify, reported distinctly) — never silently passed, never
       conflated with a measured misalignment.

    Returns
    -------
    dict with:
      ``coords`` : SkyCoord — consensus positions (component-mean frames)
      ``nexp`` : ndarray — exposures contributing per star
      ``scatter_mas`` : ndarray — per-star rms scatter of contributing positions
      ``exposures`` : list of per-exposure dicts:
          ``key`` (visit, exposure, module, filter), ``n_reliable``,
          ``component`` (int; -1 = untied island), ``internal_tie`` (bool),
          ``vs_consensus`` (measure_offset result or None), ``unverified``
          (bool — no measurable tie to the consensus), ``misaligned`` (off >
          tol AND significant), ``raoffset_meta``/``deoffset_meta`` (the im0
          alignment baked into the catalog, arcsec)
      ``n_components`` : int
      ``consensus_ok`` : bool — every exposure measurable vs the consensus
    """
    if len(exposure_tables) < min_exposures:
        raise ConsensusBuildError(
            f"visit consensus ({context}): need >= {min_exposures} exposures, "
            f"got {len(exposure_tables)}")

    entries = []
    for tbl in exposure_tables:
        keep = select_reliable_stars(tbl, snr_min=snr_min, qfit_max=qfit_max)
        _all_coords = catalog_coords(tbl)
        # A "reliable" star (good snr/qfit) can still carry a non-finite RA/Dec --
        # e.g. a saturated-core replacement / recovered row whose position solve
        # failed.  It must NOT enter the consensus: these coords are concatenated
        # straight into a cKDTree (parity-halves path) and np.median (dec_mid),
        # both of which raise on NaN/inf ("data must be finite").  Drop non-finite
        # positions here so n_reliable, coords, and flux stay consistent and every
        # downstream tie sees only finite coordinates.
        _finite = np.isfinite(_all_coords.ra.deg) & np.isfinite(_all_coords.dec.deg)
        keep = np.asarray(keep, dtype=bool) & _finite
        coords = _all_coords[keep]
        if "flux_fit" in tbl.colnames:
            flux = np.asarray(tbl["flux_fit"], dtype=float)[keep]
        else:
            flux = np.full(int(keep.sum()), np.nan)
        entries.append(dict(
            key=exposure_key(tbl), coords=coords, flux=flux,
            n_reliable=int(keep.sum()),
            raoffset_meta=_meta_lookup(tbl, "RAOFFSET", default=0.0),
            deoffset_meta=_meta_lookup(tbl, "DEOFFSET", default=0.0)))

    # Two catalogs sharing an identity are two measurements of the SAME exposure
    # blended into one consensus entry.  That is never legitimate, and it is how
    # the arches/sgrc m2 corrections were fabricated: the duplicated exposures
    # straddled the consensus and read 20-45 mas off, so the checkpoint
    # "corrected" a bookkeeping artifact into the offsets table.  Fail loudly
    # rather than silently averaging -- a silent merge is indistinguishable from
    # a real misalignment downstream.
    dupes = {k: n for k, n in Counter(e["key"] for e in entries).items() if n > 1}
    if dupes:
        raise DuplicateExposureError(
            f"visit consensus ({context}): {len(dupes)} exposure identity/ies "
            f"ingested more than once -- duplicate per-frame catalogs for the "
            f"same (visit, exposure, module, filter, vgroup): "
            f"{sorted(dupes.items())[:8]}.  Check for grouped-fit (`_group_`) "
            f"or stale module-level variants alongside the per-detector ones.")

    usable_idx = [i for i, e in enumerate(entries) if e["n_reliable"] >= min_stars]
    usable = [entries[i] for i in usable_idx]
    if len(usable) < min_exposures:
        raise ConsensusBuildError(
            f"visit consensus ({context}): only {len(usable)} exposures have >= "
            f"{min_stars} reliable stars (of {len(entries)})")

    dec_mid = float(np.median(np.concatenate([e["coords"].dec.deg for e in usable])))
    cosd = max(np.cos(np.radians(dec_mid)), 1e-6)

    def _shift(sc, dra_mas, ddec_mas):
        return SkyCoord(ra=sc.ra + (dra_mas / 3.6e6 / cosd) * u.deg,
                        dec=sc.dec + (ddec_mas / 3.6e6) * u.deg, frame="icrs")

    # 2. relative tie.  TWO paths:
    #    - small visits (<= 16 usable exposures): component-wise UNION GROWTH
    #      (robust when a tile holds only 2-3 exposures -- a pooled half could
    #      be dominated by a single bad exposure there; O(n^2) is cheap at
    #      this size);
    #    - large visits: PARITY-HALVES (2026-07-13) -- the growth loop is
    #      O(n^2) measure_offset calls each rebuilding a KD tree on a growing
    #      multi-million-star union (a 192-exposure brick filter spent 9+
    #      hours there).  Split the exposures into two fixed pooled halves
    #      (alternating by size rank so both cover the mosaic), measure each
    #      exposure ONCE against the opposite half (KD tree cached on the two
    #      fixed objects), bridge the half-frames with one half-vs-half
    #      measurement.  A bad exposure is a small minority of its half's
    #      local stars at this size, so the histogram peak stays clean.
    #      Untied exposures are ISLANDS (component -1), reference-tie only.
    rel = [None] * len(usable)
    comp_id = np.full(len(usable), -1, dtype=int)
    if len(usable) <= 16:
        remaining = set(range(len(usable)))
        n_components = 0
        while remaining:
            seed_i = max(remaining, key=lambda i: usable[i]["n_reliable"])
            comp = n_components
            comp_id[seed_i] = comp
            rel[seed_i] = dict(dra=0.0, ddec=0.0, off=0.0, ok=True, swept=False,
                               npairs=usable[seed_i]["n_reliable"],
                               contrast=float("inf"), window_arcsec=0.0,
                               dra_err=0.0, ddec_err=0.0,
                               n_peak=usable[seed_i]["n_reliable"])
            remaining.discard(seed_i)
            union = usable[seed_i]["coords"]
            # Module families already tied into this component.  A DIFFERENT
            # NIRCam module images an ADJACENT, ~disjoint sky tile (~56" away,
            # sharing ~no stars per exposure); a wide sweep across that gap locks
            # onto the footprint cross-correlation at the search-window edge --
            # the #158 alias (fake ~28-29" antisymmetric split).  Growing across
            # it also violates this builder's own invariant (components are
            # per-overlap-tile, tied to each other ONLY through the absolute
            # reference).  So a cross-module exposure may join a component ONLY
            # through a GENUINE small-offset overlap (bounded window, no sweep,
            # non-swept tie); otherwise it seeds its own component and is aligned
            # via measure_reference_tie.  Same-module exposures keep the full
            # sweep so a large single-frame WCS error (brick-1182 v001 ~20") is
            # still found.  #185's confirm_windows remains the statistical
            # backstop; this is the geometric one.
            comp_modfam = {module_family(usable[seed_i]["key"][2])}
            grew = True
            while grew and remaining:
                grew = False
                for i in sorted(remaining,
                                key=lambda j: -usable[j]["n_reliable"]):
                    _cropped = _crop_to_footprint(union, usable[i]["coords"])
                    if module_family(usable[i]["key"][2]) not in comp_modfam:
                        res = measure_offset(
                            usable[i]["coords"], _cropped,
                            sweep=False, maxsep=_CROSS_MODULE_TIE_MAXSEP,
                            context=f"{context} exp{usable[i]['key']} "
                                    f"vs component {comp} union (cross-module)")
                        if res is None or not res["ok"] or res.get("swept"):
                            continue
                    else:
                        res = measure_offset(
                            usable[i]["coords"], _cropped,
                            sweep=True, confirm_windows=True,
                            context=f"{context} exp{usable[i]['key']} "
                                    f"vs component {comp} union")
                        if res is None or not res["ok"]:
                            continue
                    rel[i] = res
                    comp_id[i] = comp
                    comp_modfam.add(module_family(usable[i]["key"][2]))
                    remaining.discard(i)
                    shifted_i = _shift(usable[i]["coords"], res["dra"], res["ddec"])
                    union = SkyCoord(
                        ra=np.concatenate([union.ra.deg, shifted_i.ra.deg]) * u.deg,
                        dec=np.concatenate([union.dec.deg, shifted_i.dec.deg]) * u.deg,
                        frame="icrs")
                    grew = True
            n_components += 1
    else:
        order = np.argsort([-e["n_reliable"] for e in usable])
        parity = np.zeros(len(usable), dtype=int)
        parity[order[1::2]] = 1
        halves = {}
        for par in (0, 1):
            sel = [i for i in range(len(usable)) if parity[i] == par]
            if sel:
                # KDTreeReference: the half's KD tree is built ONCE and reused
                # for every exposure measured against it (the per-call astropy
                # rebuild on a multi-million-star half dominated the runtime)
                halves[par] = KDTreeReference(SkyCoord(
                    ra=np.concatenate([usable[i]["coords"].ra.deg
                                       for i in sel]) * u.deg,
                    dec=np.concatenate([usable[i]["coords"].dec.deg
                                        for i in sel]) * u.deg,
                    frame="icrs"))
        half_bridge = None
        if 0 in halves and 1 in halves:
            # correction to move the ODD half onto the EVEN half's frame
            # (both halves cover the full mosaic; capped uniform subsamples
            # keep this single measure cheap without moving the peak)
            half_bridge = measure_offset(_cap_stars(halves[1].coords),
                                         _cap_stars(halves[0].coords), sweep=True,
                                         confirm_windows=True,
                                         context=f"{context} half1 vs half0")
        for i, e in enumerate(usable):
            other = halves.get(1 - parity[i])
            if other is None:
                continue
            res = measure_offset(e["coords"], other, sweep=True,
                                 confirm_windows=True,
                                 context=f"{context} exp{e['key']} vs half"
                                         f"{1 - parity[i]}")
            if res is None or not res["ok"]:
                continue
            if parity[i] == 0:
                if half_bridge is None or not half_bridge.get("ok"):
                    continue
                res = dict(res, dra=res["dra"] + half_bridge["dra"],
                           ddec=res["ddec"] + half_bridge["ddec"])
                res["off"] = float(np.hypot(res["dra"], res["ddec"]))
            rel[i] = res
            comp_id[i] = 0
        n_components = 1 if (comp_id == 0).any() else 0

    # 3. per-component association + consensus positions
    all_ra, all_dec, all_scatter, all_nexp, all_mag = [], [], [], [], []
    for comp in range(n_components):
        members = [i for i in range(len(usable)) if comp_id[i] == comp]
        # component-mean frame: MEDIAN of the member offsets, not mean -- one
        # grossly misaligned exposure must not drag the frame toward itself
        med_dra = float(np.median([rel[i]["dra"] for i in members]))
        med_ddec = float(np.median([rel[i]["ddec"] for i in members]))
        # Per-star position sums are accumulated as DELTAS from the seed star's
        # position (small degrees), NOT raw ~266 deg coordinates: sum(x^2)/n -
        # mean^2 on raw GC coordinates cancels catastrophically in float64 and
        # fabricated a ~10-15 mas noise floor in scatter_mas.
        ref_ra = ref_dec = None
        sum_dra = sum_ddec = sum_dra2 = sum_ddec2 = counts = flux0 = None
        for i in members:
            sc = _shift(usable[i]["coords"], rel[i]["dra"], rel[i]["ddec"])
            if ref_ra is None:
                ref_ra = sc.ra.deg.copy(); ref_dec = sc.dec.deg.copy()
                sum_dra = np.zeros(len(sc)); sum_ddec = np.zeros(len(sc))
                sum_dra2 = np.zeros(len(sc)); sum_ddec2 = np.zeros(len(sc))
                counts = np.ones(len(sc), dtype=int)
                flux0 = usable[i]["flux"].copy()
                seed = sc
                continue
            # nearest sc star within match_radius for each seed star.  scipy
            # tree on the SMALL member catalog + parallel query of the (large,
            # growing) seed: astropy search_around_sky rebuilt a tree on the
            # multi-million-star seed for every member (~minutes each).
            radius_arcsec = match_radius.to(u.arcsec).value
            sc_tree = cKDTree(_unit_xyz(sc))
            _, idx = sc_tree.query(_unit_xyz(seed), k=1,
                                   distance_upper_bound=_chord(radius_arcsec),
                                   workers=-1)
            found = idx < len(sc)
            matched_b = np.zeros(len(sc), dtype=bool)
            if found.any():
                ia_n = np.nonzero(found)[0]
                ib_n = idx[found]
                dra_n = sc.ra.deg[ib_n] - ref_ra[ia_n]
                ddec_n = sc.dec.deg[ib_n] - ref_dec[ia_n]
                sum_dra[ia_n] += dra_n
                sum_ddec[ia_n] += ddec_n
                sum_dra2[ia_n] += dra_n ** 2
                sum_ddec2[ia_n] += ddec_n ** 2
                counts[ia_n] += 1
                matched_b[ib_n] = True
            # unmatched stars extend the seed (mosaic tiles: coverage grows);
            # each becomes its own delta reference (delta 0 on first sighting)
            new = ~matched_b
            if new.any():
                seed = SkyCoord(
                    ra=np.concatenate([seed.ra.deg, sc.ra.deg[new]]) * u.deg,
                    dec=np.concatenate([seed.dec.deg, sc.dec.deg[new]]) * u.deg,
                    frame="icrs")
                n_new = int(new.sum())
                ref_ra = np.concatenate([ref_ra, sc.ra.deg[new]])
                ref_dec = np.concatenate([ref_dec, sc.dec.deg[new]])
                sum_dra = np.concatenate([sum_dra, np.zeros(n_new)])
                sum_ddec = np.concatenate([sum_ddec, np.zeros(n_new)])
                sum_dra2 = np.concatenate([sum_dra2, np.zeros(n_new)])
                sum_ddec2 = np.concatenate([sum_ddec2, np.zeros(n_new)])
                counts = np.concatenate([counts, np.ones(n_new, int)])
                flux0 = np.concatenate([flux0, usable[i]["flux"][new]])
        good = counts >= min_exposures
        if not good.any():
            continue
        mean_dra_deg = sum_dra[good] / counts[good]
        mean_ddec_deg = sum_ddec[good] / counts[good]
        mean_ra = ref_ra[good] + mean_dra_deg
        mean_dec = ref_dec[good] + mean_ddec_deg
        var_ra = np.maximum(sum_dra2[good] / counts[good] - mean_dra_deg ** 2, 0.0)
        var_dec = np.maximum(sum_ddec2[good] / counts[good] - mean_ddec_deg ** 2, 0.0)
        all_ra.append(mean_ra - med_dra / 3.6e6 / cosd)
        all_dec.append(mean_dec - med_ddec / 3.6e6)
        all_scatter.append(np.sqrt(var_ra * cosd ** 2 + var_dec) * 3.6e6)
        all_nexp.append(counts[good])
        with np.errstate(divide="ignore", invalid="ignore"):
            all_mag.append(-2.5 * np.log10(np.where(flux0[good] > 0,
                                                    flux0[good], np.nan)))

    n_cons = int(sum(len(r) for r in all_ra))
    if n_cons < min_stars:
        raise ConsensusBuildError(
            f"visit consensus ({context}): only {n_cons} stars matched in >= "
            f"{min_exposures} exposures across {n_components} component(s) "
            f"(need >= {min_stars})")
    consensus = SkyCoord(ra=np.concatenate(all_ra) * u.deg,
                         dec=np.concatenate(all_dec) * u.deg, frame="icrs")
    scatter_mas = np.concatenate(all_scatter)
    nexp = np.concatenate(all_nexp)
    consensus_mag = np.concatenate(all_mag)

    # 5. every exposure re-measured against the consensus.  No measurable tie
    # (an exposure whose footprint has no >=2-exposure consensus coverage) =
    # UNVERIFIED -- reported, never conflated with a measured misalignment.
    exposures = []
    consensus_ok = True
    consensus_ref = KDTreeReference(consensus)
    for pos, e in enumerate(usable):
        # BOUNDED sweep: a per-exposure tie is mas-scale by construction
        # (PER_EXPOSURE_SWEEP_WINDOWS), and any swept peak must survive an
        # independent window before it is believed (confirm_windows).
        res = measure_offset(e["coords"], consensus_ref, sweep=True,
                             sweep_windows=PER_EXPOSURE_SWEEP_WINDOWS,
                             confirm_windows=True,
                             context=f"{context} exp{e['key']} vs consensus")
        measurable = res is not None and res["ok"]
        gross = None
        if not measurable:
            # The bounded window found nothing trustworthy.  Fall back to the FULL
            # sweep so a genuinely gross frame is still FOUND -- but only adopt it
            # when the peak REPRODUCES at an independent window.  That split is the
            # whole point: a real 20" misalignment confirms (and is then flagged
            # exactly as before, for the correction ceiling to rule on), while a
            # window-edge alias does not and stays a recorded could-not-verify.
            gross = measure_offset(e["coords"], consensus_ref, sweep=True,
                                   confirm_windows=True,
                                   context=f"{context} exp{e['key']} vs consensus "
                                           f"(gross)")
            if gross is not None and gross["ok"]:
                res = gross
                measurable = True
        if not measurable:
            consensus_ok = False
        misaligned = False
        if measurable:
            err = float(np.hypot(res.get("dra_err", 0.0) or 0.0,
                                 res.get("ddec_err", 0.0) or 0.0))
            # misaligned only when BOTH large and significant: an offset with an
            # error bar bigger than itself is not a measured misalignment
            misaligned = bool(res["off"] > EXPOSURE_CONSENSUS_TOL_MAS
                              and (not np.isfinite(err) or res["off"] > 3.0 * err
                                   or res["off"] > 10.0 * EXPOSURE_CONSENSUS_TOL_MAS))
        exposures.append(dict(
            key=e["key"], n_reliable=e["n_reliable"],
            component=int(comp_id[pos]),
            internal_tie=bool(rel[pos] is not None and rel[pos]["npairs"] > 0
                              and (comp_id == comp_id[pos]).sum() > 1),
            vs_consensus=res, unverified=not measurable,
            gross_diagnostic=gross,
            misaligned=misaligned,
            raoffset_meta=float(e["raoffset_meta"] or 0.0),
            deoffset_meta=float(e["deoffset_meta"] or 0.0)))

    skipped = [e["key"] for i, e in enumerate(entries) if i not in usable_idx]
    anchor = usable[int(np.argmax([e["n_reliable"] for e in usable]))]
    return dict(coords=consensus, mag=consensus_mag, nexp=nexp,
                scatter_mas=scatter_mas, exposures=exposures,
                consensus_ok=consensus_ok, anchor_key=anchor["key"],
                n_components=n_components, skipped=skipped)


def module_family(module):
    """Detector -> NIRCam module family ('nrca1'/'nrcalong' -> 'a').

    The antisymmetry test compares MODULES, not detectors: an alias that lands
    one module on the other's stars moves all four SW detectors of that module
    together.  Anything unrecognised is returned unchanged, so a non-NIRCam or
    oddly-named module simply forms its own family (and, being alone, can never
    trip a two-family antisymmetry test).
    """
    m = str(module).strip().lower()
    if m.startswith("nrc") and len(m) > 3 and m[3] in ("a", "b"):
        return m[3]
    return m


def detect_module_antisymmetry(exposures,
                               min_mas=MODULE_ANTISYMMETRY_MIN_MAS,
                               cos_max=MODULE_ANTISYMMETRY_COS_MAX,
                               mag_tol=MODULE_ANTISYMMETRY_MAG_TOL):
    """Flag the issue-158 alias signature: per-module offsets equal and OPPOSITE.

    Per-exposure jitter is common-mode across the detectors of one exposure --
    every detector of an exposure rides the same guide-star pointing error, so
    the modules agree.  Two modules reading LARGE, equal-and-opposite offsets is
    the signature of one module tied onto the other's stars (directly, or
    through a consensus frame that a single bad tie corrupted and the median
    re-centring then split evenly between them).

    The guard keys on what IS invariant: the antisymmetry and the SCALE (see
    MODULE_ANTISYMMETRY_MIN_MAS).  The W51 alias sat at a ~56" module split --
    the lag that slides the two module FOOTPRINTS onto each other, which depends
    on the mosaic/dither pattern and so has no fixed value -- while the SIAF
    nrcalong<->nrcblong reference-point separation is ~174".  Keying on a
    hardcoded separation would have made this guard silently never fire.

    ``exposures`` is ``build_visit_consensus(...)['exposures']``.

    Returns ``dict(detected, n_pairs_tested, n_antisymmetric, keys, examples)``
    where ``keys`` is the set of exposure-key tuples involved.
    """
    by_exp = {}
    for e in exposures:
        res = e.get("vs_consensus")
        if res is None or not res.get("ok"):
            continue
        key = tuple(e["key"])
        # (visit, exposure, vgroup) -- module (key[2]) is what we compare across
        grp = (key[0], key[1], key[4] if len(key) > 4 else None)
        fam = module_family(key[2])
        by_exp.setdefault(grp, {}).setdefault(fam, []).append(
            (key, float(res["dra"]), float(res["ddec"])))

    n_tested = 0
    n_anti = 0
    flagged_keys = set()
    examples = []
    for grp, fams in sorted(by_exp.items(), key=lambda kv: str(kv[0])):
        if len(fams) != 2:
            continue
        n_tested += 1
        (fa, va), (fb, vb) = sorted(fams.items())
        a = np.array([np.mean([v[1] for v in va]), np.mean([v[2] for v in va])])
        b = np.array([np.mean([v[1] for v in vb]), np.mean([v[2] for v in vb])])
        na, nb = float(np.hypot(*a)), float(np.hypot(*b))
        if na < min_mas or nb < min_mas:
            continue                      # mas-scale split: real, not an alias
        cos = float(np.dot(a, b) / (na * nb))
        if cos > cos_max:
            continue                      # not opposed
        if abs(na - nb) > mag_tol * max(na, nb):
            continue                      # not equal in magnitude
        n_anti += 1
        for _, vals in fams.items():
            for key, _dra, _ddec in vals:
                flagged_keys.add(key)
        if len(examples) < 4:
            examples.append(dict(
                visit=grp[0], exposure=grp[1], vgroup=grp[2],
                module_a=fa, module_b=fb,
                dra_a_mas=float(a[0]), ddec_a_mas=float(a[1]),
                dra_b_mas=float(b[0]), ddec_b_mas=float(b[1]),
                separation_mas=float(np.hypot(*(a - b))), cos=cos))

    return dict(detected=bool(n_anti), n_pairs_tested=int(n_tested),
                n_antisymmetric=int(n_anti), keys=flagged_keys,
                examples=examples, min_mas=float(min_mas))


def _brightest_subset_for_spacing(coords, mag, target_spacing_arcsec):
    """Bright-magnitude cut so the subset's ESTIMATED mean source spacing
    (sqrt(footprint area / N), uniform-density estimate) is >= the target.
    Returns a boolean mask.  This is the "reasonable flux cut" that sparsifies
    a dense catalog before source-by-source matching."""
    finite = np.isfinite(mag)
    if finite.sum() == 0:
        return finite
    ra = coords.ra.deg[finite]
    dec = coords.dec.deg[finite]
    cosd = max(np.cos(np.radians(np.median(dec))), 1e-6)
    area_arcsec2 = max(
        (ra.max() - ra.min()) * cosd * 3600.0 * (dec.max() - dec.min()) * 3600.0, 1.0)
    n_max = max(int(area_arcsec2 / target_spacing_arcsec ** 2), 1)
    idx_finite = np.where(finite)[0]
    order = np.argsort(np.asarray(mag)[idx_finite])  # bright (small mag) first
    keep = np.zeros(len(coords), dtype=bool)
    keep[idx_finite[order[:n_max]]] = True
    return keep


def _magnitude_windowed_match(consensus_coords, consensus_mag, ref_coords, ref_mag,
                              global_result, match_radius=0.3 * u.arcsec,
                              min_pairs=30, sparsity_factor=3.0, context=""):
    """Careful source-by-source check against the reference, sparsified by a
    bright-flux cut (only meaningful when the JWST band overlaps VIRAC2's).

    The flux cut removes most of the ambiguity that makes dense-NN matching
    dangerous (both sides are cut until the estimated source spacing is >=
    ``sparsity_factor`` x the match radius); the residual danger is removed by
    the ``local_residual_map`` precondition — a verified, small global tie must
    already exist, so a nearest pair within ``match_radius`` is the RIGHT star.
    Returns the robust mean residual (mas) with a standard error, or None if
    not enough pairs.
    """
    radius_arcsec = match_radius.to(u.arcsec).value if hasattr(match_radius, "to") \
        else float(match_radius)
    target_spacing = sparsity_factor * radius_arcsec
    keep_c = _brightest_subset_for_spacing(consensus_coords, consensus_mag, target_spacing)
    keep_r = _brightest_subset_for_spacing(ref_coords, ref_mag, target_spacing)
    if keep_c.sum() < min_pairs or keep_r.sum() < min_pairs:
        return None
    lrm = local_residual_map(
        consensus_coords[keep_c], ref_coords[keep_r], global_result,
        cell_arcsec=1e9,  # single cell: this is the BULK flux-matched residual
        match_radius=match_radius, min_stars=min_pairs, context=context)
    if not lrm["cells"]:
        return None
    cell = max(lrm["cells"], key=lambda c: c["n"])
    return dict(dra=cell["dra_mas"], ddec=cell["ddec_mas"],
                dra_err=cell["dra_sem"], ddec_err=cell["ddec_sem"],
                off=cell["off_mas"], n=cell["n"])


def measure_reference_tie(consensus_coords, ref_coords_all, ref_coords_sparse,
                          filtername=None, consensus_mag=None, ref_mag=None,
                          agree_tol_mas=REFERENCE_AGREE_TOL_MAS,
                          gross_tol_mas=REFERENCE_CROSSCHECK_GROSS_MAS,
                          grid_nx=6, grid_ny=6, dense=True, context=""):
    """Tie the visit consensus to the absolute reference with MULTIPLE
    independent checks.  No single number signs off (CLAUDE.md).

    The reference catalog is VIRAC2-full (check A, density-immune histogram).
    Gaia-only (check B/C) is a SPARSE cross-check, NOT the reference catalog: in
    the Galactic Center Gaia defines the FRAME but is far too sparse to tie a
    dense JWST field, so it may never BLOCK a coherent VIRAC tie (memory:
    gc-gaia-frame-not-catalog).  See the ``REFERENCE_CROSSCHECK_GROSS_MAS`` note.

    Checks
    ------
    A. ``measure_offset`` vs the FULL (dense VIRAC2) reference, histogram + sweep.
       DETECTS the tie and handles large offsets, but its peak is biased several
       mas against a dense reference, so the reported bulk is refined same-star
       (see below).
    A'. same-star bulk: once A verifies a small tie, a single-cell
       ``local_residual_map`` matched-pair residual gives the UNBIASED bulk offset
       (the reported ``dra_mas``/``ddec_mas``). Falls back to A for a large/unverified
       tie. ``bulk_source`` records which was used.
    B. ``measure_offset`` vs the SPARSE reference (Gaia-only subset) — diagnostic.
    C. cross-reference agreement (A vs B).  A GROSS split (> ``gross_tol_mas``)
       means A is a spurious/window-limited peak (reference-dependent) and DOES
       block.  A fine split (~5-10 mas, the sparse-Gaia noise regime) is recorded
       but does NOT block — it is a JWST-side population effect
       (Gaia<->VIRAC2 agree ~2.3 mas over the Brick footprint).
    D. per-tile map vs the full reference (``measure_offset_grid``) — a bulk
       ~0 with a shifted half-mosaic FAILS here.
    E. (when the band overlaps VIRAC2 and magnitudes are provided) a
       flux-windowed source-by-source residual — an independent systematics
       check on the histogram peak.

    Returns
    -------
    dict with per-check results and:
      ``dra_mas``/``ddec_mas``/``dra_err_mas``/``ddec_err_mas`` — the adopted
      correction (same-star refined when the tie is small, else check-A histogram;
      on-sky mas, sign = correction to ADD to the consensus to land on the
      reference); ``bulk_source`` = ``"same-star"`` or ``"histogram"``;
      ``same_star`` = the refined matched-pair result (or None);
      ``apply_ok`` — True when A is coherent, the sparse cross-check does not
      GROSSLY disagree (C within ``gross_tol_mas``), and the per-tile check
      passes.  With a DENSE (``dense=True``) reference the per-tile check is D
      (``per_tile['clean']``).  With a Gaia-ONLY reference (``dense=False``) D is
      pure noise (see the per-tile note below) and is replaced by the same-star
      matched-pair refinement (A') succeeding -- still multi-check, since A'
      refuses unless the global tie is verified-small.  NOTE the two checks that
      survive on a Gaia-ONLY field (A' and C) are both GLOBAL statistics: D is
      the only SPATIAL one, so a localized seam -- half a mosaic displaced --
      is not detected there.  That is a real reduction in coverage, accepted
      because D against a sparse reference returns noise in every tile rather
      than catching seams.  Seam cover for those fields is the release-time
      interframe overlap gate, not this checkpoint.  The fine cross-reference
      agreement (``cross_reference['agree']``, ``agree_tol_mas``) is reported for
      diagnostics but does NOT gate apply_ok.
      An offset must never be APPLIED on a single check.
    """
    res_a = measure_offset(consensus_coords, ref_coords_all, sweep=True,
                           context=f"{context} vs full-ref")
    res_b = measure_offset(consensus_coords, ref_coords_sparse, sweep=True,
                           context=f"{context} vs sparse-ref")
    # fine cross-check (diagnostic) + gross cross-check (the only one that gates)
    agree = agree_across_references(consensus_coords, ref_coords_all,
                                    ref_coords_sparse, tol_mas=agree_tol_mas,
                                    label_a=f"{context}/full",
                                    label_b=f"{context}/sparse")
    sep_mas = agree.get("sep_mas", float("nan"))
    # gross agreement re-uses the same measured peaks (no re-measure): a real tie
    # or a mild sparse-Gaia offset both pass; only a MEASURED spurious split
    # (finite, > gross) fails.  Critically, an UNMEASURABLE sparse tie (sep_mas
    # nan -- too few Gaia stars to form a coherent peak) must NOT block: that is
    # the extreme-sparse inner-GC regime (arches/quintuplet/sgra) where Gaia is
    # untenable and this whole policy applies most.  Gating on nan would re-block
    # exactly the VIRAC tie this is meant to keep; only a finite gross split can
    # block.  (Residual risk when Gaia is unmeasurable: a spurious VIRAC peak
    # then leans entirely on check D per-tile + the sweep, since the gross-Gaia
    # backstop -- added because per-tile alone missed brick-1182 v001 at a narrow
    # window -- cannot fire without a measurable Gaia peak.  brick-1182 v001
    # itself is unaffected: Gaia WAS measurable there, sep ~700 mas, finite.)
    cross_gross_ok = bool((not np.isfinite(sep_mas)) or sep_mas <= gross_tol_mas)
    grid = measure_offset_grid(consensus_coords, ref_coords_all,
                               nx=grid_nx, ny=grid_ny,
                               context=f"{context} per-tile")

    fluxmatched = None
    if (filtername is not None and consensus_mag is not None and ref_mag is not None
            and res_a is not None and res_a.get("ok") and not res_a.get("swept")):
        lam = filter_wavelength_um(filtername)
        if VIRAC2_OVERLAP_UM[0] <= lam <= VIRAC2_OVERLAP_UM[1] and res_a["off"] < 100.0:
            fluxmatched = _magnitude_windowed_match(
                consensus_coords, np.asarray(consensus_mag, dtype=float),
                ref_coords_all, np.asarray(ref_mag, dtype=float),
                res_a, context=f"{context} flux-matched")

    # --- same-star bulk refinement (fixes the histogram's dense-reference bias) ---
    # Check A (histogram) DETECTS the tie + handles large offsets/sweep, but its
    # peak is biased several mas against dense VIRAC. Once A verifies the tie is
    # small (not swept), refine the reported bulk with a single-cell same-star
    # residual via local_residual_map -- which itself REFUSES unless the global tie
    # is verified small (so pairs are unambiguous; the sanctioned refinement, not
    # dense NN-median). A large/unverified tie keeps the histogram value.
    same_star = None
    if res_a is not None and res_a.get("ok") and not res_a.get("swept"):
        try:
            lrm = local_residual_map(
                consensus_coords, ref_coords_all, res_a, cell_arcsec=1e9,
                match_radius=SAMESTAR_MATCH_RADIUS, min_stars=SAMESTAR_MIN_PAIRS,
                context=f"{context} same-star bulk")
            if lrm.get("cells"):
                c = max(lrm["cells"], key=lambda cc: cc["n"])
                # local_residual_map returns the matched-pair residual AFTER
                # removing res_a's (biased) global offset, so the same-star TOTAL
                # = histogram offset + residual (the histogram bias cancels: the
                # residual carries the correction back to the true same-star tie).
                sdra = res_a["dra"] + c["dra_mas"]
                sddec = res_a["ddec"] + c["ddec_mas"]
                same_star = dict(dra=sdra, ddec=sddec,
                                 off=float(np.hypot(sdra, sddec)),
                                 dra_err=c["dra_sem"], ddec_err=c["ddec_sem"],
                                 n=c["n"], residual_dra=c["dra_mas"],
                                 residual_ddec=c["ddec_mas"])
        except GlobalTieNotVerifiedError:
            same_star = None   # tie too large to refine -> keep the histogram value

    # The reported bulk (the correction to ADD) is the same-star refinement when
    # available, else the histogram. Check A stays in `vs_full` for detection.
    bulk = same_star if same_star is not None else res_a
    bulk_source = ("same-star" if same_star is not None
                   else "histogram" if res_a is not None else "none")

    # Per-tile check D (``grid``) requires a DENSE reference.  Measured against a
    # Gaia-ONLY refcat (no VIRAC2 component, ``dense=False`` -- e.g. w51, outside
    # the VVV footprint, whose refcat is a pure gaia_refcat.fits) each small tile
    # pairs the wrong sparse stars and returns arcsecond-scale NOISE in every tile
    # (24/28 "coherent" w51 tiles land at 16-58 arcsec, not the true 40 mas), so it
    # cannot see a seam and MUST NOT gate -- gating on it strands the bulk-sentinel
    # Gaia tie the reducer needs.  In that regime the corroborating second check is
    # the same-star matched-pair refinement, which itself REFUSES unless the global
    # tie is verified-small (GlobalTieNotVerifiedError), so the sign-off keeps its
    # COUNT of independent checks (never a single number).
    #
    # COVERAGE GAP (Gaia-only only): D is the sole SPATIAL check; vs_full, the
    # sweep, and the same-star refinement are all GLOBAL statistics over the whole
    # field.  So for a Gaia-only reference there is no per-region seam check, and a
    # half-mosaic seam that a global bulk hides would pass here (matched-pair stats
    # are structurally weak at a localized sub-population displaced beyond the match
    # radius -- see the brick F182M matched-pair blind-spot).  On these fields seam
    # coverage moves to the release-time interframe-overlap gate / manual QC; it is
    # NOT recovered here.  A DENSE (VIRAC2) reference keeps the per-tile gate intact
    # -- that is the brick-1182 half-mosaic-seam protection, unchanged.
    per_tile_ok = bool(grid.get("clean")) if dense else (same_star is not None)
    apply_ok = bool(res_a is not None and res_a.get("ok")
                    and per_tile_ok and cross_gross_ok)
    out = dict(vs_full=res_a, vs_sparse=res_b, cross_reference=agree,
               cross_reference_gross_ok=cross_gross_ok,
               cross_reference_gross_tol_mas=gross_tol_mas,
               per_tile=grid, per_tile_ok=per_tile_ok, reference_dense=bool(dense),
               flux_matched=fluxmatched, same_star=same_star,
               bulk_source=bulk_source, apply_ok=apply_ok)
    if bulk is not None:
        out.update(dra_mas=bulk["dra"], ddec_mas=bulk["ddec"],
                   dra_err_mas=bulk.get("dra_err", float("nan")),
                   ddec_err_mas=bulk.get("ddec_err", float("nan")),
                   off_mas=bulk["off"],
                   swept=(res_a.get("swept", False) if res_a is not None else False))
    else:
        out.update(dra_mas=float("nan"), ddec_mas=float("nan"),
                   dra_err_mas=float("nan"), ddec_err_mas=float("nan"),
                   off_mas=float("nan"), swept=False)
    return out


def load_reference_catalog(path):
    """Load a gaia+virac2 seed refcat (build_gaia_virac2_refcat.py output) and
    split it into the full (dense) and Gaia-only (sparse) SkyCoord sets."""
    ref = Table.read(path)
    if "skycoord" in ref.colnames:
        coords = SkyCoord(ref["skycoord"]).icrs
    else:
        coords = SkyCoord(ra=np.asarray(ref["RA"], dtype=float) * u.deg,
                          dec=np.asarray(ref["DEC"], dtype=float) * u.deg,
                          frame="icrs")
    source = np.asarray(ref["source"]).astype(str) if "source" in ref.colnames else None
    if source is not None:
        is_gaia = np.char.startswith(np.char.upper(source), "GAIA")
        sparse = coords[is_gaia]
        # A refcat is DENSE only if it actually carries a non-Gaia (VIRAC2)
        # component.  A source-tagged file that happens to be all-Gaia, or one
        # with no source column at all (below), is Gaia-only: full == sparse.
        dense = bool((~is_gaia).any())
    else:
        sparse = coords
        dense = False   # Gaia-only refcat (no VIRAC2 dense component)
    mag = np.asarray(ref["refmag"], dtype=float) if "refmag" in ref.colnames else None
    return dict(all=coords, sparse=sparse, mag=mag, table=ref, dense=dense)
