"""Sanctioned astrometric-offset measurement — offset-histogram stacking.

This is the ONE public, guarded entry point for measuring the bulk (and per-tile)
on-sky offset between two source lists.  It is **density-immune**: it histograms
ALL pairwise offsets within a window and takes the peak, so it stays correct no
matter how large the shift.  This is the method that must be used instead of the
FORBIDDEN dense-nearest-neighbour-median (`match_to_catalog_sky(...).median()`),
which collapses toward ~0 in a crowded field and has repeatedly corrupted the
brick-1182 / prop-2221 astrometry.  See CLAUDE.md and
``reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md``.

Do NOT write ad-hoc NN-median matching.  Call ``measure_offset`` here.

Density-immune is NOT unbiased against a DENSE reference
--------------------------------------------------------

"Density-immune" here means immune to NN-COLLAPSE, the failure this module was
written to replace.  It does NOT mean the peak is unbiased.  The histogram peak
is unbiased only while the NON-matching pair background is uniform, and against
a DENSE reference (full VIRAC2) two catalogs tracing the SAME clustered stellar
field make a correlated, non-uniform wrong-pair background that PULLS THE PEAK
by several mas.  Measured on brick (2026-07-16): the histogram read ~9-10 mas
against dense VIRAC2 with a +-6.5 mas RA term whose SIGN FLIPPED between
filters -- an artifact, since a real field offset cannot flip sign per filter --
while the same-star tie over the same data was ~0 in RA and ~-5 mas in Dec.
Against SPARSE Gaia the histogram was clean.

So against a dense reference use ``measure_offset`` to DETECT that the tie is
small, then take the PRECISE bulk value same-star, from ``local_residual_map``
(single giant cell) -- or simply call
``visit_consensus.measure_reference_tie``, which does the whole detect-then-
refine sequence and reports which route produced its number
(``bulk_source``: ``"same-star"`` / ``"histogram"``).  A shipped correction must
come from the same-star bulk.  See CLAUDE.md and the
``histogram-vs-samestar-offset-bias`` note.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from scipy.spatial import cKDTree


# Contrast (peak bin / median of the OCCUPIED bins of the pair-offset
# histogram) below this = NO coherent tie (scattered pairs), i.e. the two
# frames are NOT registered at this scale.
#
# The denominator is the median over the bins holding at least one pair --
# `H[H > 0]` in `_hist_peak`, whose docstring carries the expression -- NOT the
# median over all bins, and not a background level.  In a SPARSE histogram most
# occupied bins hold exactly one pair, so the denominator is 1 and the default
# floor of 5 means "at least 5 pairs in the peak bin" rather than "5x above
# background" (issue #400).  In a dense one the denominator is a real occupancy
# and the ratio reads as a contrast.
#
# The expression lives in `_hist_peak`'s docstring rather than here on purpose:
# the module docstring above names `match_to_catalog_sky` as the forbidden
# method, and the NN-median grep-guard
# (tests/test_no_adhoc_nn_median_astrometry.py) reports a file whose
# module-level text carries BOTH a match token and a median/mean CALL.  Inside
# the function the guard attributes the reduce where it belongs.
DEFAULT_MIN_CONTRAST = 5.0


def _unit_xyz(coords):
    """(N, 3) unit-sphere cartesian positions of a SkyCoord."""
    ra = np.radians(np.asarray(coords.ra.deg, dtype=float))
    dec = np.radians(np.asarray(coords.dec.deg, dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def _chord(sep_arcsec):
    """Unit-sphere chord length equivalent to an angular separation."""
    return 2.0 * np.sin(np.radians(sep_arcsec / 3600.0) / 2.0)


class KDTreeReference:
    """A large fixed reference catalog with its KD tree built ONCE.

    ``measure_offset``'s plain path (astropy ``search_around_sky``) rebuilds
    KD trees on BOTH source lists on every call — 8x per swept measure
    (4 windows x probe+main).  Against a multi-million-star pooled reference
    measured many times (visit-consensus ties: one call per exposure of a
    192-exposure visit) the rebuilds dominate the runtime by orders of
    magnitude.  Wrap the reference once and pass the wrapper as ``b``:

        ref = KDTreeReference(consensus_coords)
        for exp in exposures:
            measure_offset(exp_coords, ref, ...)

    Queries run parallel (``workers=-1``) once there are enough of them --
    see ``_PARALLEL_MIN_QUERIES``; below that the pool spin-up dominates and
    the reuse is a slowdown, which it was on sparse fields.  Results match the
    plain path
    (same deterministic probe/subsample RNG, exact within-radius pair sets,
    same histogram) EXCEPT in the dense-subsample regime: when the pair
    budget forces subsampling, the plain path keeps a hard 2000-point floor
    while the tree path relaxes that floor to respect the total-pair budget
    (see ``_measure_at_window_tree``), so the subsample -- and hence the
    histogram -- can differ there."""

    #: Below this many QUERY points, ``workers=-1`` costs more in thread-pool
    #: spin-up than it saves.  Measured per exposure on real per-frame
    #: catalogs, plain path vs tree path:
    #:
    #:     sickle F187N   ~100 query pts   112 ms  ->  959 ms   0.12x
    #:     sickle F210M   ~471            1.38 s   -> 1.40 s    0.99x
    #:     sgrc  F162M  ~4,019           15.0 s    -> 4.05 s    3.7x
    #:     cloudc F182M ~6,195           14.3 s    -> 2.33 s    6.1x
    #:
    #: 96 F187N frames x 0.85 s is ~82 s per filter per frozen stage -- small,
    #: but sickle F187N is a sparse field and the reuse made it slower, which
    #: is the opposite of the point.
    _PARALLEL_MIN_QUERIES = 1000

    def _workers(self, xyz):
        return -1 if np.shape(xyz)[0] >= self._PARALLEL_MIN_QUERIES else 1

    def __init__(self, coords):
        self.coords = coords
        self.ra_deg = np.asarray(coords.ra.deg, dtype=float)
        self.dec_deg = np.asarray(coords.dec.deg, dtype=float)
        self._tree = cKDTree(_unit_xyz(coords))

    def __len__(self):
        return len(self.ra_deg)

    def count_within_xyz(self, xyz, sep_arcsec):
        return int(np.sum(self._tree.query_ball_point(
            xyz, _chord(sep_arcsec), workers=self._workers(xyz),
            return_length=True)))

    def pairs_within_xyz(self, xyz, sep_arcsec):
        """(ia, ib) index pairs with separation < sep_arcsec (ia into xyz,
        ib into the wrapped reference)."""
        lists = self._tree.query_ball_point(xyz, _chord(sep_arcsec),
                                            workers=self._workers(xyz))
        counts = np.fromiter((len(lst) for lst in lists), dtype=np.int64,
                             count=len(lists))
        ia = np.repeat(np.arange(len(lists)), counts)
        if counts.any():
            ib = np.concatenate([np.asarray(lst, dtype=np.int64)
                                 for lst in lists if lst])
        else:
            ib = np.zeros(0, dtype=np.int64)
        return ia, ib


class NoCoherentTieError(RuntimeError):
    """Raised when the offset-histogram peak contrast is too low to trust — the two
    source lists have no coherent astrometric tie (broken/untied WCS)."""


def _hist_peak(dra_arcsec, ddec_arcsec, maxsep_arcsec, bin_arcsec):
    """2-D histogram peak of a cloud of pair offsets.  Returns
    (dra_mas, ddec_mas, off_mas, npairs, contrast, dra_err_mas, ddec_err_mas,
    n_peak).

    ``contrast`` is ``H.max() / np.median(H[H > 0])`` -- the peak bin over the
    median of the OCCUPIED bins, which in a sparse histogram is 1.  See the
    ``DEFAULT_MIN_CONTRAST`` comment for what that means for the floor.
    """
    n = len(dra_arcsec)
    m = maxsep_arcsec
    bins = np.arange(-m, m + bin_arcsec, bin_arcsec)
    H, xe, ye = np.histogram2d(dra_arcsec, ddec_arcsec, bins=[bins, bins])
    i, j = np.unravel_index(H.argmax(), H.shape)
    bg = float(np.median(H[H > 0])) if (H > 0).any() else 0.0
    dra0 = (xe[i] + xe[i + 1]) / 2.0
    ddec0 = (ye[j] + ye[j + 1]) / 2.0
    # refine on the pairs within one bin of the peak; the robust scatter of those
    # near-peak pairs gives an error bar on the peak position (MAD-based standard
    # error -- an offset without an error bar cannot gate a threshold decision)
    near = (np.abs(dra_arcsec - dra0) < bin_arcsec) & (np.abs(ddec_arcsec - ddec0) < bin_arcsec)
    n_peak = int(near.sum())
    dra_err = ddec_err = float("nan")
    if n_peak >= 5:
        dra0 = float(np.median(dra_arcsec[near]))
        ddec0 = float(np.median(ddec_arcsec[near]))
        mad_scale = 1.4826 / np.sqrt(n_peak)
        dra_err = float(np.median(np.abs(dra_arcsec[near] - dra0)) * mad_scale * 1000.0)
        ddec_err = float(np.median(np.abs(ddec_arcsec[near] - ddec0)) * mad_scale * 1000.0)
    contrast = float(H.max() / bg) if bg else float("inf")
    return (dra0 * 1000.0, ddec0 * 1000.0, float(np.hypot(dra0, ddec0) * 1000.0),
            int(n), contrast, dra_err, ddec_err, n_peak)


# Windows (arcsec) the sweep escalates through when a narrow window shows no
# coherent tie.  A LARGE rigid offset (brick-1182 v001 was ~20") has ZERO true
# pairs inside a 3" window -> the peak is noise (low contrast, reference-dependent)
# and looks exactly like "broken/no tie".  Widening the window recovers the real
# peak.  This is the trap that made a 20" offset read as ~2"/incoherent.
DEFAULT_SWEEP_WINDOWS = (3.0, 10.0, 30.0, 60.0)


# All-pairs budget per window.  The sweep evaluates windows up to 60", where a
# dense catalog pair (e.g. a merged catalog vs the full VIRAC2 refcat) produces
# BILLIONS of pairs and kills the process.  Above this budget, ``a`` is
# subsampled (deterministically) -- the histogram PEAK is a density estimate, so
# a uniform subsample preserves both its location and its contrast statistics.
MAX_PAIRS_PER_WINDOW = 3_000_000


# --- window-edge alias (issue #158) -----------------------------------------
# The pair-offset histogram of two ADJACENT but non-overlapping footprints has
# no true peak, but it is NOT flat: the number of pairs at lag (dRA, ddec) is
# the cross-correlation of the two footprints, which grows monotonically toward
# the lag that slides one footprint onto the other.  Inside a finite search
# window that ridge is truncated at the window EDGE, and the histogram's arg-max
# lands on it.  The resulting "peak" is sharp (its bin sits on a real ridge, so
# n_peak and the MAD error bar both look excellent) and it clears the contrast
# floor -- but it is pure geometry, and it MOVES when the window moves.
#
# Measured on W51 nrcblong-vs-nrcalong (issue #158, 2026-07-28), the same
# exposure pair at windows 55/60/66/70/80/90/100":
#     off = 54.8 / 59.8 / 64.7 / 67.2 / 75.9 / 89.0 / 99.5"
# i.e. off ~ the window, every time, with contrast 6-9 throughout.  A REAL tie
# does not do that: it reports the SAME offset at every window that can contain
# it (W51 F480M reads (+20.6, -1.7) mas identically at 1/3/10/30/60").
#
# Two cheap, independent tells follow:
#   * ``window_edge_fraction`` = off / window.  ~1 means the peak is riding the
#     window edge.  Recorded on every result as a diagnostic.
#   * ``window_consistent``: re-measure at INDEPENDENT wider windows and require
#     the peak to REPRODUCE.  This is the gate (see ``confirm_peak_windows``).
WINDOW_EDGE_FRACTION = 0.85

# Confirmation probes are placed so the candidate peak sits WELL INSIDE the
# probe window (a probe barely wider than the peak would put it back on the
# edge and confirm the artifact).  1.4x and 2.2x of the measured offset.
CONFIRM_WINDOW_FACTORS = (1.4, 2.2)

# Never probe past this: beyond it the pair count explodes and no per-frame or
# per-visit tie could mean anything anyway.
CONFIRM_MAX_WINDOW_ARCSEC = 300.0

# Floor on the peak-agreement tolerance between two windows (mas).  The bin
# widens with the window (``bw = window/150``), so two windows can only be
# expected to agree to a few of the COARSER bin widths.
CONFIRM_MIN_TOL_MAS = 100.0


def _window_bin_arcsec(window_arcsec, bin_arcsec):
    """The histogram bin ``_measure_at_window`` uses at this window."""
    return max(bin_arcsec, window_arcsec / 150.0)


def confirm_peak_windows(a, b, best, bin_arcsec=0.02, min_pairs=30,
                         factors=CONFIRM_WINDOW_FACTORS,
                         max_window_arcsec=CONFIRM_MAX_WINDOW_ARCSEC):
    """Re-measure ``best``'s peak at INDEPENDENT wider windows; does it survive?

    A real rigid offset is a property of the two source lists, so it reads the
    same at every window large enough to contain it.  A window-edge artifact
    (see ``WINDOW_EDGE_FRACTION``) is a property of the WINDOW, so it slides to
    the new edge as soon as the window changes.

    Returns ``dict(consistent, probes, n_probes)``; the per-window agreement
    tolerance is ``tol_mas`` on each PROBE, not a top-level key (issue #400).
    ``consistent`` is
    True when at least one probe reproduced the peak, False when every probe
    that produced a result disagreed, and None when no probe could be run (no
    pairs, or the probe window would exceed ``max_window_arcsec``) -- an
    UNDETERMINED result must never be read as a pass or as a failure.
    """
    off_arcsec = float(best["off"]) / 1000.0
    base_w = float(best["window_arcsec"])
    probes = []
    for f in factors:
        w = max(float(f) * off_arcsec, base_w * 1.25)
        if w > max_window_arcsec or abs(w - base_w) < 0.1 * base_w:
            continue
        res = _measure_at_window(a, b, w, bin_arcsec, min_pairs)
        if res is None:
            probes.append(dict(window_arcsec=w, dra=None, ddec=None))
            continue
        tol = max(CONFIRM_MIN_TOL_MAS,
                  3000.0 * max(_window_bin_arcsec(w, bin_arcsec),
                               _window_bin_arcsec(base_w, bin_arcsec)))
        sep = float(np.hypot(res["dra"] - best["dra"], res["ddec"] - best["ddec"]))
        # ``off_mas`` is the canonical spelling of the offset magnitude in the
        # records this module emits (issue #267); ``off`` is written alongside
        # it for the readers and on-disk records that predate the convergence.
        probes.append(dict(window_arcsec=w, dra=res["dra"], ddec=res["ddec"],
                           off=res["off"], off_mas=float(res["off"]),
                           contrast=res["contrast"],
                           npairs=res["npairs"], sep_mas=sep, tol_mas=tol,
                           agrees=bool(sep <= tol)))
    measured = [p for p in probes if p.get("dra") is not None]
    if not measured:
        # UNDETERMINED -- no probe could be run.  Not a pass, not a failure.
        return dict(consistent=None, probes=probes, n_probes=0)
    return dict(consistent=any(p["agrees"] for p in measured),
                probes=probes, n_probes=len(measured))


def _measure_at_window_tree(a, ref, maxsep_arcsec, bin_arcsec, min_pairs,
                            max_pairs=MAX_PAIRS_PER_WINDOW):
    """``_measure_at_window`` against a prebuilt ``KDTreeReference`` — no tree
    rebuilds, parallel queries.  Same deterministic RNG as the plain path,
    but NOT guaranteed numerically identical when the pair budget forces
    subsampling: the plain path keeps a hard 2000-point floor whereas this
    path relaxes the floor so the total-pair budget is not blown by >~2x."""
    n_a = len(a)
    a_xyz = _unit_xyz(a)
    a_ra = np.asarray(a.ra.deg, dtype=float)
    a_dec = np.asarray(a.dec.deg, dtype=float)
    if n_a > 2000 and len(ref) > 0:
        rng = np.random.default_rng(1182)
        probe = rng.choice(n_a, 1000, replace=False)
        est_total = ref.count_within_xyz(a_xyz[probe], maxsep_arcsec) * (n_a / 1000.0)
        if est_total > max_pairs:
            keep = int(n_a * max_pairs / est_total)
            # points floor, but never let the floor blow the total-pair budget
            # by more than ~2x: a 60" window in an ultra-dense field has >1e5
            # neighbours PER point, where a hard 2000-point floor (the plain
            # path's) materializes >4e8 pairs and dominates the runtime
            per_point = max(est_total / n_a, 1.0)
            keep = max(keep, min(2000, int(2 * max_pairs / per_point)), 30)
            sel = rng.choice(n_a, min(keep, n_a), replace=False)
            a_xyz, a_ra, a_dec = a_xyz[sel], a_ra[sel], a_dec[sel]
    ia, ib = ref.pairs_within_xyz(a_xyz, maxsep_arcsec)
    if len(ia) < min_pairs:
        return None
    cosd = np.cos(np.radians(a_dec[ia]))
    ddeg = ref.ra_deg[ib] - a_ra[ia]
    ddeg = (ddeg + 180.0) % 360.0 - 180.0   # RA-wrap-safe difference
    dra = ddeg * 3600.0 * cosd
    ddec = (ref.dec_deg[ib] - a_dec[ia]) * 3600.0
    bw = max(bin_arcsec, maxsep_arcsec / 150.0)
    (dra_mas, ddec_mas, off_mas, npairs, contrast,
     dra_err_mas, ddec_err_mas, n_peak) = _hist_peak(dra, ddec, maxsep_arcsec, bw)
    return dict(dra=dra_mas, ddec=ddec_mas, off=off_mas, npairs=npairs,
                contrast=contrast, window_arcsec=maxsep_arcsec,
                dra_err=dra_err_mas, ddec_err=ddec_err_mas, n_peak=n_peak)


def _measure_at_window(a, b, maxsep_arcsec, bin_arcsec, min_pairs,
                       max_pairs=MAX_PAIRS_PER_WINDOW):
    """Single-window histogram-peak offset.  Returns a result dict or None."""
    if isinstance(b, KDTreeReference):
        return _measure_at_window_tree(a, b, maxsep_arcsec, bin_arcsec,
                                       min_pairs, max_pairs)
    n_a = len(a)
    if n_a > 2000 and len(b) > 0:
        # probe the pair density on a small subsample, then cap the total
        rng = np.random.default_rng(1182)
        probe = rng.choice(n_a, 1000, replace=False)
        ia_p, _, _, _ = search_around_sky(a[probe], b, maxsep_arcsec * u.arcsec)
        est_total = len(ia_p) * (n_a / 1000.0)
        if est_total > max_pairs:
            keep = max(int(n_a * max_pairs / est_total), 2000)
            a = a[rng.choice(n_a, min(keep, n_a), replace=False)]
    ia, ib, _, _ = search_around_sky(a, b, maxsep_arcsec * u.arcsec)
    if len(ia) < min_pairs:
        return None
    cosd = np.cos(np.radians(a[ia].dec.value))
    dra = (b[ib].ra - a[ia].ra).to(u.arcsec).value * cosd
    ddec = (b[ib].dec - a[ia].dec).to(u.arcsec).value
    # keep ~150 bins across the window so the peak stays resolved as we widen
    bw = max(bin_arcsec, maxsep_arcsec / 150.0)
    (dra_mas, ddec_mas, off_mas, npairs, contrast,
     dra_err_mas, ddec_err_mas, n_peak) = _hist_peak(dra, ddec, maxsep_arcsec, bw)
    return dict(dra=dra_mas, ddec=ddec_mas, off=off_mas, npairs=npairs,
                contrast=contrast, window_arcsec=maxsep_arcsec,
                dra_err=dra_err_mas, ddec_err=ddec_err_mas, n_peak=n_peak)


def measure_offset(a, b, maxsep=3.0 * u.arcsec, bin_arcsec=0.02, min_pairs=30,
                   min_contrast=None, raise_on_low_contrast=False, context="",
                   sweep=True, sweep_windows=None, confirm_windows=False):
    """Bulk on-sky offset to move ``a`` onto ``b`` (mas), via offset-histogram
    stacking of all pairs within ``maxsep``.  Density-immune; the sanctioned
    replacement for NN-median.

    Parameters
    ----------
    a, b : SkyCoord
        Source lists.  Offset is (b - a) at the histogram peak.
    maxsep : Quantity
        Initial pair-search window.
    bin_arcsec : float
        Base histogram bin size (arcsec); the bin widens with the window to keep
        the peak resolved.
    min_contrast : float or None
        Peak/median contrast floor for a "real" tie (default
        ``DEFAULT_MIN_CONTRAST``).  Below it the tie is flagged (``ok=False``).
    raise_on_low_contrast : bool
        If True, raise ``NoCoherentTieError`` when NO window yields a coherent tie.
    sweep : bool
        If the initial window shows no coherent tie (contrast < ``min_contrast``),
        AUTO-ESCALATE the search window through ``sweep_windows`` and keep the
        highest-contrast result.  This is ON by default: it is the guard against a
        large offset masquerading as "no tie" (the brick-1182 v001 trap), and it
        is what makes the contrast threshold meaningful: with ``sweep=False``
        the contrast comes from one narrow window, in which a large offset and a
        genuine absence of a tie are indistinguishable.
    sweep_windows : iterable of float or None
        Windows (arcsec) to escalate through (default ``DEFAULT_SWEEP_WINDOWS``).
    confirm_windows : bool
        Require a SWEPT peak to reproduce at independent, wider windows before
        it is trusted (``confirm_peak_windows``).  A window-edge alias -- a peak
        that is a property of the search window rather than of the two source
        lists -- is rejected here (``ok`` False, ``alias_rejected`` True); see
        ``WINDOW_EDGE_FRACTION`` and issue #158.  Costs one or two extra
        measurements, and ONLY on a swept result, so the common (small, un-swept)
        tie is numerically and computationally untouched.  Off by default so an
        arbitrary caller opts in, but the BULK reference tie
        (``measure_reference_tie``) now opts in: the earlier worry -- that it
        legitimately reports a large swept offset only the widest window can
        contain -- is covered by ``confirm_peak_windows`` rejecting only on a
        MEASURED disagreement, so a genuine gross shift reproduces at the wider
        probes and survives (``test_true_large_offset_survives_confirmation``,
        issues #158 and #257).

    Returns
    -------
    dict or None
        ``dict(dra, ddec, off, npairs, contrast, ok, window_arcsec, swept,
        dra_err, ddec_err, n_peak, window_edge_fraction, windows)`` (mas), or
        None if too few pairs at every window.  ``ok`` is False when NO window
        reaches ``min_contrast`` (or, with ``confirm_windows``, when a swept peak
        failed confirmation).
        ``window_arcsec`` is the window the reported peak came from -- a value
        >> your expected offset is the tell that the tie was only found after
        widening (investigate: the frame is grossly shifted).  ``dra_err`` /
        ``ddec_err`` are MAD-based standard errors of the peak position from
        the ``n_peak`` near-peak pairs.  ``window_edge_fraction`` is
        ``off / window`` -- near 1 the peak is riding the window edge.  Two
        non-overlapping footprints have a pair-density ridge at the lag that
        slides one onto the other, and the window truncates it, so the arg-max
        lands on the cut: that is footprint geometry, not a tie (issue #158).
        Widening the window MOVES such a peak instead of removing it;
        ``confirm_windows`` / ``window_consistent`` is the gate.
        ``windows`` records every evaluated window so a record can be audited
        after the fact.

        The offset magnitude of THIS dict is ``off`` only, deliberately: three
        callers refine ``dra``/``ddec`` after the measurement and recompute
        ``off`` in place (``interframe_overlap._confirm_tie`` callers,
        ``visit_consensus`` half-bridge), so a mirrored ``off_mas`` here would
        silently go stale on a refined result.  The mirrored, canonical
        ``off_mas`` is written on the derived per-cell / per-window RECORDS
        (``measure_offset_grid`` cells, ``windows`` entries), which nothing
        mutates after the fact.  See issue #267.

        ``dra``/``ddec`` here are the HISTOGRAM PEAK, which against a DENSE
        reference carries a several-mas bias -- the correlated wrong-pair
        background described in this module's docstring, distinct from the
        NN-collapse this method is immune to.  Read it as a DETECTION of the
        tie's size, not as the precise bulk: refine same-star through
        ``local_residual_map`` (single giant cell), or call
        ``visit_consensus.measure_reference_tie``, which sequences the two and
        reports ``bulk_source``.  Against a SPARSE reference (the Gaia-only
        subset) the peak is clean and no refinement is needed.  Issue #397.
    """
    min_contrast = DEFAULT_MIN_CONTRAST if min_contrast is None else min_contrast
    maxsep_arcsec = maxsep.to(u.arcsec).value if hasattr(maxsep, "to") else float(maxsep)

    windows = [maxsep_arcsec]
    if sweep:
        extra = list(sweep_windows) if sweep_windows is not None else list(DEFAULT_SWEEP_WINDOWS)
        for w in extra:
            if w > maxsep_arcsec:
                windows.append(float(w))

    # Evaluate every window and take the GLOBAL max-contrast peak. We must not stop
    # early on a moderate peak at an intermediate window: a window still smaller than
    # the true offset contains only noise, which can clear a contrast floor by chance
    # (that noise peak is what made v001's 20" offset read as ~1.6"). Only a window
    # that actually CONTAINS the offset produces the real, dominant peak.
    best = None
    evaluated = []
    for w in windows:
        res = _measure_at_window(a, b, w, bin_arcsec, min_pairs)
        if res is None:
            evaluated.append(dict(window_arcsec=w, dra=None, ddec=None))
            continue
        # both spellings of the offset magnitude -- see issue #267; ``off_mas``
        # is canonical, ``off`` is retained for already-written records
        evaluated.append(dict(window_arcsec=w, dra=res["dra"], ddec=res["ddec"],
                              off=res["off"], off_mas=float(res["off"]),
                              contrast=res["contrast"],
                              npairs=res["npairs"], n_peak=res["n_peak"]))
        if best is None or res["contrast"] > best["contrast"]:
            best = res

    if best is None:
        return None
    best["ok"] = best["contrast"] >= min_contrast
    # swept = the measured offset EXCEEDS the initial window, i.e. the narrow window
    # could not have contained it and the sweep is what found it. A True here on a
    # tie you expected to be small means the frame is grossly shifted -- investigate.
    best["swept"] = (best["off"] / 1000.0) > maxsep_arcsec
    best["windows"] = evaluated
    best["window_edge_fraction"] = (
        float(best["off"] / 1000.0 / best["window_arcsec"])
        if best["window_arcsec"] > 0 else 0.0)
    # A swept peak must REPRODUCE at an independent window or it is geometry,
    # not a tie (issue #158).  Only a measured DISAGREEMENT rejects: an
    # undetermined confirmation (no probe could run) leaves the result as it was.
    best["window_consistent"] = None
    best["alias_rejected"] = False
    if confirm_windows and best["ok"] and best["swept"]:
        conf = confirm_peak_windows(a, b, best, bin_arcsec=bin_arcsec,
                                    min_pairs=min_pairs)
        best["window_confirmation"] = conf
        best["window_consistent"] = conf["consistent"]
        if conf["consistent"] is False:
            best["ok"] = False
            best["alias_rejected"] = True
    if not best["ok"] and raise_on_low_contrast:
        if best["alias_rejected"]:
            raise NoCoherentTieError(
                f"No trustworthy astrometric tie ({context or 'unspecified'}): the "
                f"only peak ({best['off'] / 1000.0:.1f}\" at the {best['window_arcsec']:.0f}\" "
                f"window, contrast {best['contrast']:.1f}) did NOT reproduce at an "
                f"independent window -- it rode the window edge "
                f"(off/window = {best['window_edge_fraction']:.2f}), so it is the "
                f"footprint geometry, not a tie (issue #158).")
        raise NoCoherentTieError(
            f"No coherent astrometric tie ({context or 'unspecified'}) at any window "
            f"up to {windows[-1]:.0f}\": best contrast {best['contrast']:.1f} < "
            f"{min_contrast} ({best['npairs']} pairs). The two source lists are not "
            f"registered (broken/untied WCS). Do NOT trust a NN-median 'agreement'.")
    return best


def measure_offset_grid(a, b, nx=6, ny=6, ra_bounds=None, dec_bounds=None,
                        maxsep=3.0 * u.arcsec, bin_arcsec=0.02, min_pairs=30,
                        min_contrast=None, max_off_mas=None, context=""):
    """Per-tile offset map — measure ``measure_offset`` in an ``nx`` x ``ny`` grid
    over the field.  A bulk offset ~0 can HIDE a half-mosaic that is untied
    (brick-1182 visit-001), so ALWAYS map per tile before signing off.

    ``max_off_mas`` is the per-tile offset-MAGNITUDE ceiling.  A tile can have a
    razor-sharp, high-contrast peak (contrast >> ``min_contrast``) that sits at a
    real, non-zero offset — a locally MISREGISTERED tile that is perfectly
    self-consistent.  Contrast alone (the pre-2026-07 behaviour) CANNOT see this:
    it only asks "is there a coherent peak?", not "is the peak at ZERO?".  That
    blind spot let a ~90 mas local seam residual in brick-1182 F200W (visit-001
    side, where a single rigid per-visit shift left a field-dependent residual)
    pass with contrast ~20 — and the drizzle doubled every star in the overlap.
    When ``max_off_mas`` is set, a tile is ``ok`` only if it BOTH has a coherent
    peak (contrast) AND that peak is within ``max_off_mas`` of zero.  Leave it
    ``None`` for pure offset-mapping, where you want the value and not a
    verdict.

    **No production caller passes it, and this docstring used to tell you to
    pass it "for any release/QC sign-off" (issue #392).**  That instruction was
    written when this function was the interframe-overlap gate's per-tile
    estimator; it stopped being that in 0fb1958 (2026-07-13), because a
    per-tile histogram over a 40-250-star tile produces a noise peak that
    clears any contrast floor and lands anywhere in the window.  The release
    gate's magnitude limit is now ``interframe_overlap.overlap_offset_grid(...,
    tol_mas=...)``, on matched pairs.  The one remaining production caller of
    THIS parameter's owner -- the monitoring scan -- deliberately leaves it
    ``None`` and says so in a comment, so with ``max_off_mas=None``
    ``off_ok`` is unconditionally True and ``clean`` is a statement about
    contrast alone.  Whether the monitoring scan should switch it on is the
    open half of #392.

    Use a FINE grid for QC (``nx=ny`` >= ~16 so a thin overlap strip is not
    diluted inside a coarse tile); a 4x4 grid hid the brick-1182 seam.

    Returns
    -------
    dict
        ``dict(cells=[...], n_ok, n_total, worst_off_mas, min_contrast_seen,
        worst_off_cell, clean)``.  Each cell gets ``ok`` (contrast AND, if
        ``max_off_mas`` set, magnitude) and ``off_ok``.  ``clean`` is True only
        if every covered cell is ``ok``.

        Each cell's offset magnitude is available as ``off_mas`` (canonical,
        same spelling as ``local_residual_map`` cells and as the
        ``worst_off_cell`` summary) and, for back-compatibility with records
        and readers written before issue #267, also as ``off``.  Both hold the
        same value; prefer ``off_mas``.
    """
    min_contrast = DEFAULT_MIN_CONTRAST if min_contrast is None else min_contrast
    ra = a.ra.deg
    dec = a.dec.deg
    r0, r1 = ra_bounds if ra_bounds else (float(ra.min()), float(ra.max()))
    d0, d1 = dec_bounds if dec_bounds else (float(dec.min()), float(dec.max()))
    re = np.linspace(r0, r1, nx + 1)
    de = np.linspace(d0, d1, ny + 1)
    cells = []
    for i in range(nx):
        for j in range(ny):
            sel = (ra >= re[i]) & (ra < re[i + 1]) & (dec >= de[j]) & (dec < de[j + 1])
            if sel.sum() < min_pairs:
                continue
            res = measure_offset(a[sel], b, maxsep=maxsep, bin_arcsec=bin_arcsec,
                                 min_pairs=min_pairs, min_contrast=min_contrast,
                                 context=f"{context} tile[{i},{j}]")
            if res is None:
                continue
            contrast_ok = bool(res["ok"])
            off_ok = True if max_off_mas is None else (res["off"] <= max_off_mas)
            # CELL OFFSET KEY (issue #267): a cell carries the SAME number under
            # BOTH ``off_mas`` and ``off``.  ``off_mas`` is the canonical name --
            # it states the unit, it is what the ``worst_off_cell`` summary in
            # this same return dict uses, and it is what ``local_residual_map``
            # cells have always used.  ``off`` is kept only so that (a) records
            # already written to disk under the old spelling and (b) readers
            # written against them keep working; new readers should use
            # ``off_mas``.  Writing both is what lets the two spellings converge
            # without invalidating any recorded checkpoint.
            res.update(ix=i, iy=j, contrast_ok=contrast_ok, off_ok=bool(off_ok),
                       ok=bool(contrast_ok and off_ok),
                       off_mas=float(res["off"]))
            cells.append(res)
    n_ok = sum(1 for c in cells if c["ok"])
    worst = max(cells, key=lambda c: c["off_mas"], default=None)
    return dict(cells=cells, n_ok=n_ok, n_total=len(cells),
                worst_off_mas=max((c["off_mas"] for c in cells), default=float("nan")),
                worst_off_cell=(None if worst is None else dict(
                    ix=worst["ix"], iy=worst["iy"], off_mas=worst["off_mas"],
                    contrast=worst["contrast"])),
                min_contrast_seen=min((c["contrast"] for c in cells), default=float("nan")),
                clean=bool(cells) and n_ok == len(cells))


def agree_across_references(a, ref_a, ref_b, tol_mas=100.0, label_a="refA",
                           label_b="refB", **kwargs):
    """Measure the offset of ``a`` against TWO independent references and check they
    AGREE.  A spurious histogram peak (e.g. a large offset only partly captured, or
    a crowding artefact) is REFERENCE-DEPENDENT -- it moves when you swap a dense
    catalogue (VIRAC2) for a sparse one (Gaia-only).  A real rigid tie gives the
    same peak against both.  brick-1182 v001 measured at a narrow window gave
    VIRAC 2.7" vs Gaia 2.0" with dRA differing ~700 mas: that DISAGREEMENT was the
    tell it was not a clean tie (the true offset was ~20", outside the window).

    Returns
    -------
    dict
        ``dict(a=<measure_offset result>, b=<...>, ddra_mas, dddec_mas, sep_mas,
        agree)``.  ``agree`` is False if either measurement failed/low-contrast or
        the two peaks differ by more than ``tol_mas``.
    """
    ra = measure_offset(a, ref_a, context=label_a, **kwargs)
    rb = measure_offset(a, ref_b, context=label_b, **kwargs)
    out = dict(a=ra, b=rb)
    if ra is None or rb is None or not ra.get("ok") or not rb.get("ok"):
        out.update(ddra_mas=float("nan"), dddec_mas=float("nan"),
                   sep_mas=float("nan"), agree=False)
        return out
    ddra = ra["dra"] - rb["dra"]
    dddec = ra["ddec"] - rb["ddec"]
    sep = float(np.hypot(ddra, dddec))
    out.update(ddra_mas=ddra, dddec_mas=dddec, sep_mas=sep, agree=sep <= tol_mas)
    return out


class GlobalTieNotVerifiedError(RuntimeError):
    """Raised when ``local_residual_map`` is called without a verified small global
    tie.  Per-star matched-pair statistics are only meaningful AFTER the bulk
    offset is known (via ``measure_offset``) to be much smaller than the match
    radius -- otherwise the matching pairs the WRONG stars and the residual map
    fabricates false agreement (the banned dense-NN failure mode)."""


def local_residual_map(a, b, global_result, cell_arcsec=2.0,
                       match_radius=0.3 * u.arcsec, min_stars=10,
                       tol_mas=15.0, nsigma=3.0, context=""):
    """Fine-scale (default 2"x2" cell) residual-offset map from matched pairs,
    AFTER a verified global tie.  This is the sanctioned "histogram refinement"
    class of measurement: the coarse offset is measured first with the
    density-immune ``measure_offset``; only when that verified tie is much
    smaller than the match radius do per-star pairings become unambiguous, and
    the per-cell robust mean of the pair residuals (with a standard error) maps
    local distortion/misregistration at scales ``measure_offset_grid`` cannot
    reach (its histogram peak needs more pairs per cell than a 2" cell holds).

    A cell is only FLAGGED when it is both large AND significant:
    ``|mean| > tol_mas`` and ``|mean| > nsigma * sem`` and ``n >= min_stars``.
    A 15 mas "offset" carried by one star is not a measurement.

    Parameters
    ----------
    a, b : SkyCoord
        Source lists.  Residuals are (b - a) per matched pair, minus the global
        offset from ``global_result``.
    global_result : dict
        The ``measure_offset(a, b)`` result.  REQUIRED.  Must have ``ok=True``,
        ``swept=False``, and ``off`` < match_radius/3, else
        ``GlobalTieNotVerifiedError`` is raised.
    cell_arcsec : float
        Cell size of the residual map (arcsec).
    match_radius : Quantity
        Pair-search radius.  Keep small (default 0.3") -- ambiguity rises with
        radius in a dense field.
    min_stars : int
        Minimum matched pairs in a cell for the cell to be measurable.
    tol_mas : float
        Local residual-offset tolerance (mas).
    nsigma : float
        Significance requirement: a cell is only flagged when its mean residual
        exceeds ``nsigma`` times its standard error.

    Returns
    -------
    dict
        ``dict(cells=[...], n_cells, n_measured, n_pairs, n_flagged,
        worst_off_mas, worst_sig_off_mas, clean)``, plus ``reason`` on an empty
        map.  ``n_pairs`` is the number of UNAMBIGUOUS matched pairs the map
        was built from.  Each cell:
        ``dict(ra0, dec0, ix, iy, n, dra_mas, ddec_mas, dra_sem, ddec_sem,
        off_mas, significant, flagged)``.  ``clean`` is True when no cell is
        flagged AND at least one cell was measurable.

        ``off_mas`` is the canonical per-cell offset key, the same one
        ``measure_offset_grid`` cells now carry (issue #267), so one reader can
        consume cells from either producer.
    """
    if global_result is None or not global_result.get("ok"):
        raise GlobalTieNotVerifiedError(
            f"local_residual_map({context}): no verified global tie -- run "
            f"measure_offset first and fix the bulk registration before mapping "
            f"local residuals.")
    radius_arcsec = match_radius.to(u.arcsec).value if hasattr(match_radius, "to") \
        else float(match_radius)
    if global_result.get("swept"):
        raise GlobalTieNotVerifiedError(
            f"local_residual_map({context}): global tie was only found by window "
            f"SWEEP (offset {global_result['off']:.0f} mas) -- the frame is grossly "
            f"shifted; correct the bulk offset before mapping local residuals.")
    if global_result["off"] > radius_arcsec * 1000.0 / 3.0:
        raise GlobalTieNotVerifiedError(
            f"local_residual_map({context}): global offset {global_result['off']:.1f} "
            f"mas is not << match radius {radius_arcsec * 1000:.0f} mas; matched "
            f"pairs would be ambiguous. Correct the bulk offset first.")

    gdra_deg = (global_result["dra"] / 3.6e6)  # on-sky mas -> deg (Δα·cosδ)
    gddec_deg = (global_result["ddec"] / 3.6e6)

    def _no_pairs(n_pairs=0, reason=None):
        # A map can come back empty three ways, and `n_pairs` alone separates
        # only one of them (issue #400): the "0 pairs found" and "pairs found
        # but ALL ambiguous" branches both report 0, because n_pairs counts
        # UNAMBIGUOUS pairs and the second case has none either.  The third --
        # pairs binned but every cell below min_stars -- does not come through
        # here at all; it returns the full dict with `cells=[]` and a non-zero
        # n_pairs.  `reason` names which branch this is, so a caller reporting
        # "every cell held too few stars" is not naming a cause it never
        # checked.
        return dict(cells=[], n_cells=0, n_measured=0, n_flagged=0,
                    n_pairs=n_pairs, reason=reason,
                    worst_off_mas=float("nan"), worst_sig_off_mas=float("nan"),
                    clean=False)

    ia, ib, sep, _ = search_around_sky(a, b, radius_arcsec * u.arcsec)
    if len(ia) == 0:
        return _no_pairs(n_pairs=0, reason="no pairs within match_radius")
    # keep only the NEAREST b for each a (unambiguous association given the
    # verified small tie), then require uniqueness of the b partner
    order = np.lexsort((sep.arcsec, ia))
    ia_o, ib_o = ia[order], ib[order]
    first = np.concatenate(([True], ia_o[1:] != ia_o[:-1]))
    ia_n, ib_n = ia_o[first], ib_o[first]
    _, b_counts = np.unique(ib_n, return_counts=True)
    b_multi = set(np.unique(ib_n)[b_counts > 1])
    keep = np.array([bi not in b_multi for bi in ib_n])
    ia_n, ib_n = ia_n[keep], ib_n[keep]
    # The uniqueness filter can remove EVERY pair -- in a crowded field each `b`
    # can be the nearest partner of more than one `a`, and when that is true of
    # all of them nothing survives.  The `len(ia) == 0` guard above does not
    # cover this: pairs WERE found, they were just all ambiguous.  Falling
    # through leaves `ra_deg` zero-size and `ra_deg.min()` raises
    # "zero-size array to reduction operation minimum which has no identity",
    # which crashed the whole --refcat arbiter run of
    # check_interframe_overlap.py on brick F187N (2026-08-03) after it had
    # already cleared F182M -- an unhandled exception instead of the "could not
    # measure this pair" answer the caller is written to handle.
    if len(ia_n) == 0:
        return _no_pairs(n_pairs=0,
                         reason=f"all {len(ia)} pair(s) ambiguous "
                                f"(no unique nearest partner)")

    cosd = np.cos(np.radians(a[ia_n].dec.value))
    dra = (b[ib_n].ra - a[ia_n].ra).to(u.arcsec).value * cosd * 1000.0 - global_result["dra"]
    ddec = (b[ib_n].dec - a[ia_n].dec).to(u.arcsec).value * 1000.0 - global_result["ddec"]

    ra_deg = a[ia_n].ra.deg
    dec_deg = a[ia_n].dec.deg
    dec_mid = float(np.median(dec_deg))
    cell_deg_dec = cell_arcsec / 3600.0
    cell_deg_ra = cell_arcsec / 3600.0 / max(np.cos(np.radians(dec_mid)), 1e-6)
    r0, d0 = float(ra_deg.min()), float(dec_deg.min())
    ix = np.floor((ra_deg - r0) / cell_deg_ra).astype(int)
    iy = np.floor((dec_deg - d0) / cell_deg_dec).astype(int)

    cells = []
    for (cx, cy) in sorted(set(zip(ix.tolist(), iy.tolist()))):
        sel = (ix == cx) & (iy == cy)
        n = int(sel.sum())
        if n < min_stars:
            continue
        cdra = float(np.median(dra[sel]))
        cddec = float(np.median(ddec[sel]))
        mad_scale = 1.4826 / np.sqrt(n)
        dra_sem = float(np.median(np.abs(dra[sel] - cdra)) * mad_scale)
        ddec_sem = float(np.median(np.abs(ddec[sel] - cddec)) * mad_scale)
        off = float(np.hypot(cdra, cddec))
        sem = float(np.hypot(dra_sem, ddec_sem))
        significant = bool(sem > 0 and off > nsigma * sem)
        cells.append(dict(
            ra0=r0 + (cx + 0.5) * cell_deg_ra, dec0=d0 + (cy + 0.5) * cell_deg_dec,
            ix=int(cx), iy=int(cy), n=n, dra_mas=cdra, ddec_mas=cddec,
            dra_sem=dra_sem, ddec_sem=ddec_sem, off_mas=off,
            significant=significant, flagged=bool(off > tol_mas and significant)))
    flagged = [c for c in cells if c["flagged"]]
    sig = [c for c in cells if c["significant"]]
    return dict(cells=cells, n_cells=len(cells), n_measured=len(cells),
                n_pairs=int(len(ia_n)), n_flagged=len(flagged),
                reason=(None if cells else
                        f"every cell held fewer than min_stars={min_stars} "
                        f"of the {len(ia_n)} unambiguous pair(s)"),
                worst_off_mas=max((c["off_mas"] for c in cells), default=float("nan")),
                worst_sig_off_mas=max((c["off_mas"] for c in sig), default=float("nan")),
                clean=bool(cells) and not flagged)


def residual_vs_magnitude(a, b, mag, global_result, match_radius=0.3 * u.arcsec,
                          bin_mag=1.0, min_stars=30, tol_mas=5.0, nsigma=3.0,
                          context=""):
    """Positional residual (a vs b matched pairs) as a function of source
    BRIGHTNESS, after a verified global tie.

    A reference catalog for pointing (NIRSpec MSA/TA) must show NO systematic
    position-vs-brightness trend: saturation-core centroid bias, wing-fit
    substitution offsets, and detector nonlinearity all imprint exactly such a
    trend, and a bulk tie (dominated by faint stars) cannot see it.  Same
    sanctioned "histogram refinement" class as ``local_residual_map``: the
    global tie must already be verified small, so a nearest pair within
    ``match_radius`` is the right star; per-magnitude-bin robust means with
    standard errors then expose any brightness systematics.

    Parameters
    ----------
    a, b : SkyCoord
        Source lists.  Residuals are (b - a) per matched pair minus the global
        offset.  ``a`` is the catalog under test; ``mag`` aligns with ``a``.
    mag : array
        Magnitude (any consistent zero point; only the ORDERING/binning is
        used) per ``a`` source.  Non-finite entries are excluded.
    global_result : dict
        The ``measure_offset(a, b)`` result; same preconditions as
        ``local_residual_map`` (ok, not swept, off << match radius).
    bin_mag : float
        Magnitude bin width.
    min_stars : int
        Minimum matched pairs per bin for the bin to be measurable.
    tol_mas : float
        Bin-mean residual tolerance (mas).
    nsigma : float
        A bin is only flagged when its mean exceeds ``nsigma`` standard errors.

    Returns
    -------
    dict
        ``dict(bins=[...], n_bins, n_flagged, slope_dra_mas_per_mag,
        slope_ddec_mas_per_mag, slope_err_dra, slope_err_ddec,
        slope_significant, worst_off_mas, clean)``.  Each bin:
        ``dict(mag_lo, mag_hi, mag_mid, n, dra_mas, ddec_mas, dra_sem,
        ddec_sem, off_mas, significant, flagged)``.  ``clean`` requires no
        flagged bin, no significant slope, and >= 2 measurable bins.
    """
    if global_result is None or not global_result.get("ok"):
        raise GlobalTieNotVerifiedError(
            f"residual_vs_magnitude({context}): no verified global tie -- run "
            f"measure_offset first.")
    radius_arcsec = match_radius.to(u.arcsec).value if hasattr(match_radius, "to") \
        else float(match_radius)
    if global_result.get("swept"):
        raise GlobalTieNotVerifiedError(
            f"residual_vs_magnitude({context}): global tie was only found by "
            f"window SWEEP -- correct the bulk offset first.")
    if global_result["off"] > radius_arcsec * 1000.0 / 3.0:
        raise GlobalTieNotVerifiedError(
            f"residual_vs_magnitude({context}): global offset "
            f"{global_result['off']:.1f} mas is not << match radius; matched "
            f"pairs would be ambiguous.")

    mag = np.asarray(mag, dtype=float)
    finite = np.isfinite(mag)
    a_f = a[finite]
    mag_f = mag[finite]
    ia, ib, sep, _ = search_around_sky(a_f, b, radius_arcsec * u.arcsec)
    empty = dict(bins=[], n_bins=0, n_flagged=0,
                 slope_dra_mas_per_mag=float("nan"),
                 slope_ddec_mas_per_mag=float("nan"),
                 slope_err_dra=float("nan"), slope_err_ddec=float("nan"),
                 slope_significant=False, worst_off_mas=float("nan"),
                 clean=False)
    if len(ia) == 0:
        return empty
    order = np.lexsort((sep.arcsec, ia))
    ia_o, ib_o = ia[order], ib[order]
    first = np.concatenate(([True], ia_o[1:] != ia_o[:-1]))
    ia_n, ib_n = ia_o[first], ib_o[first]
    _, b_counts = np.unique(ib_n, return_counts=True)
    b_multi = set(np.unique(ib_n)[b_counts > 1])
    keep = np.array([bi not in b_multi for bi in ib_n])
    ia_n, ib_n = ia_n[keep], ib_n[keep]
    if len(ia_n) < min_stars:
        return empty

    cosd = np.cos(np.radians(a_f[ia_n].dec.value))
    dra = (b[ib_n].ra - a_f[ia_n].ra).to(u.arcsec).value * cosd * 1000.0 \
        - global_result["dra"]
    ddec = (b[ib_n].dec - a_f[ia_n].dec).to(u.arcsec).value * 1000.0 \
        - global_result["ddec"]
    m = mag_f[ia_n]

    m0 = np.floor(m.min() / bin_mag) * bin_mag
    nb = max(int(np.ceil((m.max() - m0) / bin_mag)), 1)
    bins = []
    for k in range(nb):
        lo, hi = m0 + k * bin_mag, m0 + (k + 1) * bin_mag
        sel = (m >= lo) & (m < hi)
        n = int(sel.sum())
        if n < min_stars:
            continue
        cdra = float(np.median(dra[sel]))
        cddec = float(np.median(ddec[sel]))
        mad_scale = 1.4826 / np.sqrt(n)
        dra_sem = float(np.median(np.abs(dra[sel] - cdra)) * mad_scale)
        ddec_sem = float(np.median(np.abs(ddec[sel] - cddec)) * mad_scale)
        off = float(np.hypot(cdra, cddec))
        sem = float(np.hypot(dra_sem, ddec_sem))
        significant = bool(sem > 0 and off > nsigma * sem)
        bins.append(dict(mag_lo=float(lo), mag_hi=float(hi),
                         mag_mid=float(0.5 * (lo + hi)), n=n,
                         dra_mas=cdra, ddec_mas=cddec,
                         dra_sem=dra_sem, ddec_sem=ddec_sem, off_mas=off,
                         significant=significant,
                         flagged=bool(off > tol_mas and significant)))

    # weighted linear slope across the measurable bins (a smooth mas/mag drift
    # can stay under the per-bin tolerance while accumulating across the range)
    slope = dict(dra=float("nan"), ddec=float("nan"),
                 edra=float("nan"), eddec=float("nan"))
    if len(bins) >= 2:
        x = np.array([bb["mag_mid"] for bb in bins])
        for comp, err in (("dra", "dra_sem"), ("ddec", "ddec_sem")):
            y = np.array([bb[f"{comp}_mas"] for bb in bins])
            w = 1.0 / np.maximum(np.array([bb[err] for bb in bins]), 1e-3) ** 2
            xm = np.sum(w * x) / np.sum(w)
            denom = np.sum(w * (x - xm) ** 2)
            if denom > 0:
                slope[comp] = float(np.sum(w * (x - xm) * y) / denom)
                slope["e" + comp] = float(np.sqrt(1.0 / denom))
    slope_sig = bool(
        (np.isfinite(slope["dra"]) and slope["edra"] > 0
         and abs(slope["dra"]) > nsigma * slope["edra"]
         and abs(slope["dra"]) * bin_mag * len(bins) > tol_mas)
        or (np.isfinite(slope["ddec"]) and slope["eddec"] > 0
            and abs(slope["ddec"]) > nsigma * slope["eddec"]
            and abs(slope["ddec"]) * bin_mag * len(bins) > tol_mas))

    flagged = [bb for bb in bins if bb["flagged"]]
    return dict(bins=bins, n_bins=len(bins), n_flagged=len(flagged),
                slope_dra_mas_per_mag=slope["dra"],
                slope_ddec_mas_per_mag=slope["ddec"],
                slope_err_dra=slope["edra"], slope_err_ddec=slope["eddec"],
                slope_significant=slope_sig,
                worst_off_mas=max((bb["off_mas"] for bb in bins),
                                  default=float("nan")),
                clean=bool(len(bins) >= 2 and not flagged and not slope_sig))
