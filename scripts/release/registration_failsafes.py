#!/usr/bin/env python
"""Local-registration failsafes for JWST-GC mosaics (spatially resolved).

A field-average astrometry check passes over a LOCALIZED seam/overlap misregistration
(brick 1182 F356W, 2026-07: several-arcsec junk in the module-overlap band, bulk ~0).
These checks are spatially binned and use CONFOUND-FREE truth sets (no external catalog,
so crowding/extinction can't fool them):

  1. per-module   : every bright MERGED detection must have a same-band per-module
                    (nrca/nrcb) detection within TOL.  The merged is the only place the
                    two modules are combined, so overlap-misregistration junk appears
                    here and not in the clean single-module mosaics.
  2. cross-band   : every bright detection must have a detection in ANOTHER JWST band
                    within TOL.  Same stars, JWST-internal registration is sub-mas, and
                    all bands are NIR -> no VIRAC2 color/depth decoupling.
  3. own-catalog  : every bright detection must have a source in the mosaic's OWN vetted
                    catalog within TOL (and the catalog must land on the mosaic).  A
                    mosaic must match the catalog derived from it.

Per cell: fraction of bright detections that have a truth-set match ("agreement") and the
median offset.  Agreement ~1 where registered; it COLLAPSES in a misregistered band.
FAIL if any covered cell drops below FRAC_FLOOR (or << field median) or offset > OFF_MAX.
A cell also FAILs, whatever its peak contrast, when it belongs to a connected patch of
MIN_SEAM_CELLS or more high-offset cells: a misregistration is a connected patch, while
wrong-pair noise is scattered singletons, and shape -- unlike the contrast statistic --
does not scale with star density (issue #170).
A peak beyond WINDOW_EDGE_FRAC * MX is ambiguous between a real large offset and the
arg-max of the wrong-pair background of a window too narrow to hold a true pair, so it
is RESOLVED before it is graded -- by its contrast, and by SWEEPING the window to 2x and
4x MX (issue #588).  A confirmed offset is graded at the swept value, which is how this
check reaches 2.5-10".  A cell that neither test can resolve measured nothing: it is
reported, and a connected patch of such cells makes the verdict None (could-not-verify),
never True.
Non-zero exit on FAIL (1) or could-not-verify (2) so it can gate a chain.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import namedtuple as _namedtuple

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import label as _ndlabel
from scipy.stats import poisson as _poisson
from scipy.stats import binned_statistic_2d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder

# The tree the products live under.  Read from the environment with the same
# spelling the sibling release gates use -- `check_interframe_overlap.py` and
# `check_astrometry_checkpoints.py` both take `JWST_BASE` with this default --
# so all three gates can be pointed at one tree together.  It was hardcoded
# here, and `fields.yaml` has TWO roots: `orange` (/orange/adamginsburg/jwst)
# and `blue` (/blue/adamginsburg/adamginsburg/jwst).  A `root: blue` field is
# reachable under /orange only if someone made a symlink for it (brick and
# cloudc have one; gc-treasury, root blue per #421, does not), and without one
# every glob below matched nothing -- the gate reported on an empty tree.
# The default is unchanged, so nothing moves for the fields that resolve today.
#
# `GC_BASEPATH_OVERRIDE` is deliberately NOT consulted (check_astrometry_
# checkpoints.py reads it ahead of JWST_BASE): in
# `jwst_gc_pipeline.scratch_basepath` that variable is a per-FIELD basepath --
# it already ends in the field name -- while BASE here is the root the globs
# join `{BASE}/{field}/...` onto, so honouring it would look for
# `<scratch>/brick/brick/F410M/pipeline`.
BASE = os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst")
GRID = 20
MX = 2.5 * u.arcsec              # pair-separation search radius (recovers offsets up to this)
XBIN = 0.04                      # arcsec, offset-histogram bin
MIN_PAIRS = 80                   # pairs needed in a cell to attempt a peak
MIN_PEAK_RATIO = 5.0             # peak/background below this -> cell UNVERIFIED (not a fail)
FAIL_MIN_RATIO = 10.0            # a FAIL needs peak/background >= this -- CONFIDENT contrast,
                                 # not just the verify floor. A real localized seam doubles
                                 # stars into a SHARP secondary peak (the clean brick cells
                                 # verify at median contrast ~18); a floor-level peak
                                 # (ratio ~ MIN_PEAK_RATIO) at a large offset is dense-field
                                 # wrong-pair noise in a crowded, few-detection cell, not a
                                 # seam (brick F405N: 7 bright-star cells at 80 mas / peak_bg
                                 # 5-8 were a FALSE own_catalog FAIL; the same-star m7 check
                                 # of those regions read <=22 mas, 2026-07). Coverage is
                                 # unchanged (no detections removed); only the fail bar rises.
OFF_MAX = 60.0                   # a VERIFIED cell whose peak offset exceeds this (mas) -> FAIL

# A SECOND, INDEPENDENT reason for a cell to fail, added ALONGSIDE the contrast bar
# above and never in place of it (issue #170).
#
# `ratio` is `H.max() / median(H[H>0])`, and that divisor is 1 in every cell measured:
# the 2.5" search disk holds 12,271 bins of 40x40 mas against 90-10,476 pairs per cell,
# so the median OCCUPIED bin never rises off one count (1199 of 1199 cells across brick
# F405N/F187N/F212N, measured 2026-08-22).  `ratio` is therefore the raw peak-BIN
# COUNT, which grows with the cell's star count: on those 1199 cells -- every one of
# which peaks in the ZERO-offset bin, i.e. is identically well registered -- it spans
# 3 to 127, a factor of 42, regressing against pair count with a log-log slope of 0.72.
# A fixed bar on it is a density cut.
#
# The consequence this constant addresses is measured, not modelled.  #179 injected
# +90 mas seams into real brick F405N data and read their per-cell `ratio` as 5-49 with
# medians of 12-18.  `FAIL_MIN_RATIO = 10` sits INSIDE that distribution: the seam's
# sparser cells score below it and are recorded as `unconfident_highoff` rather than as
# failures.  A seam confined to sparse regions can therefore be measured, reported, and
# not blocked.
#
# Contiguity does not depend on the amplitude scale at all.  A misregistration is a
# CONNECTED patch of cells -- a seam, a visit footprint, a module overlap -- while
# wrong-pair noise is scattered singletons.  #179's trial found it much the stronger
# axis: it fired on 365 / 179 / 45 cells of three injected seams against the contrast
# bar's 273 / 127 / 29, and on ZERO cells across ten brick bands and five cloudc bands.
#
# MIN_SEAM_CELLS = 3, not 2, and that lower bound is measured: two of the seven brick
# F405N cells that were a false own-catalog FAIL in July 2026 -- (12,13) and (13,13) --
# are 4-adjacent, so a 2-cell bar re-creates the exact false alarm #166/#172 removed.
# Real component sizes on that band were [2, 1, 1, 1, 1, 1]; the injected seams' largest
# components were 371, 184 and 26 cells, so 3 leaves ~100x headroom on a real seam while
# clearing the known false-positive population.  Those seven cells are no longer on disk
# (F405N's merged mosaic was rebuilt 2026-08-22 and every verified cell now peaks in the
# zero bin), which is why this is the axis that needs NO recalibration: it is a shape
# requirement, not a threshold in the units that moved.
#
# 4-CONNECTIVITY (edge neighbours, not corners).  Diagonal-only touching is one cell's
# worth of contact and is what a scattered pair of noise cells produces; requiring a
# shared edge is the stricter reading and is what the [2,1,1,1,1,1] figure above counts.
#
# Strictly ADDITIVE.  `fail` is OR-ed with this, so the seam axis can only add failures
# and can never turn an existing FAIL into a PASS.  Whether a LONE high-offset,
# high-contrast cell should stop failing the field -- #170's proposal 2 in full -- is a
# RELAXATION of the gate and is deliberately not part of this.
MIN_SEAM_CELLS = 3               # connected high-offset cells that FAIL regardless of contrast

# THE WINDOW-EDGE ARM (issue #588) -- the only part of this file that reads the SEARCH
# WINDOW itself.  `MX` is a FIXED 2.5" pair-separation disk, so a cell holding no true
# counterpart pair inside 2.5" still returns an arg-max: the largest bin of the
# wrong-pair background.  That background is uniform over the disk, so its arg-max
# radius follows p(r) dr ~ r dr -- median 0.71*MX = 1768 mas, 75 % of it beyond 0.5*MX.
# `gc2211_o028` F150W merged reads exactly that shape: 138 of 139 verified cells
# high-offset, median 1826 mas (0.73*MX), worst 2487 mas (0.995*MX), every one at
# peak_bg 5.0-7.0 -- the verify floor.
#
# A REAL rigid 1.3-2.5" misregistration puts its arg-max at the same RADIUS, so radius
# ALONE cannot separate the two, and a cell must never be ungraded on radius alone: that
# converts the loudest failure this gate can see -- a clean, sharp, rigid 2" shift --
# into a PASS.  CLAUDE.md's rule for the ambiguity is not to stop grading but to SWEEP
# THE WINDOW ("a real tie reads the SAME offset at every window that can contain it;
# a swept peak near the window EDGE is geometry").  So a near-edge cell is RESOLVED, by
# two independent measurements, before anything is withdrawn:
#
#   1. CONTRAST at the base window.  The background's peak is the extreme value of ~0.7
#      counts/bin over 12,271 bins: 5-7, i.e. the verify floor, which is what o028
#      reads.  A rigid shift piles true counterparts into ONE bin: the same shifts
#      injected synthetically read peak_bg 16-24 at the same pair counts, a factor of 4
#      clear.  A near-edge cell at >= `fail_min_ratio` is a measurement and is graded
#      unchanged.  For the STRICT checks (cross-band, per-module) `fail_min_ratio` IS
#      the verify floor, so every verified cell clears it and this arm never fires for
#      them at all -- the diagnosis behind it (an own-catalog truth 12.6x denser than
#      the detection list) is specific to the own-catalog leg.
#   2. THE SWEEP (`sweep_cell_windows`), for the rest.  The cell's own pairs are
#      re-histogrammed at 2x and 4x MX.  A window RESOLVES the cell when its peak
#      stands clear of that window's own rim AND is improbable under that window's own
#      wrong-pair background -- `expected <= SWEEP_MAX_EXPECTED_BINS`, the number of
#      bins of the cell's background expected to reach the observed peak, which carries
#      the window with it (the bin count grows as W^2, a fixed contrast floor does not).
#      A resolved cell is graded AT THE SWEPT VALUE -- which is how this check measures
#      the 2.5-10" regime it previously could only alias to ~0.7*MX.
#
#      What the sweep is NOT: a cross-window agreement test.  `SWEEP_FACTORS` are
#      NESTED searches on a shared 40 mas bin grid, so the wider window's pair set is a
#      superset of the narrower one's and their histograms are bin-for-bin equal inside
#      the narrower disk.  Whenever the wider window resolves (its arg-max inside half
#      its own window = inside the narrower window's disk) it therefore reports the
#      narrower window's arg-max EXACTLY, and "the offsets agree" is arithmetic rather
#      than evidence.  That was the confirmation rule when this arm landed (#758) and
#      the review of that PR refuted it; the look-elsewhere statistic replaced it.  The
#      vector comparison is retained as a consistency assertion.
#
# Only a cell that survives BOTH is withdrawn, and withdrawing is not passing.  It is
# recorded (`n_window_edge`, `window_edge_cells`, each with its `swept_windows`), and a
# 4-connected patch of `MIN_SEAM_CELLS` withdrawn cells -- a REGION that could not be
# measured, the "no tie exists inside the window" shape of brick-1182 (20"), cloudc
# F410M (4.06") and sgra-1939 (14.8") -- makes the check's `PASS` **None**, which
# `main()` returns as exit 2 and `stage_release.py` refuses with "could NOT VERIFY".
# The block / no-block boundary is therefore not moved by this arm; what changes is the
# claim attached to it (a measured 1.8" misregistration vs "this cell measured
# nothing") and the 10" of reach the sweep adds.
#
# 0.5 here against `astrometry_offsets.WINDOW_EDGE_FRACTION = 0.85`: that estimator
# sweeps 3/10/30/60", so a peak surviving at 0.85 of its window has already been checked
# against a wider one.  0.5 is where this file STARTS sweeping, not where it stops
# believing.
WINDOW_EDGE_FRAC = 0.5           # off > this * MX -> the base window alone is not
                                 # evidence; confirm by contrast or by sweeping
SWEEP_FACTORS = (2.0, 4.0)       # multiples of MX the near-edge cells are re-measured
                                 # at (5" and 10").  What the extra window buys is REACH
                                 # -- an offset of 2.5-10" has no true pair inside MX at
                                 # all, so only a wider search can see it.  It does NOT
                                 # buy an independent second opinion: the searches are
                                 # nested on a shared bin grid (see `sweep_cell_windows`)
SWEEP_AGREE_MAS = 120.0          # 3 histogram bins.  The tolerance of the cross-window
                                 # VECTOR consistency check.  Note what this is NOT: with
                                 # nested windows on a shared bin grid, two resolving
                                 # windows agree ARITHMETICALLY (see `sweep_cell_windows`),
                                 # so this is an assertion, not the evidence.
SWEEP_MAX_EXPECTED_BINS = 0.01   # LOOK-ELSEWHERE bar for a swept peak: the expected
                                 # number of bins of the cell's OWN wrong-pair background
                                 # reaching the observed peak count must be <= this.
                                 # A background arg-max sits at ~1 by construction at
                                 # EVERY window, so 0.01 is two orders of magnitude
                                 # clear of it; the real ties this file is calibrated on
                                 # (a rigid shift, and a 6-counterpart-per-cell shift in
                                 # a 15x denser truth) read far below it.  This replaces
                                 # applying `MIN_PEAK_RATIO` / `FAIL_MIN_RATIO` -- both
                                 # calibrated at the 2.5" BASE window (#179) -- at 5" and
                                 # 10", where the bin count is 4x and 16x and the
                                 # background's own maximum is correspondingly higher.
SWEEP_MAX_PAIRS = 2_000_000      # cap on the transient pair array of ONE wider-window
                                 # search.  Pairs grow as W^2, so the detection batch is
                                 # sized from the base window's measured pairs-per-
                                 # detection: a 4x window means 16x the pairs.

OVERLAP_STRIDE = 16              # pixel stride when sampling a mosaic for module overlap
MIN_OVERLAP_SAMPLES = 50         # sampled positions with real data in BOTH modules before
                                 # the modules count as overlapping.  At the i2d pixel scale
                                 # a stride-16 sample is ~0.5"; 50 samples is a few arcsec of
                                 # genuine shared sky, well under the thinnest real seam
                                 # measured (sgrc F360M, 279) and far above the 0 that two
                                 # abutting modules give (arches F212N/F323N).


def detect(path, thr=30.0):
    h = fits.open(path); sci = h["SCI"]; w = WCS(sci.header); d = sci.data.astype("float32")
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=2.5, threshold=thr * std)(d - med)
    if t is None:
        return None, None
    return SkyCoord(w.pixel_to_world(t["xcentroid"], t["ycentroid"])), np.asarray(t["flux"], float)


# Precise merged-mosaic filename parser.  The bug being fixed is that `g[0]` on
# an UNSORTED `glob.glob(jw*-o*...)` picked a non-deterministic file whenever >1
# matched.  We enumerate with a tight character-class glob (no `*` in the
# proposal/observation) and validate each name with this regex, which yields the
# proposal-observation key.  NOTE: >1 observation in one filter directory is a
# NORMAL layout, not a stray -- gc2211 is multi-observation by design and
# ngc6334 F200W carries two proposals (both o001) on purpose.  So the ambiguity
# is resolved by (a) an optional release scope and (b) a DETERMINISTIC sorted
# pick, NOT by refusing.  Distinguishing a genuine multi-observation layout from
# a misfiled stray is what the release ``observations`` scope is for; a
# within-directory obs count cannot tell them apart.  Filter class allows the
# wide-double bands (F150W2/F322W2) -- their trailing `2` was silently dropped,
# and a dropped band fails OPEN (cross-band needs >=2 bands or it warns-not-fails).
_MOSAIC_RE = re.compile(
    r"^jw(?P<prop>\d{5})-o(?P<obs>\d{3})_t001_nircam_clear-"
    r"(?P<filt>f\d{3,4}[wmn]2?)-(?P<module>merged|nrca|nrcb|nrcalong|nrcblong)"
    r"_i2d\.fits$")


def _mosaic_candidates(field, filt, module, observations=None):
    """On-disk mosaics for (field, filt, module) as sorted (obs_key, path),
    name-validated.  ``obs_key`` = ``"<proposal>-<observation>"``.  When
    ``observations`` (a set of obs_keys) is given, only in-scope mosaics are
    returned -- this is how a misfiled stray from another observation is
    excluded (brick's 2221 o002), while a legitimate multi-observation layout
    keeps all its in-scope mosaics."""
    # tight glob: 5-digit proposal, 3-digit observation -- no `*` in either
    pat = (f"{BASE}/{field}/{filt}/pipeline/"
           f"jw[0-9][0-9][0-9][0-9][0-9]-o[0-9][0-9][0-9]_t001_nircam_clear-"
           f"{filt.lower()}-{module}_i2d.fits")
    out = []
    for p in sorted(glob.glob(pat)):
        m = _MOSAIC_RE.match(os.path.basename(p))
        if m and m.group("filt") == filt.lower() and m.group("module") == module:
            key = f"{m.group('prop')}-{m.group('obs')}"
            if observations is not None and key not in observations:
                continue
            out.append((key, p))
    return sorted(out)


def mosaic(field, filt, module="merged", observations=None):
    """The merged mosaic for (field, filt, module).  Deterministic: a sorted
    pick of the (in-scope) name-validated candidates -- fixes the non-
    deterministic `g[0]` on an unsorted glob.  Returns None when none match."""
    cands = _mosaic_candidates(field, filt, module, observations=observations)
    return cands[0][1] if cands else None


def catalog_module_tokens(view):
    """Catalog module tokens for a scan *view*.

    The vetted catalog name carries the module in the slot after the filter, the
    same way the mosaic name does -- and a field that never made a ``merged``
    product has no ``merged`` catalog either.  sickle is nrcb-only and writes
    ``f187n_nrcb_...``; sgrb2 keeps eight nrcb-only bands.  Asking for
    ``merged`` in a per-module view therefore matched nothing for them, and a
    no-match was indistinguishable from "this field has no catalog at all".

    Both spellings are returned for a module view because a field names its LW
    products with either token (``nrcb`` or ``nrcblong``) -- the ambiguity
    ``module_family`` already absorbs on the mosaic side.
    """
    if not str(view).startswith("module-"):
        return ("merged",)
    fam = str(view).split("-", 1)[1]
    return (f"nrc{fam}", f"nrc{fam}long")


def catalog_candidates(field, filt, view="merged"):
    """Sorted m7 vetted catalogs for (field, filt, view).

    An observation token appears in ONE OF TWO slots -- after the module
    (``f200w_nrcb_o050_indivexp_...``, the post-#469 spelling) or at the end
    (``..._dao_basic_o050_vetted.fits``, the pre-#469 one) -- and gc2211 o050
    carries BOTH on disk simultaneously, so a pattern that pins one placement
    matches half of its own field.  All three forms are enumerated, with the
    token as a 3-digit character class rather than ``*`` so it cannot span a
    neighbouring name segment.
    """
    out = []
    for mod in catalog_module_tokens(view):
        stem = f"{filt.lower()}_{mod}"
        tail = "indivexp_merged_resbgsub_m7_dao_basic"
        for pat in (f"{stem}_{tail}_vetted.fits",
                    f"{stem}_{tail}_o[0-9][0-9][0-9]_vetted.fits",
                    f"{stem}_o[0-9][0-9][0-9]_{tail}_vetted.fits"):
            out += glob.glob(f"{BASE}/{field}/catalogs/{pat}")
    return sorted(set(out))


def no_catalog_note(field, filt, view):
    """The line a band with no vetted catalog contributes to the report.

    Its own function so the wording is testable without standing up a mosaic:
    the point of the line is that the absence is STATED, and a source-grep for a
    phrase that happens to straddle two f-string fragments would pass on a
    version that says nothing.
    """
    return (f"{filt}: no m7 vetted catalog under {field}/catalogs matching view "
            f"{view} ({'/'.join(catalog_module_tokens(view))}); own-catalog "
            f"check not run")


def catalog_sc(field, filt, view="merged"):
    """Positions from the m7 vetted catalog for (field, filt, view), or None.

    Deterministic when more than one candidate matches: a sorted pick, as
    ``mosaic`` does.
    """
    g = catalog_candidates(field, filt, view)
    if not g:
        return None
    t = Table.read(g[0])
    for c in ("skycoord", "skycoord_ref"):
        m = [x for x in t.colnames if x.lower() == c]
        if m:
            return SkyCoord(t[m[0]])
    return None


def seam_mask(highoff, min_cells=MIN_SEAM_CELLS):
    """Cells belonging to a 4-connected component of ``min_cells`` or more.

    ``highoff`` is the boolean grid of VERIFIED cells whose peak offset exceeds
    ``OFF_MAX``.  Returns ``(mask, sizes)``: the same-shaped boolean grid keeping only
    cells in a large-enough component, and the sorted (descending) list of the sizes of
    the components kept.

    A shape test, not an amplitude test -- see ``MIN_SEAM_CELLS``.  It reads only the
    grids ``per_cell`` already computes, so it costs nothing to measure.
    """
    highoff = np.asarray(highoff, bool)
    if not highoff.any():
        return np.zeros_like(highoff), []
    # 4-connectivity: edge neighbours only.  scipy's default structure is exactly this.
    lab, n = _ndlabel(highoff)
    keep = np.zeros_like(highoff)
    sizes = []
    for k in range(1, n + 1):
        member = lab == k
        size = int(member.sum())
        if size >= min_cells:
            keep |= member
            sizes.append(size)
    return keep, sorted(sizes, reverse=True)


_PeakStats = _namedtuple("_PeakStats", "off ratio dra_mas dde_mas peak lam n_bins expected")


def could_not_verify_patch(highoff, edge, min_cells=MIN_SEAM_CELLS):
    """The could-not-verify shape test: which withdrawn cells form a REGION.

    Scattered withdrawn singletons are the wrong-pair background and are only reported.
    A 4-connected patch is a region of the mosaic this estimator could not measure at
    all -- the brick-1182 "no tie exists inside the window" shape -- and that is not a
    pass: it takes ``PASS`` to None (exit 2, "could NOT VERIFY"), never to True.

    The QUORUM is counted on the UNION of the two populations, not on the withdrawn
    cells alone.  One physical misregistration lands partly in ``highoff`` (cells the
    estimator MEASURED) and partly in ``edge`` (cells it COULD NOT), and testing the two
    halves against ``min_cells`` SEPARATELY can leave a real region below the quorum on
    both axes at once: 2 measured + 2 withdrawn is a 4-cell region that neither 3-cell
    test sees, so it blocks on neither.  On the union it is one component of 4 and it
    blocks.  (Review of #758, B5.)

    Cells keep their own verdict inside a quorate region -- ``highoff`` members fail on
    the seam axis exactly as before, ``edge`` members make ``PASS`` None -- so this can
    only ADD blocking, and a cell is never reported as a MEASURED misregistration on the
    strength of neighbours that measured nothing.

    Returns ``(edge_patch, edge_sizes, region, region_sizes)``.
    """
    highoff = np.asarray(highoff, bool)
    edge = np.asarray(edge, bool)
    region, region_sizes = seam_mask(highoff | edge, min_cells=min_cells)
    return (region & edge, _component_sizes_touching(region, edge),
            region, region_sizes)


def _component_sizes_touching(region, members):
    """Sizes of the 4-connected components of ``region`` that contain a ``members`` cell.

    ``region`` is the quorate union of the measured-high-offset and could-not-measure
    populations; ``members`` picks one of them.  Reporting the size of the WHOLE region
    a withdrawn cell belongs to (rather than of the withdrawn cells alone) is the point:
    it is the size of the mosaic patch involved, which is what a reader needs.
    """
    region = np.asarray(region, bool)
    members = np.asarray(members, bool)
    if not (region & members).any():
        return []
    lab, n = _ndlabel(region)
    return sorted({int((lab == k).sum()) for k in np.unique(lab[region & members])
                   if k > 0}, reverse=True)


def _hist_peak_stats(dra, dde, half_window_mas, bin_mas=XBIN * 1000):
    """Full statistics of ONE cell's offset-histogram peak inside a window (mas).

    Returns a ``_PeakStats``: the peak's offset MAGNITUDE and its two COMPONENTS
    (``dra_mas``, ``dde_mas`` -- two peaks at the same radius in opposite directions are
    not the same measurement), the peak/background ratio, and the LOOK-ELSEWHERE
    statistic ``expected``.

    ``expected`` is how many bins of this cell's OWN wrong-pair background are expected
    to reach the observed peak count: ``n_bins * P(Poisson(lam) >= peak)``, with
    ``lam = n_pairs / n_bins`` and ``n_bins`` the bins covering the searched DISK (the
    pairs come from a radius search, so they populate a disk, not the square).  It is
    ~1 for a background arg-max by construction -- that is what an extreme value IS --
    and orders of magnitude below 1 for a real tie.

    This is the statistic the SWEEP needs and ``ratio`` is not.  ``n_bins`` grows as
    W^2, so a contrast floor calibrated at one window (``MIN_PEAK_RATIO`` /
    ``FAIL_MIN_RATIO``, both calibrated at the 2.5" base window, #179) is a DIFFERENT,
    weaker bar when applied at 5" or 10": the background's own maximum rises with the
    number of bins searched while the floor does not.  ``expected`` carries the window
    with it.  See ``sweep_cell_windows``.

    LIMIT: the null is Poisson and uniform over the disk.  A clustered stellar field's
    wrong-pair background is over-dispersed relative to that, so ``expected`` is a lower
    bound on the true look-elsewhere probability -- which is why the bar it is compared
    against (``SWEEP_MAX_EXPECTED_BINS``) sits two orders of magnitude below the null's
    own maximum rather than at it, and why the ``MIN_PEAK_RATIO`` floor is kept
    alongside it rather than replaced by it.
    """
    hb = np.arange(-half_window_mas, half_window_mas + bin_mas, bin_mas)
    H, xb, yb = np.histogram2d(dra, dde, bins=[hb, hb])
    n_bins = max(1.0, np.pi * float(half_window_mas) ** 2 / float(bin_mas) ** 2)
    if not (H > 0).any():
        return _PeakStats(np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, n_bins, np.inf)
    bg = np.median(H[H > 0])
    pi, pj = np.unravel_index(H.argmax(), H.shape)
    peak = float(H.max())
    ratio = float(peak / bg) if bg > 0 else np.inf
    dra_c = float((xb[pi] + xb[pi + 1]) / 2)
    dde_c = float((yb[pj] + yb[pj + 1]) / 2)
    lam = float(np.size(dra)) / n_bins
    expected = float(n_bins * _poisson.sf(peak - 1, lam))
    return _PeakStats(float(np.hypot(dra_c, dde_c)), ratio, dra_c, dde_c,
                      peak, lam, n_bins, expected)


def _hist_peak(dra, dde, half_window_mas, bin_mas=XBIN * 1000):
    """Offset-histogram peak of ONE cell's pair set inside +-``half_window_mas`` (mas).

    Returns ``(offset_mas, peak_over_background)``; ``(nan, nan)`` for an empty cell.

    Factored out of ``per_cell`` so the SAME statistic can be re-run on one cell at a
    WIDER window (``sweep_cell_windows``).  The sweep compares a cell against itself
    across windows, so base and swept measurements have to be the identical estimator
    or their agreement means nothing.
    """
    s = _hist_peak_stats(dra, dde, half_window_mas, bin_mas)
    return s.off, s.ratio


def _det_cell_batches(cell_of_det, cells, chunk):
    """Detection indices for ``cells``, grouped into batches of <= ``chunk`` detections.

    A batch NEVER splits a cell: the sweep histograms a cell's pairs as one set, and a
    cell measured twice from two halves of its own detections is not the same
    measurement.  A single cell larger than ``chunk`` is its own batch.
    """
    sel = np.flatnonzero(np.isin(cell_of_det, np.array(sorted(cells), dtype=int)))
    if sel.size == 0:
        return []
    sel = sel[np.argsort(cell_of_det[sel], kind="stable")]
    cd = cell_of_det[sel]
    ucell, cstart = np.unique(cd, return_index=True)
    cstop = np.append(cstart[1:], cd.size)
    batches, cur, size = [], [], 0
    for a, b in zip(cstart, cstop):
        if cur and size + (b - a) > chunk:
            batches.append(np.concatenate(cur))
            cur, size = [], 0
        cur.append(sel[a:b])
        size += b - a
    if cur:
        batches.append(np.concatenate(cur))
    return batches


def sweep_cell_windows(det, truth, cell_of_det, cells, base,
                       factors=SWEEP_FACTORS, window_edge_frac=WINDOW_EDGE_FRAC,
                       agree_mas=SWEEP_AGREE_MAS, pairs_per_det=1.0,
                       max_pairs=SWEEP_MAX_PAIRS,
                       max_expected_bins=SWEEP_MAX_EXPECTED_BINS):
    """SWEEP the search window for the cells whose base-window peak rode its rim.

    This is the step that keeps a window-edge peak from being waved through.  A peak at
    0.7*MX has two possible causes -- a REAL misregistration of 1.8 arcsec, or the
    arg-max of the wrong-pair background of a window too narrow to hold any true pair --
    and the radius is the same in both.  CLAUDE.md's rule for that ambiguity is to widen
    the window: *a real tie reads the SAME offset at every window that can contain it*,
    while the background's arg-max radius scales WITH the window (median 0.71*W).

    ``cells``      flat cell keys (``i * GRID + j``) to re-measure.
    ``cell_of_det`` the flat cell key of every DETECTION (not of every pair).
    ``base``       ``{cell: _PeakStats}`` from the base MX window.
    ``pairs_per_det`` pairs per detection measured at the BASE window, used to size the
                   detection batches so one wider search stays under ``max_pairs``.

    Each cell is re-measured at each of ``factors`` x MX from a fresh pair search at
    that radius, restricted to that cell's detections.  A measurement RESOLVES an offset
    when its peak stands clear of its OWN rim (``off <= window_edge_frac * W``) AND is
    improbable under that window's own wrong-pair background: the LOOK-ELSEWHERE
    statistic ``expected`` (see ``_hist_peak_stats``) must be ``<= max_expected_bins``.
    A cell is CONFIRMED -- there is a tie, and the gate goes back to grading it, at the
    swept value -- when at least one window resolves and every resolving window reads
    the same offset VECTOR to within ``agree_mas``.  It is graded at the WIDEST
    resolving window.

    WHY NOT "the offsets agree across windows", which is what this function did when it
    landed (#758) and what the review of that PR refuted:

    * ``SWEEP_FACTORS`` are NESTED searches on a SHARED 40 mas bin grid (the 5" grid's
      edges are a subset of the 10" one's, both being multiples of the bin), so the 10"
      pair set is a superset of the 5" one and their histograms are bin-for-bin equal
      wherever both are defined.  If the 10" arg-max lands inside radius 5000 -- which
      is exactly the condition for the 10" window to "resolve" at
      ``window_edge_frac = 0.5`` -- then it is also the arg-max of the 5" histogram.
      Agreement between the two is then arithmetic, not evidence.  Measured on a uniform
      pair background: of the trials where both windows resolved, 3 of 3 "reproduced".
    * and in the 2.5-5" band only the widest window can resolve at all, so there was no
      second measurement to agree with and confirmation fell through to a raw contrast
      floor calibrated at the 2.5" base window (#179) -- a floor the background's own
      maximum rises past as the bin count grows with W^2.

    ``expected`` is the statistic that does carry the window: it asks how many bins of
    this cell's own background are expected to reach the observed peak, so it is ~1 for
    a background arg-max at EVERY window and tiny for a real tie.  The cross-window
    vector comparison is kept as a consistency assertion (and now compares vectors, not
    magnitudes, so two peaks at the same radius in opposite directions no longer count
    as the same reading), but it is no longer the evidence.

    Returns ``(confirmed, measurements)``: ``{cell: (off_mas, ratio)}`` for confirmed
    cells, and ``{cell: [(window_mas, _PeakStats), ...]}`` for every swept cell.
    """
    mx_mas = MX.to(u.mas).value
    meas = {int(k): [(mx_mas, base[int(k)])] for k in cells}
    if not meas:
        return {}, {}
    for fac in factors:
        w = float(fac) * mx_mas
        rad = (w / 1000.0) * u.arcsec
        # Pairs grow as W^2, so the batch shrinks as 1/fac^2 off the base window's
        # measured pairs-per-detection.  Without this a dense field's 4x search builds
        # a pair array 16x the base one in a single allocation.
        chunk = max(1, int(max_pairs / max(1.0, pairs_per_det * fac * fac)))
        batches = _det_cell_batches(cell_of_det, set(meas), chunk)
        for block in batches:
            sub = det[block]
            ia, ib, _, _ = search_around_sky(sub, truth, rad)
            if len(ia) == 0:
                continue
            dra = ((truth[ib].ra - sub[ia].ra).to(u.arcsec).value
                   * np.cos(sub[ia].dec.rad) * 1000)
            dde = (truth[ib].dec - sub[ia].dec).to(u.arcsec).value * 1000
            kk = cell_of_det[block][ia]
            o = np.argsort(kk, kind="stable")
            kk, dra, dde = kk[o], dra[o], dde[o]
            uk, start = np.unique(kk, return_index=True)
            stop = np.append(start[1:], kk.size)
            for k, a, b in zip(uk, start, stop):
                if int(k) not in meas or b - a < MIN_PAIRS:
                    continue
                meas[int(k)].append((w, _hist_peak_stats(dra[a:b], dde[a:b], w)))
    confirmed = {}
    for k, ms in meas.items():
        # The merged rule's two conditions (clear of its own rim, above the verify
        # floor) AND the look-elsewhere bar.  Strictly narrower than what shipped, so a
        # cell can only move from confirmed-and-graded to withdrawn -- which is why the
        # withdrawn population has to share the seam axis's quorum (see `per_cell`),
        # or a small real region could fall below both quorums at once.
        resolving = sorted([(w, st) for (w, st) in ms
                            if np.isfinite(st.off) and np.isfinite(st.ratio)
                            and st.ratio >= MIN_PEAK_RATIO
                            and st.off <= window_edge_frac * w
                            and st.expected <= max_expected_bins],
                           key=lambda t: t[0])
        if not resolving:
            continue
        # Grade at the WIDEST resolving window.  The narrowest is the one whose true
        # counterpart is most likely to lie outside it, i.e. the one most likely to be
        # reading an alias.
        best = resolving[-1][1]
        # Vector consistency, not magnitude: |(dra, dde) - (dra, dde)|, so a peak at the
        # same radius in the opposite direction is a disagreement.
        consistent = all(np.hypot(st.dra_mas - best.dra_mas,
                                  st.dde_mas - best.dde_mas) <= agree_mas
                         for (_, st) in resolving)
        if consistent:
            confirmed[k] = (best.off, best.ratio)
    return confirmed, meas


def per_cell(det, flux, truth, label, bright_pct=None, fail_min_ratio=MIN_PEAK_RATIO,
             min_seam_cells=MIN_SEAM_CELLS, window_edge_frac=WINDOW_EDGE_FRAC,
             sweep=True, sweep_factors=SWEEP_FACTORS):
    """Per-cell registration offset by pair-separation HISTOGRAM cross-correlation.

    For every det-truth pair within MX, bin by the detection's spatial cell and by the
    offset (dRA*cos, dDec).  In each cell the REAL counterparts pile into a peak at the
    true offset; chance coincidences form a flat background -> crowding-robust (NOT
    nearest-neighbour, which just measures the chance-NN distance in a dense field).

    A cell is VERIFIED only if it has >=MIN_PAIRS and peak/background >=MIN_PEAK_RATIO;
    otherwise it is UNVERIFIED (reported, never a fail).  A verified cell FAILs if its
    peak offset exceeds OFF_MAX *and* its contrast >= ``fail_min_ratio``.  Field FAIL =
    any cell fails.

    ``fail_min_ratio`` (default ``MIN_PEAK_RATIO`` -> the historic strict behaviour) is
    raised to ``FAIL_MIN_RATIO`` ONLY for the own-catalog check, where a floor-level
    peak in a dense bright-star cell is wrong-pair noise, not a seam.  The cross-band
    and per-module checks keep the strict floor, so a real seam that own-catalog's
    relaxed bar might miss is still caught by them (defense in depth).

    A cell ALSO fails, whatever its contrast, when it belongs to a 4-connected patch of
    ``min_seam_cells`` or more high-offset verified cells (issue #170).  The contrast
    bar is arithmetically the raw peak-BIN COUNT and so is a density cut; real injected
    seams score 5-49 on it, straddling ``FAIL_MIN_RATIO``.  Shape does not scale with
    density.  The two axes are OR-ed, so this can only add failures.

    WINDOW EDGE (issue #588).  A peak beyond ``window_edge_frac * MX`` is not trusted
    from the base window alone -- at that radius a real misregistration and the arg-max
    of the wrong-pair background of a window too narrow to hold a true pair are the same
    number.  Such a cell is resolved, never merely dropped:

    * a CONFIDENT peak there (``ratio >= fail_min_ratio``) is a tie and is graded
      unchanged -- which is every verified cell for the strict checks, where
      ``fail_min_ratio`` IS the verify floor, so this arm is inert for them;
    * otherwise the cell is SWEPT (``sweep_cell_windows``: 2x and 4x MX).  A peak that
      stands clear of its own window's rim AND is improbable under that window's own
      background (the look-elsewhere statistic, which scales with the window as a fixed
      contrast floor does not) is confirmed and graded AT THE SWEPT VALUE, which is how
      this check reaches the 2.5-10 arcsec regime it used to alias;
    * only a cell that survives both is withdrawn, as ``n_window_edge`` /
      ``window_edge_cells`` -- and withdrawing is not passing.  A 4-connected patch of
      ``min_seam_cells`` withdrawn cells makes ``PASS`` **None**: could-not-verify,
      which ``main`` returns as exit 2 and ``stage_release`` refuses on.

    The ``min_seam_cells`` quorum is counted on the UNION of the high-offset and
    withdrawn populations, because one physical misregistration lands in BOTH: testing
    the two halves separately can leave a real region under the quorum on each axis at
    once (2 + 2 is a 4-cell region that neither 3-cell test sees).  Cells keep their own
    verdict inside a quorate region -- high-offset ones fail, withdrawn ones make
    ``PASS`` None -- so a cell is never called a measured misregistration on the
    strength of neighbours that measured nothing.

    ``PASS`` is tri-state: False (a cell failed), None (nothing failed but a coherent
    region could not be measured), True.
    """
    if det is None or truth is None or len(det) < 200 or len(truth) < 200:
        return dict(label=label, error="missing detections/truth")
    ia, ib, sep, _ = search_around_sky(det, truth, MX)
    if len(ia) < 2000:
        return dict(label=label, error="too few pairs")
    dra = (truth[ib].ra - det[ia].ra).to(u.arcsec).value * np.cos(det[ia].dec.rad) * 1000
    dde = (truth[ib].dec - det[ia].dec).to(u.arcsec).value * 1000
    pra, pde = det[ia].ra.deg, det[ia].dec.deg

    xe = np.linspace(det.ra.deg.min(), det.ra.deg.max(), GRID + 1)
    ye = np.linspace(det.dec.deg.min(), det.dec.deg.max(), GRID + 1)
    ci = np.clip(np.digitize(pra, xe) - 1, 0, GRID - 1)
    cj = np.clip(np.digitize(pde, ye) - 1, 0, GRID - 1)
    mx_mas = MX.to(u.mas).value
    # The cell of every DETECTION (ci/cj above are per PAIR).  The sweep re-searches
    # from the detections, so it needs this map.
    cell_of_det = (np.clip(np.digitize(det.ra.deg, xe) - 1, 0, GRID - 1) * GRID
                   + np.clip(np.digitize(det.dec.deg, ye) - 1, 0, GRID - 1))

    off = np.full((GRID, GRID), np.nan)      # peak offset (mas)
    ratio = np.full((GRID, GRID), np.nan)    # peak/background
    npair = np.zeros((GRID, GRID), int)
    base_stats = {}                          # cell -> _PeakStats at the BASE window
    order = np.lexsort((cj, ci))
    ci, cj, dra, dde = ci[order], cj[order], dra[order], dde[order]
    keyc = ci * GRID + cj
    bnd = np.searchsorted(keyc, np.arange(GRID * GRID + 1))
    for k in range(GRID * GRID):
        s, e = bnd[k], bnd[k + 1]
        npair[k // GRID, k % GRID] = e - s
        if e - s < MIN_PAIRS:
            continue
        st = _hist_peak_stats(dra[s:e], dde[s:e], mx_mas)
        base_stats[k] = st
        off[k // GRID, k % GRID], ratio[k // GRID, k % GRID] = st.off, st.ratio

    verified = np.isfinite(ratio) & (ratio >= MIN_PEAK_RATIO) & (npair >= MIN_PAIRS)
    # WINDOW EDGE (issue #588).  A peak beyond half the FIXED MX disk is ambiguous
    # between a real large offset and the background arg-max of a window too narrow to
    # hold a true pair, so it is RESOLVED before it is graded -- never ungraded on
    # radius alone, which would turn the loudest failure this gate can see (a clean
    # rigid 1.3-2.5" shift) into a PASS.
    near_edge = verified & (off > window_edge_frac * mx_mas)
    # 1. CONTRAST.  The background peak is the extreme value of ~0.7 counts/bin over
    #    12,271 bins: 5-7, the verify floor (gc2211_o028 reads exactly that).  A rigid
    #    shift piles true counterparts into ONE bin: the same offsets injected
    #    synthetically read 16-24 at the same pair counts.  A confident near-edge peak
    #    is a measurement and is graded unchanged.  For the STRICT checks (cross-band,
    #    per-module) `fail_min_ratio` is the verify floor itself, so every verified cell
    #    clears this and the arm never fires there -- the diagnosis behind it
    #    (a 12.6x denser per-exposure catalog) is own-catalog-specific.
    confident_edge = near_edge & (ratio >= fail_min_ratio)
    unresolved_edge = near_edge & ~confident_edge
    # 2. SWEEP the rest at 2x and 4x MX.  A real tie reads the same offset at every
    #    window that can contain it; the background's arg-max moves with the window.
    #    A confirmed cell is graded AT THE SWEPT OFFSET, so a real 2.5-10" shift -- the
    #    brick-1182 / cloudc-F410M / sgra-1939 class, which the base window could only
    #    alias to ~0.7*MX -- is now measured and failed rather than aliased.
    swept = {}
    if sweep and unresolved_edge.any():
        cand = [int(i) * GRID + int(j) for i, j in zip(*np.where(unresolved_edge))]
        confirmed, swept = sweep_cell_windows(
            det, truth, cell_of_det, cand, {k: base_stats[k] for k in cand},
            factors=sweep_factors, window_edge_frac=window_edge_frac,
            pairs_per_det=len(ia) / max(1, len(det)))
        for k, (o, r) in confirmed.items():
            off[k // GRID, k % GRID] = o
            ratio[k // GRID, k % GRID] = r
            unresolved_edge[k // GRID, k % GRID] = False
    # 3. What is left measured nothing.  Reported, and refused (below) when it is
    #    shaped like a region rather than scattered noise.
    edge = unresolved_edge
    highoff = verified & (off > OFF_MAX) & ~edge
    # A FAIL requires a large offset AND confident contrast. A real localized seam
    # doubles stars into a sharp high-contrast peak; a bright-star-crowded, sparse
    # cell yields a floor-level peak (ratio ~ MIN_PEAK_RATIO) at a spurious offset.
    # Sub-FAIL_MIN_RATIO high-offset cells stay verified-but-not-failed (reported).
    fail = highoff & (ratio >= fail_min_ratio)
    # SECOND AXIS (issue #170): a connected patch of >= min_seam_cells high-offset
    # cells fails whatever its contrast.  The contrast bar is a raw peak-bin count and
    # therefore a density cut, and real injected seams score 5-49 on it -- straddling
    # FAIL_MIN_RATIO = 10 -- so a seam in sparse cells is measured and not blocked.
    # Shape does not scale with density.  OR-ed in, so this can only ADD failures.
    seam, seam_sizes = seam_mask(highoff, min_cells=min_seam_cells)
    fail = fail | seam
    # High offset but sub-fail_min_ratio contrast AND not part of a seam-shaped patch:
    # NOT a fail, but reported so a real low-contrast issue is never silently hidden by
    # the margin.  Excluding `seam` here keeps the two reports disjoint -- a cell that
    # now fails must not also be listed as a tolerated sub-margin cell.
    unconfident = highoff & (ratio < fail_min_ratio) & ~seam
    # COULD-NOT-VERIFY: the same shape test the seam axis uses, on the withdrawn cells.
    edge_patch, edge_sizes, region, region_sizes = could_not_verify_patch(
        highoff, edge, min_cells=min_seam_cells)
    worst = [dict(ra=float((xe[i] + xe[i + 1]) / 2), dec=float((ye[j] + ye[j + 1]) / 2),
                  offset_mas=round(float(off[i, j]), 0), peak_bg=round(float(ratio[i, j]), 1),
                  npairs=int(npair[i, j]))
             for i, j in sorted(zip(*np.where(fail)), key=lambda c: -off[c])][:8]
    unconfident_cells = [dict(ra=float((xe[i] + xe[i + 1]) / 2), dec=float((ye[j] + ye[j + 1]) / 2),
                              offset_mas=round(float(off[i, j]), 0), peak_bg=round(float(ratio[i, j]), 1),
                              npairs=int(npair[i, j]))
                         for i, j in sorted(zip(*np.where(unconfident)), key=lambda c: -off[c])][:8]
    edge_cells = [dict(ra=float((xe[i] + xe[i + 1]) / 2), dec=float((ye[j] + ye[j + 1]) / 2),
                       offset_mas=round(float(off[i, j]), 0), peak_bg=round(float(ratio[i, j]), 1),
                       npairs=int(npair[i, j]),
                       window_edge_fraction=float(round(off[i, j] / mx_mas, 3)),
                       swept_windows=[[round(w, 0), round(st.off, 0), round(st.ratio, 1),
                                       float(f"{st.expected:.3g}")]
                                      for (w, st) in swept.get(i * GRID + j, [])])
                  for i, j in sorted(zip(*np.where(edge)), key=lambda c: -off[c])][:8]
    n_fail = int(fail.sum())
    # Tri-state.  A fail outranks could-not-verify (both block; FAIL is the more
    # specific statement).  True only when every verified cell was actually measured.
    verdict = False if n_fail else (None if edge_patch.any() else True)
    return dict(label=label, verified_cells=int(verified.sum()),
                unverified_cells=int((npair >= MIN_PAIRS).sum() - verified.sum()),
                median_verified_offset_mas=round(float(np.nanmedian(off[verified])), 1) if verified.any() else None,
                # ... and the same median over the cells the gate actually GRADED,
                # which is the number a reader should compare against OFF_MAX.
                median_graded_offset_mas=(round(float(np.nanmedian(off[verified & ~edge])), 1)
                                          if (verified & ~edge).any() else None),
                n_fail=n_fail, PASS=verdict, worst=worst,
                n_unconfident_highoff=int(unconfident.sum()),
                unconfident_highoff_cells=unconfident_cells,
                # COULD-NOT-MEASURE (issue #588): cells whose peak rode the unswept MX
                # window AND was neither confident nor reproduced by the sweep.  Neither
                # fail axis reads them, they are disjoint from `unconfident_highoff_cells`,
                # and a `min_seam_cells` patch of them makes PASS None rather than True.
                n_window_edge=int(edge.sum()),
                window_edge_cells=edge_cells,
                window_edge_frac=float(window_edge_frac),
                n_window_edge_patch=int(edge_patch.sum()),
                window_edge_component_sizes=edge_sizes,
                # The quorate UNION of the measured-high-offset and could-not-measure
                # populations -- the mosaic region actually involved, which is what the
                # `min_seam_cells` quorum is now counted on.
                n_blocking_region=int(region.sum()),
                blocking_region_component_sizes=region_sizes,
                # Near-edge cells the two resolution steps GRADED rather than withdrew.
                n_edge_confident=int(confident_edge.sum()),
                n_edge_swept_confirmed=int(near_edge.sum() - confident_edge.sum()
                                           - edge.sum()),
                swept=bool(sweep),
                # The seam axis, reported separately so a reader can tell WHICH test
                # failed the field.  `n_fail_seam_only` counts cells that fail on shape
                # alone -- the ones the contrast bar would have let through.
                n_fail_seam=int(seam.sum()),
                n_fail_seam_only=int((seam & (ratio < fail_min_ratio)).sum()),
                seam_component_sizes=seam_sizes,
                min_seam_cells=int(min_seam_cells),
                _g=(off, verified, (xe, ye)))


def build_truths(field, filt, xband, observations=None):
    det, flux = detect(mosaic(field, filt, "merged", observations=observations))
    truths = {}
    # 1. per-module
    pm = []
    for m in ("nrca", "nrcb", "nrcalong", "nrcblong"):
        p = mosaic(field, filt, m, observations=observations)
        if p:
            s, _ = detect(p)
            if s is not None:
                pm.append(s)
    if pm:
        truths["per-module"] = SkyCoord(np.concatenate([s.ra.deg for s in pm]) * u.deg,
                                        np.concatenate([s.dec.deg for s in pm]) * u.deg)
    # 2. cross-band
    if xband:
        p = mosaic(field, xband, "merged", observations=observations)
        if p:
            s, _ = detect(p)
            truths[f"cross-band({xband})"] = s
    # 3. own catalog
    c = catalog_sc(field, filt)
    if c is not None:
        truths["own-catalog"] = c
    return det, flux, truths


def plot_all(results, out):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        if "_g" not in r:
            ax.set_title(f"{r['label']}: {r.get('error','')}"); continue
        off, verified, (xe, ye) = r["_g"]
        shown = np.where(verified, off, np.nan)
        im = ax.pcolormesh(xe, ye, shown.T, cmap="inferno", vmin=0, vmax=max(OFF_MAX * 2, 100))
        ax.invert_xaxis(); plt.colorbar(im, ax=ax, label="verified peak offset [mas]")
        # Tri-state: None is could-not-verify, which is not a pass and not a FAIL.
        v = ("PASS" if r["PASS"] else
             (f"UNVERIFIED {r.get('n_window_edge', 0)}" if r["PASS"] is None
              else f"FAIL {r['n_fail']}"))
        med = r.get("median_verified_offset_mas")
        colour = "green" if r["PASS"] else ("orange" if r["PASS"] is None else "red")
        ax.set_title(f"{r['label']}\nmed {med} mas — {v}", color=colour)
    fig.tight_layout(); fig.savefig(out, dpi=100); print("wrote", out)


def field_bands(field):
    """Filters with a merged mosaic on disk for this field.  Enumerates with a
    tight character-class glob (no `*` in proposal/observation) and validates
    each name with ``_MOSAIC_RE``; the mosaic's parsed filter must match its
    ``<field>/<FILT>/pipeline`` directory."""
    out = []
    pat = (f"{BASE}/{field}/*/pipeline/"
           f"jw[0-9][0-9][0-9][0-9][0-9]-o[0-9][0-9][0-9]_t001_nircam_clear-"
           f"*-merged_i2d.fits")
    for p in glob.glob(pat):
        m = _MOSAIC_RE.match(os.path.basename(p))
        if m is None or m.group("module") != "merged":
            continue
        filt = m.group("filt").upper()
        d = os.path.basename(os.path.dirname(os.path.dirname(p)))   # <field>/<FILT>/pipeline
        if d.upper() == filt:
            out.append(filt)
    return sorted(set(out))


def field_band_mosaics(field, observations=None):
    """``{FILT: {module_token: path}}`` for every validly-named mosaic on disk.

    ``field_bands`` lists only the bands that have a ``merged`` mosaic, so a band
    drizzled per-module and never merged is invisible to it -- and therefore
    silently ungated.  That is not a rare state: cloudc F182M, sgrc F115W/F162M,
    cloudef F162M/F210M and sgrb2 F150W are all in it (2026-08-03), and arches
    and sickle have no merged mosaic in ANY band.  Enumerate by module instead
    and let the caller decide what it can check with what is present.
    """
    out = {}
    pat = (f"{BASE}/{field}/*/pipeline/"
           f"jw[0-9][0-9][0-9][0-9][0-9]-o[0-9][0-9][0-9]_t001_nircam_clear-"
           f"*_i2d.fits")
    for p in sorted(glob.glob(pat)):
        m = _MOSAIC_RE.match(os.path.basename(p))
        if m is None:
            continue
        filt = m.group("filt").upper()
        d = os.path.basename(os.path.dirname(os.path.dirname(p)))   # <field>/<FILT>/pipeline
        if d.upper() != filt:
            continue
        if observations is not None \
                and f"{m.group('prop')}-{m.group('obs')}" not in observations:
            continue
        out.setdefault(filt, {}).setdefault(m.group("module"), p)
    return out


def module_family(token):
    """``nrca``/``nrcalong`` -> ``'a'``; ``nrcb``/``nrcblong`` -> ``'b'``.

    A field names its per-module mosaics with the SW tokens in both channels
    (arches writes ``f323n-nrca``, not ``f323n-nrcalong``), so the family, not
    the token, is what identifies "the same piece of sky".
    """
    return "a" if token.startswith("nrca") else "b"


def _sampled_valid_sky(path, stride=OVERLAP_STRIDE):
    """(ra, dec) of stride-sampled pixels that carry real data, plus (wcs, data).

    ``i2d`` mosaics are a rectified plain ``RA---TAN`` grid with no SIP, so
    ``WCS(header)`` is exact here -- the GWCS rule (ASTROMETRY RULE #2) exempts
    them explicitly.
    """
    with fits.open(path) as hdul:
        for h in hdul:
            if h.data is not None and h.data.ndim == 2 and h.header.get("CTYPE1"):
                data, hdr = np.asarray(h.data), h.header
                break
        else:
            return None
    ww = WCS(hdr)
    ys, xs = np.mgrid[0:data.shape[0]:stride, 0:data.shape[1]:stride]
    good = np.isfinite(data[ys, xs]) & (data[ys, xs] != 0)
    if not good.any():
        return None
    ra, dec = ww.all_pix2world(xs[good].astype(float), ys[good].astype(float), 0)
    return dict(ra=ra, dec=dec, wcs=ww, data=data)


def modules_overlap(path_a, path_b, stride=OVERLAP_STRIDE):
    """Do the two per-module mosaics share sky where BOTH carry real data?

    Not a bounding-box test: two abutting NIRCam modules drizzled onto their own
    grids can have boxes that touch or overlap while no pixel holds data from
    both.  Sample A's real pixels, map them into B, and count the ones that land
    on real data there.

    Returns ``None`` when either mosaic cannot be read (unknown, not "no").
    """
    a = _sampled_valid_sky(path_a, stride)
    b = _sampled_valid_sky(path_b, stride)
    if a is None or b is None:
        return None
    x, y = b["wcs"].all_world2pix(a["ra"], a["dec"], 0)
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    inside = ((xi >= 0) & (yi >= 0)
              & (xi < b["data"].shape[1]) & (yi < b["data"].shape[0]))
    if inside.any():
        d = b["data"][yi[inside], xi[inside]]
        n_both = int((np.isfinite(d) & (d != 0)).sum())
    else:
        n_both = 0
    return dict(n_sampled=int(len(x)), n_in_bbox=int(inside.sum()),
                n_both=n_both, overlaps=bool(n_both >= MIN_OVERLAP_SAMPLES))


def field_module_geometry(field, observations=None, verbose=False):
    """Whether this field's nrca and nrcb footprints share sky.

    ``mode``:

    * ``'single-module'`` — only one module family was observed (sickle: nrcb
      only).  There is no inter-module seam, so a per-module gate is complete.
    * ``'disjoint'`` — both modules observed, no band shows shared data (arches,
      quintuplet: the two modules image adjacent, non-overlapping sky).  A merged
      mosaic would add nothing the per-module mosaics do not already carry, so a
      per-module gate is again complete.
    * ``'overlapping'`` — some band has real data from both modules.  The seam
      between them is exactly where the misregistration this script exists to
      catch would live, so a band in this field needs its MERGED mosaic to be
      fully gated.
    * ``'merged-only'`` — no per-module mosaics were kept, so the geometry cannot
      be measured from disk.  The merged mosaic is all there is, and gating it is
      both the only option and the right one (brick, w51's single-module bands).
    * ``'unknown'`` — both modules exist but no band had both readable.
    """
    inv = field_band_mosaics(field, observations=observations)
    fams = set()
    for mods in inv.values():
        fams.update(module_family(t) for t in mods if t != "merged")
    if not fams:
        mode = "merged-only" if any("merged" in m for m in inv.values()) else "unknown"
        return dict(mode=mode, families=[], evidence={})
    if len(fams) == 1:
        return dict(mode="single-module", families=sorted(fams), evidence={})
    evidence, seen = {}, False
    for filt in sorted(inv):
        mods = inv[filt]
        pa = mods.get("nrca") or mods.get("nrcalong")
        pb = mods.get("nrcb") or mods.get("nrcblong")
        if not (pa and pb):
            continue
        r = modules_overlap(pa, pb)
        if r is None:
            continue
        seen = True
        evidence[filt] = r
        if verbose:
            print(f"  module overlap {field} {filt}: {r['n_both']} shared "
                  f"samples of {r['n_sampled']} -> "
                  f"{'OVERLAPPING' if r['overlaps'] else 'disjoint'}", flush=True)
    if not seen:
        return dict(mode="unknown", families=sorted(fams), evidence=evidence)
    mode = ("overlapping" if any(r["overlaps"] for r in evidence.values())
            else "disjoint")
    return dict(mode=mode, families=sorted(fams), evidence=evidence)


def _channel(f):
    """SW and LW detect different stellar populations and have independent distortion
    solutions, so a SW-vs-LW cross-match yields chance pairs -> spurious offsets that
    false-FAIL an internally-consistent field (e.g. gc2211 F200W vs F277W ~89 mas is an
    artifact; the within-channel + inter-module audit FLAGS none). Cross-band truth must
    therefore be pooled WITHIN channel only."""
    return "SW" if int(f[1:4]) <= 212 else "LW"


def _scan_view(field, view, band_paths, verbose, images_only):
    """Cross-band + own-catalog checks over one coherent set of mosaics.

    A *view* is a set of same-geometry mosaics that can serve as one another's
    cross-band truth: either every band's ``merged`` mosaic, or every band's
    mosaic for one module family.  Mixing the two would cross-match a merged
    mosaic against a single module's, whose non-overlapping parts have no
    counterpart to find.
    """
    bands = sorted(band_paths)
    if len(bands) < 2:
        # A single-band view has no cross-band truth and never will: how many
        # filters an observation used is a fact about the program.  A verdict of
        # "could not verify, therefore blocked" here would block a field for
        # having been observed the way it was observed.  Reported, not blocking;
        # the per-pair inter-frame overlap gate, the m2-m7 checkpoint ladder and
        # the absolute-frame refcat check are unaffected and still run.
        #
        # NOT the arches/quintuplet case, despite the resemblance: those fields
        # have TWO bands, so this branch is never taken for them.  Their bands
        # are one SW and one LW, and `_channel` refuses to cross-match across
        # that boundary, which leaves each band the sole member of its channel
        # and reaches the `if not graded` exemption further down instead.  Two-
        # filter programs are the NORM -- JWST 10678, the Treasury program, is
        # two filters throughout -- and that is the branch which keeps them from
        # being blocked for it.
        return dict(view=view, bands=bands, PASS=True, report={},
                    unchecked=[], n_graded=0,
                    unavailable=[f"{bands[0] if bands else '(none)'}: only band "
                                 f"in view {view}, nothing to cross-band against"])
    dets = {}
    for b in bands:
        s, f = detect(band_paths[b])
        dets[b] = (s, f)
        if verbose:
            print(f"  detect {field} [{view}] {b}: {0 if s is None else len(s)}",
                  flush=True)

    report, any_fail, unchecked, unavailable = {}, False, [], []
    n_graded = 0
    for b in bands:
        d, fl = dets[b]
        if d is None:
            # An unreadable/empty mosaic is NOT "locally misregistered" -- calling
            # it FAIL makes stage_release print a diagnosis that is simply wrong
            # about the file.  Could-not-verify, per this script's own tri-state.
            report[b] = {"error": "no detections"}
            unchecked.append(f"{b}: no detections in view {view} (mosaic empty, "
                             f"truncated, or unreadable)")
            continue
        # Whether a same-channel partner EXISTS AT ALL for this band, taken from
        # the band names rather than from `others`.  The two differ in exactly
        # the case that matters below: a sibling whose mosaic would not open
        # leaves `others` empty but `siblings` non-empty, and that is a real
        # defect which must keep blocking.
        siblings = [o for o in bands if o != b and _channel(o) == _channel(b)]
        others = [dets[o][0] for o in siblings if dets[o][0] is not None]
        checks = {}
        if others:
            tru = SkyCoord(np.concatenate([s.ra.deg for s in others]) * u.deg,
                           np.concatenate([s.dec.deg for s in others]) * u.deg)
            r = per_cell(d, fl, tru, f"{b} vs cross-band [{view}]"); r.pop("_g", None)
            checks["cross_band"] = r
        if not images_only:
            cat = catalog_sc(field, b, view)
            if cat is not None:
                r = per_cell(d, fl, cat, f"{b} vs own-catalog [{view}]",
                             fail_min_ratio=FAIL_MIN_RATIO); r.pop("_g", None)
                checks["own_catalog"] = r
            else:
                # A band with no vetted catalog to match against SAYS SO.  It
                # used to drop out of `checks` with nothing recorded, so a field
                # whose catalogs this pattern never matched (sickle, the gc2211
                # observations, sgrb2's nrcb-only bands) reported a green gate on
                # a check that had never run -- the silent-pass hole this script
                # exists to close, one level up from the `per_cell` one below.
                # UNAVAILABLE rather than unchecked: an m7 vetted catalog that
                # has not been produced yet is a state of the campaign, not a
                # defect in the mosaics, and blocking on it would block every
                # field that reaches this gate before its catalogs do.
                unavailable.append(no_catalog_note(field, b, view))
        # A check that MATCHED NOTHING is not a pass.  ``per_cell`` returns
        # ``dict(error=...)`` with no ``PASS`` key for "too few pairs" / "missing
        # detections", and ``.get("PASS", True)`` used to read those as passes --
        # the same silent-pass hole this script exists to close, one level down.
        # Reachable: gc2211's SW view pools F150W (o028) and F200W (o023) mosaics
        # 13.5 arcmin apart, so there are zero pairs to match.
        errored = {k: c for k, c in checks.items() if "error" in c}
        for k, c in errored.items():
            unchecked.append(f"{b}: {k} could not be evaluated in view {view} "
                             f"({c['error']})")
        graded = {k: c for k, c in checks.items() if k not in errored}
        # Nothing graded.  Two very different reasons, and only one of them is a
        # defect:
        #
        #   NO SAME-CHANNEL PARTNER EXISTS.  The cross-band check compares a band
        #   against the pooled detections of the field's OTHER bands in the same
        #   channel (SW-vs-LW is excluded on purpose -- see `_channel`).  A field
        #   observed in one SW and one LW filter therefore has no partner for
        #   either band, and no re-reduction can produce one: it is a property of
        #   the observing program.  arches and quintuplet (F212N + F323N) are
        #   exactly that, and with `--images-only` removing the own-catalog leg as
        #   well, every band came back ungated and the field verdict was None,
        #   which BLOCKS.  A gate that a correct field cannot pass under any
        #   circumstances is not a gate -- it is a standing instruction to reach
        #   for --allow-registration-fail, which is the one habit this script
        #   exists to prevent.  So it is recorded as UNAVAILABLE and does not
        #   block.  What still gates these fields is unchanged and is not weaker
        #   for it: `check_interframe_overlap.py` (reference-free, per pair, PER
        #   TILE, swept -- the blocking gate of checklist item 0), the m2/m3-m7
        #   astrometry checkpoint ladder upstream, and the absolute-frame check
        #   against the field's Gaia-tied refcat.
        #
        #   A PARTNER EXISTS BUT PRODUCED NOTHING.  Its mosaic would not open, or
        #   the match found too few pairs.  That is a defect in this release, it
        #   is fixable, and it keeps blocking exactly as before.
        if not graded:
            reason = (f"{b}: no check available in view {view} "
                      f"(sole {_channel(b)} band"
                      f"{', no own-catalog' if images_only else ''})")
            (unavailable if not siblings else unchecked).append(reason)
        n_graded += len(graded)
        # Tri-state per check (issue #588).  `PASS is False` is a measured failure;
        # `PASS is None` means the check could not MEASURE a coherent region of the
        # mosaic -- every verified cell there peaked beyond half the search window and
        # neither its contrast nor the sweep to 4x could resolve it.  That is not a
        # pass: it joins `unchecked`, which becomes the field's `unresolved` and takes
        # the field verdict to None -> exit 2 -> stage_release "could NOT VERIFY".
        # Reading None as False here would print a diagnosis ("locally misregistered")
        # that the estimator did not measure; reading it as True would ship the field.
        bad = any(c.get("PASS") is False for c in graded.values())
        for k, c in graded.items():
            if c.get("PASS") is None:
                unchecked.append(
                    f"{b}: {k} could not resolve a tie in view {view}: "
                    f"{c.get('n_window_edge')} verified cells peaked beyond "
                    f"{c.get('window_edge_frac')}*{MX}, and when the window was swept "
                    f"to {max(SWEEP_FACTORS):.0f}x ({max(SWEEP_FACTORS) * MX}) no peak "
                    f"stood clear of its own rim above that window's own background "
                    f"(look-elsewhere bar {SWEEP_MAX_EXPECTED_BINS}); in connected "
                    f"patches of "
                    f"{(c.get('window_edge_component_sizes') or [])[:5]} cells -- no "
                    f"tie exists inside the widest window this check can search, so "
                    f"the registration of those cells is UNKNOWN, not good")
        report[b] = checks
        any_fail = any_fail or bad
        if verbose:
            def _tag(k, v):
                verdict = ("PASS" if v.get("PASS") else
                           ("CANNOT-VERIFY" if v.get("PASS") is None
                            else "FAIL:" + str(v.get("n_fail"))))
                s = f"{k}={verdict}"
                nu = v.get("n_unconfident_highoff") or 0
                s += (f"(unconf={nu})" if nu else "")   # high-off, sub-margin cells
                # Cells whose peak rode the window and that neither contrast nor the
                # sweep could resolve: said on the line, never dropped silently (#588).
                ne = v.get("n_window_edge") or 0
                if ne:
                    s += (f"(window_edge={ne},patch="
                          f"{v.get('window_edge_component_sizes')})")
                nc = v.get("n_edge_swept_confirmed") or 0
                s += (f"(swept_confirmed={nc})" if nc else "")
                # Which axis failed it.  `seam` cells fail on SHAPE; `seam_only` are
                # the ones the contrast bar alone would have passed (issue #170).
                ns = v.get("n_fail_seam") or 0
                if ns:
                    s += (f"[seam={ns},only={v.get('n_fail_seam_only') or 0},"
                          f"comp={v.get('seam_component_sizes')}]")
                return s
            tags = " ".join(_tag(k, v) for k, v in checks.items())
            unver = any(c.get("PASS") is None for c in graded.values())
            state = "FAIL" if bad else ("UNVERIFIED" if unver else "ok")
            print(f"  {field} [{view}] {b}: {state}  {tags}", flush=True)
    return dict(view=view, bands=bands, PASS=bool(not any_fail), report=report,
                unchecked=unchecked, unavailable=unavailable,
                # How many checks were actually GRADED. `PASS: True` with
                # `n_graded: 0` is a pass on no evidence -- nothing was wrong
                # because nothing was measured -- and a caller could not tell
                # that from a pass backed by own-catalog and cross-band checks.
                # It is the `--images-only` verdict for a one-SW-one-LW field:
                # cross-band is impossible by construction and own-catalog is
                # switched off, so every band lands in `unavailable`. Still not
                # blocking (that is the point of this PR), but distinguishable.
                n_graded=n_graded)


def scan_field(field, verbose=True, images_only=False, observations=None):
    """Run the cross-band + own-catalog failsafes on EVERY band of a field.

    Cross-band truth for band F = the pooled detections of all OTHER bands of the field
    (same stars, JWST-internal registration).  Detects each band once.

    The mosaics are grouped into *views* according to the field's module geometry:

    * modules that OVERLAP — the seam between them is what this script exists to
      catch, so the ``merged`` mosaic (the only place the two modules are
      combined) is the thing that must be checked.  A band with no merged mosaic
      cannot be fully gated here; it is checked per module for what that is worth
      and reported as ungated.
    * modules that are DISJOINT (arches, quintuplet) or a field that used only
      ONE module (sickle) — there is no seam to catch, and each module's own
      mosaic is a complete object, so the gate is PER MODULE and every module
      must pass on its own.  The merged mosaic is gated TOO where one exists:
      it is not needed for the seam, but it SHIPS, and a merged drizzle that
      places module B at the wrong offset is invisible in the per-module views.
      A merged view covering fewer than 2 bands is dropped rather than gated —
      it has nothing to cross-band-check against — and the drop is printed.

    ``images_only``: gate an IMAGE-ONLY release -- run the reference-free cross-band
    (image-to-image) check only, and SKIP own-catalog.  An image-only release ships the
    mosaics without the catalog, so a mosaic<->catalog mismatch (own_catalog FAIL) is not
    a reason to block; the images can still be internally consistent and shippable.

    ``PASS`` is tri-state.  ``True``/``False`` are a verified pass/fail; ``None``
    means the field could not be verified either way -- no mosaics, no view with
    >=2 bands, a check that errored, or overlapping modules with a band whose
    merged mosaic is missing.  ``None`` BLOCKS: ambiguity is not a pass.

    There is deliberately NO "not covered here, and that is fine" verdict: every
    view admitted is gated, and anything that cannot be gated is either dropped
    before it becomes a view (the <2-band merged case above) or reported as
    ungated, which blocks.  The distinction is made when the view is BUILT, not
    when it is judged, because by judging time a view that cannot be checked is
    indistinguishable from one that failed to be.
    """
    inv = field_band_mosaics(field, observations=observations)
    if not inv:
        return dict(field=field, bands=[], PASS=None,
                    error="no validly-named mosaics on disk")
    geom = field_module_geometry(field, observations=observations, verbose=verbose)
    if verbose:
        print(f"  {field} module geometry: {geom['mode']} "
              f"(families {geom['families']})", flush=True)

    views, ungated = {}, []
    if geom["mode"] in ("disjoint", "single-module"):
        for fam in geom["families"]:
            paths = {}
            for filt, mods in inv.items():
                cand = [t for t in mods
                        if t != "merged" and module_family(t) == fam]
                if cand:
                    paths[filt] = mods[sorted(cand)[0]]
            if paths:
                views[f"module-{fam}"] = paths
        # The per-module views account for the module IMAGES, but the merged
        # product also SHIPS, and a merged drizzle that places module B at the
        # wrong offset -- or writes a wrong output WCS -- is invisible in them.
        # Gating only per-module opened zero merged mosaics for m92 (4 bands),
        # gc2211 (3) and sgra (2), all of which the previous gate did open.
        # This went unnoticed because arches and quintuplet, the two fields the
        # disjoint branch was written for, have NO merged mosaics at all, so
        # there the dict comes back empty and nothing changes.
        merged = {f: m["merged"] for f, m in inv.items() if "merged" in m}
        # >= 2, the same bar the per-module views below use.  A view with ONE band
        # cannot serve as its own cross-band truth, so it is not a view -- and
        # admitting it is not neutral: it lands in `unresolved` and the field
        # verdict becomes None, which BLOCKS.  sickle is the case: one merged
        # mosaic, five bands passing on the module view, and no re-reduction short
        # of producing four more merged mosaics could clear it.  A gate a correct
        # field cannot pass is a gate that teaches people to use the override.
        if len(merged) >= 2:
            views["merged"] = merged
        elif merged:
            print(f"  {field}: only {sorted(merged)} has a merged mosaic -- a "
                  f"one-band merged view cannot be cross-band-checked against "
                  f"anything, so it is not gated here (the module views below "
                  f"still gate every band)", flush=True)
    else:
        # overlapping / merged-only / unknown: merged is the object to gate
        merged = {f: m["merged"] for f, m in inv.items() if "merged" in m}
        if merged:
            views["merged"] = merged
        why = {"overlapping": "this field's modules overlap",
               "merged-only": "no per-module mosaics were kept, so the module "
                              "geometry could not be measured",
               "unknown": "this field's module geometry could not be measured"}
        for filt, mods in sorted(inv.items()):
            if "merged" not in mods:
                ungated.append(
                    f"{filt}: no merged mosaic, and "
                    f"{why.get(geom['mode'], geom['mode'])}"
                    f" -- the inter-module seam of this band is NOT covered here "
                    f"(present: {sorted(mods)})")
        # Only when something is ungated is it worth also running the per-module
        # views: they cannot see the seam (that is what merged is for), so for a
        # fully-merged field they would triple the runtime and add nothing.  When
        # a band HAS no merged mosaic, they are the only look at it available.
        if ungated:
            for fam in geom["families"]:
                paths = {}
                for filt, mods in inv.items():
                    cand = [t for t in mods
                            if t != "merged" and module_family(t) == fam]
                    if cand:
                        paths[filt] = mods[sorted(cand)[0]]
                if len(paths) >= 2:
                    views[f"module-{fam}"] = paths

    if not views:
        # Same reasoning as the single-band view above, one level up: a field
        # whose mosaics never form a 2-band view has no cross-band truth
        # available anywhere, which is a property of the program rather than of
        # this release.  A field with NO mosaics at all is a different thing and
        # is still blocked -- that is caught by the `if not inv` return above.
        return dict(field=field, bands=sorted(inv), PASS=True,
                    geometry=geom["mode"], views={}, unresolved=[],
                    unavailable=[f"no view with >=2 bands to cross-match "
                                 f"(bands present: {sorted(inv)})"],
                    n_graded=0, evidence="none", report={})

    results = {name: _scan_view(field, name, paths, verbose, images_only)
               for name, paths in sorted(views.items())}
    # In per-module mode EVERY module must pass on its own -- that is the whole
    # point of accepting the modules separately.
    any_fail = any(r.get("PASS") is False for r in results.values())
    unresolved = [f"view {n}: {r['error']}" for n, r in results.items()
                  if r.get("PASS") is None]
    unresolved += ungated
    unavailable = []
    for r in results.values():
        unresolved += r.get("unchecked", [])
        unavailable += r.get("unavailable", [])

    if any_fail:
        passed = False
    elif unresolved:
        passed = None
    else:
        passed = True
    if verbose and unresolved:
        for u in unresolved:
            print(f"  {field} UNGATED: {u}", flush=True)
    # Reported, never blocking -- see `_scan_view`.  Printed under a different
    # word from UNGATED on purpose: UNGATED names something that should have
    # been checked and was not, and reading the two as one line is how a real
    # defect would get waved through as "oh, that field always says that".
    if verbose and unavailable:
        for u in unavailable:
            print(f"  {field} NO-PARTNER (not blocking): {u}", flush=True)
    n_graded = sum(r.get("n_graded", 0) for r in results.values())
    # A pass and the evidence behind it are two different facts. `PASS: True`
    # with nothing graded says "no check found a problem" only in the sense that
    # no check ran -- the `--images-only` verdict for a one-SW-one-LW field,
    # where cross-band cannot exist and own-catalog is switched off. It stays a
    # pass (a gate a correct field can never satisfy teaches people to reach for
    # the override), and `evidence` lets a caller, a log reader or a later gate
    # tell it apart from a pass backed by graded checks.
    evidence = "graded" if n_graded else "none"
    if verbose and passed and not n_graded:
        print(f"  {field} PASS ON NO EVIDENCE: no check was graded "
              f"({len(unavailable)} unavailable). Registration for this field "
              f"rests on the inter-frame overlap gate and the astrometry "
              f"checkpoints, not on this one.", flush=True)
    return dict(field=field, bands=sorted(inv), geometry=geom["mode"],
                module_families=geom["families"],
                overlap_evidence=geom.get("evidence", {}),
                views=results, PASS=passed, unresolved=unresolved,
                unavailable=unavailable, n_graded=n_graded, evidence=evidence,
                # flattened single-view report, for callers that read `report`
                report=(results.get("merged") or
                        list(results.values())[0]).get("report", {}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", required=True)
    ap.add_argument("--filter", default=None, help="single band (omit for --scan)")
    ap.add_argument("--xband", default=None, help="cross-band reference filter (e.g. F200W)")
    ap.add_argument("--scan", action="store_true", help="scan EVERY band of the field (gate mode)")
    ap.add_argument("--images-only", action="store_true",
                    help="cross-band (image-to-image) check only; skip own-catalog "
                         "(gate for an image-only release)")
    ap.add_argument("--observations", default=None,
                    help="csv of <proposal>-<observation> keys (e.g. "
                         "02221-001,01182-004) to scope mosaic selection to; a "
                         "misfiled stray from another observation in a shared "
                         "target dir is excluded.  Omit to pick deterministically "
                         "(sorted) among whatever validly-named mosaics are present.")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    obs = set(args.observations.split(",")) if args.observations else None

    if args.scan or not args.filter:
        res = scan_field(args.field, images_only=args.images_only, observations=obs)
        if args.json:
            json.dump(res, open(args.json, "w"), indent=2, default=str)
        print(json.dumps({"field": res.get("field"), "PASS": res.get("PASS"),
                          "geometry": res.get("geometry"),
                          "error": res.get("error"),
                          "unresolved": res.get("unresolved"),
                          # bands with no same-channel partner to cross-check
                          # against; reported so the summary line never implies
                          # a check ran that could not have
                          "unavailable": res.get("unavailable"),
                          # "graded"/"none": whether the PASS above rests on
                          # checks that actually ran. A pass on no evidence is
                          # still a pass, and is not the same thing.
                          "evidence": res.get("evidence"),
                          "n_graded": res.get("n_graded")}, default=str))
        # PASS is tri-state and only True is a pass.  `None` (could not verify:
        # no mosaics, <2 bands, a band with no merged mosaic in an overlapping-
        # module field) used to return 0 and let staging proceed -- a gate that
        # goes green because it never ran.  Ambiguity is not a pass; it blocks,
        # and stage_release's --allow-registration-fail + ALLOW_REGISTRATION_FAIL=1
        # is the deliberate, two-key way past it.
        if res.get("PASS") is None:
            for u in (res.get("unresolved") or [res.get("error") or "unspecified"]):
                print(f"UNVERIFIED: {u}", file=sys.stderr)
            return 2    # exit 2 = could-not-verify -> gate blocks staging
        return 0 if res.get("PASS") else 1   # exit 1 = FAIL -> gate blocks staging

    det, flux, truths = build_truths(args.field, args.filter, args.xband,
                                     observations=obs)
    # own-catalog gets the relaxed fail bar; per-module / cross-band stay strict.
    results = [per_cell(det, flux, t, f"{args.filter} vs {name}",
                        fail_min_ratio=(FAIL_MIN_RATIO if name == "own-catalog"
                                        else MIN_PEAK_RATIO))
               for name, t in truths.items()]
    if args.plot:
        plot_all(results, args.plot)
    any_fail = False
    unverified = False
    for r in results:
        r.pop("_g", None)
        print(json.dumps(r, indent=2, default=str))
        # Tri-state, matching --scan: False = FAIL (1), None = could-not-verify (2).
        any_fail = any_fail or (r.get("PASS") is False)
        unverified = unverified or (r.get("PASS") is None)
    if any_fail:
        return 1
    return 2 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
