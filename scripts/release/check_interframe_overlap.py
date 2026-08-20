#!/usr/bin/env python
"""BLOCKING gate: reference-free inter-frame overlap registration.

The brick-1182 F200W seam failure (2026-07-12) proved that our existing gates are
STRUCTURALLY BLIND to a per-visit residual:

- ``registration_failsafes.py`` matches the mosaic against its OWN merged catalog.
  Both are built from the same ``_crf`` frames, so if visit-001 carries a ~90 mas
  residual, the catalog inherits it too -> mosaic and catalog AGREE (offset ~0) ->
  PASS, even though both are wrong. A self-referential truth cannot see a shared
  error. It also searches only +-2.5" (no sweep) so a gross overlap offset reads as
  "no overlap", not FAIL.
- Bulk / coarse-grid vs-reference checks average the two visits together in the
  overlap and read ~0.

The ONLY check that sees it is REFERENCE-FREE and PAIRWISE: detect on each
per-exposure ``_crf`` frame (each on its own corrected GWCS), pool by
exposure-group (visit x module), and histogram-stack every OVERLAPPING pair's
MUTUAL offset. Overlapping same-instrument groups must co-register to < 30 mas.
Non-zero exit on FAIL so it gates a release/staging chain.

A pair NEITHER reference-free layer can measure (a near-non-overlap sliver: 0
mutual-coverage tiles) is DEFERRED to the external same-star arbiter, which is
scoped to that pair's OWN overlap footprint and ties each exposure group to the
reference there, then differences the two star by star -- the reference is only a
common anchor, so its local systematics cancel.  A field-wide residual map cannot
do this job: a sliver is a minority of every cell it touches, and one field-wide
boolean cleared every deferred pair at once (issue #174).

Usage::

    python check_interframe_overlap.py --field brick --filter F200W
    python check_interframe_overlap.py --field brick --scan           # all filters
    # scope to specific released observations (default: auto-derived from the
    # field's merged mosaics, so stray crf from other programs are excluded):
    python check_interframe_overlap.py --field brick --scan --observations 02221-001,01182-004
    # optional external absolute cross-check (fine grid vs VIRAC2/Gaia):
    python check_interframe_overlap.py --field brick --filter F200W --refcat <path>
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.detection import find_peaks

from jwst_gc_pipeline import fields
from jwst_gc_pipeline.photometry.naming import MIRI_FILTERS
from jwst_gc_pipeline.frame_wcs import frame_wcs
from jwst_gc_pipeline.photometry.interframe_overlap import (
    overlap_offset_grid, pairwise_overlap_offsets, DEFAULT_OVERLAP_TOL_MAS)
from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset_grid, measure_offset, local_residual_map)

BASE = os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst")
TOL_MAS = float(os.environ.get("OVERLAP_TOL_MAS", DEFAULT_OVERLAP_TOL_MAS))
# external-reference fine-grid gate
GRID_N = int(os.environ.get("OVERLAP_GRID_N", 16))
# GROSS ceiling on the GLOBAL (field-wide) tie vs the reference.  This is NOT the
# per-cell sensitivity: a field whose bulk tie is 80 mas off is a different (and
# much louder) failure than a 30 mas local seam, and conflating the two made the
# arbiter blind to everything between TOL_MAS and 80 mas (issue #174).
GRID_MAX_OFF_MAS = float(os.environ.get("OVERLAP_GRID_MAX_OFF_MAS", 80.0))
# Ladder of same-star residual-map cell sizes (arcsec).  A single 30" cell
# averages a sliver narrower than itself away (a 4"-wide strip shifted 150 mas
# read clean, worst=0.0), and the deferred pairs this arbiter is authoritative
# for are thin slivers BY CONSTRUCTION.  Measure at every scale the star density
# supports: a scale with no cell holding >= SAMESTAR_MIN_STARS is UNMEASURABLE at
# that scale (it contributes nothing), never "clean".
SAMESTAR_CELLS_ARCSEC = tuple(
    float(x) for x in os.environ.get("OVERLAP_SAMESTAR_CELLS", "2,4,8,16,30").split(","))
SAMESTAR_MIN_STARS = int(os.environ.get("OVERLAP_SAMESTAR_MIN_STARS", 20))
# Same-star matching radii for the NN-collapse consistency check.  A localized
# sub-population shifted by ~the match radius pairs with the WRONG neighbour at
# 0.3" (residuals scatter, median ~0 -> reads clean) but simply STOPS matching at
# 0.1" -- so its matched-pair count collapses with radius while the rest of the
# field's does not.  That ratio is the tell; it needs no offset histogram.
SAMESTAR_MATCH_RADII = tuple(
    float(x) for x in os.environ.get("OVERLAP_SAMESTAR_RADII", "0.1,0.2,0.3").split(","))
SAMESTAR_MATCH_RADIUS = max(SAMESTAR_MATCH_RADII)
# A cell keeping less than this FRACTION of the field-typical matched-pair count
# when the radius shrinks is matching-ambiguous, not clean.
SAMESTAR_COLLAPSE_FRAC = float(os.environ.get("OVERLAP_SAMESTAR_COLLAPSE_FRAC", 0.5))
# Inside a deferred pair's footprint the two frames are matched to the SAME
# reference stars, so they must keep comparable FRACTIONS of their matches as the
# radius shrinks.  A gap wider than this (and statistically significant) means one
# frame has a sub-population sitting at ~the match radius: not verifiable.
# Measured on brick F405N's real deferred-pair geometry: 0.837 vs 0.849 (gap
# 0.012) for the frames as reduced.
SAMESTAR_KEEP_MARGIN = float(os.environ.get("OVERLAP_SAMESTAR_KEEP_MARGIN", 0.12))


def _detect(path, nsigma=8.0, box=5):
    # GWCS, not the SCI header's SIP fit.  This is the BLOCKING inter-frame
    # overlap gate: it compares frame-vs-frame star positions at a 30 mas
    # tolerance, so a 5-8 mas position-dependent SIP-fit error (different per
    # detector and per filter, i.e. different for the two frames being
    # compared) is a large fraction of the gate's own budget.
    with fits.open(path, memmap=True) as h:
        d = np.asarray(h["SCI"].data)
        w = frame_wcs(h)
    fin = np.isfinite(d)
    if fin.sum() < 2000:
        return None
    _, med, std = sigma_clipped_stats(d[fin], sigma=3.0, maxiters=3)
    tb = find_peaks(np.where(fin, d, 0.0), threshold=med + nsigma * std, box_size=box)
    if tb is None or len(tb) < 20:
        return None
    sky = w.pixel_to_world(np.array(tb["x_peak"]), np.array(tb["y_peak"]))
    return SkyCoord(sky.ra, sky.dec)


# Precise per-exposure crf filename parser.  JWST products are named
#   jw<PPPPP><OOO><VVV>_<GGSAA>_<EEEEE>_<detector>[_<lineage>...]_o<OOO>_crf.fits
# (PPPPP=proposal, OOO=observation, VVV=visit; the trailing _oOOO_ repeats the
# observation).  We anchor on that exact structure rather than a permissive
# `jw*_*_nrc*_o*` glob: a wildcard `o*` matches EVERY observation, so a shared
# target directory (the brick dir also holds 2221 o002 = cloudc crf) leaks stray
# frames from other observations/programs into the verdict.  The regex both
# validates a name and yields (proposal, observation, visit, module) exactly.
_CRF_RE = re.compile(
    r"^jw(?P<prop>\d{5})(?P<obs>\d{3})(?P<visit>\d{3})_(?P<vgroup>\d+)_(?P<exp>\d+)_"
    r"(?P<det>mirimage|nrc[ab](?:long|[1-4])?)"
    r"(?P<lineage>(?:_[a-z0-9]+)*?)_o(?P<obs2>\d{3})_crf\.fits$")

#: Lineage tokens of the RETIRED post-resample realignment path
#: (``realign_to_vvv`` / ``sync_gwcs_to_fits_wcs``, removed from the code
#: 2026-07-11 -- see ASTROMETRY_WCS_CORRECTION_FLOW.md).  Its OUTPUTS are older
#: than that removal (the cloudc set is dated 2023-08-01), and still on disk, and
#: still match ``jw*_crf.fits``.
#:
#: They must never be pooled into a registration verdict: in those files the FITS
#: header and the ASDF GWCS disagree with EACH OTHER by arcseconds, because that
#: path realigned one representation and not the other.  Measured 2026-07-29 on
#: cloudc/F405N, all 32 ``*_realigned_to_vvv_*_crf.fits``:
#:
#:     visit 001 (16 frames):  7.972 - 8.067 arcsec FITS-vs-GWCS
#:     visit 002 (16 frames):  4.091 - 4.170 arcsec
#:     live *_destreak_o002_crf (32):  6.681 - 7.972 mas, median 7.327
#:
#: The ~3.9 arcsec VISIT-TO-VISIT differential matters more here than the
#: absolute size, because this gate groups by (obs+visit, module) -- a
#: visit-dependent shift is precisely what it exists to catch.  Those files also
#: lack the ``-SIP`` CTYPE suffix, so astropy warns when reading them: a second,
#: independent reason no reader gets the intended answer out of them.
#:
#: SCOPE, stated plainly: this blocklist removes 32 files in ONE directory
#: (cloudc/F405N is the only directory archive-wide with retired-path crf).  It
#: does NOT solve lineage staleness in general -- 46 of 127 pipeline directories
#: still admit more than one lineage copy of the same exposure (mostly
#: ``['', '_destreak']`` or ``['_align', '_destreak']``), and in cloudc/F405N the
#: ``bare`` copy sits 8.5 arcsec from the live ``_destreak`` one by same-pixel sky
#: position.  Selecting a single family per exposure is the general fix; see the
#: follow-up issue.  Do not read this blocklist as more than it is.
_RETIRED_LINEAGE_TOKENS = ("realigned", "refcat", "vvv")


def _retired_lineage(name):
    """True if ``name`` is a retired-path crf (the reason _parse_crf rejected it)."""
    m = _CRF_RE.match(os.path.basename(name))
    if m is None:
        return False
    return bool(set(m.group("lineage").split("_")) & set(_RETIRED_LINEAGE_TOKENS))


def _parse_crf(name):
    """Parse a per-exposure crf basename into its identifying fields, or None
    if it is not a well-formed crf name.  ``module`` collapses the detector to
    the physical NIRCam module (nrca/nrcb) or ``mirimage``; SW and LW share a
    module label but never share a filter, so the per-filter check keeps them
    apart.  ``obs_key`` = ``"<proposal>-<observation>"`` is the release-scoping
    key (proposal-aware, so 1182-o004 and 2221-o001 never collide)."""
    m = _CRF_RE.match(os.path.basename(name))
    if m is None:
        return None
    # the leading obs and the trailing _oOOO_ must agree; a name whose tokens
    # disagree (a product-named crf copied into a per-exposure name) is a parse
    # failure, not a coin-flip on whichever token the code happens to read
    if m.group("obs2") != m.group("obs"):
        return None
    # Retired-path products (see _RETIRED_LINEAGE_TOKENS): their FITS header and
    # their GWCS disagree by arcseconds, so pooling them makes the verdict a
    # coin-flip on which representation the reader used.
    lineage = set(m.group("lineage").split("_")) - {""}
    if lineage & set(_RETIRED_LINEAGE_TOKENS):
        return None
    det = m.group("det")
    module = "mirimage" if det == "mirimage" else det[:4]
    return dict(prop=m.group("prop"), obs=m.group("obs"), visit=m.group("visit"),
                det=det, module=module,
                obs_key=f"{m.group('prop')}-{m.group('obs')}")


def exposure_identity(name):
    """Which PHYSICAL exposure a crf name refers to, ignoring its lineage.

    One exposure is normally on disk several times over: the reduction has been
    re-run with different settings across three years, and each run writes its
    own copy under a name that differs only by a lineage token
    (``_destreak``, ``_align``, or none at all).  Those copies are the same
    photons on different sky coordinates -- in cloudc/F405N, up to 8.5 arcsec
    apart -- so pooling them into one registration verdict compares an exposure
    against stale copies of itself.

    Returns ``None`` for a name that is not a well-formed crf.
    """
    m = _CRF_RE.match(os.path.basename(name))
    if m is None or m.group("obs2") != m.group("obs"):
        return None
    # A retired-path product is not a copy of this exposure that competes on
    # merit: its FITS header and its GWCS disagree by arcseconds, so it is
    # rejected outright (see _RETIRED_LINEAGE_TOKENS).  Giving it an identity
    # would let it into a selection that is only supposed to choose between
    # frames a reader can get a consistent answer out of.
    if set(m.group("lineage").split("_")) & set(_RETIRED_LINEAGE_TOKENS):
        return None
    return (m.group("prop"), m.group("obs"), m.group("visit"),
            m.group("vgroup"), m.group("exp"), m.group("det"))


def _lineage_token(name):
    """``'_destreak'``, ``'_align'``, or ``''`` for the bare name."""
    m = _CRF_RE.match(os.path.basename(name))
    return "" if m is None else m.group("lineage")


def lineage_separation_mas(a, b, samples=((256, 256), (1024, 1024), (1792, 1792))):
    """How far apart two copies of one exposure place the same pixel, in
    milliarcseconds.

    Read from each frame's authoritative coordinate solution (the GWCS carried
    in the file's ASDF extension, via ``frame_wcs``), NOT from the approximate
    FITS header keywords -- those are a fitted approximation and disagree with
    the GWCS by several mas on frames written before 2026-07-29.

    Returns ``None`` when either frame cannot be read, or when the sampled
    pixels fall outside a frame's valid area (a subarray exposure), because a
    missing measurement must not be reported as agreement.
    """
    try:
        wa, wb = frame_wcs(a), frame_wcs(b)
    except (OSError, ValueError, KeyError):
        return None
    seps = []
    for x, y in samples:
        ca, cb = wa.pixel_to_world(x, y), wb.pixel_to_world(x, y)
        sep = ca.separation(cb).to(u.mas).value
        if np.isfinite(sep):
            seps.append(float(sep))
    return max(seps) if seps else None


def _reduction_lineage(field, filt, obs, detector):
    """The lineage token this field's own reduction writes, or ``None``.

    Which frames a field ships is not a judgement call and not a preference
    list: ``destreak_policy`` already records, per (field, filter), whether
    stage 1 applies its streak-removal step -- and therefore whether the
    reduced frame is called ``*_destreak_o<obs>_crf.fits`` or
    ``*_align_o<obs>_crf.fits``.  The cataloguing stage picks its inputs from
    that same function, so reading it here is what keeps this gate and the
    catalogue looking at the same files.

    ``None`` for MIRI: streak removal is a NIRCam stage-1 step and the policy
    does not name MIRI products, so applying its answer to a MIRI frame would
    ask for a lineage that never exists.
    """
    if str(detector).startswith("miri"):
        return None
    try:
        from jwst_gc_pipeline.reduction.destreak_policy import crf_suffix
    except ImportError:
        return None
    suffix = crf_suffix(field, filt, obs)          # 'destreak_o002_crf'
    return "_" + suffix.split("_o")[0]             # '_destreak'


class OutOfReleaseScope(Exception):
    """This filter directory's products belong to other observations entirely.

    Not a verification failure and not a missing product: the frames are on
    disk and well-formed, they are simply not part of the release being gated.
    Carried as an exception rather than an empty result so it cannot be
    confused with "found nothing", which must keep failing closed.
    """

    def __init__(self, field, filt, derived, requested):
        self.field, self.filt = field, filt
        self.derived, self.requested = set(derived), set(requested)
        super().__init__(
            f"{field}/{filt}: products belong to "
            f"{', '.join(sorted(self.derived))}, which this release does not "
            f"claim ({', '.join(sorted(self.requested))})")


class UnparseableFrameError(ValueError):
    """A frame reached the selector whose name it cannot identify."""


def select_one_copy_per_exposure(frames, field, filt):
    """Keep one processed copy of each physical exposure.  ``(kept, dropped)``.

    ``dropped`` is a list of ``(path, why)`` so the caller can say what it left
    out -- a silently smaller frame set must stay distinguishable from a
    smaller directory.

    **The rule is a single recorded fact, not a preference order.** Which of
    the reduction's two output variants a field produces was decided per
    (field, filter) long ago and is recorded in
    ``reduction/destreak_policy.py``; the cataloguing stage reads that same
    record to choose which frames to photometer.  Reading it here is what makes
    the registration gate and the catalogue examine the same files by
    construction, and it decides every one of the 43 affected directories.

    When the recorded variant is absent for an exposure -- a partly-finished
    re-reduction, or MIRI, whose products the policy does not name -- the
    newest copy is kept and the caller is told, because reaching that case
    means nothing recorded decided it.

    **What this deliberately does NOT use.** An earlier version preferred the
    copy carrying ``RAOFFSET`` (the keyword ``fix_alignment`` writes when it
    applies a field's pointing correction).  Measured across the archive, that
    keyword does not predict the coordinate solution in either direction: two
    w51/F444W copies both record ``RAOFFSET=(0,0)`` and place the same pixel
    37.6 mas apart, while a wd2/F150W copy with no ``RAOFFSET`` at all is
    identical to its corrected twin to 0.0 mas.  Whatever differs between two
    reduction runs is not captured by that keyword, so selecting on it would be
    guessing with a confident-looking rule.  The separation between the kept
    and dropped copies is instead MEASURED and reported -- see
    ``lineage_separation_mas`` and the caller -- which is the cross-check that
    would catch this rule pointing at the wrong copy.

    Raises ``UnparseableFrameError`` rather than dropping a frame whose name it
    cannot identify: vanishing silently is the failure this whole change exists
    to remove.  This is a guard on this function's own contract, not a fix for
    any directory today -- ``build_groups`` filters with ``_parse_crf`` first,
    which accepts exactly the same names, so nothing unparseable reaches here
    from the gate.  (wd1/F200W's frames are rejected by that earlier filter and
    the directory reads as empty, which the gate already fails closed on; see
    issue #376.)
    """
    by_exposure = {}
    unparseable = []
    for path in frames:
        identity = exposure_identity(path)
        if identity is None:
            unparseable.append(path)
            continue
        by_exposure.setdefault(identity, []).append(path)
    if unparseable:
        raise UnparseableFrameError(
            f"{len(unparseable)} frame(s) reached the lineage selector with a "
            f"name it cannot identify, e.g. "
            f"{os.path.basename(unparseable[0])}.  Refusing to drop them "
            f"silently: a smaller frame set must stay distinguishable from a "
            f"smaller directory.")

    kept, dropped = [], []
    for identity, copies in sorted(by_exposure.items()):
        if len(copies) == 1:
            kept.append(copies[0])
            continue
        obs, detector = identity[1], identity[5]
        wanted = _reduction_lineage(field, filt, obs, detector)
        chosen = [p for p in copies if _lineage_token(p) == wanted] if wanted else []
        why = f"this field's reduction writes {wanted or '?'}_o{obs}_crf"
        if len(chosen) != 1:
            # Nothing recorded decided it: MIRI, a partly-finished
            # re-reduction, or (impossible today) two files with one lineage.
            # (mtime, path) not mtime alone: equal timestamps are common
            # among copies written by one run, and `max` returns the FIRST
            # maximal element, so a bare mtime key makes the winner depend on
            # the order the directory happened to be listed in.
            chosen = [max(chosen or copies,
                          key=lambda p: (os.path.getmtime(p), p))]
            why = ("newest -- no recorded reduction setting decided this one"
                   if not wanted else
                   f"newest -- this field's reduction writes "
                   f"{wanted}_o{obs}_crf, which is not on disk for this exposure")
        kept.append(chosen[0])
        dropped += [(p, why) for p in copies if p != chosen[0]]
    return sorted(kept), dropped


def report_lineage_disagreement(dropped, kept, label, threshold_mas=100.0):
    """Measure how far each discarded copy sits from the one kept, and say so.

    This is the check issue #205 asked for, and it is the only part of the
    selection that consults the data rather than a filename: if the recorded
    reduction setting ever points at the stale copy, the rule itself cannot
    notice, but this will.  A pair beyond ``threshold_mas`` is not a lineage
    question -- at that size one of the two frames is simply wrong.

    **This reports; it does not gate.** Issue #205 asks a second question --
    whether copies disagreeing beyond some tolerance should make the gate
    REFUSE rather than pick one -- and that is deliberately not answered here,
    because the tolerance was to be set from measurements that did not exist
    until this function produced them.  So a directory can print "one of the
    two frames is wrong" for most of its discarded copies and still return a
    passing verdict.  Choosing the refusal threshold is the follow-up this
    data is for.

    Returns the list of ``(dropped_path, separation_mas)`` it measured.
    """
    by_identity = {exposure_identity(p): p for p in kept}
    measured = []
    for path, _why in dropped:
        twin = by_identity.get(exposure_identity(path))
        if twin is None:
            continue
        sep = lineage_separation_mas(twin, path)
        measured.append((path, sep))
    usable = [s for _p, s in measured if s is not None]
    if not usable:
        return measured
    over = sorted(((p, s) for p, s in measured
                   if s is not None and s > threshold_mas),
                  key=lambda t: -t[1])
    print(f"  {label}: discarded copies sit "
          f"{min(usable):.1f}-{max(usable):.1f} mas from the one kept "
          f"(median {float(np.median(usable)):.1f})", flush=True)
    if over:
        print(f"      {len(over)} beyond {threshold_mas:.0f} mas -- at that "
              f"size one of the two frames is wrong, not merely older:",
              flush=True)
        for path, sep in over[:3]:
            print(f"        {sep:9.1f} mas  {os.path.basename(path)}", flush=True)
    unmeasured = sum(1 for _p, s in measured if s is None)
    if unmeasured:
        print(f"      {unmeasured} pair(s) could not be measured (unreadable "
              f"frame, or the sampled pixels lie outside a subarray)",
              flush=True)
    return measured


def _group_key(crf_path):
    """Group per-exposure crf by (obs+visit, module).  Filename like
    jw01182004001_04101_00001_nrca3_destreak_o004_crf.fits -> '004001:nrca'."""
    p = _parse_crf(crf_path)
    if p is None:
        return "det?:det?"
    return f"{p['obs']}{p['visit']}:{p['module']}"


# regex for a merged science mosaic of EITHER instrument (nircam merged_i2d /
# miri ..._data_i2d), incl. the combined multi-observation form
# jwPPPPP-oOOO-MMM (an association of obs OOO and MMM -> both are released).
_MOSAIC_OBS_RE = re.compile(
    r"^jw(?P<prop>\d{5})-o(?P<obs>\d{3})(?:-(?P<obs2>\d{3}))?_t001_"
    r"(?:nircam|miri)_")


def _mosaic_obs_keys(name):
    """The ``"<proposal>-<observation>"`` key(s) a merged-mosaic basename
    encodes (a combined ``-oOOO-MMM`` mosaic encodes both)."""
    m = _MOSAIC_OBS_RE.match(os.path.basename(name))
    if m is None:
        return set()
    keys = {f"{m.group('prop')}-{m.group('obs')}"}
    if m.group("obs2"):
        keys.add(f"{m.group('prop')}-{m.group('obs2')}")
    return keys


# explicit sentinel: "scoping disabled -- accept every well-formed crf".
# Distinct from ``None`` (derive the scope) so a derivation that finds NOTHING
# is never silently indistinguishable from a deliberate opt-out.
NO_SCOPE = object()


def _field_observations(field, filt):
    """The ``"<proposal>-<observation>"`` keys RELEASED in ONE filter directory,
    from the merged science mosaics on disk there.  Derivation is PER FILTER
    DIRECTORY (not per field): a ``{field}/{filt}/pipeline`` dir holds one
    instrument's products, so a MIRI filter derives MIRI observations and a
    NIRCam filter derives NIRCam ones -- and a stray obs sharing a proposal-obs
    key across instruments (brick MIRI F2550W and the cloudc NIRCam crf are both
    2221 o002) can never leak between them.  ``images-merged/`` is included for
    the per-observation release layout (gc2211's o050 lives there).

    Matches ONLY the canonical science-mosaic form
    ``..._t001_<instr>_clear-<filt>-...`` -- a shared dir also holds
    NON-canonical stray products (the cloudc combined
    ``jw02221-o002_t001_nircam_f405n-f444w_i2d.fits`` has no ``clear`` and is not
    a release mosaic); matching those would re-derive the stray's observation."""
    obs = set()
    fl = filt.lower()
    for pat in (f"{BASE}/{field}/{filt}/pipeline/jw*-o*_t001_nircam_clear-{fl}-*_i2d.fits",
                f"{BASE}/{field}/{filt}/pipeline/jw*-o*_t001_miri_clear-{fl}-*_i2d.fits",
                f"{BASE}/{field}/images-merged/jw*-o*_t001_nircam_clear-{fl}-*_i2d.fits",
                f"{BASE}/{field}/images-merged/jw*-o*_t001_miri_clear-{fl}-*_i2d.fits"):
        for p in glob.glob(pat):
            obs |= _mosaic_obs_keys(p)
    return obs


def build_groups(field, filt, observations=None):
    """Detect on each per-exposure crf and pool by (obs+visit, module).

    Scoping (excludes stray crf from other programs/observations in a shared
    target dir -- the brick dir also holds 2221 o002 = cloudc crf):

    - ``observations=None`` (default): scope to the RELEASED observations derived
      from THIS filter directory's mosaics (``_field_observations``);
    - an explicit iterable of ``"<proposal>-<observation>"`` keys: RESTRICT the
      derived scope to their intersection (restrict-only -- a broad field-level
      set, e.g. one that also lists a MIRI observation, can never RE-ADMIT a
      stray that the per-directory derivation already excluded);
    - ``NO_SCOPE``: disable scoping, accept every well-formed crf.

    A derivation that yields nothing (no mosaic on disk) is announced loudly and
    leaves scoping OFF for that filter -- on a shared target dir that is exactly
    when a stray would slip in, so it must not pass silently.
    """
    if observations is NO_SCOPE:
        scope = None
    else:
        derived = _field_observations(field, filt)
        if observations is None:
            scope = derived or None
            if scope is None:
                print(f"  WARNING: {field}/{filt}: could not derive released "
                      f"observations (no mosaic on disk) -- scoping DISABLED; a "
                      f"stray crf from another observation could pollute this "
                      f"verdict", flush=True)
        else:
            passed = set(observations)
            # restrict-only: intersect with the per-directory derivation when it
            # exists; fall back to the passed set only when nothing was derived
            scope = (derived & passed) if derived else passed
            if derived and not scope:
                # The directory HAS products and they are well-formed; they just
                # belong to observations this release does not claim.  Returning
                # an empty scope filters every frame out and the caller reports
                # "NO crf frames matched -- glob mismatch / missing products?",
                # which names the wrong cause and refuses the whole field.
                #
                # Live case: cloudc ships NIRCam 2221-o002 and its F770W
                # directory holds 8 well-formed 2526-o021 crf.  The empty
                # intersection refused a NIRCam-only release over a MIRI band
                # that release never touched, while the frames sat on disk and
                # parsed cleanly.
                #
                # Distinct from a derivation that yields NOTHING (no mosaic on
                # disk), which stays fail-closed above: there the glob really
                # may have drifted.
                raise OutOfReleaseScope(field, filt, derived, passed)
    # Enumerate broadly, then let the precise _parse_crf regex decide -- a name
    # that is not a well-formed crf, or belongs to an out-of-scope observation,
    # is dropped.  MIRI crf carry no _destreak token; the parser covers them
    # (excluding MIRI once silently PASSED the F2550W doubled-star saga).
    frames = []
    n_retired = 0
    for fn in sorted(glob.glob(f"{BASE}/{field}/{filt}/pipeline/jw*_crf.fits")):
        p = _parse_crf(fn)
        if p is None:
            # _parse_crf returns None both for "not a well-formed crf name" and
            # for "deliberately excluded retired-path product".  Count the second
            # so a silently smaller frame set stays distinguishable from an empty
            # directory: check_filter fails closed on nframes == 0, but a PARTIAL
            # exclusion (a visit losing all its frames) would pass with fewer
            # groups and no trace.
            if _retired_lineage(fn):
                n_retired += 1
            continue
        if scope is not None and p["obs_key"] not in scope:
            continue
        frames.append(fn)
    if n_retired:
        print(f"  {field}/{filt}: excluded {n_retired} retired-path crf "
              f"(realign_to_vvv lineage; FITS header and GWCS disagree by "
              f"arcseconds in those files)", flush=True)
    # One copy per physical exposure.  A working directory normally holds the
    # same exposure several times over, written by reductions run months or
    # years apart, and pooling them compares an exposure against stale copies of
    # itself.  Announced rather than silent, for the same reason as the retired
    # count above: a smaller frame set must stay distinguishable from a smaller
    # directory.
    frames, superseded = select_one_copy_per_exposure(frames, field, filt)
    if superseded:
        reasons = {}
        for path, why in superseded:
            reasons.setdefault(why, []).append(path)
        print(f"  {field}/{filt}: {len(superseded)} superseded lineage "
              f"copies excluded, keeping one per exposure:", flush=True)
        for why, paths in sorted(reasons.items()):
            example = os.path.basename(paths[0])
            print(f"      {len(paths):4d} dropped because {why}  (e.g. "
                  f"{example})", flush=True)
        # Measure what was discarded rather than only naming it.  The selection
        # above reads filenames and a recorded setting; this is the only part
        # that consults the frames, and so the only part that could notice the
        # setting pointing at the wrong copy.
        #
        # It is not free: two GWCS loads per discarded copy at ~0.5-2 s each,
        # so ~3 min on a 192-exposure filter directory such as brick/F212N,
        # ahead of the detection pass that reopens the kept frames anyway.
        # Worth it while the disagreements are this large and this unmapped;
        # revisit if a full --scan becomes routine.
        report_lineage_disagreement(superseded, frames, f"{field}/{filt}")
    groups = {}
    ndet = {}
    for fn in frames:
        s = _detect(fn)
        if s is None:
            continue
        k = _group_key(fn)
        groups.setdefault(k, []).append(s)
        ndet[k] = ndet.get(k, 0) + len(s)
    pooled = {k: SkyCoord(np.concatenate([c.ra.deg for c in v]) * u.deg,
                          np.concatenate([c.dec.deg for c in v]) * u.deg)
              for k, v in groups.items()}
    return pooled, ndet, len(frames)


def _refcat(path):
    t = Table.read(path)
    cols = {c.lower(): c for c in t.colnames}
    ra, dec = t[cols["ra"]], t[cols["dec"]]
    rc = SkyCoord(ra.data * u.deg if ra.unit is None else ra,
                  dec.data * u.deg if dec.unit is None else dec)
    src = cols.get("source")
    gaia = None
    if src is not None:
        gm = np.array([s in (b"GaiaDR3", "GaiaDR3") for s in t[src]])
        gaia = rc[gm] if gm.any() else None
    return rc, gaia


def _val_mas(x):
    try:
        return float(x.to(u.mas).value)
    except AttributeError:
        return float(x)


def _verdict(clean, measurable, worst_off_mas=float("nan"), n_ok=0, n_total=0,
             reason="", **extra):
    """Uniform arbiter verdict.  ``measurable`` is the third state the gate needs:
    a map with no cell holding enough stars is COULD-NOT-VERIFY, which is a
    different answer from ``clean``, and the gate must not treat them alike."""
    out = dict(clean=bool(clean), measurable=bool(measurable),
               worst_off_mas=float(worst_off_mas), n_ok=int(n_ok),
               n_total=int(n_total), reason=reason)
    out.update(extra)
    return out


def _match_radius_consistency(src, ref, cell_arcsec, min_stars=SAMESTAR_MIN_STARS,
                              radii=SAMESTAR_MATCH_RADII,
                              collapse_frac=SAMESTAR_COLLAPSE_FRAC):
    """NN-collapse detector: per-cell matched-REFERENCE-star counts as the match
    radius shrinks (issue #174 option (b)).

    A localized sub-population shifted by ~the match radius (300 mas ~ the
    dense-field NN spacing) pairs with the WRONG neighbour at 0.3": the residuals
    scatter, the per-cell median lands near zero, and the residual map reports
    CLEAN.  Shrink the radius to 0.1" and those stars stop matching altogether --
    their cell's matched count collapses while the rest of the field's barely
    moves.  The statistic is a RATIO OF SAME-STAR MATCH COUNTS within a cell,
    normalised to the field, so it carries none of the density coupling of an
    offset histogram (issue #170).

    Cells are indexed on the REFERENCE positions so the grid does not move with
    the radius.  Returns ``dict(ok, n_ambiguous, worst_ratio, field_ratio,
    n_cells)``; ``ok=False`` means at least one cell is matching-ambiguous and the
    map's "clean" there cannot be believed.
    """
    radii = sorted(float(r) for r in radii)
    if len(radii) < 2 or len(src) == 0 or len(ref) == 0:
        return dict(ok=True, n_ambiguous=0, worst_ratio=float("nan"),
                    field_ratio=float("nan"), n_cells=0)
    r_small, r_big = radii[0], radii[-1]
    dec_mid = float(np.median(ref.dec.deg))
    cell_dec = cell_arcsec / 3600.0
    cell_ra = cell_arcsec / 3600.0 / max(np.cos(np.radians(dec_mid)), 1e-6)
    r0, d0 = float(ref.ra.deg.min()), float(ref.dec.deg.min())
    ix = np.floor((ref.ra.deg - r0) / cell_ra).astype(np.int64)
    iy = np.floor((ref.dec.deg - d0) / cell_dec).astype(np.int64)
    key = ix * (iy.max() + 2) + iy
    counts = {}
    for rr in (r_small, r_big):
        iref, _, _, _ = search_around_sky(ref, src, rr * u.arcsec)
        matched = np.unique(iref)
        counts[rr] = dict(zip(*np.unique(key[matched], return_counts=True))) \
            if len(matched) else {}
    big, small = counts[r_big], counts[r_small]
    cells = [(k, n) for k, n in big.items() if n >= min_stars]
    if not cells:
        return dict(ok=True, n_ambiguous=0, worst_ratio=float("nan"),
                    field_ratio=float("nan"), n_cells=0)
    tot_big = sum(n for _, n in cells)
    tot_small = sum(small.get(k, 0) for k, _ in cells)
    field_ratio = tot_small / tot_big if tot_big else 0.0
    ratios = [(small.get(k, 0) / n) for k, n in cells]
    worst = min(ratios)
    n_amb = sum(1 for x in ratios if x < collapse_frac * field_ratio)
    return dict(ok=bool(n_amb == 0), n_ambiguous=int(n_amb),
                worst_ratio=float(worst), field_ratio=float(field_ratio),
                n_cells=len(cells))


def _global_tie(allsrc, ref, max_off_mas,
                match_radius_arcsec=SAMESTAR_MATCH_RADIUS):
    """Density-immune global tie, with every precondition ``local_residual_map``
    imposes checked HERE.  It raises ``GlobalTieNotVerifiedError`` unless the tie
    has ``ok=True``, ``swept=False`` and ``off < match_radius/3``; the arbiter
    used to check only the last two, so a LOW-CONTRAST tie escaped as an uncaught
    traceback out of a blocking gate.  A tie we cannot verify is a
    could-not-verify verdict, not a crash and not a pass.

    Returns ``(g, verdict)``: exactly one is None."""
    g = measure_offset(allsrc, ref, maxsep=3.0 * u.arcsec)
    if g is None:
        return None, _verdict(False, measurable=False,
                              reason="no global tie could be measured vs the reference")
    goff = float(np.hypot(_val_mas(g["dra"]), _val_mas(g["ddec"])))
    if not g.get("ok"):
        # low contrast == the peak is not a tie at all; local_residual_map would
        # raise GlobalTieNotVerifiedError.
        return None, _verdict(False, measurable=False, worst_off_mas=goff,
                              reason=f"global tie NOT verified (low contrast "
                                     f"{g.get('contrast')}); cannot map residuals")
    if g.get("swept") or goff > max_off_mas:
        # a gross or window-swept global tie == a genuine absolute-frame offset
        # (brick-1182 v001 ~20" class); reference cross-check is DIRTY.
        return None, _verdict(False, measurable=True, worst_off_mas=goff, n_total=1,
                              reason=f"global tie {goff:.0f} mas "
                                     f"(swept={bool(g.get('swept'))}) > {max_off_mas:.0f} mas")
    if goff > match_radius_arcsec * 1000.0 / 3.0:
        # same precondition, stated by the library; keep it out of the traceback
        return None, _verdict(False, measurable=False, worst_off_mas=goff,
                              reason=f"global tie {goff:.0f} mas is not << the "
                                     f"{match_radius_arcsec * 1000:.0f} mas match radius; "
                                     f"matched pairs would be ambiguous")
    return g, None


def _samestar_ref_grid(allsrc, ref, max_off_mas, tol_mas=None, cells_arcsec=None,
                       min_stars=SAMESTAR_MIN_STARS, check_radii=True):
    """Absolute cross-check of ``allsrc`` vs a reference catalog by the SAME-STAR
    residual map (density-immune).

    ``measure_offset_grid`` (per-cell histogram) is fooled by the dense-field
    wrong-pair bias in sparse cells: on brick F405N it read a 58" worst cell
    while the same-star tie of the identical data is 3 mas.  Here we (1) get the
    density-immune global tie and verify every precondition, (2) fail if it is
    gross / window-swept (the field really is off the reference), else (3) map
    per-cell SAME-STAR residuals with ``local_residual_map`` (real matched pairs,
    per-cell standard errors -- the noise-peak failure mode does not exist) at a
    LADDER of cell sizes, and (4) cross-check that the same-star matching itself
    is not NN-collapsing.

    Two thresholds, two jobs (issue #174): ``max_off_mas`` is the GROSS ceiling on
    the global tie; ``tol_mas`` (default ``TOL_MAS``, the gate's own tolerance) is
    the PER-CELL sensitivity.  Passing the gross ceiling through as the per-cell
    tolerance made every local seam between 30 and 80 mas read clean.

    Returns ``clean, measurable, worst_off_mas, n_ok, n_total`` (+ diagnostics).
    ``measurable=False`` is could-not-verify, NOT clean."""
    tol_mas = TOL_MAS if tol_mas is None else float(tol_mas)
    cells_arcsec = SAMESTAR_CELLS_ARCSEC if cells_arcsec is None else cells_arcsec
    g, bad = _global_tie(allsrc, ref, max_off_mas)
    if bad is not None:
        return bad
    scales, n_meas_tot, n_flag_tot, worst = [], 0, 0, float("nan")
    for cell in sorted(float(c) for c in cells_arcsec):
        lr = local_residual_map(allsrc, ref, g, cell_arcsec=cell,
                                match_radius=SAMESTAR_MATCH_RADIUS * u.arcsec,
                                min_stars=min_stars, tol_mas=tol_mas)
        n_meas = int(lr.get("n_measured", 0))
        n_flag = int(lr.get("n_flagged", 0))
        w = lr.get("worst_off_mas")
        scales.append(dict(cell_arcsec=cell, n_measured=n_meas, n_flagged=n_flag,
                           worst_off_mas=(float(w) if w is not None else float("nan"))))
        n_meas_tot += n_meas
        n_flag_tot += n_flag
        if n_meas and w is not None and np.isfinite(w):
            worst = float(w) if not np.isfinite(worst) else max(worst, float(w))
    if n_meas_tot == 0:
        # no scale had a cell with enough stars: UNMEASURABLE, not clean.
        return _verdict(False, measurable=False, global_tie=g, scales=scales,
                        reason=f"no residual-map cell held >= {min_stars} matched "
                               f"stars at any scale {tuple(cells_arcsec)}\"")
    amb = (_match_radius_consistency(allsrc, ref,
                                     cell_arcsec=max(float(c) for c in cells_arcsec),
                                     min_stars=min_stars)
           if check_radii else dict(ok=True, n_ambiguous=0, n_cells=0,
                                    worst_ratio=float("nan"), field_ratio=float("nan")))
    # FIELD-WIDE the radius-collapse ratio is a DIAGNOSTIC, not a gate: a stray
    # exposure group from another program sharing the target directory (brick
    # F405N carries o002 cloudc frames) collapses it on its own -- measured 0.16
    # for that group against 0.83-0.86 for the brick's own visit.  It GATES where
    # it is well posed: inside a deferred pair's own footprint
    # (``_samestar_pair_footprint``), where the population is the pair's.  Set
    # OVERLAP_SAMESTAR_RADII_GATE=1 to make it gate field-wide too.
    radii_gate = os.environ.get("OVERLAP_SAMESTAR_RADII_GATE", "") == "1"
    clean = bool(n_flag_tot == 0 and (amb["ok"] or not radii_gate))
    reason = ""
    if n_flag_tot:
        reason = f"{n_flag_tot} significant cell(s) > {tol_mas:.0f} mas"
    elif not amb["ok"]:
        reason = (f"{amb['n_ambiguous']} cell(s) whose same-star match count "
                  f"collapses with the match radius (worst {amb['worst_ratio']:.2f} "
                  f"vs field {amb['field_ratio']:.2f}) -- possible NN-collapsed "
                  f"sub-population"
                  + ("; not verifiable as clean" if radii_gate else " [diagnostic]"))
    return _verdict(clean, measurable=True,
                    worst_off_mas=(0.0 if not np.isfinite(worst) else worst),
                    n_ok=n_meas_tot - n_flag_tot, n_total=n_meas_tot,
                    reason=reason, global_tie=g, scales=scales, radii=amb)


def _nearest_residuals(src, ref, radius_arcsec, global_result):
    """Per REFERENCE star, the residual (ref - nearest src detection) in mas with
    the verified global tie removed.  Indexed on the reference so two exposure
    groups can be differenced star-by-star.

    Deliberately allows a source to serve several reference stars: ``src`` here
    is a pooled per-exposure detection list in which every star appears once per
    dither, so the library's a-side uniqueness filter would throw away the
    overlap regions -- exactly where a deferred pair lives."""
    isrc, iref, sep, _ = search_around_sky(src, ref, radius_arcsec * u.arcsec)
    if len(isrc) == 0:
        return np.array([], int), np.array([]), np.array([])
    order = np.lexsort((sep.arcsec, iref))
    iref_o, isrc_o = iref[order], isrc[order]
    first = np.concatenate(([True], iref_o[1:] != iref_o[:-1]))
    iref_n, isrc_n = iref_o[first], isrc_o[first]
    cosd = np.cos(np.radians(ref[iref_n].dec.value))
    dra = ((ref[iref_n].ra - src[isrc_n].ra).to(u.arcsec).value * cosd * 1000.0
           - _val_mas(global_result["dra"]))
    ddec = ((ref[iref_n].dec - src[isrc_n].dec).to(u.arcsec).value * 1000.0
            - _val_mas(global_result["ddec"]))
    return iref_n, dra, ddec


def _robust(x):
    """Median and its standard error (MAD-based) -- the same estimator
    ``local_residual_map`` uses per cell."""
    if len(x) == 0:
        return float("nan"), float("nan")
    m = float(np.median(x))
    return m, float(1.4826 * np.median(np.abs(x - m)) / np.sqrt(len(x)))


def pair_overlap_footprint(a_src, b_src, radius_arcsec=3.0):
    """The pair's own mutual-coverage footprint: the ``a`` detections that have a
    ``b`` detection within ``radius_arcsec``.  A deferred pair is a sliver, and a
    sliver is a minority of the FIELD but the whole population of its own
    footprint -- which is why the arbiter must be scoped to it."""
    if len(a_src) == 0 or len(b_src) == 0:
        return None
    ia, _, _, _ = search_around_sky(a_src, b_src, radius_arcsec * u.arcsec)
    if len(ia) == 0:
        return None
    return a_src[np.unique(ia)]


def _samestar_pair_footprint(a_src, b_src, ref, global_result, tol_mas=None,
                             abs_tol_mas=None, min_stars=SAMESTAR_MIN_STARS,
                             radii=SAMESTAR_MATCH_RADII, nsigma=3.0):
    """Same-star arbiter SCOPED TO ONE DEFERRED PAIR'S OVERLAP FOOTPRINT.

    The field-wide map is a single boolean for the whole filter: one clean map
    cleared EVERY deferred pair, and a sliver-sized offset is diluted inside a
    field-sized cell anyway.  Here each group is tied to the reference over the
    pair's own footprint only, and the two are DIFFERENCED star by star: the
    difference is the inter-frame offset in the overlap, with the reference used
    purely as a common anchor, so the reference's own local systematics (and the
    blend bias of a seeing-limited catalog) cancel.

    Measured at several match radii; if the answer moves with the radius the
    matching is NN-collapsing and the verdict is could-not-verify, not clean."""
    tol_mas = TOL_MAS if tol_mas is None else float(tol_mas)
    # The INTER-FRAME difference is what this gate owns, and it is measured to a
    # few mas, so it is held to TOL_MAS.  The ABSOLUTE excursion of the footprint
    # vs the reference is a different quantity (the field-wide map owns it) and it
    # carries the reference's own local systematics -- brick F405N's clean
    # deferred pair sits 17 mas off VIRAC2 there -- so it is held to the gross
    # ceiling, not to TOL_MAS.
    abs_tol_mas = GRID_MAX_OFF_MAS if abs_tol_mas is None else float(abs_tol_mas)
    fp = pair_overlap_footprint(a_src, b_src)
    if fp is None:
        return _verdict(False, measurable=False,
                        reason="no mutual-coverage footprint (groups share no sky)")
    iref, _, _, _ = search_around_sky(ref, fp, 1.5 * u.arcsec)
    if len(iref) == 0:
        return _verdict(False, measurable=False, n_footprint_ref=0,
                        reason="no reference stars inside the pair's overlap footprint")
    rsub = ref[np.unique(iref)]
    per_radius = []
    for rr in sorted(float(r) for r in radii):
        ia, dra_a, ddec_a = _nearest_residuals(a_src, rsub, rr, global_result)
        ib, dra_b, ddec_b = _nearest_residuals(b_src, rsub, rr, global_result)
        common, ja, jb = np.intersect1d(ia, ib, return_indices=True)
        n = len(common)
        if n < min_stars:
            per_radius.append(dict(radius=rr, n=n, n_a=len(ia), n_b=len(ib),
                                   off_mas=float("nan"), sem_mas=float("nan"),
                                   mad_mas=float("nan"), a_off_mas=float("nan"),
                                   b_off_mas=float("nan")))
            continue
        ddra, dsem = _robust(dra_a[ja] - dra_b[jb])
        dddec, dsemd = _robust(ddec_a[ja] - ddec_b[jb])
        # Scatter of the per-star differential, recorded as a DIAGNOSTIC.  It is
        # the obvious second NN-collapse tell (a collapsing sub-population pairs
        # with whatever lies inside the radius, so its residuals fill the search
        # annulus), but measured on real pooled crf lists it barely moves -- 1.15
        # clean vs 1.31 with half the frame shifted 300 mas -- because at ~14
        # detections/arcsec^2 the wrong neighbour is usually close to the right
        # position anyway.  Reported, not gated on.
        mad = float(np.hypot(dsem, dsemd) * np.sqrt(n))
        aoff = float(np.hypot(_robust(dra_a[ja])[0], _robust(ddec_a[ja])[0]))
        boff = float(np.hypot(_robust(dra_b[jb])[0], _robust(ddec_b[jb])[0]))
        per_radius.append(dict(radius=rr, n=n, n_a=len(ia), n_b=len(ib),
                               off_mas=float(np.hypot(ddra, dddec)),
                               sem_mas=float(np.hypot(dsem, dsemd)), mad_mas=mad,
                               a_off_mas=aoff, b_off_mas=boff))
    ok_r = [p for p in per_radius if p["n"] >= min_stars]
    if not ok_r:
        best = max(per_radius, key=lambda p: p["n"])
        return _verdict(False, measurable=False, n_footprint_ref=len(rsub),
                        per_radius=per_radius, n_common=best["n"],
                        reason=f"only {best['n']} common same-star match(es) in the "
                               f"pair's footprint (need {min_stars}) -- the overlap "
                               f"holds too few reference stars to arbitrate")
    widest = ok_r[-1]
    offs = [p["off_mas"] for p in ok_r]
    spread = float(max(offs) - min(offs))
    # NN-COLLAPSE tell, posed as a comparison of the TWO GROUPS against each other
    # over the SAME reference stars: shrink the match radius and count how many of
    # each group's matches survive.  A sub-population of one frame sitting at ~the
    # match radius pairs with the wrong neighbour at 0.3" (so its median residual
    # reads ~0, "clean") but simply stops matching at 0.1" -- its keep-ratio
    # collapses while the other frame's does not.  Both ratios are measured on the
    # same stars in the same sky, so nothing here is density-coupled.
    keep, nbig = {}, {}
    for side in ("n_a", "n_b"):
        big = per_radius[-1][side]
        nbig[side] = big
        keep[side] = (per_radius[0][side] / big) if big else float("nan")
    ratios = [v for v in keep.values() if np.isfinite(v)]
    keep_gap = float(max(ratios) - min(ratios)) if len(ratios) == 2 else float("nan")
    pbar = float(np.mean(ratios)) if len(ratios) == 2 else float("nan")
    sig = (np.sqrt(max(pbar * (1 - pbar), 0.0)
                   * (1 / max(nbig["n_a"], 1) + 1 / max(nbig["n_b"], 1)))
           if np.isfinite(pbar) else float("nan"))
    keep_consistent = bool(len(ratios) == 2 and
                           keep_gap <= max(SAMESTAR_KEEP_MARGIN, 3.0 * sig))
    flagged = any(p["off_mas"] > tol_mas and p["off_mas"] > nsigma * p["sem_mas"]
                  for p in ok_r)
    # a localized ABSOLUTE excursion inside the footprint (both frames off there)
    # is a real registration error too, and the field-wide map dilutes it.
    abs_flag = any(max(p["a_off_mas"], p["b_off_mas"]) > abs_tol_mas for p in ok_r)
    consistent = spread <= tol_mas and len(ok_r) == len(per_radius)
    worst = max(max(p["off_mas"], p["a_off_mas"], p["b_off_mas"]) for p in ok_r)
    if flagged or abs_flag:
        reason = (f"inter-frame same-star offset {widest['off_mas']:.0f} mas "
                  f"(+-{widest['sem_mas']:.0f}) over {widest['n']} common stars in the "
                  f"pair's own overlap footprint" if flagged else
                  f"footprint sits {worst:.0f} mas off the reference while the field "
                  f"tie does not")
        return _verdict(False, measurable=True, worst_off_mas=worst,
                        n_ok=0, n_total=widest["n"], n_common=widest["n"],
                        n_footprint_ref=len(rsub), per_radius=per_radius,
                        reason=reason)
    if not consistent or not keep_consistent:
        why = (f"same-star offset moves {spread:.0f} mas across match radii "
               f"{[p['radius'] for p in per_radius]} (or a radius lost its matches)"
               if not consistent else
               f"the two frames keep very different fractions of their same-star "
               f"matches when the radius shrinks to {per_radius[0]['radius']}\" "
               f"(A {keep['n_a']:.2f} vs B {keep['n_b']:.2f}, gap {keep_gap:.2f} > "
               f"{max(SAMESTAR_KEEP_MARGIN, 3.0 * sig):.2f}) -- a sub-population of "
               f"one frame sits at ~the match radius")
        return _verdict(False, measurable=False, worst_off_mas=worst,
                        n_common=widest["n"], n_footprint_ref=len(rsub),
                        per_radius=per_radius, keep_ratio=keep,
                        reason=f"{why} -- NN-collapsing, not verifiable as clean")
    return _verdict(True, measurable=True, worst_off_mas=worst,
                    n_ok=widest["n"], n_total=widest["n"], n_common=widest["n"],
                    n_footprint_ref=len(rsub), per_radius=per_radius,
                    keep_ratio=keep,
                    reason=f"{widest['n']} common same-star matches, inter-frame "
                           f"{widest['off_mas']:.0f} mas")


def check_filter(field, filt, refcat=None, verbose=True, observations=None):
    try:
        pooled, ndet, nframes = build_groups(field, filt,
                                             observations=observations)
    except OutOfReleaseScope as out:
        # Reported, never blocking, and never counted as verified: a band this
        # release does not ship has nothing for this gate to say about it.
        if verbose:
            print(f"  {field} {filt}: NOT IN THIS RELEASE -- {out}", flush=True)
        # PASS is None, not True: this gate did not verify anything here and
        # must not claim it did.  `not_in_release` is what the scan reads to
        # skip it; anything that only looks at PASS sees "no verdict".
        return dict(field=field, filt=filt, PASS=None, not_in_release=True,
                    derived=sorted(out.derived),
                    note=f"not in this release's observations "
                         f"({', '.join(sorted(out.derived))})")
    # FAIL-CLOSED on "found nothing": a gate that goes green because its glob
    # matched zero files (renamed products, naming drift) is the silent
    # false-agreement class this repo bans.  Distinguish it from a genuine
    # single-group field (frames found, nothing to pairwise-check).
    if nframes == 0:
        if verbose:
            print(f"  {field} {filt}: NO crf frames matched -- cannot verify "
                  f"inter-frame registration (glob mismatch / missing products?)",
                  flush=True)
        return dict(field=field, filt=filt, PASS=False, could_not_verify=True,
                    note="no crf frames matched")
    if not pooled:
        if verbose:
            print(f"  {field} {filt}: {nframes} crf but detection produced NO "
                  f"usable groups -- cannot verify", flush=True)
        return dict(field=field, filt=filt, PASS=False, could_not_verify=True,
                    note="no detections from any crf")
    if len(pooled) < 2:
        if verbose:
            print(f"  {field} {filt}: single exposure-group from {nframes} crf "
                  f"-- nothing to overlap-check", flush=True)
        return dict(field=field, filt=filt, PASS=True,
                    note=f"single exposure-group ({nframes} crf)")

    # PER-TILE (local) reference-free check: a per-visit residual is spatially
    # varying, so a field-pooled single offset can average below tol while a thin
    # seam is ~90 mas off. Grid it.
    # TWO-LAYER frame-vs-frame check, both reference-free:
    #   fine  : per-tile grid on mutual-coverage cells (owns the 30-100 mas
    #           seam regime; blind BY CONSTRUCTION to offsets > its cell
    #           margin, which empty the mutual-coverage cells);
    #   gross : field-pooled SWEPT histogram over the intersection populations
    #           (owns the >margin regime -- the brick-1182 v001 ~20" class --
    #           a gross rigid offset still overlaps in the strip, so the swept
    #           peak recovers it with no reference catalog).
    # A pair verifies only if a layer POSITIVELY measured it; a pair NEITHER
    # layer can measure is could-not-verify -> exit 2 (fail closed: "a
    # failsafe that cannot run is not a passing failsafe").
    res = overlap_offset_grid(pooled, tol_mas=TOL_MAS, nx=GRID_N, ny=GRID_N,
                              maxsep=3.0 * u.arcsec)
    pw = {(r["a"], r["b"]): r for r in pairwise_overlap_offsets(
        pooled, tol_mas=TOL_MAS, maxsep=3.0 * u.arcsec)}
    GROSS_MAS = 3.0 * 3000.0  # beyond the grid's ~3*maxsep cell margin
    bad, unverifiable, overlapped = [], [], []
    for r in res:
        if not r["overlap"]:
            continue
        overlapped.append(r)
        p = pw.get((r["a"], r["b"])) or pw.get((r["b"], r["a"])) or {}
        r["pairwise"] = {k: p.get(k) for k in
                         ("off_mas", "dra_mas", "ddec_mas", "contrast",
                          "n_peak", "measurable", "swept", "ok",
                          "window_consistent", "window_edge_fraction",
                          "alias_rejected")}
        # Authoritative discriminant: did ANY mutual-coverage TILE get measured?
        # n_total == 0 -> a sparse / thin overlap where NEITHER the per-tile
        # same-star layer NOR the pooled frame-vs-frame histogram is trustworthy
        # (against a dense clustered field the wrong-pair bias yields a
        # _confirm_tie-CONFIRMED but SPURIOUS pooled peak: brick F405N
        # nrca-long|nrcb-long read 55 mas pooled over 0 tiles, yet same-star vs
        # VIRAC = 3 mas).  Defer such a flagged pair to the external SAME-STAR
        # reference below rather than hard-failing on the pooled number alone.
        #
        # EXCEPT when the pooled peak is WINDOW-CONFIRMED (issue #158): a real
        # rigid offset reads the SAME value at every window large enough to hold
        # it, while a pair-density ridge slides with the window.  Deferring a
        # window-confirmed gross offset would report the brick-1182 v001 ~20"
        # class as "could not verify" (rc=2, unknown) rather than as the measured
        # misregistration it is (rc=1, actionable) -- and with no --refcat both
        # collapse to the same refusal, so the operator loses the one signal that
        # says which it was.  A confirmed gross peak is a MEASUREMENT; fail on it.
        n_tiles_measured = int(r.get("n_total", 0) or 0)
        if n_tiles_measured == 0:
            gross_measured = bool(p.get("measurable")) and not p.get("ok", True)
            if gross_measured and p.get("window_consistent") is True:
                r["fail_reason"] = (
                    f"gross pairwise offset {p['off_mas']:.0f} mas "
                    f"(swept={p.get('swept')}, reproduced at an independent "
                    f"window) over a 0-tile overlap")
                r["could_not_verify"] = False
                bad.append(r)
            elif r.get("could_not_verify") or not r.get("ok", True):
                if gross_measured:
                    r["fail_reason"] = (
                        f"gross pairwise offset {p['off_mas']:.0f} mas "
                        f"(swept={p.get('swept')}, window_consistent="
                        f"{p.get('window_consistent')}) over a 0-tile overlap -- "
                        f"pooled histogram not authoritative; deferred to reference")
                unverifiable.append(r)
            # ok + 0 tiles = clean non-overlap: nothing to check.
        else:
            # Real mutual coverage measured -> the reference-free verdict IS
            # authoritative (this is the brick-1182 F200W dense-seam regime).
            if not r["ok"]:
                r["fail_reason"] = (r.get("fail_reason")
                                    or "per-tile misregistration")
                bad.append(r)
            elif (p.get("measurable") and p.get("off_mas") is not None
                    and p["off_mas"] > GROSS_MAS):
                # tiles measured fine locally but the pooled swept peak sits
                # beyond the grid's sight -- gross regime, fail
                r["fail_reason"] = f"gross pairwise offset {p['off_mas']:.0f} mas"
                bad.append(r)
    if verbose:
        print(f"  {field} {filt}: {nframes} crf -> {len(pooled)} groups, "
              f"{len(overlapped)} overlapping pairs, {len(bad)} FAIL, "
              f"{len(unverifiable)} could-not-verify (tol {TOL_MAS:.0f} mas, "
              f"{GRID_N}x{GRID_N} tiles + pooled swept)", flush=True)
        for r in sorted(overlapped, key=lambda r: -(r["worst_off_mas"] or 0))[:8]:
            tag = ("FAIL" if r in bad
                   else ("could-not-verify" if r in unverifiable else "ok"))
            _w = r["worst_off_mas"]
            _p = r.get("pairwise", {})
            print(f"      {tag}: {r['a']} | {r['b']}  worst tile off="
                  f"{'n/a' if _w is None else f'{_w:.0f} mas'} "
                  f"({r['n_ok']}/{r['n_total']} tiles ok, "
                  f"{r.get('n_no_coverage', 0)} no-mutual-coverage cells; "
                  f"pooled off={'n/a' if _p.get('off_mas') is None else f'{_p['off_mas']:.0f} mas'}"
                  f"{' MEASURABLE' if _p.get('measurable') else ''}"
                  f"{'; ' + r['fail_reason'] if r.get('fail_reason') else ''})",
                  flush=True)
        for r in unverifiable:
            print(f"      COULD NOT VERIFY: {r['a']} | {r['b']} -- footprints "
                  f"intersect but neither the per-tile grid (no mutual-coverage "
                  f"cells) nor the pooled swept histogram (no measurable peak) "
                  f"could measure the pair.  Fail-closed: requires the external "
                  f"reference map (--refcat) or a fixed reduction to stage.",
                  flush=True)

    ext_fail = False
    ext_ran = False
    field_clean = False
    if refcat:
        rc, gaia = _refcat(refcat)
        allsrc = SkyCoord(np.concatenate([p.ra.deg for p in pooled.values()]) * u.deg,
                          np.concatenate([p.dec.deg for p in pooled.values()]) * u.deg)
        # GC reference-frame policy (gc-gaia-frame-not-catalog): VIRAC2 is the GC
        # reference catalog and the ONLY absolute frame that GATES here. Gaia is
        # the FRAME but far too sparse (~1.8k over the Brick vs ~113k VIRAC2) to
        # per-tile a dense field -- on a fine grid its cells are star-starved and
        # false-fail, so it is measured as a DIAGNOSTIC only and never sets
        # ext_fail. A real absolute-frame problem shows up against dense VIRAC2.
        for rn, rr, gates in (("VIRAC2", rc, True), ("Gaia", gaia, False)):
            if rr is None:
                continue
            g = _samestar_ref_grid(allsrc, rr, max_off_mas=GRID_MAX_OFF_MAS)
            if gates:
                # a map that could not be measured did NOT run: it clears nothing
                ext_ran = bool(g["measurable"])
                field_clean = bool(g["clean"])
            if verbose:
                tag = "" if gates else " [diagnostic, non-gating: Gaia too sparse]"
                scales = ",".join(f"{s['cell_arcsec']:g}\":{s['n_measured']}"
                                  for s in g.get("scales", []))
                print(f"      same-star residual map vs {rn}: clean={g['clean']} "
                      f"measurable={g['measurable']} "
                      f"worst_off={g['worst_off_mas']:.0f} mas "
                      f"(cell tol {TOL_MAS:.0f}, global max {GRID_MAX_OFF_MAS:.0f}) "
                      f"n_ok={g['n_ok']}/{g['n_total']} cells [{scales}]{tag}"
                      f"{'; ' + g['reason'] if g.get('reason') else ''}", flush=True)
                rad = g.get("radii") or {}
                if rad.get("n_ambiguous"):
                    # non-gating field-wide: a stray group from another program in
                    # a shared target dir collapses this ratio all by itself
                    # (brick F405N: the o002 cloudc frames read 0.16 vs 0.85 for
                    # the brick's own visit).  Loud, because it is usually real.
                    print(f"        [diagnostic] {rad['n_ambiguous']}/{rad['n_cells']} "
                          f"cell(s) lose their same-star matches as the match radius "
                          f"shrinks (worst {rad['worst_ratio']:.2f} vs field "
                          f"{rad['field_ratio']:.2f}) -- a sub-population sitting at "
                          f"~the match radius there; inspect those frames.", flush=True)
            if gates and g["measurable"] and not g["clean"]:
                ext_fail = True
            # PER-PAIR scoping (issue #174 item 3): the field-wide map is ONE
            # boolean for the whole filter, so a single clean map cleared EVERY
            # deferred pair -- and on brick F405N every cell it could measure sat
            # in a stray o002 observation, not in the brick at all.  Re-measure
            # each deferred pair inside its OWN overlap footprint, where the
            # sliver is the whole population instead of a diluted minority.
            if gates and unverifiable and g.get("global_tie") is not None:
                for r in unverifiable:
                    v = _samestar_pair_footprint(pooled[r["a"]], pooled[r["b"]], rr,
                                                 g["global_tie"])
                    r["ext_pair"] = v
                    if verbose:
                        print(f"        footprint-scoped same-star arbiter "
                              f"{r['a']} | {r['b']}: clean={v['clean']} "
                              f"measurable={v['measurable']} "
                              f"worst={v['worst_off_mas']:.0f} mas -- {v['reason']}",
                              flush=True)

    # FAIL-CLOSED: a pair NEITHER frame-vs-frame layer could measure only passes
    # when the external reference POSITIVELY measured it.  Per-pair now: its own
    # footprint verdict decides, and only a pair whose footprint holds too few
    # reference stars falls back to the field-wide map (loudly).
    still_open, cleared = [], 0
    for r in unverifiable:
        v = r.get("ext_pair")
        if v is not None and v["measurable"] and not v["clean"]:
            r["fail_reason"] = (f"{r.get('fail_reason', '')}; " if r.get("fail_reason")
                                else "") + f"footprint-scoped reference arbiter: {v['reason']}"
            bad.append(r)
            continue
        if v is not None and v["clean"]:
            cleared += 1                  # cleared on its OWN footprint
            continue
        if ext_ran and field_clean:
            cleared += 1
            if verbose:
                why = v["reason"] if v is not None else "external reference not run"
                print(f"      WARNING: pair {r['a']} | {r['b']} could NOT be arbitrated "
                      f"in its own overlap footprint ({why}); cleared only by the "
                      f"FIELD-WIDE same-star map, which does not resolve this "
                      f"pair's sliver.", flush=True)
            continue
        still_open.append(r)
    could_not_verify = bool(still_open)
    if cleared and verbose:
        print(f"      {cleared}/{len(unverifiable)} deferred pair(s) accepted by the "
              f"external same-star reference", flush=True)
    return dict(field=field, filt=filt,
                PASS=bool(not bad and not ext_fail and not could_not_verify),
                could_not_verify=could_not_verify,
                n_fail=len(bad), pairs=res)


def field_filters(field):
    """Filters this field actually HAS products for.

    Enumerating by directory alone counts a `<field>/<FILT>/pipeline/` that was
    created and never populated.  ``check_filter`` then reports "NO crf frames
    matched -- cannot verify" and, fail-closed, the whole field exits 2.  An
    empty directory is not a band that failed verification; it is not a band.

    w51 (2026-08-03) carries four such leftovers -- F115W, F200W, F212N, F356W,
    every one of them containing zero files -- and they alone were enough to
    block a field whose eleven real NIRCam bands all pass.

    The fail-closed intent is kept for the case it was written for: a directory
    that holds ANY file but whose crf glob matches nothing is a REAL mismatch
    (wrong suffix, or a half-finished reduction that wrote e.g. only ``_asn.json``
    before dying) and still blocks.  The emptiness test is therefore ``os.listdir``
    (nothing at all), NOT ``*.fits`` -- a half-finished run has files and zero
    ``.fits``, and must reach ``check_filter`` rather than be dropped as "not a
    band".  Empirically the two coincide today (the four w51 leftovers are truly
    empty; no archive directory holds files but no ``.fits``), so this keeps code
    and docstring aligned rather than fixing an observed failure.

    An empty directory is skipped ONLY when the band is NOT declared for the
    field in ``fields.yaml``.  A DECLARED band with an empty directory is a
    reduction that produced nothing -- indistinguishable on disk from a
    just-``mkdir``'d run -- and must still reach ``check_filter`` so the field
    blocks; skipping it would route around the "declared but nothing there"
    check.  The four w51 leftovers (F115W/F200W/F212N/F356W) are undeclared, so
    this changes no verdict today; it keeps fail-closed for the declared case.

    "Declared" here means declared FOR THE INSTRUMENTS THIS TREE HOLDS -- NIRCam
    and MIRI.  NIRISS declares its bands separately and reduces to a different
    layout, so a NIRISS-only band can never have a directory here; counting one
    as "declared, never reduced" would block a correct field with no reduction
    able to clear it (sgrc declares F158M/F200W/F356W for NIRISS alone).  A gate
    a correct field cannot pass is a gate that teaches people to use the
    override.

    Names read off disk keep their DIRECTORY casing; the declared names appended
    below are upper-cased.  That mixed provenance is deliberate: the returned
    name is used as a PATH COMPONENT (``check_filter`` globs
    ``{BASE}/{field}/{filt}/pipeline/...``), so upper-casing a lower-case
    directory would make its glob match nothing and block a correct field.  The
    appended declared names have no directory to match by construction, which is
    exactly what makes them block.  Consumers comparing by name upper-case first.
    """
    declared = fields.declared_filters(field)   # NIRCam + MIRI, upper-cased
    out = []
    for p in sorted(glob.glob(f"{BASE}/{field}/*/pipeline/")):
        filt = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if not filt.upper().startswith("F"):
            continue
        try:
            empty = not os.listdir(p)
        except OSError as exc:
            # glob proved the dir existed a moment ago; if it is now unreadable
            # or gone we cannot determine emptiness -- report it, do not skip
            # (fail-closed): let check_filter decide rather than silently drop.
            print(f"  {field} {filt}: cannot read pipeline directory ({exc}) "
                  f"-- reporting, not skipping (fail-closed)", flush=True)
            out.append(filt)
            continue
        if empty:
            if filt.upper() in declared:
                print(f"  {field} {filt}: DECLARED band with an empty pipeline "
                      f"directory -- reduction produced nothing (blocks)",
                      flush=True)
                out.append(filt)
                continue
            print(f"  {field} {filt}: empty pipeline directory, nothing at all "
                  f"-- undeclared, not a band, skipping (not a verification "
                  f"failure)", flush=True)
            continue
        out.append(filt)
    # Declared but NO pipeline directory at all -- the same "declared but nothing
    # there" class as an empty declared directory, one step further out (the loop
    # above only sees directories that exist).  A band the registry declares and
    # the archive never reduced must be noticed at release time, so report it and
    # let check_filter block.  (cloudef F2100W/F770W, sgrc F158M/F200W/F356W on the
    # 2026-08 archive.)  Undeclared missing directories are simply not bands.
    for filt in sorted(declared - {f.upper() for f in out}):
        print(f"  {field} {filt}: DECLARED band with no pipeline directory at all "
              f"-- never reduced (blocks)", flush=True)
        out.append(filt)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--scan", action="store_true", help="every filter of the field")
    ap.add_argument("--instrument", default=None, choices=["nircam", "miri"],
                    help="restrict --scan to one instrument's bands. NIRCam and "
                         "MIRI are independent observations of the same sky: a "
                         "MIRI band that cannot be verified says nothing about "
                         "the NIRCam mosaics, so the caller gates them "
                         "separately and withholds only what failed.")
    ap.add_argument("--refcat", default=None,
                    help="optional external refcat for the fine-grid absolute cross-check")
    ap.add_argument("--observations", default=None,
                    help="csv of <proposal>-<observation> keys (e.g. "
                         "02221-001,01182-004) to scope the check to; stray crf "
                         "from other programs/observations in a shared target dir "
                         "are excluded.  Default: the field's released "
                         "observations (derived from the merged mosaics on disk).")
    args = ap.parse_args(argv)

    filts = field_filters(args.field) if args.scan else [args.filter]
    if args.instrument and args.scan:
        want_miri = args.instrument == "miri"
        filts = [f for f in filts
                 if (str(f).lower() in MIRI_FILTERS) == want_miri]
        if not filts:
            # rc 0, deliberately, and NOT the all-skipped case #452 made rc 2.
            # There, bands existed and every one was excluded by a scope that
            # could be wrong -- nothing was measured but something should have
            # been.  Here the caller asked about an instrument this field has no
            # bands for at all, which is a true and complete answer.  The caller
            # only asks per instrument PRESENT IN ITS MANIFEST, so an empty list
            # means the manifest and the filter directories disagree; that is
            # the listed-source gate's job, and it refuses before reaching here.
            print(f"  {args.field}: no {args.instrument} bands to check",
                  flush=True)
            return 0
    if not filts or filts == [None]:
        print("ERROR: give --filter or --scan", file=sys.stderr)
        return 2
    any_fail = False
    any_noverify = False
    # How many bands this scan actually MEASURED.  Skipping out-of-release bands
    # without counting them let a scan whose every band was skipped fall through
    # to `return 0` -- eight "NOT IN THIS RELEASE" lines and a pass, which
    # `stage_release` (refusing only on rc != 0) reads as the gate having run.
    # That is the same false-agreement this file's own could-not-verify message
    # argues against, arrived at from the other side: a wrong
    # `_release_observations` derivation used to REFUSE a good field, and would
    # then have PASSED one.
    checked = 0
    for f in filts:
        r = check_filter(args.field, f, refcat=args.refcat,
                         observations=(set(args.observations.split(","))
                                       if args.observations else None))
        if r.get("not_in_release"):
            continue          # not this release's band; neither passed nor failed
        checked += 1
        if r.get("could_not_verify"):
            any_noverify = True
        elif not r.get("PASS"):
            any_fail = True
    if any_fail:
        print(f"\nOVERLAP GATE: FAIL for {args.field} -- inter-frame misregistration "
              f"(> {TOL_MAS:.0f} mas). Do NOT stage; re-examine per-visit alignment.",
              flush=True)
        return 1
    if filts and not checked:
        print(f"\nOVERLAP GATE: COULD NOT VERIFY {args.field} -- every band was "
              f"skipped as belonging to other observations, so nothing was "
              f"measured. A scope that excludes the whole field is a wrong scope, "
              f"not a passing gate.", flush=True)
        return 2
    if any_noverify:
        # exit 2 = could-not-verify: distinct from a measured FAIL, but still
        # refused by stage_release (its rc != 0 branch) -- fail closed, never
        # green-because-the-glob-matched-nothing.
        print(f"\nOVERLAP GATE: COULD NOT VERIFY {args.field} -- at least one filter "
              f"had no matchable crf frames / no detections. Fix the products or the "
              f"glob; a gate that finds nothing is not a passing gate.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
