"""The detector-frame exposures a released mosaic was actually drizzled from.

A release ships mosaics, which are resampled onto a rectified sky grid.  Anyone
who wants to re-drizzle, re-fit, or check a per-exposure systematic needs the
frames in the ORIGINAL DETECTOR FRAME instead -- the last product before
``resample``, carrying the full GWCS distortion chain and this pipeline's
astrometric solution baked in.

WHICH FILE THAT IS DEPENDS ON THE FIELD, AND IS NOT GUESSED HERE
================================================================

The reduction writes several detector-frame products per exposure::

    *_cal.fits                 Stage-2 output, raw ``assign_wcs`` frame
    *_destreak.fits            + 1/f destreak, + this pipeline's WCS correction
    *_align.fits               same, on fields where destreaking is off
    *_<obs>_crf.fits           + Stage-3 outlier/CR flags   <- normally the last

so "the final detector-frame version" is field- and filter-dependent:
``destreak_policy`` alone decides destreak-vs-align, MIRI runs a different
Stage-3, and at least one shipped band (wd1 F150W) was drizzled straight from
``_cal.fits`` with no ``_crf`` ever written.

Rather than reconstruct that from policy -- which is how a release ends up
offering frames that are not the ones behind its own mosaic -- this module reads
it out of the mosaic itself.  Every drizzled product records the association it
was built from in its ``ASNTABLE`` header keyword, and that association lists its
members by name.  So the input list is PROVENANCE, not inference: the frames
offered for a mosaic are, by construction, the frames that mosaic came from.

Measured over every live science mosaic in v1.1/v1.2 (77 mosaics, 13 fields) the
``ASNTABLE`` resolved beside the mosaic in 77 of 77 cases, and the member forms
were exactly three:

    NIRCam   ``*_destreak.fits`` / ``*_align.fits`` / ``*_cal.fits``, asn_id
             ``oNNN``, with the Stage-3 twin ``<stem>_<asn_id>_crf.fits`` present
             for all but wd1 F150W
    MIRI     members that are ALREADY ``*_<obs>_crf.fits`` (absolute paths),
             asn_id ``a3001``, with no twin to look for

Hence the rule in ``final_detector_frame``: a member that is already a ``_crf``
IS the final frame; otherwise prefer its ``_crf`` twin; otherwise the member
itself.  Every branch is a file that exists on disk -- nothing is emitted for a
name that was merely computed.

NOT FROZEN, DELIBERATELY
========================

``releases/<version>/`` is a frozen tree of COPIES (see its
``DO_NOT_DELETE.README``): a published mosaic must stay byte-identical even
after the pipeline regenerates its source.  Exposures are staged as SYMLINKS
instead, and ``stage_release.stage`` keeps them symlinks even under ``--copy``.
Two reasons, and the second is why this is safe:

* one field-filter is ~20 GB of exposures against ~500 MB of mosaics, so copying
  them would multiply the frozen tree by ~40x, and
* a re-reduction rewrites their HEADERS (WCS, DQ) rather than producing a
  different observation, so following the symlink to the current frame is what a
  user re-drizzling the field actually wants.

The consequence is that an exposure link is NOT a frozen, checksummed
deliverable and must not be presented as one.  They carry no ``sha256``
(hashing TB of symlinked frames at staging time is not the same check as
hashing a frozen copy), and the page says what they are.

WITHHOLDING TRACKS THE MOSAIC
=============================

Each exposure item records ``parent_dest`` -- the staged mosaic it was drizzled
into.  The pipeline's astrometric correction is baked into these frames' WCS, so
when a mosaic is withheld because its source was superseded, the frames behind it
carry the same superseded solution and are withheld with it.  ``parent_dest`` is
what couples the two; without it the page would pull a mosaic down for bad
astrometry while still offering the frames that produced it.
"""
import json
import os
from pathlib import Path

#: Header keyword every drizzled JWST product carries, naming its association.
ASN_KEYWORD = "ASNTABLE"

#: Manifest category for a detector-frame exposure.  Distinct from ``image`` so
#: that every existing gate, freshness audit and page table -- all of which
#: select on ``category == "image"`` -- keeps seeing mosaics only.
EXPOSURE_CATEGORY = "exposure"
EXPOSURE_KIND = "detector_frame"


def _asn_name(mosaic):
    """``ASNTABLE`` from a mosaic's primary header, or ``None``."""
    from astropy.io import fits
    try:
        return fits.getheader(str(mosaic), 0).get(ASN_KEYWORD)
    except (OSError, ValueError):
        return None


def asn_for_mosaic(mosaic, search_root=None):
    """Path to the association a mosaic was drizzled from, or ``None``.

    Looked for beside the mosaic first, which is where all 77 live v1.1/v1.2
    science mosaics keep it.  ``search_root`` (a field's ``data_dir``) adds a
    bounded fallback for the one shipped layout where the mosaic was moved away
    from its pipeline directory afterwards -- brick's MIRI F2550W sits in
    ``brick/images/`` while its association stayed in a ``*/pipeline/`` dir.
    The fallback is two fixed globs, not a walk of a multi-TB tree.
    """
    mosaic = Path(mosaic)
    name = _asn_name(mosaic)
    if not name:
        return None
    beside = mosaic.parent / str(name)
    if beside.is_file():
        return beside
    if search_root is not None:
        for pattern in (f"*/pipeline/{name}", f"pipeline/{name}"):
            for hit in sorted(Path(search_root).glob(pattern)):
                if hit.is_file():
                    return hit
    return None


def final_detector_frame(expname, asn_dir, asn_id):
    """The last detector-frame product for one association member.

    ``expname`` is taken verbatim from the association (absolute for MIRI,
    relative to the association's own directory for NIRCam).  Returns an
    existing ``Path``, or ``None`` when neither the member nor its ``_crf`` twin
    is on disk.
    """
    member = Path(expname)
    if not member.is_absolute():
        member = Path(asn_dir) / member.name
    # MIRI's Stage-3 association already lists `_crf` frames: there is no twin
    # to look for and `<stem>_<asn_id>_crf.fits` would be a name that never
    # existed.
    if member.name.endswith("_crf.fits"):
        return member if member.is_file() else None
    if asn_id:
        twin = member.with_name(f"{member.name[:-len('.fits')]}_{asn_id}_crf.fits")
        if twin.is_file():
            return twin
    # No `_crf` was written for this band (wd1 F150W drizzles `_cal` directly),
    # so the association member IS the final detector-frame product.
    return member if member.is_file() else None


def exposures_for_mosaic(mosaic, search_root=None):
    """``(frames, problem)`` for one science mosaic.

    ``frames`` is the de-duplicated, sorted list of existing detector-frame
    products behind it; ``problem`` is a one-line description when the input
    list could not be established, in which case ``frames`` is empty.

    An unresolvable association is REPORTED, never guessed around.  Globbing the
    pipeline directory for plausible-looking frames would happily offer another
    observation's exposures for a multi-pointing field -- they share the
    directory -- and a wrong input list is worse than a missing one.
    """
    mosaic = Path(mosaic)
    asn_path = asn_for_mosaic(mosaic, search_root=search_root)
    if asn_path is None:
        name = _asn_name(mosaic)
        detail = f"ASNTABLE={name!r} not found on disk" if name else "no ASNTABLE header"
        return [], f"{mosaic.name}: {detail}"
    try:
        asn = json.loads(asn_path.read_text())
        products = asn["products"]
        asn_id = asn.get("asn_id")
    except (OSError, ValueError, KeyError, TypeError) as err:
        return [], f"{mosaic.name}: unreadable association {asn_path.name} ({err})"

    frames, absent = {}, 0
    for product in products:
        for member in product.get("members", []):
            expname = member.get("expname")
            if not expname or member.get("exptype", "science") != "science":
                continue
            frame = final_detector_frame(expname, asn_path.parent, asn_id)
            if frame is None:
                absent += 1
                continue
            frames[frame.name] = frame
    if not frames:
        return [], (f"{mosaic.name}: association {asn_path.name} lists "
                    f"{absent} member(s), none of them on disk")
    problem = None
    if absent:
        # A partial input list is a fact about the release, not a detail to
        # swallow: it means some frames behind a shipped mosaic have been
        # removed or renamed since it was drizzled.
        problem = (f"{mosaic.name}: {absent} of {absent + len(frames)} "
                   f"association member(s) are not on disk")
    return [frames[k] for k in sorted(frames)], problem


def discover_exposures(science_items, search_root=None, problems=None):
    """Exposure manifest items for every staged science mosaic.

    ``science_items`` are the already-discovered ``category == "image"`` /
    ``kind == "science"`` deliverables; each exposure inherits its mosaic's
    filter, observation and instrument, and records the mosaic in
    ``parent_src``.  Descriptions of mosaics whose input list could not be
    established are appended to ``problems`` when a list is passed.

    A frame shared by two shipped mosaics is emitted ONCE, against the first
    mosaic in order.  Fields whose modules are drizzled separately (arches,
    quintuplet) list disjoint members per module, so this is a guard rather
    than a routine case -- but emitting a duplicate would put two rows and two
    identical destinations in the manifest for one file.
    """
    items, claimed = [], set()
    for mosaic_item in science_items:
        frames, problem = exposures_for_mosaic(mosaic_item["src"],
                                               search_root=search_root)
        if problem and problems is not None:
            problems.append(problem)
        for frame in frames:
            key = str(frame)
            if key in claimed:
                continue
            claimed.add(key)
            items.append({
                "category": EXPOSURE_CATEGORY,
                "kind": EXPOSURE_KIND,
                "filter": mosaic_item.get("filter"),
                "iteration": None,
                "observation": mosaic_item.get("observation"),
                "instrument": mosaic_item.get("instrument"),
                "src": key,
                # provenance both ways: which mosaic these frames made, and
                # therefore which mosaic's withholding they follow.
                "parent_src": mosaic_item["src"],
            })
    return items


def link_parents(items):
    """Fill each exposure's ``parent_dest`` from its parent mosaic's ``dest``.

    Called after ``dest`` is assigned to every item, because that is the key the
    page withholds on.  An exposure whose parent is not in ``items`` keeps
    ``parent_dest`` unset and is therefore never withheld by association -- a
    state the caller should not be able to reach, since exposures are only
    discovered from mosaics that are being staged.
    """
    dest_by_src = {it["src"]: it.get("dest") for it in items
                   if it.get("category") == "image"}
    for it in items:
        if it.get("category") == EXPOSURE_CATEGORY and it.get("parent_src"):
            parent = dest_by_src.get(it["parent_src"])
            if parent:
                it["parent_dest"] = parent
    return items


def summarize(items):
    """``{(observation, filter): (count, total_bytes)}`` over exposure items."""
    out = {}
    for it in items:
        if it.get("category") != EXPOSURE_CATEGORY:
            continue
        key = (it.get("observation") or "", it.get("filter") or "")
        count, total = out.get(key, (0, 0))
        out[key] = (count + 1, total + (it.get("size_bytes") or 0))
    return out


def suffix_histogram(items):
    """``{suffix: n}`` over exposure items -- which product each frame is.

    Shown at staging time so a field that silently fell back off ``_crf`` (a
    band drizzled from ``_cal``) is visible in the log rather than only in the
    file names.
    """
    out = {}
    for it in items:
        if it.get("category") != EXPOSURE_CATEGORY:
            continue
        stem = os.path.basename(it["src"])
        suffix = "_".join(stem[:-len(".fits")].split("_")[4:]) or "?"
        out[suffix] = out.get(suffix, 0) + 1
    return out
