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
import re
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
        # Three distinguishable states, and saying the wrong one sends someone
        # looking in the wrong place: a mosaic that is not there at all (a
        # dangling staged symlink reads exactly like this), one that opens but
        # names no association, and one that names an association that is gone.
        if not mosaic.is_file():
            return [], f"{mosaic.name}: mosaic not readable at {mosaic}"
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
        # ``asn_source`` lets a caller read the association out of a DIFFERENT
        # copy of the same mosaic -- ``add_to_release`` uses the frozen copy in
        # the release tree, which is present and byte-exact even when the
        # original pipeline product has since been quarantined or re-drizzled.
        # ``parent_src`` stays the manifest's ``src`` either way, because that is
        # what the freshness audit and the page key on.
        frames, problem = exposures_for_mosaic(
            mosaic_item.get("asn_source") or mosaic_item["src"],
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


#: ``jw<proposal:5><observation:3><visit:3>_<exposure>_<n>_<detector>_...``
EXPOSURE_NAME_RE = re.compile(r"^jw(?P<prop>\d{5})(?P<obs>\d{3})(?P<visit>\d{3})_")

#: The detector-frame products, in the order the LAST one wins.  Matches the
#: fallback order in ``final_detector_frame``: Stage-3 flags where they were
#: written, otherwise the frame the mosaic would have been drizzled from.
DETECTOR_FRAME_SUFFIXES = ("cal", "align", "destreak", "align_crf", "destreak_crf")


def field_observation_keys(field_cfg):
    """``{(proposal, observation)}`` this field's release covers.

    Read from ``proposal_prefix`` (``jw02045-o001_t001_...``, possibly a list)
    plus ``observations`` (``jw02211`` + ``["o023", ...]``), which are the two
    shapes the registry uses.  These are what scope a pipeline-directory scan to
    THIS release: a multi-pointing field keeps every observation's frames in one
    directory, so an unscoped glob would hand o023's release o046's exposures.
    """
    prefixes = field_cfg.get("proposal_prefix") or []
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    observations = field_cfg.get("observations")
    keys = set()
    for prefix in prefixes:
        match = re.match(r"^jw(?P<prop>\d{5})(?:-o?(?P<obs>\d{3}))?", str(prefix))
        if match is None:
            continue
        prop = match.group("prop")
        if observations:
            for obs in observations:
                keys.add((prop, str(obs).lstrip("o").zfill(3)))
        elif match.group("obs"):
            keys.add((prop, match.group("obs")))
    return keys


def enumerate_field_exposures(field_cfg, target, filters=None, requested=True):
    """Detector frames for a field found WITHOUT reading any mosaic.

    ``{(observation, FILTER): [Path, ...]}``, with ``observation`` ``None`` for a
    single-pointing field and ``oNNN`` where the registry lists observations --
    the same key the mosaic path produces, so the frames land in
    ``exposures/<obs>/<FILTER>/`` beside ``images/<obs>/<FILTER>/`` either way.
    Splitting by observation is not cosmetic: gc2211 keeps 352 F200W frames from
    o023 AND o049 in one pipeline directory, and pooling them would put one
    pointing's exposures under the other's heading.

    The normal path derives frames from a mosaic's
    ``ASNTABLE``, which is exact but requires the mosaic to exist.  Detector
    frames are a DEPENDENCY of the mosaic -- they are produced first -- so they
    can and should be releasable before it: a field mid-reduction, or one whose
    band is in an m2 correct-and-requarantine cycle with no live mosaic at all,
    still has its ``_cal``/``_destreak``/``_crf`` frames sitting on disk and
    there is nothing about them to wait for.

    Two things keep this from being the loose glob that the ASN path exists to
    avoid:

    * the frames are SCOPED to this release's (proposal, observation) pairs,
      parsed out of the exposure filename itself.  A multi-pointing field keeps
      every observation's frames in one pipeline directory, so this is what stops
      o023's release being handed o046's exposures.
    * within a filter, ONE product is chosen for the whole set -- the most
      processed one present, per ``destreak_policy`` -- rather than mixing a
      ``_crf`` for one exposure with a ``_cal`` for another.

    It remains less exact than the association: it answers "which frames belong
    to this field/observation/filter", not "which frames went into that mosaic".
    Prefer ``exposures_for_mosaic`` whenever a mosaic exists.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jwst_gc_pipeline.reduction import destreak_policy

    data_dir = Path(field_cfg["data_dir"])
    keys = field_observation_keys(field_cfg)
    out = {}
    if not data_dir.is_dir():
        return out
    for filter_dir in sorted(data_dir.iterdir()):
        if not re.fullmatch(r"F\d{3,4}[WMN]2?", filter_dir.name):
            continue
        if filters and filter_dir.name.upper() not in {f.upper() for f in filters}:
            continue
        pipeline = filter_dir / "pipeline"
        if not pipeline.is_dir():
            continue
        # `destreak_policy` is the single authority on destreak-vs-align, and it
        # is per (field, filter) -- sickle destreaks its short filters and not
        # its long ones -- so it is asked per filter rather than per field.
        token = "destreak" if destreak_policy.destreaks(
            target, filter_dir.name, requested) else "align"
        multi = bool(field_cfg.get("observations"))
        by_product = {}
        for path in pipeline.glob("jw*.fits"):
            match = EXPOSURE_NAME_RE.match(path.name)
            if match is None:
                continue
            if keys and (match.group("prop"), match.group("obs")) not in keys:
                continue
            stem = path.name[:-len(".fits")]
            tail = "_".join(stem.split("_")[4:])
            if tail == "cal":
                product = "cal"
            elif tail == token:
                product = token
            elif re.fullmatch(rf"{token}_o\d+_crf", tail) or re.fullmatch(r"o\d+_crf", tail):
                product = f"{token}_crf"
            else:
                continue
            obs = f"o{match.group('obs')}" if multi else None
            by_product.setdefault(product, {}).setdefault(obs, []).append(path)
        # One product for the whole filter, most-processed first, rather than a
        # `_crf` for one exposure and a `_cal` for another.
        for product in reversed(DETECTOR_FRAME_SUFFIXES):
            if product in by_product:
                for obs, paths in by_product[product].items():
                    out[(obs, filter_dir.name.upper())] = sorted(paths)
                break
    return out


def add_to_release(field_dir, assign_dest, collection_root, https_base,
                   search_root=None, problems=None):
    """Add (or refresh) ONLY the detector-frame exposures of a staged release.

    Returns ``(exposure_items, manifest)`` without writing anything; the caller
    stages the links and rewrites ``MANIFEST.json``.

    WHY THIS EXISTS SEPARATELY FROM A NORMAL RE-STAGE
    =================================================

    A full ``--stage`` re-derives the whole deliverable set and re-runs every
    mosaic gate, so a field that cannot currently ship a mosaic cannot receive
    its exposures either.  arches is exactly that: its F212N mosaic is in an m2
    correct-and-requarantine cycle and has no live product, so the listed-source
    gate refuses the field outright -- while its ALREADY-PUBLISHED v1.2 release
    sits on disk, gated and frozen weeks ago.

    Adding frames to that release asserts nothing new about registration.  The
    mosaics are untouched, the gates that admitted them already ran, and each
    frame is read from the association of the mosaic it belongs to.  So this
    path deliberately does NOT re-run those gates -- there is no mosaic decision
    for them to make -- and it cannot change which mosaics ship: it only ever
    adds ``category == "exposure"`` entries.

    THE ASSOCIATION IS READ FROM THE STAGED COPY
    ============================================

    Not from ``src``.  A published release is a tree of COPIES, so the staged
    mosaic is present and byte-exact even when the pipeline has since renamed or
    re-drizzled the original -- which is the whole situation this function is
    for.  Falling back to ``src`` covers a symlink-mode release, where the
    staged path is the original.

    A consequence worth stating: the association is NEVER beside the staged copy,
    because it stayed in the pipeline directory the mosaic was drizzled in.  So
    ``search_root`` is REQUIRED on this path, where it is merely a fallback on
    the normal one; without it every mosaic reports its ``ASNTABLE`` as not found
    and the field silently gains no frames.

    ``built`` IS PRESERVED BY THE CALLER
    ====================================

    Not a detail: ``release_freshness`` compares a quarantine twin's mtime
    against ``built`` to decide whether a staged copy predates a repudiation.
    Stamping a fresh ``built`` here would make every existing twin older than
    "staging" and silently flip the field's QUARANTINED images back to LIVE --
    re-publishing, as a side effect of adding a symlink, exactly the mosaics the
    astrometry checkpoint pulled.  arches would have had both superseded F212N
    mosaics returned to its page.
    """
    field_dir = Path(field_dir)
    manifest = json.loads((field_dir / "MANIFEST.json").read_text())
    science = []
    for entry in manifest.get("files", []):
        if entry.get("category") != "image" or entry.get("kind") != "science":
            continue
        staged = field_dir / entry["dest"]
        item = dict(entry)
        # `is_file()` follows symlinks, so a dangling staged link falls through
        # to `src` rather than being read as present.
        item["asn_source"] = str(staged) if staged.is_file() else entry["src"]
        science.append(item)

    exposures = discover_exposures(science, search_root=search_root,
                                   problems=problems)
    for item in exposures:
        item["dest"] = str(assign_dest(item))
        src = Path(item["src"])
        item["size_bytes"] = src.stat().st_size if src.is_file() else None
        item.setdefault("version", manifest.get("version"))
        rel = (field_dir / item["dest"]).relative_to(collection_root)
        item["globus_path"] = "/" + str(rel)
        item["url"] = https_base + item["globus_path"]
    # parent_dest needs the mosaics' dests, which are already in the manifest
    link_parents(list(manifest.get("files", [])) + exposures)
    return exposures, manifest


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
