"""Has a staged product been SUPERSEDED since it was staged?

Staging copies a mosaic into a frozen release tree and records its source path
in ``MANIFEST.json``.  Nothing afterwards ever looked at that source again.  So
when the pipeline decides a mosaic's astrometry is wrong -- the m2 checkpoint
corrects the offsets table and renames every mosaic built before it to
``*_i2d_im0_badastrom.fits`` -- the release keeps serving the old bytes, and the
web page keeps presenting them as the field's astrometry.

That is the exact inversion of what a release is for.  Measured 2026-08-05:

    v1.1  cloudc    0 live / 6 quarantined      <- every image, and the field
                                                   has no current mosaic at all
    v1.1  gc2211    2 live / 2 quarantined      (o050, both bands)
    v1.1  sgrc      6 live / 2 quarantined      (F115W, F162M)
    v1.1  sgrb2    13 live / 1 quarantined      (F150W)
    v1.1  w51      11 live / 1 quarantined      (F210M)
    v1.1  sickle    7 live / 1 quarantined      (F210M)

Cloud C's published images predate the 2026-07-12 astrometry fix, so the page
was showing ~4" errors as evidence that the astrometry is sound.

The check is deliberately cheap and needs no FITS reading.  Two ways a staged
copy goes stale, and the second is the larger:

* the source was RENAMED to a quarantine twin -- resolve ``src``, and if it is
  gone with a twin in its place, the staged copy is superseded;
* the source was REBUILT IN PLACE under the same name, which is a perfectly
  good file.  Presence is not freshness.  The manifest records ``size_bytes``
  for every entry, so one ``stat`` says whether the bytes on disk are still
  the bytes that were staged.  52 of 115 live entries fail that comparison.

Nothing is deleted -- a superseded image stops being PUBLISHED, and the file
stays where it is.

SCOPE -- this detects CHANGE SINCE STAGING, never CORRECTNESS.  A clean audit
here is not an astrometry pass.  The worst live case in the tree at the time of
writing is one this module is structurally blind to: ``v1.2-2026.08/sgra``
reads ``live`` for both images and they are ~14.8" off (#324) -- their sources
were never renamed and never rebuilt, they were simply never corrected, because
proposal 1939 was missing from ``ALIGNMENT_CONFIG``.  Nothing here should try to
catch that; the registration gates and the m2 checkpoint are what look at
pixels.  This module only answers "are the bytes on the page still the bytes
that were signed off?".
"""
import glob
import json
import os

#: How the pipeline renames a product whose astrometry was superseded.
#: ``rename_stale_mosaics.py`` uses the second form.
QUARANTINE_GLOBS = (
    "{stem}_im0_badastrom*.fits",
    "{src}_badastrometry_stale",
    "{src}.STALE_*",
)

LIVE = "live"
#: The pipeline REPUDIATED the file: it renamed it to a quarantine twin.  This
#: is the only state that says anything about astrometry.
QUARANTINED = "quarantined"
#: The source was rebuilt in place under the same name -- a re-run, a re-chunk,
#: a new stage.  The staged bytes are simply older than the source's; nothing in
#: a ``stat`` says why, so nothing here may claim a reason.
REBUILT = "rebuilt"
MISSING = "missing"

#: Both mean "do not publish these bytes", and they are the majority/minority
#: the other way round from what the first version of this module assumed: 52
#: of the 116 non-live entries are REBUILT, not quarantined.  They were reported
#: under one name, and the page then asserted the astrometry checkpoint had
#: quarantined 23 of brick's v1.0 images when nothing of the sort had happened.
SUPERSEDED_STATES = (QUARANTINED, REBUILT)

# There is deliberately no `SUPERSEDED` constant.  An alias for it was added
# here for "back-compat" and pointed at QUARANTINED, which is worse than not
# having one: the name used to mean EITHER stale state, so every comparison
# against it silently stopped matching the rebuilt case -- 54 of the 114
# non-live entries, the majority this split exists to name.  It also shipped the
# branch red, because the two regression tests pinning the overwrite fix compare
# against it.  Removing the name fails loudly at import instead.  Use
# `is_superseded()` for "either", or the two states for "which".


def is_superseded(state):
    return state in SUPERSEDED_STATES


#: A rebuilt mosaic differs from the staged one by far more than this.  The
#: tolerance exists only so a byte-identical re-copy (rsync, restore from tape)
#: is not read as a rebuild.
SIZE_TOLERANCE_BYTES = 0


def source_state(src, recorded_size=None):
    """``live`` / ``superseded`` / ``missing`` for one staged file's source.

    PRESENCE IS NOT FRESHNESS.  The first version of this resolved supersession
    as "the source is gone and a quarantined twin is in its place", which
    catches only the RENAME.  A mosaic rebuilt IN PLACE under the same name is
    a perfectly good file, and the guard called it live.

    Measured across the whole release tree: 115 live entries, of which **52**
    had a source whose size no longer matched what was staged.  On cloudc --
    the field this check is named for -- five of six sources were rebuilt the
    same morning, each to a different size, and every one read `live`:

        f187n  recorded 3342309120  now 3340854720
        f212n  recorded 3342107520  now 3339990720
        f405n  recorded  552960000  now  552686400
        f410m  recorded  553383360  now  552888000
        f466n  recorded  553178880  now  552686400

    The page kept serving July's bytes with the notice saying nothing, which is
    the failure this module exists to prevent.  The manifest records
    ``size_bytes`` for 115 of 115 entries, so comparing it costs one ``stat``
    and no FITS read -- the same budget as before.  ``sha256`` is also recorded
    and deliberately NOT used: these are multi-GB mosaics and hashing them at
    page-build time is a different kind of check.
    """
    if not src:
        return MISSING
    if os.path.isfile(src):
        if recorded_size is None:
            return LIVE
        try:
            now = os.path.getsize(src)
        except OSError:
            return LIVE          # unreadable size is not evidence of staleness
        if abs(now - int(recorded_size)) > SIZE_TOLERANCE_BYTES:
            return REBUILT
        return LIVE
    stem = src[:-5] if src.endswith(".fits") else src
    for pattern in QUARANTINE_GLOBS:
        if glob.glob(pattern.format(stem=stem, src=src)):
            return QUARANTINED
    return MISSING


def audit_manifest(manifest, categories=("image",)):
    """``{dest: state}`` for every staged file of the given categories."""
    out = {}
    for entry in manifest.get("files", []):
        if categories and entry.get("category") not in categories:
            continue
        out[entry.get("dest")] = source_state(entry.get("src"),
                                              entry.get("size_bytes"))
    return out


def superseded_files(manifest, categories=("image",)):
    """Staged files whose source is no longer the one they were staged from."""
    return sorted(dest for dest, state in audit_manifest(manifest, categories).items()
                  if is_superseded(state))


def superseded_reasons(manifest, categories=("image",)):
    """``{dest: QUARANTINED | REBUILT}`` -- WHY each withheld file is withheld.

    The page has to say something about every image it withholds, and the two
    states support different sentences.  ``QUARANTINED`` means the pipeline
    renamed the file to a quarantine twin, which is a statement about its
    astrometry.  ``REBUILT`` means one ``stat`` disagrees with the recorded
    size; that can be a re-run, a re-chunk or a new stage, and asserting a
    quarantine there is a claim the check cannot support.
    """
    return {dest: state
            for dest, state in audit_manifest(manifest, categories).items()
            if is_superseded(state)}


def load_manifest(field_dir):
    path = os.path.join(str(field_dir), "MANIFEST.json")
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def field_report(field_dir):
    """``(n_live, superseded, missing)`` for a staged field directory."""
    manifest = load_manifest(field_dir)
    if manifest is None:
        return 0, [], []
    states = audit_manifest(manifest)
    return (sum(1 for s in states.values() if s == LIVE),
            sorted(d for d, s in states.items() if is_superseded(s)),
            sorted(d for d, s in states.items() if s == MISSING))


def describe(field, field_dir):
    """One human line, or ``None`` when everything the field ships is current."""
    manifest = load_manifest(field_dir)
    states = audit_manifest(manifest) if manifest else {}
    live, superseded, missing = field_report(field_dir)
    if not superseded and not missing:
        return None
    parts = [f"{field}: {live} current"]
    quarantined = [d for d in superseded if states.get(d) == QUARANTINED]
    rebuilt = [d for d in superseded if states.get(d) == REBUILT]
    if quarantined:
        parts.append(f"{len(quarantined)} QUARANTINED "
                     f"(source renamed as bad-astrometry since staging)")
    if rebuilt:
        parts.append(f"{len(rebuilt)} REBUILT "
                     f"(source rebuilt in place; these are the older bytes)")
    if missing:
        parts.append(f"{len(missing)} with a missing source")
    return ", ".join(parts)
