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
SUPERSEDED = "superseded"
MISSING = "missing"


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
            return SUPERSEDED
        return LIVE
    stem = src[:-5] if src.endswith(".fits") else src
    for pattern in QUARANTINE_GLOBS:
        if glob.glob(pattern.format(stem=stem, src=src)):
            return SUPERSEDED
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
    """Staged files whose source has since been quarantined."""
    return sorted(dest for dest, state in audit_manifest(manifest, categories).items()
                  if state == SUPERSEDED)


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
            sorted(d for d, s in states.items() if s == SUPERSEDED),
            sorted(d for d, s in states.items() if s == MISSING))


def describe(field, field_dir):
    """One human line, or ``None`` when everything the field ships is current."""
    live, superseded, missing = field_report(field_dir)
    if not superseded and not missing:
        return None
    parts = [f"{field}: {live} current"]
    if superseded:
        parts.append(f"{len(superseded)} SUPERSEDED "
                     f"(source quarantined as bad-astrometry since staging)")
    if missing:
        parts.append(f"{len(missing)} with a missing source")
    return ", ".join(parts)
