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

The check is deliberately cheap and needs no FITS reading: resolve each staged
image's ``src``; if it is gone and a quarantined twin is in its place, the
staged copy is superseded.  Nothing is deleted -- a superseded image stops being
PUBLISHED, and the file stays where it is.
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


def source_state(src):
    """``live`` / ``superseded`` / ``missing`` for one staged file's source."""
    if not src:
        return MISSING
    if os.path.isfile(src):
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
        out[entry.get("dest")] = source_state(entry.get("src"))
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
