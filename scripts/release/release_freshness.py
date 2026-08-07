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

The check is deliberately cheap and needs no FITS reading.  Three signals, in
order of how much they establish:

* a quarantine TWIN newer than the release -- ``*_im0_badastrom.fits`` beside
  the source, created after ``built``.  The staged copy predates a repudiation.
  This is decisive whatever the size says, because a drizzled ``i2d``'s length
  is fixed by the output grid: a re-drizzle after a mas-level correction writes
  an IDENTICAL byte count, so the size comparison sees nothing.  39 staged
  mosaics were reading ``live`` with such a twin unread beside them.
* the source is GONE with a twin in its place -- the plain rename.
* the source was REBUILT IN PLACE under the same name, which is a perfectly
  good file.  Presence is not freshness: the manifest records ``size_bytes``,
  so one ``stat`` says whether the bytes on disk are still the staged ones.
  With no twin newer than staging, that is all it is -- a re-run, and the page
  says nothing about why.

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
import datetime
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
#: The pipeline REPUDIATED a product of this name: a quarantine twin exists,
#: either in place of a vanished source or created after this release was
#: staged.  This is the only state that says anything about astrometry.
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


def source_state(src, recorded_size=None, staged_at=None):
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
    # The quarantine twin is looked for FIRST, whether or not the source still
    # exists.  Testing `isfile` first was wrong for the commonest shape in the
    # tree: the m2 checkpoint quarantines a mosaic and the field is then
    # RE-DRIZZLED under the same name, so a repudiated product reads REBUILT
    # while its `*_im0_badastrom.fits` twin sits in the same directory.  49 of
    # the 54 rebuilt entries are that -- including all 23 of brick v1.0 and all
    # 6 of cloudc, i.e. exactly the fields whose staged copies predate the
    # 2026-07-12 astrometry fix.  Reporting those as "rebuilt, no claim made
    # about their astrometry" understates what is known about them.
    twin = newest_quarantine_twin(src)
    # A twin that was created AFTER this release was staged is the decisive
    # signal, and it is decisive whatever the size says.  Drizzled `i2d` output
    # length is fixed by the output grid, so a re-drizzle after a mas-level
    # offsets correction writes a file of IDENTICAL byte length -- the size
    # comparison sees nothing, and 39 staged mosaics whose sources were
    # repudiated after staging were reading `live` with the twin unread beside
    # them.  Verified by sha256 on a sample: recorded == staged, source !=
    # recorded, sizes equal.
    repudiated_since_staging = (twin is not None and staged_at is not None
                                and twin > staged_at)
    if os.path.isfile(src):
        if repudiated_since_staging:
            return QUARANTINED
        if recorded_size is None:
            return LIVE
        try:
            now = os.path.getsize(src)
        except OSError:
            return LIVE          # unreadable size is not evidence of staleness
        if abs(now - int(recorded_size)) <= SIZE_TOLERANCE_BYTES:
            # Same bytes on disk as were staged, and no repudiation since.  A
            # twin OLDER than staging does not condemn these: the field was
            # corrected and re-staged, which is what the quarantine was for.
            return LIVE
        # The staged copy differs from disk and there is a twin, but the twin
        # predates staging -- so it belongs to a repudiation this staged copy
        # already came after.  Calling that a quarantine would put "the m2
        # checkpoint repudiated this" on a page for what is a plain re-run, the
        # same false cause the two-state split was introduced to remove.
        return REBUILT
    return QUARANTINED if twin is not None else MISSING


def newest_quarantine_twin(src):
    """mtime of the newest ``*_im0_badastrom.fits``-style twin, or ``None``.

    The TIME matters, not just existence: a twin older than the release says a
    repudiation was corrected before staging, a twin newer says the staged copy
    predates one.
    """
    if not src:
        return None
    stem = src[:-5] if src.endswith(".fits") else src
    times = []
    for pattern in QUARANTINE_GLOBS:
        for path in glob.glob(pattern.format(stem=stem, src=src)):
            try:
                times.append(os.path.getmtime(path))
            except OSError:
                times.append(float("inf"))   # present but unreadable: assume recent
    return max(times) if times else None


def has_quarantine_twin(src):
    """Was a product of this name ever repudiated by the pipeline?

    Independent of whether the source exists now: a quarantine that was later
    corrected and re-drizzled leaves the twin behind, and that twin is the only
    on-disk evidence that the bytes staged BEFORE the correction are the bad
    ones.
    """
    if not src:
        return False
    stem = src[:-5] if src.endswith(".fits") else src
    return any(glob.glob(pattern.format(stem=stem, src=src))
               for pattern in QUARANTINE_GLOBS)


def _staged_at(manifest):
    """When this release was built, as epoch seconds, or ``None``."""
    built = (manifest or {}).get("built")
    if not built:
        return None
    try:
        return datetime.datetime.fromisoformat(
            str(built).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def audit_manifest(manifest, categories=("image",)):
    """``{dest: state}`` for every staged file of the given categories."""
    out = {}
    staged_at = _staged_at(manifest)
    for entry in manifest.get("files", []):
        if categories and entry.get("category") not in categories:
            continue
        out[entry.get("dest")] = source_state(entry.get("src"),
                                              entry.get("size_bytes"),
                                              staged_at=staged_at)
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
                     f"(a bad-astrometry twin exists for the source)")
    if rebuilt:
        parts.append(f"{len(rebuilt)} REBUILT "
                     f"(source rebuilt in place; these are the older bytes)")
    if missing:
        parts.append(f"{len(missing)} with a missing source")
    return ", ".join(parts)
