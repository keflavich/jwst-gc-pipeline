"""Refuse a per-frame catalog write that would replace another observation's.

A per-frame catalog is named
``{band}_{detector}[_obs]_visit{NNN}_vgroup{G}_exp{N}...``, and the ``_obs``
slot is filled only for the proposals in ``naming.MULTIOBS_PROPOSALS``, the
per-observation exceptions in ``naming.PER_OBS_PERFRAME_FIELDS`` and ngc6334's
shared tree (``_j{proposal}``).  Every other ``(proposal, basepath)`` holding
two observations spells one filename for both, so two observations that reuse a
``(visit, vgroup, exposure, detector)`` tuple write the same file and the
second write replaces the first.

It happened.  cloudef's obs-005 recatalog on 2026-08-19 wrote over 528 of
obs-002's per-frame catalogs; F480M kept none of its own and its offsets table
could not be rebuilt.  Nothing raised, and nothing in the writers refused
(issue #718).  The collision is live in three of sickle's MIRI filters (#719).

``cataloging``'s existing collision guard compares the frames of ONE run
against each other, so it cannot see a file a PREVIOUS run of a different
observation left on disk.  This one reads the file it is about to replace.
Every per-frame writer already stamps the exposure it measured into
``meta['filename']`` -- ``FILENAME`` in the ext-1 header -- and a JWST
per-exposure name ``jw<PPPPP><OOO><VVV>_<vgroup>_<exp>_<detector>_...`` carries
the observation, so the check needs no rename and no migration: the ~161k
catalogs already on disk answer it as they are.  (The rename is #316; #459 was
closed for changing merged-level naming, and nothing here touches a name.)

What it does NOT claim to catch:

* an existing catalog whose provenance is unreadable or absent -- a truncated
  file from a killed job, or a table written before the ``FILENAME`` stamp.
  There is nothing to compare, and refusing every such write would stop runs
  over files that are very likely the run's own.
* a write whose SOURCE is not a per-exposure product.  The merged-module
  catalogs go through the same writer with a mosaic (``jw02221-o001_t001_...``)
  as their source; that name carries an observation but the merged names are a
  separate concern with its own history (#459), deliberately left alone here.
* two frames of the SAME observation colliding.  That is the in-run guard's
  job (``cataloging``: "per-frame output name COLLISION"), and it sees the
  whole frame set at once, which this cannot.
"""
import os
import re


class ForeignObservationOverwriteError(RuntimeError):
    """A per-frame catalog write would replace a different observation's file."""


#: ``jw<proposal:5><observation:3><visit:3>_<vgroup>_<exposure>_<detector>_...``
#: -- the JWST per-exposure product name.  Anchored at the start of the
#: BASENAME and requiring the four underscore-separated fields, so a resampled
#: mosaic (``jw02221-o001_t001_nircam_...``, hyphen after the proposal) and a
#: hand-made table do not match.
_EXPOSURE_RE = re.compile(r'^jw(\d{5})(\d{3})(\d{3})_[^_]+_[^_]+_[^_]+_')


def exposure_observation(path):
    """``(proposal, observation)`` named by a JWST per-exposure file, or None.

    Read from the BASENAME: a basepath such as ``.../gc2211_o023/`` carries an
    observation of its own, and the exposure's is the one that matters here.
    Both fields come back zero-stripped on the proposal (``'02092'`` ->
    ``'2092'``, the spelling ``options.proposal_id`` uses) and three-digit on
    the observation (``'002'``, the spelling every token and glob uses).

    None means "this name does not identify an exposure", which every caller
    treats as "no comparison to make".
    """
    m = _EXPOSURE_RE.match(os.path.basename(str(path or '')))
    if m is None:
        return None
    return str(int(m.group(1))), m.group(2)


def recorded_source_exposure(catalog_path):
    """The source exposure a per-frame catalog on disk records, or None.

    HEADER-ONLY for FITS: ``FILENAME`` lives in the ext-1 header that
    ``Table.write`` builds from ``meta``, so the rows never need reading --
    this runs once per per-frame write, in front of the highest-frequency
    output in the pipeline.  (Same property, and the same measurement, as
    ``cataloging._catalog_source_frame``, which reads the stamp for the
    checkpoint's input filter.  Kept separate rather than imported:
    ``cataloging`` imports ``crowdsource_catalogs_long``, which imports this,
    so the import would be a cycle.)

    Anything unreadable is None -- a zero-length file a killed job left, a
    table with no stamp.  The caller allows the write in that case.
    """
    from astropy.io import fits
    if str(catalog_path).endswith(('.fits', '.fit', '.fits.gz')):
        try:
            return str(fits.getheader(catalog_path, ext=1).get('FILENAME')
                       or '') or None
        except (OSError, KeyError, IndexError, ValueError):
            return None
    from astropy.table import Table
    try:
        return str(Table.read(catalog_path).meta.get('FILENAME') or '') or None
    except (OSError, ValueError, KeyError):
        return None


def assert_no_foreign_observation_overwrite(out_path, source_exposure):
    """Refuse to write ``out_path`` over another observation's per-frame catalog.

    ``source_exposure`` is the exposure this catalog was measured on -- the
    same value the writer stamps into ``meta['filename']``.

    Returns silently when there is nothing to compare: no file at ``out_path``
    (the ordinary case), a source or an existing stamp that does not name an
    exposure, or an existing file recorded against the SAME observation, which
    is an ordinary re-run overwriting its own output.
    """
    if not out_path or not os.path.exists(out_path):
        return
    mine = exposure_observation(source_exposure)
    if mine is None:
        return
    existing_source = recorded_source_exposure(out_path)
    theirs = exposure_observation(existing_source)
    if theirs is None or theirs == mine:
        return
    raise ForeignObservationOverwriteError(
        f"per-frame catalog write would replace ANOTHER OBSERVATION's catalog "
        f"(issue #718):\n"
        f"    writing  {out_path}\n"
        f"    measured on {os.path.basename(str(source_exposure))} "
        f"-> proposal {mine[0]}, observation {mine[1]}\n"
        f"    existing file records {os.path.basename(str(existing_source))} "
        f"-> proposal {theirs[0]}, observation {theirs[1]}\n"
        f"The per-frame name carries no observation token here, so both "
        f"observations spell this one filename and the write would destroy the "
        f"other one's catalog -- how cloudef obs-005 destroyed 528 of obs-002's "
        f"on 2026-08-19.  Either move the existing catalogs aside if they are "
        f"expendable, or give this (proposal, observation) a per-frame "
        f"observation token via naming.PER_OBS_PERFRAME_FIELDS so the two names "
        f"differ.")
