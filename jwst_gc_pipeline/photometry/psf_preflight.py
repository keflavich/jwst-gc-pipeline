"""Fail fast, and truthfully, when the local stpsf data tree is incomplete.

The gc2211 o050 m12 finalize (38895067) died after 1 h 21 m -- with its
astrometry checkpoint already PASSED -- building a diagnostic residual mosaic::

    MergedcatMosaicError: [m12] nrcb/F277W: mergedcat residual / model i2d
    mosaic build failed: Failed to download PSF after 11 attempts; last error:
    ValueError: File wss_target_phase_fp6.fits, not found under
    /orange/adamginsburg/repos/webbpsf/data/.

Nothing was downloaded and nothing could have been.  ``wss_target_phase_fp6``
is local data that was absent from the tree ``STPSF_PATH`` points at and present
in its sibling ``stpsf-data`` tree (#346, fixed on disk).  Two separate defects
turned that into 81 wasted minutes and a misleading cause:

1. The retry loop in ``get_psf_model`` catches every exception the same way, so
   a file that is *not on this filesystem* was retried eleven times with backoff
   against a network that was never involved, and then reported as a download
   failure -- the same shape as #327, where a cached-file read presented as a
   CRDS outage.

2. It fires at the END of a run.  The OPD file a run needs is decided by
   ``load_wss_opd_by_date(obsdate)``, which is seconds of work; doing it up
   front fails in seconds instead of after the science.

This module supplies both halves: a classifier the retry loop can ask, and a
preflight the pipeline can run before it does any work.
"""
import os
import re

#: stpsf/WebbPSF raise a bare ``ValueError`` for an absent data file, so there
#: is no exception type to match on.  The message is stable and specific -- it
#: names the file AND the root it searched -- and matching it is what lets a
#: local-data problem be told from a network one.  Anchored on both halves so a
#: generic "not found" from elsewhere does not qualify.
_MISSING_LOCAL = re.compile(
    r"File\s+(?P<name>\S+?),?\s+not found under\s+(?P<root>\S+)", re.IGNORECASE)


def missing_local_data(exc):
    """``(filename, root)`` if ``exc`` is stpsf's absent-data-file error, else None."""
    m = _MISSING_LOCAL.search(str(exc))
    if not m:
        return None
    return m.group("name"), m.group("root").rstrip(".")


def missing_local_data_message(exc):
    """The message to raise instead of 'Failed to download PSF'.

    Names the sibling tree explicitly: on this account the file was present in
    ``stpsf-data`` and absent from ``data``, the two are byte-identical where
    they overlap, and every minute spent reading the wrong message was a minute
    not spent copying one file.
    """
    found = missing_local_data(exc)
    if found is None:
        return None
    name, root = found
    lines = [
        f"stpsf data file {name!r} is not present under {root} -- this is a "
        f"MISSING LOCAL FILE, not a failed download, and retrying cannot fix it.",
        f"  STPSF_PATH = {os.environ.get('STPSF_PATH', '<unset>')}",
        f"  WEBBPSF_PATH = {os.environ.get('WEBBPSF_PATH', '<unset>')}",
    ]
    for sibling in _sibling_trees(root):
        hit = _find_under(sibling, name)
        if hit:
            lines.append(f"  present in a sibling tree: {hit}")
            lines.append(f"  copy it into place under {root} (the trees are "
                         f"the same data; see #346)")
    return "\n".join(lines)


def _sibling_trees(root):
    """Candidate data trees next to ``root``.

    The account keeps ``webbpsf/data`` and ``webbpsf/stpsf-data`` side by side,
    and the file that was missing from one was sitting in the other.
    """
    parent = os.path.dirname(os.path.normpath(root))
    if not os.path.isdir(parent):
        return []
    return [os.path.join(parent, d) for d in sorted(os.listdir(parent))
            if os.path.isdir(os.path.join(parent, d))
            and os.path.normpath(os.path.join(parent, d)) != os.path.normpath(root)]


def _find_under(tree, name, max_depth=4):
    for dirpath, _dirs, files in os.walk(tree):
        depth = dirpath[len(tree):].count(os.sep)
        if depth > max_depth:
            continue
        if name in files:
            return os.path.join(dirpath, name)
    return None


class PSFDataMissingError(RuntimeError):
    """A local stpsf data file the run needs is not on this filesystem."""


def preflight_psf_data(instrument, filtername, obsdate, verbose=True):
    """Load the OPD this run will need, now, so a missing file costs seconds.

    Which OPD file is needed depends on ``obsdate`` -- it is chosen inside
    ``load_wss_opd_by_date`` -- so there is no static manifest to check.  Doing
    the lookup itself is the check, and it is the cheap part of building a PSF
    grid (no ``psf_grid`` call, no fov, no oversampling).

    Returns True when the data is there.  Raises `PSFDataMissingError` naming
    the file, the tree, and any sibling tree that has it.  Anything else --
    stpsf not importable, an instrument this does not know, a network hiccup in
    an unrelated part of the load -- returns False and is left to the real call
    site: a preflight that fails a run for a reason the run itself would have
    survived is worse than no preflight.
    """
    if not obsdate:
        return False
    try:
        import stpsf as _psf
    except ImportError:
        try:
            import webbpsf as _psf
        except ImportError:
            return False
    factory = {'NIRCAM': 'NIRCam', 'MIRI': 'MIRI', 'NIRISS': 'NIRISS'}.get(
        str(instrument).upper())
    if factory is None or not hasattr(_psf, factory):
        return False
    inst = getattr(_psf, factory)()
    try:
        inst.load_wss_opd_by_date(f'{obsdate}T00:00:00')
    except ValueError as ex:
        msg = missing_local_data_message(ex)
        if msg is None:
            return False
        raise PSFDataMissingError(
            f"PSF preflight for {instrument}/{filtername} at {obsdate}:\n{msg}"
        ) from ex
    except (OSError, KeyError, AttributeError, TypeError):
        # Not a missing-data failure this can speak to.  Say nothing and let the
        # real PSF build report whatever it finds.
        return False
    if verbose:
        print(f"PSF preflight OK: {instrument} OPD for {obsdate} is present "
              f"under {os.environ.get('STPSF_PATH', '<unset>')}", flush=True)
    return True
