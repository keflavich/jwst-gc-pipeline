"""Pipeline-version provenance stamped into every FITS file we write.

So that any catalog or image can be traced to the exact code that produced it,
we record the running jwst-gc-pipeline git commit in the FITS keyword
``GCPIPEV`` of the primary header of every FITS file written in the process.

Mechanism: a one-time monkeypatch of ``astropy.io.fits.HDUList.writeto`` (the
single chokepoint every FITS write funnels through -- ``fits.writeto``,
single-HDU ``HDU.writeto``, ``Table.write(format='fits')`` via
``io.fits.connect``, and stdatamodels' ``DataModel.save`` all call it), which
sets ``GCPIPEV`` on the primary header just before the bytes hit disk.
Installed from ``jwst_gc_pipeline/__init__.py`` at import time.
"""
import functools
import os
import subprocess

GCPIPEV_KEY = 'GCPIPEV'
# Comment kept short: a full 40-char hash (+ '-dirty') plus the keyword leaves
# <=19 cols for the comment in an 80-col FITS card; a longer comment triggers a
# VerifyWarning (comment truncated) on every write.  The value is never affected.
_GCPIPEV_COMMENT = 'gc-pipeline commit'


@functools.lru_cache(maxsize=1)
def get_pipeline_commit():
    """Return the running jwst-gc-pipeline git commit id for ``GCPIPEV``.

    Resolution order (cached for the process):
      1. ``git rev-parse HEAD`` in the repo that contains this package (the
         editable-install / worktree case), with a ``-dirty`` suffix if the
         working tree has uncommitted tracked changes;
      2. the installed package version (setuptools_scm embeds the commit, e.g.
         ``0.1.dev34+gabcdef0``) when there is no ``.git`` (wheel install);
      3. ``'unknown'`` if neither is available.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(pkg_dir)
    # ``.git`` is a directory in a normal clone and a file in a worktree; both
    # mean ``git -C repo_dir`` will resolve the right HEAD.
    if os.path.exists(os.path.join(repo_dir, '.git')):
        try:
            commit = subprocess.check_output(
                ['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL, text=True).strip()
            if commit:
                dirty = subprocess.call(
                    ['git', '-C', repo_dir, 'diff', '--quiet'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
                return commit + ('-dirty' if dirty else '')
        except (subprocess.SubprocessError, OSError):
            pass

    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version('jwst_gc_pipeline')
        except PackageNotFoundError:
            pass
    except ImportError:
        pass

    return 'unknown'


def stamp_header(header):
    """Set ``GCPIPEV`` on a FITS ``header`` in place (idempotent overwrite)."""
    try:
        header[GCPIPEV_KEY] = (get_pipeline_commit(), _GCPIPEV_COMMENT)
    except (ValueError, TypeError):
        # A header that rejects the card (exotic / read-only) must never break
        # the write; provenance is best-effort, the data is not.
        pass


_HOOK_INSTALLED = False


def install_fits_provenance_hook():
    """Monkeypatch ``HDUList.writeto`` once so every FITS write stamps GCPIPEV.

    Idempotent: calling more than once (e.g. re-import) is a no-op.
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    from astropy.io.fits.hdu.hdulist import HDUList

    _orig_writeto = HDUList.writeto

    @functools.wraps(_orig_writeto)
    def _writeto_with_provenance(self, *args, **kwargs):
        if len(self) and getattr(self[0], 'header', None) is not None:
            stamp_header(self[0].header)
        return _orig_writeto(self, *args, **kwargs)

    HDUList.writeto = _writeto_with_provenance
    _HOOK_INSTALLED = True
