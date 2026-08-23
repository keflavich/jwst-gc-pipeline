"""Where the gc2211 satstar-deblend inputs live after the #469 split.

#469 split ``/orange/adamginsburg/jwst/gc2211`` into one tree per observation
-- ``gc2211_o023``, ``_o028``, ``_o046``, ``_o049``, ``_o050`` -- and moved
69,815 frame products out of the shared tree.  The shared tree still EXISTS,
so a path built from it raises nothing; the frame globs simply return zero
matches, which is the "globs successfully, reads nothing" shape the split was
meant to remove (issue #470)::

    /orange/adamginsburg/jwst/gc2211/F200W/pipeline/*_cal.fits        0 files
    /orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline/*_cal.fits  32 files

Two roots, because the split moved two kinds of product apart:

* **frames** -- ``_cal``, ``_ramp``, ``_crf`` and everything else under
  ``<FILTER>/`` and ``<FILTER>/pipeline/`` -- are per observation, so they are
  addressed through :func:`obs_root`, which takes the observation from the
  exposure name itself.  These scripts read frames from four different
  observations (023, 028, 046, 049), so one module-level constant cannot name
  the tree for all of them.
* the **pooled pre-split catalogues** these scripts match against did not
  move: ``f<filt>_merged_indivexp_merged_dao_basic.fits`` and
  ``GALACTICNUCLEUS_2021_gc2211.fits`` are single tables covering every
  observation, and there is no per-observation equivalent to substitute --
  ``gc2211_o023/catalogs/`` holds only phase-labelled, ``_o023``-tokened
  products.  :data:`CATALOGS` therefore stays on the shared tree, which is
  where those files are.

The PSF grids are the same case as the catalogues for a different reason: a
grid is keyed by detector and filter, not by observation, and the shared
``psfs/`` cache is the complete one (38 grids, including the ``fovp512``/
``fovp1024`` ones the fitting scripts ask for, against 10 in
``gc2211_o023/psfs/``).  :data:`PSFS` points there.

These are analysis scripts rather than pipeline code, so they address the data
directly rather than through ``jwst_gc_pipeline.fields``.  That is also why
they went stale in the first place -- the registry guard
(``test_every_region_basepath_matches_the_registry``) does not see them --
and why ``tests/test_satstar_deblend_paths.py`` pins these roots instead.
"""
import os
import re

#: Root of the JWST data collection.  Overridable so the module can be
#: exercised against a fixture tree.
JWST_ROOT = os.environ.get('JWST_DATA_ROOT', '/orange/adamginsburg/jwst')

#: The shared pre-split tree.  Frames are gone from it; catalogues and the PSF
#: cache are not.
SHARED_ROOT = f'{JWST_ROOT}/gc2211'

#: Pooled catalogues covering every observation, written before the split.
CATALOGS = f'{SHARED_ROOT}/catalogs'

#: PSF grid cache, keyed by detector and filter rather than by observation.
PSFS = f'{SHARED_ROOT}/psfs/'

#: Observations of program 2211 that own a tree of their own after the split.
OBSERVATIONS = ('023', '028', '046', '049', '050')

#: ``jw02211023001_02201_00001_nrca1_cal.fits`` -> ``023``.  Also matches the
#: globs these scripts build, ``jw02211028*_nrca1_cal.fits``.
_OBS_RE = re.compile(r'jw02211(\d{3})')


def observation_of(name):
    """The three-digit observation an exposure name (or glob) belongs to.

    Accepts the observation itself (``'023'``, ``'o023'``) unchanged, so a
    caller that knows the observation need not build a filename to say so.
    """
    text = str(name)
    if re.fullmatch(r'o?\d{3}', text):
        return text.lstrip('o')
    match = _OBS_RE.search(os.path.basename(text))
    if match is None:
        raise ValueError(
            f"{text!r} carries no 'jw02211<observation>' token; cannot tell "
            f"which post-split gc2211 tree it belongs to")
    return match.group(1)


def obs_root(name):
    """The per-observation tree holding an exposure's frames.

    ``obs_root('jw02211023001_02201_00001_nrca1_cal.fits')`` ->
    ``/orange/adamginsburg/jwst/gc2211_o023``.
    """
    return f'{JWST_ROOT}/gc2211_o{observation_of(name)}'


def frame(name, filtername, suffix='cal'):
    """Path to a ``<FILTER>/`` frame product of one exposure."""
    return f'{obs_root(name)}/{filtername}/{name}_{suffix}.fits'


def pipeline(name, filtername):
    """The ``<FILTER>/pipeline/`` directory an exposure's products sit in."""
    return f'{obs_root(name)}/{filtername}/pipeline'


def frame_glob(pattern, filtername):
    """A ``<FILTER>/`` glob pattern, rooted in the observation it names."""
    return f'{obs_root(pattern)}/{filtername}/{pattern}'


def all_obs_frame_globs(pattern, filtername):
    """The same glob against EVERY post-split tree, for a pattern with no
    observation in it (``'jw02211*nrca1_cal.fits'``)."""
    return [f'{JWST_ROOT}/gc2211_o{obs}/{filtername}/{pattern}'
            for obs in OBSERVATIONS]


def catalog(name):
    """Path to one of the pooled pre-split catalogues."""
    return f'{CATALOGS}/{name}'
