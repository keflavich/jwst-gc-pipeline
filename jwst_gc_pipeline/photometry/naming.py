"""Filename-token helpers shared across the photometry pipeline.

These build the small, unambiguous tokens embedded in per-frame / merged output
filenames (iteration label, background-subtraction mode, spatial seed chunk) so
that products from different iterations and modes never collide and can be
glob-matched exactly.

Factored out of ``crowdsource_catalogs_long.py`` (2026-06-09 restructure); the
old module now imports these names from here so there is a single source of
truth.  Pure string/regex helpers with no astronomy dependencies.
"""
import os
import re

# Match _chunkXXofYY (any width) when stripping the chunk suffix from an
# iteration_label or filename component.
_CHUNK_TOKEN_RE = re.compile(r'_chunk\d+of\d+')

# Single source of truth for which JWST filters are MIRI vs NIRCam.  Lives here
# (a heavy-import-free module) so merge_catalogs.py can import it without pulling
# in crowdsource_catalogs_long.py's webbpsf chain.
MIRI_FILTERS = frozenset(['f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w',
                          'f1500w', 'f1800w', 'f2100w', 'f2550w'])

# Canonical instrument names as used downstream (PSF branch, SVO filterID,
# filename token = lowercased).
_CANONICAL_INSTRUMENT = {'nircam': 'NIRCam', 'niriss': 'NIRISS', 'miri': 'MIRI'}


def _instrument_override():
    """Process-global instrument override.

    NIRISS shares filter names with NIRCam (F158M/F200W/F356W/F480M), so the
    instrument CANNOT be derived from the filter name.  A single reduction/
    cataloging process only ever handles ONE instrument, so -- exactly like
    ``GC_BASEPATH_OVERRIDE`` (jwst_gc_pipeline.scratch_basepath) -- a process-wide
    env override is safe and avoids threading an ``instrument`` argument through
    every ``_inst_token`` / ``_svo_filter_id`` / ``_instrument_from_filter`` call
    site.  Set ``GC_INSTRUMENT_OVERRIDE=niriss`` (via ``--instrument niriss``) to
    force the NIRISS branch.  Empty/unset -> filter-name heuristic (historical
    NIRCam/MIRI behavior)."""
    val = os.environ.get('GC_INSTRUMENT_OVERRIDE', '').strip().lower()
    return _CANONICAL_INSTRUMENT.get(val) if val else None


def _instrument_from_filter(filtername, instrument=None):
    """Return 'MIRI', 'NIRCam', or 'NIRISS'.

    Precedence: explicit ``instrument`` arg > ``GC_INSTRUMENT_OVERRIDE`` env >
    filter-name heuristic (MIRI filter set -> MIRI, else NIRCam).  NIRISS can only
    arrive via the arg or the env (its filter names are shared with NIRCam)."""
    if instrument is not None:
        return _CANONICAL_INSTRUMENT.get(str(instrument).lower(), instrument)
    override = _instrument_override()
    if override is not None:
        return override
    return 'MIRI' if str(filtername).lower() in MIRI_FILTERS else 'NIRCam'


def _inst_token(filtername, instrument=None):
    """Lowercased instrument token used in JWST i2d filename conventions."""
    return _instrument_from_filter(filtername, instrument=instrument).lower()


def _svo_filter_id(filtername, instrument=None):
    """SVO FPS filterID (e.g. 'JWST/NIRCam.F480M', 'JWST/NIRISS.F200W')."""
    return f'JWST/{_instrument_from_filter(filtername, instrument=instrument)}.{filtername.upper()}'


def _chunk_token(chunk_index, n_seed_chunks):
    """Filename token for spatial seed chunking.

    Returns '' when n_seed_chunks <= 1; otherwise '_chunk{i:02d}of{n:02d}'.
    Two-digit fields keep the token sortable and unambiguous.
    """
    n = int(n_seed_chunks) if n_seed_chunks else 1
    if n <= 1:
        return ''
    i = int(chunk_index)
    if i < 0 or i >= n:
        raise ValueError(
            f'seed_chunk_index={i} out of range for n_seed_chunks={n}')
    return f'_chunk{i:02d}of{n:02d}'


def _strip_chunk(label):
    """Remove a trailing/embedded ``_chunkXXofYY`` token from a string.

    Used to recover the *base* iteration label (e.g. ``'iter3'``) from
    a chunk-suffixed compound label (e.g. ``'iter3_chunk03of08'``) so
    semantic checks like ``is_iter3`` continue to fire when chunking is on.
    """
    if label is None:
        return None
    return _CHUNK_TOKEN_RE.sub('', str(label))


def _iteration_token(iteration_label):
    if iteration_label in (None, ''):
        return ''

    token = str(iteration_label)
    if token.startswith('_'):
        return token
    return f'_{token}'


def _bgsub_token_from_flags(bgsub, resbgsub=False):
    """Filename token for the background-subtraction mode(s) in effect.

    * ``--bgsub`` (global Background2D subtraction)        -> ``_bgsub``
    * ``--use-iter3-residual-bg`` (iter3 residual-smoothed
      background subtraction)                             -> ``_resbgsub``

    Both can be set; the tokens concatenate in a fixed order so output
    catalog/residual/model/diagnostic filenames are unambiguous and the
    skip-if-done prediction (_predict_output_tokens) stays in sync with the
    names actually written by do_photometry_step.  ``_bgsub`` is never a
    substring of ``_resbgsub`` so exact-token matching does not collide.

    This is the canonical flags-based form; ``merge_catalogs`` imports it as
    ``_bgsub_token`` (it works with explicit booleans) and the producer side
    uses the ``_bgsub_token(options)`` wrapper below.
    """
    token = '_bgsub' if bgsub else ''
    if resbgsub:
        token += '_resbgsub'
    return token


def _bgsub_token(options):
    """``_bgsub_token_from_flags`` reading the flags off an options object."""
    return _bgsub_token_from_flags(
        getattr(options, 'bgsub', False),
        getattr(options, 'use_iter3_residual_bg', False))


# --- residual-i2d product-name family --------------------------------------
# The cataloging products follow a fixed residual-i2d naming convention.  These
# centralize the (otherwise scattered, inline ``.replace(...)``) transforms so
# the convention has a single source of truth.

def residual_to_smoothed_bg_i2d(residual_i2d_path):
    """``..._residual_i2d.fits`` -> ``..._residual_smoothed_bg_i2d.fits``."""
    return residual_i2d_path.replace('_residual_i2d.fits',
                                     '_residual_smoothed_bg_i2d.fits')


def smoothed_bg_to_detection_i2d(smoothed_bg_i2d_path):
    """``..._residual_smoothed_bg_i2d.fits`` -> ``..._residual_i2d.fits`` (the
    detection image sits next to the smoothed-bg, differing only by the infix)."""
    return smoothed_bg_i2d_path.replace('_smoothed_bg_i2d.fits', '_i2d.fits')


def residual_to_model_i2d(residual_i2d_path):
    """``..._residual_i2d.fits`` -> ``..._model_i2d.fits``."""
    return residual_i2d_path.replace('_residual_i2d.fits', '_model_i2d.fits')


def residual_to_infilled_i2d(residual_i2d_path):
    """``..._residual_i2d.fits`` -> ``..._residual_infilled_i2d.fits``."""
    return residual_i2d_path.replace('_residual_i2d.fits',
                                     '_residual_infilled_i2d.fits')


def vetted_to_i2dseed(vetted_path):
    """``..._vetted.fits`` -> ``..._i2dseed.fits``."""
    return vetted_path.replace('_vetted.fits', '_i2dseed.fits')


# --- observation scoping ---------------------------------------------------
# Which proposals put more than one observation under ONE basepath with
# RESTARTED (visit, vgroup, exposure) numbering, so per-frame catalog names
# need the ``_o{field}`` disambiguator ``crowdsource_catalogs_long.obs_token``
# inserts.  2211 = gc2211 (5 GC pointings); 10678 = the GC Treasury program
# (139 tiles sharing the gc-treasury tree; issue #416).  Lives here (a
# heavy-import-free module) so merge_catalogs.py can consult it without a
# circular import of crowdsource_catalogs_long.
MULTIOBS_PROPOSALS = ('2211', '10678')

#: The subset whose MERGED catalogs are per-observation too.  gc2211 is multi-
#: obs at the per-frame level but pools all five pointings into one untokened
#: merged catalog by design; at 139 tiles that pooling is itself the corruption
#: mode, so 10678 scopes the merged catalogs to one observation as well.
PER_OBS_MERGED_PROPOSALS = ('10678',)


def merged_catalog_obs_token(proposal_id, field):
    """Observation token baked into the MERGED catalog names, post-module slot.

    Only ``PER_OBS_MERGED_PROPOSALS`` get one: 10678's 139 tiles share the
    gc-treasury tree, and pooling another tile's frames into a merge is the
    corruption class the obs scoping exists to prevent, so every per-filter
    merged catalog is scoped to one observation
    (``{filt}_{module}_o{field}_indivexp_merged...``).  gc2211 keeps its
    all-obs UNTOKENED merged names ('' here): its five pointings are pooled at
    merge by design (the ``_o*`` glob in ``merge_individual_frames``) and
    scoped afterwards at the vetting step (``_vtok`` in cataloging.py).
    Writers (``merge_individual_frames``' ``out_obs_``) and every reader of a
    merged-catalog name (``cataloging._merged_path``, the m7 seed reader,
    ``merge_daophot``'s input glob) must agree on this token.
    """
    if str(proposal_id) in PER_OBS_MERGED_PROPOSALS and field not in (None, ''):
        return f'_o{field}'
    return ''


# --- reading a per-frame catalog name back --------------------------------
# Every reader that parses `{band}_{detector}_visit{NNN}_vgroup...` out of a
# per-frame catalog name has to allow for the per-observation token `obs_token`
# inserts BETWEEN the detector and the visit (`_o023`, `_j6778`).  A pattern
# that requires `_visit` immediately after the detector silently SKIPS every
# tokened frame -- it does not raise, so the reader just solves on a subset.
# Measured on the live trees (2026-08-07):
#
#     gc2211  F200W   592 globbed   192 parsed   400 skipped   (68%)
#     ngc6334 F200W   560 globbed   280 parsed   280 skipped   (50%)
#
# `cataloging.py::_DETECTOR_TOKEN_RE` was fixed for this in #302; these are the
# same property in the readers #316 lists.

#: Optional `_o023` / `_j6778` segment, as a non-capturing regex group.
OBS_TOKEN_PATTERN = r'(?:_(?:o\d{3}|j\d{4,5}))?'

#: Same segment, but CAPTURING the token (or None when absent).  Readers that
#: KEY on the frame identity need this rather than OBS_TOKEN_PATTERN: gc2211's
#: five pointings reuse the same (visit, vgroup, exposure) tuples, so admitting
#: tokened names into a key that cannot hold the token trades "silently skips
#: those frames" for "silently overwrites them", which is not an improvement.
OBS_TOKEN_CAPTURE = r'(?:_(o\d{3}|j\d{4,5}))?'

#: Glob fragment covering the same optional segment.  `*` rather than the
#: regex: a glob cannot express "optional", and a bare `*` here would also span
#: `_visit001_vgroup...`, so the caller must keep the following literal.
OBS_TOKEN_GLOB = '*'


def perframe_name_re(band_pattern=r'[a-z0-9]+',
                     detector_pattern=r'nrc[ab](?:[0-9]|long)'):
    """Compiled regex for ``{band}_{detector}[_obs]_visit{NNN}_vgroup{G}_exp{N}``.

    Groups: ``(band, detector, visit, vgroup, exposure)``.  The observation
    token is matched but not captured -- readers that need it should use
    ``obs_token``-aware parsing rather than re-deriving it here.
    """
    return re.compile(
        rf'({band_pattern})_({detector_pattern}){OBS_TOKEN_PATTERN}'
        rf'_visit(\d+)_vgroup(\w+)_exp(\d+)')
