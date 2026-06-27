"""Filename-token helpers shared across the photometry pipeline.

These build the small, unambiguous tokens embedded in per-frame / merged output
filenames (iteration label, background-subtraction mode, spatial seed chunk) so
that products from different iterations and modes never collide and can be
glob-matched exactly.

Factored out of ``crowdsource_catalogs_long.py`` (2026-06-09 restructure); the
old module now imports these names from here so there is a single source of
truth.  Pure string/regex helpers with no astronomy dependencies.
"""
import re

# Match _chunkXXofYY (any width) when stripping the chunk suffix from an
# iteration_label or filename component.
_CHUNK_TOKEN_RE = re.compile(r'_chunk\d+of\d+')

# Single source of truth for which JWST filters are MIRI vs NIRCam.  Lives here
# (a heavy-import-free module) so merge_catalogs.py can import it without pulling
# in crowdsource_catalogs_long.py's webbpsf chain.
MIRI_FILTERS = frozenset(['f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w',
                          'f1500w', 'f1800w', 'f2100w', 'f2550w'])


def _instrument_from_filter(filtername):
    """Return 'MIRI' or 'NIRCam' based on filter name (no header read needed)."""
    return 'MIRI' if str(filtername).lower() in MIRI_FILTERS else 'NIRCam'


def _inst_token(filtername):
    """Lowercased instrument token used in JWST i2d filename conventions."""
    return _instrument_from_filter(filtername).lower()


def _svo_filter_id(filtername):
    """SVO FPS filterID (e.g. 'JWST/NIRCam.F480M', 'JWST/MIRI.F770W')."""
    return f'JWST/{_instrument_from_filter(filtername)}.{filtername.upper()}'


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
