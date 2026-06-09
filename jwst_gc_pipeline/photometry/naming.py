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


def _bgsub_token(options):
    """Filename token for the background-subtraction mode(s) in effect.

    * ``--bgsub`` (global Background2D subtraction)        -> ``_bgsub``
    * ``--use-iter3-residual-bg`` (iter3 residual-smoothed
      background subtraction)                             -> ``_resbgsub``

    Both can be set; the tokens concatenate in a fixed order so output
    catalog/residual/model/diagnostic filenames are unambiguous and the
    skip-if-done prediction (_predict_output_tokens) stays in sync with the
    names actually written by do_photometry_step.  ``_bgsub`` is never a
    substring of ``_resbgsub`` so exact-token matching does not collide.
    """
    token = '_bgsub' if options.bgsub else ''
    if getattr(options, 'use_iter3_residual_bg', False):
        token += '_resbgsub'
    return token
