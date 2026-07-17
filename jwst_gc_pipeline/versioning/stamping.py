"""Emit provenance at a stage's final write: sidecar + mirrored FITS keys.

Call ``stamp_product`` (image products) or ``stamp_catalog`` (table catalogs)
immediately AFTER a stage writes its final output.  Each:

1. computes the product's output facet hashes (data/wcs/meta or table),
2. assembles the full provenance record (tag + stage + input fingerprints +
   output facets) and writes ``<product>.prov.json`` (``prov_sidecar``),
3. mirrors the compact subset (``GCSTAGE``/``GCDATAH``/``GCWCSH``/``GCMETAH``)
   into the product's FITS primary header, so the product stays self-describing
   if the sidecar is lost.  (``GCTAG``/``GCPIPEV`` are already stamped by the
   global ``provenance`` write hook.)

Everything here is BEST-EFFORT and fail-soft: a stamping failure must never
break or corrupt the science write.  The caller wraps it in
``try_stamp_*``-style guards, and the internals only catch the specific FITS/IO
errors that stamping itself can raise.
"""
import os
import warnings

from . import fingerprint as _fp
from . import prov_sidecar

# Compact facet keys mirrored into the FITS primary header.
GCSTAGE_KEY = 'GCSTAGE'
GCDATAH_KEY = 'GCDATAH'
GCWCSH_KEY = 'GCWCSH'
GCMETAH_KEY = 'GCMETAH'
# A full 64-hex sha256 + keyword leaves no room for a comment in an 80-col FITS
# card (VerifyWarning on every write).  The card is only for eyeballing; the FULL
# hash is in the sidecar, so mirror a 16-hex prefix (ample to disambiguate).
_HASH_CARD_LEN = 16

# Header keys that carry the imaging "generation" environment (jwst version /
# CRDS context / DVA-correction marker).  Mirrors astrometry_utils.GENERATION_KEYS.
_ENV_HEADER_KEYS = {'jwst_version': 'CAL_VER', 'crds_context': 'CRDS_CTX',
                    'dva': 'DVACORR'}


def env_from_header(header):
    """Extract the imaging env fingerprint (jwst/crds/dva) from a FITS header."""
    env = {}
    for name, key in _ENV_HEADER_KEYS.items():
        if key in header:
            env[name] = str(header[key])
    return env


def _mirror_keys(fits_path, stage, facets):
    """Write GCSTAGE/GCDATAH/GCWCSH/GCMETAH into the primary header, best-effort."""
    from astropy.io import fits
    try:
        with fits.open(fits_path, mode='update') as hdul:
            h = hdul[0].header
            h[GCSTAGE_KEY] = (stage, 'gc-pipeline stage')
            if facets.get('data'):
                h[GCDATAH_KEY] = (facets['data'][:_HASH_CARD_LEN], 'data facet sha256')
            if facets.get('wcs'):
                h[GCWCSH_KEY] = (facets['wcs'][:_HASH_CARD_LEN], 'wcs facet sha256')
            if facets.get('meta'):
                h[GCMETAH_KEY] = (facets['meta'][:_HASH_CARD_LEN], 'meta facet sha256')
    except (OSError, ValueError, KeyError) as exc:
        warnings.warn(f'stamp: could not mirror facet keys into {fits_path}: {exc}',
                      stacklevel=3)


def _resolve_tag_and_code(stage, repo_dir):
    from .tags import get_pipeline_tag
    tag = get_pipeline_tag(repo_dir)
    try:
        code = _fp.code_hash(stage, repo_dir=repo_dir)
    except (KeyError, FileNotFoundError) as exc:
        # Unknown stage or a moved source file: record no code hash rather than
        # abort stamping (the sidecar is still useful for data/wcs/meta).
        warnings.warn(f'stamp: code_hash unavailable for {stage!r}: {exc}',
                      stacklevel=3)
        code = None
    return tag, code


def stamp_product(fits_path, stage, env=None, params=None, upstream=None,
                  exts=_fp.DEFAULT_DATA_EXTS, repo_dir=None, mirror=True):
    """Stamp an IMAGE product (``_i2d``/``_crf``); return the written record.

    ``env`` defaults to the imaging generation keys read from the product header
    (``env_from_header``).  ``params``/``upstream`` are optional input
    fingerprints supplied by the caller when in scope.
    """
    facets = _fp.facet_hashes(fits_path, exts=exts)
    if env is None:
        from astropy.io import fits
        with fits.open(fits_path, memmap=True) as hdul:
            env = env_from_header(hdul[0].header)
            for hdu in hdul[1:]:
                # SCI header often carries CAL_VER/CRDS_CTX for cal products.
                for k, v in env_from_header(hdu.header).items():
                    env.setdefault(k, v)
    tag, code = _resolve_tag_and_code(stage, repo_dir)
    record = prov_sidecar.build_record(
        stage, tag, facets, env=env, code=code, params=params,
        upstream=upstream)
    path = prov_sidecar.write_sidecar(fits_path, record)
    if mirror:
        _mirror_keys(fits_path, stage, facets)
    return record, path


def stamp_catalog(fits_path, stage, exclude_cols=('ra', 'dec', 'skycoord_ra',
                                                  'skycoord_dec'),
                  env=None, params=None, upstream=None, repo_dir=None,
                  mirror=True):
    """Stamp a TABLE catalog product; return the written record.

    The ``data`` facet is the table's column content (``table_hash``) with the
    reprojectable sky columns excluded, so a post-hoc RA/Dec refresh does not
    read as a data change.  A catalog has no image WCS, so ``wcs`` is None; the
    ``meta`` facet is the primary-header metadata.
    """
    from astropy.io import fits
    data = _fp.table_hash(fits_path, exclude_cols=exclude_cols)
    with fits.open(fits_path, memmap=True) as hdul:
        meta = _fp.meta_hash(hdul[0].header)
    facets = {'data': data, 'wcs': None, 'meta': meta}
    tag, code = _resolve_tag_and_code(stage, repo_dir)
    record = prov_sidecar.build_record(
        stage, tag, facets, env=env or {}, code=code, params=params,
        upstream=upstream)
    path = prov_sidecar.write_sidecar(fits_path, record)
    if mirror:
        _mirror_keys(fits_path, stage, facets)
    return record, path


def try_stamp_product(fits_path, stage, **kwargs):
    """Best-effort ``stamp_product`` that never raises (for hot write paths).

    Returns the record on success, ``None`` on any handled failure (a warning is
    emitted).  Use this at in-pipeline call sites so a provenance hiccup can
    never break the run.
    """
    if not os.path.exists(fits_path):
        warnings.warn(f'stamp: product does not exist: {fits_path}', stacklevel=2)
        return None
    try:
        record, _ = stamp_product(fits_path, stage, **kwargs)
        return record
    except (OSError, ValueError, KeyError) as exc:
        warnings.warn(f'stamp: failed to stamp {stage} product {fits_path}: {exc}',
                      stacklevel=2)
        return None


def try_stamp_catalog(fits_path, stage, **kwargs):
    """Best-effort ``stamp_catalog`` that never raises (for hot write paths)."""
    if not os.path.exists(fits_path):
        warnings.warn(f'stamp: catalog does not exist: {fits_path}', stacklevel=2)
        return None
    try:
        record, _ = stamp_catalog(fits_path, stage, **kwargs)
        return record
    except (OSError, ValueError, KeyError) as exc:
        warnings.warn(f'stamp: failed to stamp {stage} catalog {fits_path}: {exc}',
                      stacklevel=2)
        return None
