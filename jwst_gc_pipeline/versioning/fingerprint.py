"""Per-facet content fingerprints for pipeline products.

The rerun-skip engine needs to know not just *whether* a product changed, but
*which facet* changed, because different facets demand different (cheaper)
downstream actions:

* ``data_hash`` -- sha256 over the science pixel arrays ONLY (header-excluded).
  A change here means the pixels differ -> any downstream PSF fit must re-run.
* ``wcs_hash``  -- sha256 over the canonicalized WCS header cards (incl. the
  repo's ``RAOFFSET``/``DEOFFSET`` bulk-shift keywords).  A change here with an
  UNCHANGED ``data_hash`` means only the astrometric solution moved -> the
  detector-space fits are still valid and only ``x,y -> ra,dec`` needs
  recomputing (``astrometry_utils.reproject_xy_to_world``), UNLESS the shift is a
  pre-seed shift at m12 (see ``rerun`` / VERSIONING_PROVENANCE.md).
* ``meta_hash`` -- sha256 over the remaining header cards.  A change here alone
  means nothing was recomputed -> re-stamp only.

Plus two *input* fingerprints:

* ``code_hash``   -- git blob hashes of the source files implementing a stage.
* ``params_hash`` -- the stage-relevant CLI/options.

These are deliberately COARSE (per-area code sets, near-full options): a coarse
input hash only forces the owning stage to *recompute*; the resulting
``data_hash`` comparison is what actually prunes the downstream cascade, so
over-invalidation costs one stage's compute, never a spurious full re-run.
"""
import hashlib
import json
import os
import re
import subprocess

# --------------------------------------------------------------------------
# Header-keyword classification
# --------------------------------------------------------------------------
# Keywords that define the WCS / astrometric solution.  Matched case-insensitively
# against the full keyword.  Includes SIP (A_*/B_*/AP_*/BP_*), PV/PS distortion,
# and the repo's bulk-shift keywords.
_WCS_KEY_RE = re.compile(
    r'^(WCSAXES|CRVAL\d+|CRPIX\d+|CDELT\d+|CTYPE\d+|CUNIT\d+|CD\d+_\d+|'
    r'PC\d+_\d+|CROTA\d+|PV\d+_\d+|PS\d+_\d+|LONPOLE|LATPOLE|RADESYS|EQUINOX|'
    r'WCSNAME|MJDREF|A_\d+_\d+|B_\d+_\d+|AP_\d+_\d+|BP_\d+_\d+|A_ORDER|B_ORDER|'
    r'AP_ORDER|BP_ORDER|A_DMAX|B_DMAX|RAOFFSET|DEOFFSET)$',
    re.IGNORECASE)

# Volatile / provenance keywords excluded from meta_hash (they are stamped BY the
# provenance layer and would make meta_hash self-referential / nondeterministic).
_META_EXCLUDE = {
    'GCPIPEV', 'GCTAG', 'GCSTAGE', 'GCDATAH', 'GCWCSH', 'GCMETAH',
    'CHECKSUM', 'DATASUM', 'DATE', 'HISTORY', 'COMMENT', '',
}

# FITS extensions that carry science content feeding a PSF fit.  ERR/DQ affect
# the fit (weights, masking) so they are part of data identity; VAR_* planes are
# derived from ERR and omitted to keep the hash cheap.
DEFAULT_DATA_EXTS = ('SCI', 'ERR', 'DQ')


def _is_wcs_key(key):
    return bool(_WCS_KEY_RE.match(str(key)))


def _hexdigest(chunks):
    """sha256 over an ordered iterable of bytes/str chunks."""
    h = hashlib.sha256()
    for c in chunks:
        if isinstance(c, str):
            c = c.encode('utf-8')
        h.update(c)
        h.update(b'\x00')  # unambiguous separator
    return h.hexdigest()


# --------------------------------------------------------------------------
# Output-facet hashes
# --------------------------------------------------------------------------
def _array_bytes(arr):
    """Platform-stable little-endian contiguous bytes of a numpy array."""
    import numpy as np
    arr = np.ascontiguousarray(arr)
    # Normalise byte order so the same values hash identically across machines.
    # '|' = not-applicable (1-byte); '<' = already little-endian.  Everything
    # else -- '>' (big) AND '=' (native, which is big-endian on a BE host) -- is
    # converted to explicit little-endian so the "platform-stable" claim holds.
    if arr.dtype.byteorder not in ('<', '|'):
        arr = arr.astype(arr.dtype.newbyteorder('<'), copy=False)
        arr = np.ascontiguousarray(arr)
    return arr.dtype.str, arr.shape, arr.tobytes(order='C')


def data_hash(source, exts=DEFAULT_DATA_EXTS):
    """sha256 over the science arrays of a FITS product (header-independent).

    ``source`` is an open ``HDUList`` or a path.  Hashes every HDU whose
    ``EXTNAME`` is in ``exts`` (or, if none match and the product is a simple
    single-image file, the first HDU that has data), in HDU order, each tagged by
    ``(extname, extver, dtype, shape)`` so a re-ordering or dtype change is
    detected.  Returns ``None`` if the product has no array data (e.g. a
    table-only catalog -- use :func:`table_hash` for those).
    """
    from astropy.io import fits
    close = False
    if not hasattr(source, '__getitem__') or isinstance(source, (str, bytes, os.PathLike)):
        source = fits.open(source, memmap=True)
        close = True
    try:
        chunks = []
        wanted = {e.upper() for e in exts}
        matched = False
        for hdu in source:
            name = str(getattr(hdu, 'name', '') or '').upper()
            if name in wanted and getattr(hdu, 'data', None) is not None:
                matched = True
                ver = hdu.header.get('EXTVER', 1)
                dstr, shape, raw = _array_bytes(hdu.data)
                chunks += [name, str(ver), dstr, repr(shape), raw]
        if not matched:
            # Fallback for a plain single-image file (no named SCI ext).
            for hdu in source:
                if getattr(hdu, 'data', None) is not None:
                    dstr, shape, raw = _array_bytes(hdu.data)
                    chunks += ['DATA', dstr, repr(shape), raw]
                    matched = True
                    break
        if not matched:
            return None
        return _hexdigest(chunks)
    finally:
        if close:
            source.close()


def _canonical_card(key, value):
    # repr() gives a stable, precise string for floats/ints/bools/str.
    return f'{str(key).upper()}={value!r}'


def wcs_hash(header):
    """sha256 over the canonicalized WCS keywords of a FITS ``header``.

    Cards are selected by :func:`_is_wcs_key`, sorted by keyword, and rendered
    with :func:`_canonical_card`, so reordering or comment changes do not affect
    the hash but any WCS *value* change does.  Returns ``None`` if the header
    carries no WCS keywords.
    """
    cards = sorted(
        (_canonical_card(k, header[k]) for k in header if _is_wcs_key(k)))
    if not cards:
        return None
    return _hexdigest(cards)


def meta_hash(header):
    """sha256 over the non-WCS, non-volatile header cards.

    Everything that is neither a WCS keyword nor an excluded volatile/provenance
    keyword, sorted and canonicalized.  This is the "header metadata only"
    facet: a change here alone triggers a re-stamp and nothing else.
    """
    cards = []
    for k in header:
        ku = str(k).upper()
        if not ku or ku in _META_EXCLUDE or _is_wcs_key(ku):
            continue
        cards.append(_canonical_card(k, header[k]))
    return _hexdigest(sorted(cards))


def table_hash(source, exclude_cols=()):
    """sha256 over an astropy table's column data (header/meta-independent).

    For catalog products.  Hashes each column's name + dtype + bytes in column
    order; ``exclude_cols`` (e.g. volatile RA/Dec recomputed by reproject) are
    skipped so a WCS-only refresh does not read as a data change.
    """
    from astropy.table import Table
    if not isinstance(source, Table):
        source = Table.read(source)
    chunks = []
    for col in source.colnames:
        if col in exclude_cols:
            continue
        dstr, shape, raw = _array_bytes(source[col].data)
        chunks += [col, dstr, repr(shape), raw]
    return _hexdigest(chunks)


def facet_hashes(fits_path, exts=DEFAULT_DATA_EXTS):
    """Return ``{'data','wcs','meta'}`` facet hashes for a FITS product.

    * ``wcs``  is taken from the header that actually carries the WCS (the first
      extension with WCS keywords, else the primary header).
    * ``meta`` is hashed over EVERY HDU header (each tagged by index), so a
      non-WCS metadata change living in an extension header (e.g. ``BUNIT`` or an
      exposure keyword on the SCI HDU) is captured -- not just primary-header
      changes.
    """
    from astropy.io import fits
    with fits.open(fits_path, memmap=True) as hdul:
        wcs_hdr = None
        for hdu in hdul:
            if any(_is_wcs_key(k) for k in hdu.header):
                wcs_hdr = hdu.header
                break
        if wcs_hdr is None:
            wcs_hdr = hdul[0].header
        meta_chunks = []
        for i, hdu in enumerate(hdul):
            meta_chunks += [f'HDU{i}', meta_hash(hdu.header)]
        return {
            'data': data_hash(hdul, exts=exts),
            'wcs': wcs_hash(wcs_hdr),
            'meta': _hexdigest(meta_chunks),
        }


# --------------------------------------------------------------------------
# Input fingerprints: code + params
# --------------------------------------------------------------------------
# Curated per-area source sets.  Cataloging stages share one set: a coarse
# cataloging-code change forces m12 to RECOMPUTE, and the output data_hash prunes
# whether m3.. actually re-run (see module docstring).  m7/m8 add their own
# cross-band/forced-fill files so a change scoped there does not disturb m12..m6.
_CATALOG_BASE = [
    'jwst_gc_pipeline/photometry/cataloging.py',
    'jwst_gc_pipeline/photometry/catalog_long.py',
    'jwst_gc_pipeline/photometry/merge_catalogs.py',
    'jwst_gc_pipeline/photometry/naming.py',
    'jwst_gc_pipeline/photometry/incremental_refit.py',
    'jwst_gc_pipeline/photometry/astrometry_checkpoint.py',
    'jwst_gc_pipeline/photometry/astrometry_offsets.py',
    'jwst_gc_pipeline/photometry/measure_offsets.py',
    'jwst_gc_pipeline/astrometry_utils.py',
]
STAGE_CODE_FILES = {
    'imaging': [
        'jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py',
        'jwst_gc_pipeline/reduction/PipelineMIRI.py',
        'jwst_gc_pipeline/reduction/dva_correction.py',
        'jwst_gc_pipeline/reduction/align_to_catalogs.py',
        'jwst_gc_pipeline/reduction/fetch_refs.py',
        'jwst_gc_pipeline/astrometry_utils.py',
    ],
    'm12': _CATALOG_BASE,
    'm3': _CATALOG_BASE,
    'm4': _CATALOG_BASE,
    'm5': _CATALOG_BASE,
    'm6': _CATALOG_BASE,
    'm7': _CATALOG_BASE + ['jwst_gc_pipeline/photometry/forced_fill.py'],
    'm8': _CATALOG_BASE + ['jwst_gc_pipeline/photometry/forced_fill.py'],
}

# Options that never affect a product's content -- excluded from params_hash so
# a different job name / worker count / output path does not spuriously
# invalidate a stage.  Matched case-insensitively against option attribute names.
_VOLATILE_PARAMS = {
    'ncores', 'njobs', 'nworkers', 'workers', 'verbose', 'debug', 'job_name',
    'jobname', 'basepath', 'base_path', 'log', 'logfile', 'profile_memory',
    'dry_run', 'manual_frame_shard', 'manual_skip_finalize',
    'manual_finalize_only', 'manual_start_phase', 'manual_stop_after_phase',
}


def _blob_hash(repo_dir, relpath):
    """git blob hash of the WORKING-TREE content of ``relpath`` (dev-aware).

    Uses ``git hash-object`` so an uncommitted edit changes the hash (a dev run
    must not reuse a release product).  Raises if the file is missing (a
    fingerprint must never silently degrade).
    """
    path = os.path.join(repo_dir, relpath)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'code-fingerprint file missing: {relpath} (under {repo_dir})')
    try:
        return subprocess.check_output(
            ['git', '-C', repo_dir, 'hash-object', path],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        # Fall back to REPLICATING git's blob object id so the value matches
        # ``git hash-object`` from a run where git WAS available (otherwise a
        # git-vs-no-git split across the recording and comparing runs would
        # mismatch code_hash and force a spurious REFIT).  git's default object
        # id is sha1 over ``b'blob <len>\0' + content``.
        with open(path, 'rb') as fh:
            content = fh.read()
        h = hashlib.sha1()
        h.update(b'blob %d\0' % len(content))
        h.update(content)
        return h.hexdigest()


def code_hash(stage, repo_dir=None):
    """sha256 over the (sorted) git blob hashes of ``stage``'s source files."""
    if repo_dir is None:
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_dir = os.path.dirname(pkg)
    files = STAGE_CODE_FILES.get(stage)
    if files is None:
        raise KeyError(f'unknown stage {stage!r}; known: {sorted(STAGE_CODE_FILES)}')
    pairs = sorted((f, _blob_hash(repo_dir, f)) for f in files)
    return _hexdigest([f'{f}:{h}' for f, h in pairs])


def params_hash(options, stage=None, exclude=_VOLATILE_PARAMS):
    """sha256 over the non-volatile options that affect content.

    ``options`` may be an argparse ``Namespace`` or a dict.  Values are
    JSON-serialized (with a repr fallback for exotic types) so the hash is
    deterministic.  ``stage`` is currently informational (all stages read the
    same options object); it is recorded for future per-stage param scoping.
    """
    if hasattr(options, '__dict__') and not isinstance(options, dict):
        items = vars(options)
    else:
        items = dict(options)
    excl = {e.lower() for e in exclude}
    kept = {}
    for k, v in items.items():
        if k.lower() in excl:
            continue
        try:
            json.dumps(v)
            kept[k] = v
        except (TypeError, ValueError):
            kept[k] = repr(v)
    payload = json.dumps(kept, sort_keys=True, default=repr)
    return _hexdigest([payload])
