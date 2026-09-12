"""Cross-phase reuse of a saturated-star fit, keyed on the fit's INPUTS.

The satstar fit runs once per cataloging phase (m12, m3, m4, m5, m6, m7) and
every phase redoes the identical work.  ``remove_saturated_stars`` opens the
ORIGINAL ``_crf.fits`` and reads ``fh['SCI'].data``; no phase writes that array,
not even the ``_resbgsub_m5/6/7`` phases, and the starting parameters are
recomputed from the frame's DQ each call, so nothing carries a prior solution
in.  The per-frame cache in ``load_or_make_satstar_catalog`` nonetheless misses,
because the cache FILENAME carries the phase token::

    satstar_file_suffix = f'{bgsub}{_iteration_token(satstar_label)}'

Measured on disk (brick F182M ``jw02221001001_07101_00001_nrca2``, 235 rows):
the six per-phase catalogs agree to maxdiff 0.000e+00 on ``x_init``, ``y_init``,
``flux_init`` AND ``x_fit``, ``y_fit``, ``flux_fit``.  Identical start, identical
answer.  Campaign-wide there are 57,619 per-exposure satstar catalogs across 31
field trees, 44,633 (77%) of them m3-m7 repeats, carrying ~3,000-3,900 core-hours.

This module supplies the content key that lets a later phase ADOPT an earlier
phase's products instead of refitting.  A reuse costs one ``Table.read``
(measured 0.030 s against a 124.4 s fit) and is bit-identical by construction --
the file is copied, not recomputed.

The key covers every input that can change the fit: the frame's SCI and DQ
arrays, the PSF grid, the recovery signature, the partner-band seed positions
(``partner_sky`` -- these genuinely differ between m12, where the partner
catalogs do not exist yet, and m3+, where they do), the seed-gate image, the
forced-fit geometry, and the environment switches cataloging sets per run
(``NIRCAM_SATSTAR_LOCK_POS``, ``NIRCAM_SATSTAR_TIGHT_BOUND``,
``NIRCAM_SATSTAR_RECOVERED_CAP``, ...).  ``outside_star_pixels`` and
``flux_overrides``/``flux_drops`` already force ``overwrite=True`` upstream and
so never reach this path, and are keyed anyway.

Why the alternative was rejected.  Warm-starting the fitter from the previous
pass's ``x_fit``/``y_fit``/``flux_fit`` saves ~29% of satstar CPU on a full
production frame (brick F182M nrca4, 205 seeds: 124.4 s -> 88.5 s) but MOVES the
photometry -- median 0.55 mmag, max 7.99 mmag, 30/114 matched sources shifted
>1 mmag -- against a cold-vs-cold baseline that is EXACTLY 0.000 on all 191
sources.  ``leastsq`` stops on a relative tolerance, so a different start stops
at a different nearby point, and the loop subtracts each accepted model from
``data_working`` before fitting the next source, chaining a ~1e-7 termination
difference forward into the mmag range.  Reuse has no such term.

Default OFF: set ``SATSTAR_CROSS_PHASE_CACHE=1`` to enable.
"""
import glob
import hashlib
import os
import shutil

import numpy as np
from astropy.io import fits

#: Env var that enables cross-phase reuse.  Default OFF.
CROSS_PHASE_CACHE_ENV = 'SATSTAR_CROSS_PHASE_CACHE'

#: FITS header/meta keyword carrying the content key of the fit that wrote a
#: satstar catalog.  Sits beside ``SATRECOV`` (the recovery signature), which
#: keys the EXISTING same-suffix cache.
CONTENT_KEY_META = 'SATKEY'

#: Every per-suffix product one satstar fit writes (saturated_star_finding.py,
#: ``remove_saturated_stars``).  A reuse copies whichever of these the donor
#: phase actually produced, so the adopting phase sees exactly the file set a
#: fit would have left behind -- the model in particular, which
#: ``_prepare_frame_for_photometry`` reads back by suffix and subtracts.
SATSTAR_PRODUCT_SUFFIXES = (
    '_satstar_catalog.fits',
    '_satstar_model.fits',
    '_satstar_residual.fits',
    '_satstar_rejected.fits',
    '_satstar_flags.fits',
    '_wingcal_calibrators.fits',
    '_extended_satstar_catalog.fits',
    '_extended_satstar_model.fits',
)

#: Catalog suffixes a donor search will accept, in the precedence
#: ``load_or_make_satstar_catalog`` itself uses (extended first).
_CATALOG_SUFFIXES = ('_extended_satstar_catalog.fits', '_satstar_catalog.fits')

#: Environment variables outside the ``*SATSTAR*`` family that steer the fit.
#: Every var whose NAME contains ``SATSTAR`` is keyed automatically, so a future
#: switch is covered without editing this list; these are the ones that are not
#: named for it.
_EXTRA_KEYED_ENV = (
    'MIRI_FIRSTGROUP_SAT_DQ',
    'MIRI_DROP_OFFFP_SATSTAR',
    'MIRI_EDGE_DETECT_MARGIN',
    'STPSF_PATH',
)


def cross_phase_cache_enabled(env=None):
    """True when cross-phase satstar reuse is switched on.  Default OFF."""
    env = os.environ if env is None else env
    return str(env.get(CROSS_PHASE_CACHE_ENV, '0')).lower() not in (
        '0', '', 'false', 'no', 'off')


def keyed_env(env=None):
    """The environment switches that steer the satstar fit, as sorted pairs."""
    env = os.environ if env is None else env
    names = {n for n in env if 'SATSTAR' in n} | {
        n for n in _EXTRA_KEYED_ENV if n in env}
    return tuple(sorted((n, str(env[n])) for n in names))


def _feed_array(h, arr):
    if arr is None:
        h.update(b'\x00none')
        return
    arr = np.ascontiguousarray(arr)
    h.update(f'{arr.dtype.str}{arr.shape}'.encode())
    h.update(memoryview(arr).cast('B'))


def _feed_stats(h, paths):
    """Feed (basename, size, mtime_ns) for each path -- identity of an input
    FILE we do not want to read in full (a PSF grid is ~100 MB)."""
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f'{os.path.basename(p)}:{st.st_size}:{st.st_mtime_ns}'.encode())


def _psf_grid_candidates(path_prefix, header, use_merged_psf_for_merged):
    """PSF grid files that COULD be the one this fit uses.

    ``get_psf`` builds its filename from instrument/detector/filter plus
    fov/oversample/npsf, and may fall back to the merged grid.  Globbing the
    instrument+detector+filter family (and the merged family when it is in play)
    is a superset of the single file used -- over-conservative in the safe
    direction: rebuilding that detector's grid invalidates the reuse.
    """
    if not path_prefix or not os.path.isdir(path_prefix):
        return []
    inst = str(header.get('INSTRUME', '')).lower()
    det = str(header.get('DETECTOR', '')).lower()
    filt = str(header.get('FILTER', '')).lower()
    if not (inst and det and filt):
        return []
    pats = [f'{inst}_{det}_{filt}_*.fits']
    if use_merged_psf_for_merged:
        pats.append(f'*merged*{filt}*.fits')
    out = set()
    for pat in pats:
        out.update(glob.glob(os.path.join(path_prefix, pat)))
    return sorted(out)


def satstar_content_key(filename, *, path_prefix, use_merged_psf_for_merged=False,
                        recovery_signature=None, partner_sky=None,
                        outside_star_pixels=None, outside_star_fit_box=512,
                        forced_grid_search_radius=5, flux_overrides=None,
                        flux_drops=None, oversub_clamp_percentile=10.0,
                        seed_gate_image=None, deblend_with_zeroframe=False,
                        env=None):
    """Digest of everything that determines this frame's satstar fit.

    Two runs with the same key would run the identical fitter over the identical
    arrays, so one may adopt the other's products.  The key deliberately does
    NOT include the phase token, the merge label, the bgsub token or the output
    filename: those name the OUTPUT, and the whole point is that the phases
    produce the same output.

    It also does not include the frame's WCS.  A cached catalog already survives
    an offsets-table correction and is re-projected onto the frame's current
    GWCS by ``load_or_make_satstar_catalog._on_current_frame`` (issue #193); the
    fit itself is done in pixel space and is unaffected.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b'satstar-content-key-v1\n')

    with fits.open(filename, memmap=False) as fh:
        header = fh[0].header
        _feed_array(h, fh['SCI'].data if 'SCI' in fh else None)
        _feed_array(h, fh['DQ'].data if 'DQ' in fh else None)

    # PSF grid + the sibling ramp (ZEROFRAME core recovery / first-group DQ).
    _feed_stats(h, _psf_grid_candidates(path_prefix, header,
                                        use_merged_psf_for_merged))
    from jwst_gc_pipeline.reduction.saturated_star_finding import _find_ramp_for
    ramp = _find_ramp_for(filename)
    _feed_stats(h, [ramp] if ramp else [])

    # Partner-band seeds: absent at m12, present at m3+ -- a real inter-phase
    # input difference, and the one the key exists to catch.
    if partner_sky is None:
        h.update(b'partner:none')
    else:
        h.update(f'partner:{len(partner_sky)}'.encode())
        _feed_array(h, np.asarray(partner_sky.ra.deg, dtype=float))
        _feed_array(h, np.asarray(partner_sky.dec.deg, dtype=float))

    _feed_array(h, seed_gate_image)

    for name, value in (('outside_star_pixels', outside_star_pixels),
                        ('flux_overrides', flux_overrides),
                        ('flux_drops', flux_drops)):
        h.update(f'{name}:{_stable_repr(value)}'.encode())

    for name, value in (('recovery_signature', recovery_signature),
                        ('use_merged_psf_for_merged', bool(use_merged_psf_for_merged)),
                        ('outside_star_fit_box', outside_star_fit_box),
                        ('forced_grid_search_radius', forced_grid_search_radius),
                        ('oversub_clamp_percentile', float(oversub_clamp_percentile)),
                        ('deblend_with_zeroframe', bool(deblend_with_zeroframe))):
        h.update(f'{name}={value!r};'.encode())

    for name, value in keyed_env(env):
        h.update(f'env:{name}={value};'.encode())

    return h.hexdigest()


def _stable_repr(value):
    """Order-independent repr for the dict/sequence options, so two equal
    override maps built in different orders key the same."""
    if value is None:
        return 'none'
    if isinstance(value, dict):
        return repr(sorted((str(k), repr(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return repr(sorted(repr(v) for v in value))
    return repr(value)


def _suffix_of(catalog_path, filename, catalog_suffix):
    """The file_suffix (phase token) a donor catalog was written under."""
    stem = filename[:-len('.fits')]
    if not (catalog_path.startswith(stem) and catalog_path.endswith(catalog_suffix)):
        raise ValueError(f"{catalog_path} is not a {catalog_suffix} product of "
                         f"{filename}")
    return catalog_path[len(stem):-len(catalog_suffix)]


def stamp_content_key(catalog_path, key):
    """Write ``key`` into a just-written satstar catalog's table header."""
    fits.setval(catalog_path, CONTENT_KEY_META, value=str(key), ext=1)


def read_content_key(catalog_path):
    """Content key stamped on a satstar catalog, or None for an unstamped one.

    An unstamped catalog is NEVER reused: it was written before the key existed
    (or by a path that does not stamp), so nothing is known about the inputs it
    was fitted from.
    """
    header = fits.getheader(catalog_path, ext=1)
    return header.get(CONTENT_KEY_META) or None


def find_reusable_satstar_catalog(filename, want_suffix, key):
    """A satstar catalog for this frame, written under a DIFFERENT suffix, whose
    stamped content key equals ``key``.  None when there is no such donor.

    Donors are searched extended-first, matching the precedence
    ``load_or_make_satstar_catalog`` applies to its own same-suffix cache.
    """
    stem = filename[:-len('.fits')]
    for catalog_suffix in _CATALOG_SUFFIXES:
        for path in sorted(glob.glob(f'{stem}*{catalog_suffix}')):
            if (catalog_suffix == '_satstar_catalog.fits'
                    and path.endswith('_extended_satstar_catalog.fits')):
                continue  # counted under the extended pass, with its own family
            if _suffix_of(path, filename, catalog_suffix) == want_suffix:
                continue  # the caller has already looked for its own
            if read_content_key(path) == key:
                return path
    return None


def adopt_satstar_products(donor_catalog, filename, want_suffix,
                           catalog_suffix=None):
    """Copy a donor phase's satstar products onto ``want_suffix``'s names.

    Returns the list of paths written, the requested catalog first.  The copy is
    byte-for-byte, so the adopting phase's catalog and model are identical to
    what its own fit would have produced -- that is the correctness argument,
    and it does not depend on a fitter tolerance.
    """
    if catalog_suffix is None:
        catalog_suffix = next(s for s in _CATALOG_SUFFIXES
                              if donor_catalog.endswith(s))
    stem = filename[:-len('.fits')]
    donor_suffix = _suffix_of(donor_catalog, filename, catalog_suffix)
    written = []
    for product in SATSTAR_PRODUCT_SUFFIXES:
        src = f'{stem}{donor_suffix}{product}'
        dst = f'{stem}{want_suffix}{product}'
        if src == dst or not os.path.exists(src):
            continue
        shutil.copyfile(src, dst)
        written.append(dst)
    want_catalog = f'{stem}{want_suffix}{catalog_suffix}'
    written.sort(key=lambda p: p != want_catalog)
    return written
