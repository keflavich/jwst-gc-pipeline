"""Read the **GWCS** of a detector-frame product, not its SIP approximation.

Why
---
Every JWST detector-frame product carries two WCS representations:

* the **GWCS** in the ASDF extension (``model.meta.wcs``) -- the authoritative
  one: the full SIAF distortion polynomial, velocity aberration, and tangent
  projection, evaluated exactly; and
* a FITS ``RA---TAN-SIP`` header -- a *fitted low-order polynomial
  approximation* of the GWCS, kept so that plain-``astropy.wcs`` consumers
  (DS9/CARTA, ``reproject``) can read something.

SIP cannot represent the JWST distortion; it can only fit it.  Both directions
of the fit carry error:

* **forward** (``all_pix2world``): the fit residual.  With the tight fit this
  package now writes it is ~0.00 mas; on every frame written before that fix it
  is 5-8 mas, position-dependent and different per detector *and* per filter --
  i.e. exactly the shape of error the astrometric gates are looking for
  (2 mas m2 consensus, 5 mas m7 cross-filter, 30 mas overlap).
* **inverse** (``all_world2pix``): SIP's inverse is a *separate* fitted
  polynomial (``AP_*``/``BP_*``) refined by an iterative solver.  Measured on a
  brick ``_crf``: up to 176 millipixels of error, and it **raises
  ``NoConvergence``** for positions outside the frame -- the failure that
  aborted the whole W51 m8 fill (#187).

The GWCS has neither problem.  Measured on the same frames: inverse exact to
<1 millipixel, and out-of-footprint positions return ``NaN`` (bounding box)
rather than raising.  It is also no slower -- 50k transforms took 11 ms
(GWCS) vs 7 ms (SIP) forward, and 14 ms vs 14 ms inverse.

So: **read the GWCS wherever it exists.**  The SIP header is for display and
for external tools.

Usage
-----
Drop-in for ``astropy.wcs.WCS(hdul['SCI'].header)``::

    from jwst_gc_pipeline.frame_wcs import frame_wcs
    ww = frame_wcs(filename)               # or frame_wcs(hdulist)
    sky = ww.pixel_to_world(x, y)          # exact, via the GWCS
    x, y = ww.all_world2pix(ra, dec, 0)    # exact; NaN off-footprint, no raise

:func:`frame_wcs` returns a :class:`FrameWCS`: coordinate transforms go through
the GWCS, everything else (``.wcs.crval``, slicing, ``to_header()``,
``proj_plane_pixel_area()``, ``celestial``, ...) is delegated to the FITS WCS,
which the tight-SIP fix keeps faithful to <0.5 mas.  When a product has no GWCS
(a plain FITS image, an i2d cutout written by an external tool) the FITS WCS is
returned directly, so callers need no branching.

``i2d`` mosaics are exempt from all of this: ``resample`` writes a rectified
plain ``RA---TAN`` grid with no SIP, so their FITS WCS is exact.
"""
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy import wcs as astropy_wcs

__all__ = ['FrameWCS', 'frame_wcs', 'gwcs_from_file', 'has_gwcs']

#: Set to '0' to fall back to the FITS/SIP WCS everywhere (debugging only --
#: it reinstates the 5-8 mas forward error and the NoConvergence failure mode).
_USE_GWCS = os.environ.get('FRAME_WCS_USE_GWCS', '1') != '0'


class MissingGwcsWarning(UserWarning):
    """A detector-frame product had no GWCS, so the SIP approximation was used."""


def _sip_wcs_from_header(header):
    """``astropy.wcs.WCS`` honouring SIP.

    ``relax=True`` matters: a header whose CTYPE lost the ``-SIP`` suffix still
    carries the ``A_*``/``B_*`` terms, and without ``relax`` astropy silently
    drops the distortion (~0.1" at the detector corners).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', astropy_wcs.FITSFixedWarning)
        return astropy_wcs.WCS(header, relax=True)


_GWCS_CACHE = {}


def _bbox_center(gw):
    """Mid-point of ``gw``'s bounding box, or None if it has none."""
    bb = getattr(gw, 'bounding_box', None)
    if bb is None:
        return None
    try:
        intervals = [tuple(iv) for iv in bb]
    except TypeError:
        return None
    if len(intervals) != 2:
        return None
    return tuple(0.5 * (float(lo) + float(hi)) for lo, hi in intervals)


def gwcs_from_file(filename, use_cache=True):
    """The GWCS stored in ``filename``'s ASDF extension, or None.

    Reads the ``ASDF`` BinTable extension's bytes and deserialises them
    in-memory, rather than instantiating a ``jwst`` datamodel: no schema
    validation and no pixel data, so it does not pay the datamodel open cost
    and holds no handle on the file.

    Results are memoised on ``(path, mtime, size)``.  A file rewritten in place
    (``fix_alignment`` overwrites its input) therefore invalidates its own entry.
    """
    try:
        import asdf
    except ImportError:                                  # pragma: no cover
        return None

    try:
        st = os.stat(filename)
        key = (os.path.abspath(filename), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if use_cache and key in _GWCS_CACHE:
        return _GWCS_CACHE[key]

    try:
        with fits.open(filename, memmap=False, lazy_load_hdus=True) as hdul:
            if 'ASDF' not in hdul:
                return None
            buf = hdul['ASDF'].data['ASDF_METADATA'].tobytes()
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        # not an ASDF-in-FITS product, or the extension is unreadable
        return None

    import io
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # lazy_load=True is required: the tree's `data`/`err`/`dq` nodes are
            # external references to the FITS extensions, which cannot resolve
            # from an in-memory buffer.  Nothing here touches them.
            with asdf.open(io.BytesIO(buf), lazy_load=True, memmap=False,
                           ignore_missing_extensions=True) as af:
                gw = af.tree.get('meta', {}).get('wcs')
                if gw is not None:
                    # Exercise both directions while the buffer is still open,
                    # so any lazily-loaded array inside the transform (e.g. a
                    # tabular distortion term) is materialised now rather than
                    # failing at first use after the buffer is gone.
                    _c = _bbox_center(gw)
                    if _c is not None:
                        gw.invert(*gw(*_c))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None

    if use_cache and gw is not None:
        _GWCS_CACHE[key] = gw
    return gw


def has_gwcs(filename):
    """True if ``filename`` carries a GWCS in its ASDF extension."""
    return gwcs_from_file(filename) is not None


class FrameWCS:
    """GWCS-backed WCS with an ``astropy.wcs.WCS``-shaped interface.

    Coordinate transforms (``pixel_to_world``, ``world_to_pixel``,
    ``all_pix2world``, ``all_world2pix``, ``wcs_pix2world``, ``wcs_world2pix``,
    and their ``*_values`` forms) are evaluated with the **GWCS** -- exact, and
    ``NaN`` rather than an exception for positions outside the frame.

    Every other attribute is delegated to the FITS ``WCS`` built from the SCI
    header (``.wcs.crval``, ``celestial``, ``proj_plane_pixel_area()``,
    ``to_header()``, ``__getitem__`` slicing, ...).  Note that a *sliced*
    ``FrameWCS`` degrades to a plain FITS WCS -- ``astropy``'s slicing machinery
    knows nothing about GWCS -- so slice only for display.
    """

    def __init__(self, gwcs_obj, fits_wcs, filename=None):
        self._gwcs = gwcs_obj
        self._fits = fits_wcs
        self.filename = filename

    # -- introspection ----------------------------------------------------
    @property
    def gwcs(self):
        """The underlying `gwcs.WCS`."""
        return self._gwcs

    @property
    def fits_wcs(self):
        """The underlying FITS/SIP `astropy.wcs.WCS` (display / external tools)."""
        return self._fits

    def __repr__(self):
        return (f"<FrameWCS gwcs-backed"
                f"{' ' + os.path.basename(self.filename) if self.filename else ''}>")

    def __getattr__(self, name):
        # only reached for attributes not defined on the class
        return getattr(self._fits, name)

    def __getitem__(self, item):
        # astropy slicing cannot carry a GWCS; degrade explicitly.
        return self._fits[item]

    # -- pixel -> world ---------------------------------------------------
    def pixel_to_world(self, *args, **kwargs):
        return self._gwcs.pixel_to_world(*args, **kwargs)

    def pixel_to_world_values(self, *args, **kwargs):
        return self._gwcs.pixel_to_world_values(*args, **kwargs)

    def all_pix2world(self, *args, **kwargs):
        return self._pix2world(*args, **kwargs)

    def wcs_pix2world(self, *args, **kwargs):
        return self._pix2world(*args, **kwargs)

    def _pix2world(self, *args, **kwargs):
        """``(x, y, origin)`` or ``(xy_array, origin)``, as astropy accepts."""
        kwargs.pop('ra_dec_order', None)
        xy, origin = _split_origin(args, kwargs)
        if len(xy) == 1:
            arr = np.asarray(xy[0], dtype=float)
            ra, dec = self._gwcs(arr[:, 0] - origin, arr[:, 1] - origin)
            return np.column_stack([ra, dec])
        x, y = (np.asarray(v, dtype=float) for v in xy)
        return self._gwcs(x - origin, y - origin)

    # -- world -> pixel ---------------------------------------------------
    def world_to_pixel(self, *args, **kwargs):
        return self._gwcs.world_to_pixel(*args, **kwargs)

    def world_to_pixel_values(self, *args, **kwargs):
        return self._gwcs.world_to_pixel_values(*args, **kwargs)

    def all_world2pix(self, *args, **kwargs):
        return self._world2pix(*args, **kwargs)

    def wcs_world2pix(self, *args, **kwargs):
        return self._world2pix(*args, **kwargs)

    def _world2pix(self, *args, **kwargs):
        # astropy's SIP-inverse-only knobs; meaningless for an exact inverse.
        for k in ('quiet', 'tolerance', 'maxiter', 'adaptive', 'detect_divergence',
                  'ra_dec_order'):
            kwargs.pop(k, None)
        rd, origin = _split_origin(args, kwargs)
        if len(rd) == 1:
            arr = np.asarray(rd[0], dtype=float)
            x, y = self._gwcs.invert(arr[:, 0], arr[:, 1])
            return np.column_stack([np.asarray(x) + origin, np.asarray(y) + origin])
        ra, dec = (np.asarray(v, dtype=float) for v in rd)
        x, y = self._gwcs.invert(ra, dec)
        return np.asarray(x) + origin, np.asarray(y) + origin


def _split_origin(args, kwargs):
    """Peel astropy's trailing ``origin`` argument off ``args``."""
    if 'origin' in kwargs:
        return list(args), int(kwargs.pop('origin'))
    if len(args) >= 2:
        return list(args[:-1]), int(args[-1])
    raise TypeError("WCS transform requires an 'origin' argument (0 or 1)")


def frame_wcs(source, ext='SCI', *, require_gwcs=False, warn_missing=True):
    """WCS for a detector-frame product, GWCS-backed where possible.

    ``source`` may be a filename, an open ``HDUList``, or an already-built WCS
    (returned unchanged, so call sites can accept either).

    Returns a :class:`FrameWCS` when a GWCS is available, otherwise a plain
    ``astropy.wcs.WCS`` read with ``relax=True``.  Set ``require_gwcs=True`` to
    raise instead of falling back -- appropriate for astrometric gates, where
    silently dropping to a 5-8 mas approximation defeats the measurement.
    """
    if isinstance(source, (FrameWCS, astropy_wcs.WCS)):
        return source

    if isinstance(source, fits.HDUList):
        hdulist, filename = source, getattr(source, 'filename', lambda: None)()
    else:
        hdulist, filename = None, os.fspath(source)

    if hdulist is not None:
        header = hdulist[ext].header
    else:
        header = fits.getheader(filename, ext)
    sip = _sip_wcs_from_header(header)

    gw = None
    if _USE_GWCS and filename:
        gw = gwcs_from_file(filename)

    if gw is None:
        msg = (f"no GWCS available for {filename or 'this HDUList'}; falling "
               f"back to the FITS/SIP approximation. SIP is a fitted "
               f"approximation of the true distortion -- positions carry its "
               f"residual, and all_world2pix can fail to converge "
               f"off-footprint.")
        if require_gwcs:
            raise ValueError(msg)
        if warn_missing and _USE_GWCS:
            warnings.warn(msg, MissingGwcsWarning)
        return sip

    return FrameWCS(gw, sip, filename=filename)
