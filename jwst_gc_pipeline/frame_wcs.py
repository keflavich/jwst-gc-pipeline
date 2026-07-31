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

**1. The fit residual.**  ``all_pix2world`` through the SIP header differs from
the GWCS by, on a brick F182M nrca1 ``_crf`` (20k random pixels), a median of
0.82 mas and a max of 5.1 mas -- equivalently 26 and 165 **millipixels** at the
31.2 mas/px SW plate scale.  Those are the *same* number in two units, not two
separate defects: SIP's own forward->inverse round trip closes to 0.000
millipixels, so the ``AP_*``/``BP_*`` inverse fit and its solver add nothing
measurable.  There is one error -- the degree of the polynomial -- and it shows
up in whichever direction you look.

It is position-dependent and a different surface per detector *and* per filter,
so no bulk tie removes it; that is exactly the shape of error the astrometric
gates look for (2 mas m2 consensus, 5 mas m7 cross-filter, 30 mas overlap).
With the tight fit this package now writes, the residual is ~0.00 mas; on every
frame written before that fix it is 5-8 mas.

**2. Off-footprint behaviour -- genuinely independent, and worse than a crash.**
SIP's inverse is solved iteratively and diverges outside the frame.  What
happens then depends on the call path, and none of the three is safe:

===========================================  ==========================================
call                                         result for a point ~2 deg off-frame
===========================================  ==========================================
``all_world2pix(ra, dec, 0)``                raises ``NoConvergence`` (the W51 m8 abort)
``all_world2pix(ra, dec, 0, quiet=True)``    returns ``[-5438745, -123680]`` -- **finite
                                             garbage, with no warning at all**
``world_to_pixel(skycoord)``                 same finite garbage, with a warning
===========================================  ==========================================

The silent finite value is the dangerous one: it propagates into a catalog
instead of stopping the run.  The GWCS returns ``NaN`` on all paths -- the
correct sentinel, which the callers' in-bounds tests already drop.

**Cost.**  Not a wash in the way you might guess: forward, GWCS is ~1.3x slower
(50k transforms, best of 7: 8.4 ms vs 6.4 ms); inverse, GWCS is ~1.1x *faster*
(14.0 ms vs 15.7 ms), because the analytic inverse beats SIP's iterative solve.
Absolute numbers vary with machine load; the directions are stable.

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

__all__ = ['FrameWCS', 'frame_wcs', 'gwcs_from_file', 'has_gwcs',
           'wcs_provenance_cards', 'MissingGwcsWarning', 'SlicedFrameWCSWarning']

#: Set to '0' to fall back to the FITS/SIP WCS everywhere (debugging only --
#: it reinstates the 5-8 mas forward error and the NoConvergence failure mode).
_USE_GWCS = os.environ.get('FRAME_WCS_USE_GWCS', '1') != '0'


class MissingGwcsWarning(UserWarning):
    """A detector-frame product had no GWCS, so the SIP approximation was used."""


class SlicedFrameWCSWarning(UserWarning):
    """A FrameWCS was sliced, degrading it to the FITS/SIP approximation."""


#: Run-level tally of how frame WCSes actually resolved, so a product can record
#: whether it was built by reading GWCSes or fell back to SIP anywhere.  See
#: :func:`wcs_provenance_cards`.
_RESOLUTION_TALLY = {'gwcs': 0, 'sip': 0}


def wcs_provenance_cards():
    """FITS cards recording how this process resolved its frame WCSes.

    A catalog built before the GWCS-first change carries **no** ``WCSSRC`` card
    at all, so the card's presence-and-value is what makes a product
    self-identifying across the change (the alternative -- inferring it from the
    build date -- is exactly what we do not want at staging time).

    ``('WCSSRC', 'GWCS'|'FITS-SIP'|'MIXED'|'NONE')`` plus the fallback count.
    """
    n_g, n_s = _RESOLUTION_TALLY['gwcs'], _RESOLUTION_TALLY['sip']
    if n_g and n_s:
        src = 'MIXED'
    elif n_g:
        src = 'GWCS'
    elif n_s:
        src = 'FITS-SIP'
    else:
        src = 'NONE'
    return [
        ('WCSSRC', src, 'frame WCS source used to build this product'),
        ('WCSNGW', n_g, 'frames whose GWCS was read'),
        ('WCSNSIP', n_s, 'frames that fell back to the SIP approximation'),
    ]


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


def _bbox_intervals(gw):
    """``gw``'s bounding box as [(lo, hi), (lo, hi)], or None."""
    bb = getattr(gw, 'bounding_box', None)
    if bb is None:
        return None
    try:
        intervals = [(float(lo), float(hi)) for lo, hi in bb]
    except (TypeError, ValueError):
        return None
    return intervals if len(intervals) == 2 else None


def _bbox_center_and_extent(gw):
    """``(nx, ny)`` implied by ``gw``'s bounding box, or None."""
    intervals = _bbox_intervals(gw)
    if intervals is None:
        return None
    (x0, x1), (y0, y1) = intervals
    return (x1 - x0), (y1 - y0)


def gwcs_from_file(filename, use_cache=True):
    """The GWCS of ``filename``, via ``stdatamodels.jwst.datamodels``, or None.

    This deliberately uses the **upstream** datamodels reader rather than
    parsing the ASDF extension by hand.  ``meta.wcs`` is the documented,
    STScI-maintained way to get a JWST product's WCS; a hand-rolled reader has
    to reimplement ASDF-in-FITS block resolution, extension handling and lazy
    loading, and would need re-checking against every ``asdf``/``gwcs``/``jwst``
    release.  Same rule as ``adjust_wcs`` in ASTROMETRY_WCS_CORRECTION_FLOW.md:
    use STScI tools.

    Cost is not a reason to hand-roll -- ``datamodels.open`` + ``meta.wcs`` is
    ~0.15 s per frame, which the memoisation below removes for repeat reads.

    Results are memoised on ``(path, mtime, size)``, so a file rewritten in
    place (``fix_alignment`` overwrites its input) invalidates its own entry.
    """
    try:
        import stdatamodels.jwst.datamodels as _dm
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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # memmap=False so nothing in the returned transform can depend on a
            # mapping of a file we are about to close.
            with _dm.open(filename, memmap=False) as model:
                gw = getattr(model.meta, 'wcs', None)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # not a JWST datamodel product, or it carries no WCS
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
        # Only reached for attributes not defined on the class.  Dunders must
        # NOT be delegated: `pickle` looks up __reduce_ex__/__getstate__, which
        # would be answered by the wrapped FITS WCS -- pickling then recursed
        # forever, and copy.deepcopy silently returned a plain astropy WCS,
        # i.e. every worker process would quietly fall back to SIP.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return getattr(self._fits, name)

    # Explicit state handling so multiprocessing / dask workers get a FrameWCS
    # back, not a bare FITS WCS.  gwcs objects are picklable (astropy models).
    def __reduce__(self):
        return (_rebuild_frame_wcs, (self._gwcs, self._fits, self.filename))

    def __copy__(self):
        return FrameWCS(self._gwcs, self._fits, filename=self.filename)

    def __deepcopy__(self, memo):
        import copy as _copy
        return FrameWCS(_copy.deepcopy(self._gwcs, memo),
                        _copy.deepcopy(self._fits, memo),
                        filename=self.filename)

    def __getitem__(self, item):
        # astropy slicing cannot carry a GWCS, so a sliced FrameWCS silently
        # becomes a SIP WCS again.  Warn rather than leave "only used for
        # display" enforced by convention alone.
        warnings.warn(
            f"slicing a FrameWCS{' (' + os.path.basename(self.filename) + ')' if self.filename else ''} "
            f"returns a plain FITS/SIP WCS -- astropy's slicing machinery "
            f"cannot carry a GWCS. Positions from the slice carry the SIP fit "
            f"residual. Use it for display/cutout bookkeeping only; for "
            f"astrometry slice the pixel coordinates instead and transform "
            f"with the unsliced FrameWCS.", SlicedFrameWCSWarning, stacklevel=2)
        return self._fits[item]

    def calc_footprint(self, header=None, undistort=True, axes=None, center=True):
        """Sky corners from the **GWCS**, so the footprint and the transforms
        agree.

        Delegating this to the FITS WCS would leave a FrameWCS whose
        ``pixel_to_world`` and whose ``calc_footprint`` come from *different*
        representations, differing by the SIP fit residual -- harmless for
        footprint-intersection gating, but it looks exactly like a bug to
        whoever eventually compares a footprint corner against a transformed
        corner.  Falls back to the FITS WCS when the array extent is unknown.
        """
        if axes is not None:
            nx, ny = float(axes[0]), float(axes[1])
        else:
            bb = _bbox_center_and_extent(self._gwcs)
            if bb is None:
                return self._fits.calc_footprint(header=header,
                                                 undistort=undistort,
                                                 axes=axes, center=center)
            nx, ny = bb
        # astropy's convention: corners of the array, `center` selects whether
        # they are pixel centres (0..n-1) or outer edges (-0.5..n-0.5).
        lo, hix, hiy = (0.0, nx - 1.0, ny - 1.0) if center else (-0.5, nx - 0.5, ny - 0.5)
        xs = np.array([lo, lo, hix, hix])
        ys = np.array([lo, hiy, hiy, lo])
        ra, dec = self._gwcs(xs, ys)
        return np.column_stack([ra, dec])

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


def _rebuild_frame_wcs(gwcs_obj, fits_wcs, filename):
    """Unpickle helper for :class:`FrameWCS` (module-level so it is picklable)."""
    return FrameWCS(gwcs_obj, fits_wcs, filename=filename)


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
        _RESOLUTION_TALLY['sip'] += 1
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

    _RESOLUTION_TALLY['gwcs'] += 1
    return FrameWCS(gw, sip, filename=filename)
