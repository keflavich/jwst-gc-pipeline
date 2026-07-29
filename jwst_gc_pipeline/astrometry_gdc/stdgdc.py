"""Loader/evaluator for Jay Anderson's STDGDC NIRCam distortion-correction maps.

File format (verified on ``STDGDC_NRCB1_F212N.fits`` and ``STDGDC_NRCBL_F277W.fits``,
library mirror at ``/orange/adamginsburg/jwst/distortion/jayander_stdgdc/NIRCam``):

- HDU0: metadata only. ``XGC_0``/``YGC_0`` are the corrected-frame coordinates
  (1-based pixel units) of the FIRST pixel of the reverse maps;
  ``NDIM_XCG``/``NDIM_YCG`` their dimensions.  Verified relation:
  ``LTV1 = 1 - XGC_0`` and ``LTV2 = 1 - YGC_0`` (NRCB1/F212N: XGC_0=0, YGC_0=-4,
  LTV=1/5; NRCBL/F277W: XGC_0=-11, YGC_0=-12, LTV=12/13).
- HDU1 ``XGC``: forward X map, 2048x2048, BSCALE 1e-5.  ``XGC[j, i]`` (numpy
  ``[y, x]``, 0-based) is the distortion-CORRECTED X position -- in Jay's
  1-BASED pixel convention -- of the raw 1-based detector pixel
  ``(X=i+1, Y=j+1)``.  Verified against the 5x5 spot table embedded in the
  HDU1/2 COMMENT cards ("X-FORWARD DISTORTION MAPPING" at X=0001..2048 /
  Y=0001..2048): e.g. XGC[0, 0] = -0.955 = table (X=0001, Y=0001);
  XGC[2047, 2047] = 2041.383 = table (X=2048, Y=2048).
- HDU2 ``YGC``: forward Y map, same convention.
- HDU3 ``MGC``: pixel-area correction in MAGNITUDES (BSCALE 1e-4); the value is
  ADDED to the instrumental magnitude (peppar convention, see below).
- HDU4/5 ``XCG``/``YCG``: reverse maps on the corrected-frame grid, dimensions
  ``NDIM_XCG x NDIM_YCG`` (e.g. 2056x2057) with ``LTV1``/``LTV2`` offsets.
  A corrected 1-based position ``(xc, yc)`` samples the reverse arrays at
  0-based index ``[yc + LTV2 - 1, xc + LTV1 - 1]``; the value is the raw
  1-based position.  Measured forward->reverse round-trip on NRCB1/F212N:
  max |dx|,|dy| ~ 5e-5 pix (limited by the BSCALE 1e-5 quantisation).
- ``v2`` variants (``SWC/<FILT>/v2/STDGDC_<DET>_<FILT>_2.fits``) have the same
  6-HDU structure (no COMMENT spot tables, no DATE); available for
  F182M/F200W/F212N only in the current mirror.

Application convention -- extracted from peppar (Matt Hosek), local checkout
``/blue/adamginsburg/adamginsburg/repos/peppar``, ``peppar/combo_starlists.py``
function ``apply_distortion_to_starlists`` (lines 876-1043):

- Maps are documented "[Y,X]" oriented (combo_starlists.py:894) and are sampled
  with ``scipy.interpolate.RegularGridInterpolator`` built on 0-based integer
  grids ``np.arange(0, shape)`` (combo_starlists.py:926-935) and evaluated at
  the RAW starlist positions directly, first axis y:
  ``xcor_interp((starlist['y'], starlist['x']))`` (combo_starlists.py:964-966).
  peppar starlists come from photutils ``PSFPhotometry`` ``x_fit``/``y_fit``
  (peppar/peppar.py:37,43,1063-1064), i.e. 0-BASED pixel positions.  Because
  map index i corresponds to 1-based raw pixel X=i+1, indexing the map at the
  0-based position IS the correct sample point.
- Interpolation: ``method='cubic'`` by default (combo_starlists.py:877,930).
  We default to bilinear (``method='linear'``); on these smooth 1-px-sampled
  maps the two differ well below the 1e-5-pix file quantisation matters
  (measured < 1e-3 pix), and bilinear is exact at grid nodes either way.
- The corrected values REPLACE ``x``/``y`` and the originals are kept as
  ``x_raw``/``y_raw``/``m_raw`` (combo_starlists.py:970-974).  peppar does NOT
  subtract 1 from the returned (1-based-convention) values -- its corrected
  frame therefore carries a benign uniform +1-pixel offset that flystar's
  polynomial transform absorbs.  This module DOES convert back to 0-based
  (``forward()`` returns map value - 1) so raw and corrected positions share
  the same 0-based convention; any residual uniform offset is absorbed by the
  affine anchoring in ``gdc_wcs.py``.
- The MGC magnitude correction IS applied by peppar:
  ``starlist['m'] += mag_offset`` (combo_starlists.py:966,975).
- The REVERSE maps (HDU4/5) are NOT used by peppar -- commented out with the
  note "Testing shows that reverse solutions don't actually reverse solution?"
  (combo_starlists.py:937-947).  With the LTV convention above they DO reverse
  the forward maps to ~5e-5 pix; peppar's commented-out code omitted the LTV
  offsets, which is the likely reason it appeared broken.
- ``XGC_0``/``YGC_0`` are not read by peppar at all.
- File resolution (combo_starlists.py:896-919): filter -> ``SWC/``/``LWC/``
  subdir; LWC has ONLY F277W solutions (``STDGDC_NRCAL_F277W.fits`` /
  ``STDGDC_NRCBL_F277W.fits``, det NRCA5->NRCAL, NRCB5->NRCBL, lines 905-908
  with a loud "only F277W distortion solution available for LW" warning);
  SWC files live in ``SWC/<FILT>/`` and a ``v2/`` subdir is PREFERRED when it
  exists (lines 910-916, ``glob('STDGDC*<DET>*.fits')`` -- the glob is
  detector-position agnostic, which matters because the library mixes
  ``STDGDC_<DET>_<FILT>.fits`` and ``STDGDC_<FILT>_<DET>.fits`` naming).
  ``distortion_dir`` points at the directory containing ``SWC/``/``LWC/``
  (docstring lines 105-107).

Public API: :class:`STDGDC` -- ``STDGDC.load(detector, filter)`` then
``forward(x, y)`` / ``reverse(x, y)`` / ``pixel_area_mag(x, y)``, all 0-based
in AND out.
"""
import glob
import os
import warnings

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

__all__ = ['STDGDC', 'GDCFileNotFoundError', 'DEFAULT_GDC_ROOT',
           'resolve_gdc_file', 'detector_filter_from_header']

#: Default library root: the directory containing ``SWC/`` and ``LWC/``.
DEFAULT_GDC_ROOT = '/orange/adamginsburg/jwst/distortion/jayander_stdgdc/NIRCam'

#: Environment variable overriding the library root.
GDC_ROOT_ENV = 'JWST_GDC_ROOT'

# SW detector short names accepted by the library file names.
_SW_DETECTORS = ('NRCA1', 'NRCA2', 'NRCA3', 'NRCA4',
                 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4')
# Long-wavelength detector aliases -> library short name
# (peppar combo_starlists.py:905-908 maps NRCA5/NRCB5 the same way).
_LW_DETECTOR_ALIASES = {'NRCA5': 'NRCAL', 'NRCALONG': 'NRCAL', 'NRCAL': 'NRCAL',
                        'NRCB5': 'NRCBL', 'NRCBLONG': 'NRCBL', 'NRCBL': 'NRCBL'}


class GDCFileNotFoundError(FileNotFoundError):
    """No STDGDC file exists for the requested (detector, filter, version)."""


def _gdc_root(root=None):
    root = root or os.environ.get(GDC_ROOT_ENV) or DEFAULT_GDC_ROOT
    # Robustness: accept a root either containing SWC/LWC directly or one
    # extra "NIRCam" level up (the wget mirror keeps NIRCam/SWC, NIRCam/LWC).
    if os.path.isdir(os.path.join(root, 'SWC')):
        return root
    nircam = os.path.join(root, 'NIRCam')
    if os.path.isdir(os.path.join(nircam, 'SWC')):
        return nircam
    raise GDCFileNotFoundError(
        f"GDC root {root!r} has no SWC/ subdirectory (checked {root!r} and "
        f"{nircam!r}); set ${GDC_ROOT_ENV} or pass root=")


def _normalize_detector(detector):
    det = str(detector).upper()
    if det in _SW_DETECTORS:
        return det, 'SWC'
    if det in _LW_DETECTOR_ALIASES:
        return _LW_DETECTOR_ALIASES[det], 'LWC'
    raise ValueError(f"Unrecognized NIRCam detector {detector!r}")


def resolve_gdc_file(detector, filtername, root=None, version='auto'):
    """Path of the STDGDC FITS file for (detector, filter).

    ``version``: ``'auto'`` (peppar behaviour: prefer the ``v2/`` file for this
    detector when present, else the top-level v1 file), ``'v1'``, or ``'v2'``.
    Raises :class:`GDCFileNotFoundError` when no file matches (e.g. F187N,
    which has no STDGDC solution).
    """
    if version not in ('auto', 'v1', 'v2'):
        raise ValueError(f"version must be 'auto', 'v1' or 'v2', got {version!r}")
    root = _gdc_root(root)
    det, channel = _normalize_detector(detector)
    filt = str(filtername).upper()

    if channel == 'LWC':
        # Only F277W solutions exist for LW (peppar combo_starlists.py:902-908).
        if filt != 'F277W':
            raise GDCFileNotFoundError(
                f"No LW STDGDC solution for {filt}; the library provides "
                f"F277W only (detector {det})")
        candidates = [os.path.join(root, 'LWC', f'STDGDC_{det}_F277W.fits')]
    else:
        filt_dir = os.path.join(root, 'SWC', filt)
        if not os.path.isdir(filt_dir):
            raise GDCFileNotFoundError(
                f"No STDGDC directory for SW filter {filt} under {filt_dir!r}")
        # Library mixes STDGDC_<DET>_<FILT> and STDGDC_<FILT>_<DET> naming, so
        # glob on the detector token (peppar combo_starlists.py:914,916).
        v1 = sorted(glob.glob(os.path.join(filt_dir, f'STDGDC*{det}*.fits')))
        v2 = sorted(glob.glob(os.path.join(filt_dir, 'v2', f'STDGDC*{det}*.fits')))
        if version == 'v1':
            candidates = v1
        elif version == 'v2':
            candidates = v2
        else:
            candidates = v2 or v1
    candidates = [c for c in candidates if os.path.isfile(c)]
    if len(candidates) == 0:
        raise GDCFileNotFoundError(
            f"No STDGDC file for detector={det} filter={filt} "
            f"version={version} under {root!r}")
    if len(candidates) > 1:
        raise GDCFileNotFoundError(
            f"Ambiguous STDGDC match for detector={det} filter={filt}: {candidates}")
    return candidates[0]


def detector_filter_from_header(header):
    """(detector, filter) for GDC lookup from a cal/crf primary header.

    Uses ``DETECTOR`` plus ``FILTER``, falling back to ``PUPIL`` when the
    filter wheel holds CLEAR (pupil-wheel filters like F162M/F164N).
    """
    detector = header['DETECTOR']
    filt = str(header.get('FILTER', '')).upper()
    pupil = str(header.get('PUPIL', '')).upper()
    if (not filt or filt.startswith('CLEAR')) and pupil.startswith('F'):
        filt = pupil
    if not filt:
        raise ValueError("Header has neither a usable FILTER nor PUPIL")
    return detector, filt


class STDGDC:
    """One STDGDC solution: forward/reverse pixel maps + pixel-area mags.

    All public methods take and return 0-BASED pixel coordinates (python /
    photutils convention); the 1-based bookkeeping of the underlying maps is
    handled internally (see module docstring).
    """

    _cache = {}

    def __init__(self, xgc, ygc, mgc, xcg=None, ycg=None, ltv1=None, ltv2=None,
                 meta=None, method='linear'):
        self.xgc = np.asarray(xgc, dtype=float)
        self.ygc = np.asarray(ygc, dtype=float)
        self.mgc = np.asarray(mgc, dtype=float)
        self.xcg = None if xcg is None else np.asarray(xcg, dtype=float)
        self.ycg = None if ycg is None else np.asarray(ycg, dtype=float)
        self.ltv1 = ltv1
        self.ltv2 = ltv2
        self.meta = dict(meta or {})
        self.method = method
        if self.xgc.shape != self.ygc.shape or self.xgc.shape != self.mgc.shape:
            raise ValueError("XGC/YGC/MGC shapes differ")

        ny, nx = self.xgc.shape
        axes = (np.arange(ny, dtype=float), np.arange(nx, dtype=float))
        interp_kw = dict(method=method, bounds_error=False, fill_value=np.nan)
        # Data are [y, x]; first interpolation axis is y (peppar
        # combo_starlists.py:894,964).
        self._fwd_x = RegularGridInterpolator(axes, self.xgc, **interp_kw)
        self._fwd_y = RegularGridInterpolator(axes, self.ygc, **interp_kw)
        self._mag = RegularGridInterpolator(axes, self.mgc, **interp_kw)
        if self.xcg is not None:
            if self.ltv1 is None or self.ltv2 is None:
                raise ValueError("Reverse maps require ltv1/ltv2")
            rny, rnx = self.xcg.shape
            raxes = (np.arange(rny, dtype=float), np.arange(rnx, dtype=float))
            self._rev_x = RegularGridInterpolator(raxes, self.xcg, **interp_kw)
            self._rev_y = RegularGridInterpolator(raxes, self.ycg, **interp_kw)
        else:
            self._rev_x = self._rev_y = None

    # ------------------------------------------------------------------ load
    @classmethod
    def from_file(cls, path, method='linear'):
        with fits.open(path) as hdul:
            if len(hdul) != 6:
                raise ValueError(
                    f"{path}: expected the 6-HDU STDGDC layout, got {len(hdul)}")
            meta = {'gdc_file': os.path.abspath(path)}
            for key in ('NAME_GDC', 'XGC_0', 'YGC_0', 'DATE'):
                if key in hdul[0].header:
                    meta[key.lower()] = hdul[0].header[key]
            return cls(hdul[1].data, hdul[2].data, hdul[3].data,
                       xcg=hdul[4].data, ycg=hdul[5].data,
                       ltv1=hdul[4].header['LTV1'], ltv2=hdul[4].header['LTV2'],
                       meta=meta, method=method)

    @classmethod
    def load(cls, detector, filtername, root=None, version='auto',
             fallback_filter=None, method='linear'):
        """Load (with caching) the STDGDC for ``(detector, filtername)``.

        ``fallback_filter``: OFF by default.  If the exact filter has no GDC
        (e.g. F187N), pass a designated neighbour (e.g. ``'F212N'``) to use its
        solution instead -- a loud warning is emitted and the substitution is
        recorded in ``meta['filter_fallback']``.
        """
        try:
            path = resolve_gdc_file(detector, filtername, root=root, version=version)
            used_filter = str(filtername).upper()
            fallback_from = None
        except GDCFileNotFoundError:
            if fallback_filter is None:
                raise
            warnings.warn(
                f"STDGDC: no solution for filter {filtername} on {detector}; "
                f"FALLING BACK to neighbour filter {fallback_filter}. The "
                f"distortion field is filter-dependent at the ~sub-mas level; "
                f"treat results as approximate.", UserWarning, stacklevel=2)
            path = resolve_gdc_file(detector, fallback_filter, root=root,
                                    version=version)
            used_filter = str(fallback_filter).upper()
            fallback_from = str(filtername).upper()
        key = (os.path.abspath(path), method)
        if key not in cls._cache:
            gdc = cls.from_file(path, method=method)
            gdc.meta.update(detector=str(detector).upper(), filter=used_filter,
                            version_requested=version)
            if fallback_from is not None:
                gdc.meta['filter_fallback'] = f'{fallback_from}->{used_filter}'
            cls._cache[key] = gdc
        return cls._cache[key]

    # ------------------------------------------------------------- transforms
    def forward(self, x, y):
        """Distortion-corrected 0-based (x, y) for raw 0-based detector (x, y).

        Raw 0-based position (x, y) == raw 1-based (x+1, y+1) == map index
        [y, x]; the stored value is the corrected 1-based position, so subtract
        1 to return to the 0-based convention.  Out-of-detector inputs -> NaN.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.stack(np.broadcast_arrays(y, x), axis=-1)
        return self._fwd_x(pts) - 1.0, self._fwd_y(pts) - 1.0

    def reverse(self, x, y):
        """Raw 0-based detector (x, y) for distortion-corrected 0-based (x, y).

        Corrected 1-based position p samples the reverse map at 0-based index
        ``p + LTV - 1``; with 0-based input that is ``x + LTV1`` / ``y + LTV2``.
        The stored value is the raw 1-based position (subtract 1).
        """
        if self._rev_x is None:
            raise ValueError("This STDGDC instance has no reverse maps")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.stack(np.broadcast_arrays(y + self.ltv2, x + self.ltv1), axis=-1)
        return self._rev_x(pts) - 1.0, self._rev_y(pts) - 1.0

    def pixel_area_mag(self, x, y):
        """Pixel-area magnitude correction at raw 0-based (x, y).

        peppar ADDS this to the instrumental magnitude
        (``starlist['m'] += mag_offset``, combo_starlists.py:966,975).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.stack(np.broadcast_arrays(y, x), axis=-1)
        return self._mag(pts)

    def __repr__(self):
        tag = self.meta.get('gdc_file', '<from arrays>')
        return f"STDGDC({os.path.basename(str(tag))}, method={self.method!r})"
