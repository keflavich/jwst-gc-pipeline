"""Affine-anchored corrected sky solution for one cal/crf frame.

Design: the STDGDC forward maps define a superior distortion FIELD but carry
their own (1-pass master-frame) zero-point/scale/rotation, and the frame's
pointing is owned by our VIRAC2-tied offsets machinery -- so the GDC must not
be allowed to move the frame in bulk.  We therefore:

1. sample the frame's EXISTING WCS (CRDS gwcs, or its SIP approximation) on a
   sparse pixel grid;
2. push the same grid through the GDC forward maps;
3. fit a 6-parameter affine mapping GDC-corrected pixels -> tangent-plane
   offsets of the ORIGINAL WCS sky positions.

The least-squares affine (with intercept) preserves the frame's mean position,
scale and rotation by construction -- only the higher-order distortion field
changes.  The per-grid-point residual of that fit IS the CRDS-vs-GDC
distortion delta map, returned for diagnostics.

This is a starlist-level correction (peppar-style), NOT a CRDS reference-file
swap: the frame's WCS on disk is untouched and downstream drizzling is
unaffected.
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from .stdgdc import STDGDC, detector_filter_from_header

__all__ = ['GDCSkySolution', 'load_frame_wcs', 'gdc_sky',
           'MAX_SANE_CORRECTION_PIX']

# Real STDGDC corrections are < ~10 px; the library marks UNMEASURED border
# pixels with 0 (e.g. NRCB4/F212N: the y=2047 row and the x=2047 column above
# y~1535), which reads as a ~2048 px "correction" and, unmasked, poisons the
# affine anchor for the whole frame (measured: affine rms 11.9" instead of
# 0.8 mas).  Any |corrected - raw| above this is treated as an invalid map
# sample.
MAX_SANE_CORRECTION_PIX = 50.0


def load_frame_wcs(cal_file, prefer_gwcs=True):
    """The frame's existing sky solution: gwcs if loadable, else SCI FITS WCS.

    Returns ``(wcs_like, primary_header)`` where ``wcs_like`` has an APE-14
    ``pixel_to_world(x, y)`` (0-based).  The FITS-SIP fallback in cal/crf
    headers is the pipeline's own fit to the gwcs (sub-mas over the detector),
    adequate for anchoring.
    """
    from astropy.io import fits

    with fits.open(cal_file) as hdul:
        header = hdul[0].header.copy()
    if prefer_gwcs:
        try:
            from stdatamodels.jwst import datamodels
            with datamodels.open(cal_file) as model:
                gw = model.meta.wcs
            if gw is not None:
                return gw, header
        except (ImportError, ValueError, AttributeError, OSError) as err:
            import warnings
            warnings.warn(f"gwcs unavailable for {cal_file} ({err}); "
                          f"falling back to SCI FITS-SIP WCS", UserWarning,
                          stacklevel=2)
    from astropy.wcs import WCS
    with fits.open(cal_file) as hdul:
        w = WCS(hdul['SCI'].header)
    return w, header


class GDCSkySolution:
    """GDC distortion field, affine-anchored to a frame's existing WCS.

    Parameters
    ----------
    wcs_like : APE-14 WCS (gwcs or astropy) with ``pixel_to_world(x, y)``
    gdc : STDGDC
    grid_n : int
        Anchor grid is ``grid_n x grid_n`` (default 32) spanning the detector.
    shape : (ny, nx) or None
        Detector shape; defaults to the GDC forward-map shape (2048x2048 for
        the real library).
    """

    def __init__(self, wcs_like, gdc, grid_n=32, shape=None):
        self.wcs = wcs_like
        self.gdc = gdc
        ny, nx = shape if shape is not None else gdc.xgc.shape
        gx = np.linspace(0.0, nx - 1.0, grid_n)
        gy = np.linspace(0.0, ny - 1.0, grid_n)
        gxx, gyy = np.meshgrid(gx, gy)
        self.grid_x = gxx
        self.grid_y = gyy

        sky0 = SkyCoord(wcs_like.pixel_to_world(gxx.ravel(), gyy.ravel())).icrs
        self.center = SkyCoord(wcs_like.pixel_to_world((nx - 1) / 2.0,
                                                       (ny - 1) / 2.0)).icrs
        dlon, dlat = self.center.spherical_offsets_to(sky0)
        xi = dlon.to_value(u.arcsec)
        eta = dlat.to_value(u.arcsec)

        xc, yc = gdc.forward(gxx.ravel(), gyy.ravel())
        valid = self._forward_valid(gxx.ravel(), gyy.ravel(), xc, yc)
        self.n_anchor_invalid = int((~valid).sum())
        if self.n_anchor_invalid:
            import warnings
            warnings.warn(
                f"GDC forward map invalid/unmeasured at "
                f"{self.n_anchor_invalid}/{valid.size} anchor grid points "
                f"(library border holes, e.g. NRCB4/F212N); fitting the "
                f"affine on the valid points only", UserWarning, stacklevel=2)
        if valid.sum() < 64:
            raise ValueError(
                f"GDC forward map valid at only {int(valid.sum())} of "
                f"{valid.size} anchor grid points -- map unusable")

        # 6-parameter affine: [xi, eta] = A @ [xc, yc] + b, least squares,
        # on the VALID map samples only.
        design = np.column_stack([np.ones_like(xc), xc, yc])
        self.coef_x, *_ = np.linalg.lstsq(design[valid], xi[valid], rcond=None)
        self.coef_y, *_ = np.linalg.lstsq(design[valid], eta[valid], rcond=None)

        fit_xi = design @ self.coef_x
        fit_eta = design @ self.coef_y
        # Residual field = affine-anchored GDC minus the original (CRDS) WCS:
        # the distortion delta map.  Mean is 0 by construction (intercept).
        # Invalid map samples are NaN in the delta map.
        dxi = np.where(valid, fit_xi - xi, np.nan)
        deta = np.where(valid, fit_eta - eta, np.nan)
        self.delta_xi_mas = dxi.reshape(gxx.shape) * 1000.0
        self.delta_eta_mas = deta.reshape(gxx.shape) * 1000.0
        self.affine_rms_mas = float(np.sqrt(np.nanmean(
            dxi ** 2 + deta ** 2)) * 1000.0)

    @staticmethod
    def _forward_valid(x, y, xc, yc):
        """True where the GDC forward map sample is finite and sane.

        The STDGDC library stores unmeasured border pixels as 0, which shows
        up as a ~2000 px 'correction'; bilinear interpolation smears garbage
        into their neighbourhood, so anything beyond
        ``MAX_SANE_CORRECTION_PIX`` is invalid.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (np.isfinite(xc) & np.isfinite(yc)
                & (np.abs(xc - x) < MAX_SANE_CORRECTION_PIX)
                & (np.abs(yc - y) < MAX_SANE_CORRECTION_PIX))

    def _tangent(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xc, yc = self.gdc.forward(x, y)
        valid = self._forward_valid(x, y, xc, yc)
        xi = self.coef_x[0] + self.coef_x[1] * xc + self.coef_x[2] * yc
        eta = self.coef_y[0] + self.coef_y[1] * xc + self.coef_y[2] * yc
        return xi, eta, valid

    def gdc_sky(self, x, y):
        """GDC-corrected, affine-anchored SkyCoord for raw 0-based (x, y).

        Stars falling on an invalid/unmeasured region of the GDC map (library
        border holes) FALL BACK to the frame's original WCS position (i.e. a
        zero distortion delta); their count is stored on
        ``self.n_star_fallback`` after each call.
        """
        xi, eta, valid = self._tangent(x, y)
        self.n_star_fallback = int(np.sum(~valid))
        if self.n_star_fallback:
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            inv = ~valid
            sky_orig = SkyCoord(
                self.wcs.pixel_to_world(x[inv], y[inv])).icrs
            oxi, oeta = self.center.spherical_offsets_to(sky_orig)
            xi = np.array(xi, dtype=float, copy=True)
            eta = np.array(eta, dtype=float, copy=True)
            xi[inv] = oxi.to_value(u.arcsec)
            eta[inv] = oeta.to_value(u.arcsec)
        return self.center.spherical_offsets_by(xi * u.arcsec, eta * u.arcsec)

    def delta_map(self):
        """(grid_x, grid_y, delta_xi_mas, delta_eta_mas): CRDS-vs-GDC field."""
        return self.grid_x, self.grid_y, self.delta_xi_mas, self.delta_eta_mas

    def provenance(self):
        """Provenance dict for catalog metadata."""
        return {
            'gdc_file': self.gdc.meta.get('gdc_file', ''),
            'gdc_detector': self.gdc.meta.get('detector', ''),
            'gdc_filter': self.gdc.meta.get('filter', ''),
            'gdc_version_requested': self.gdc.meta.get('version_requested', ''),
            'gdc_filter_fallback': self.gdc.meta.get('filter_fallback', ''),
            'gdc_affine_x': list(np.asarray(self.coef_x, float)),
            'gdc_affine_y': list(np.asarray(self.coef_y, float)),
            'gdc_affine_rms_mas': self.affine_rms_mas,
        }


def gdc_sky(x_fit, y_fit, cal_file, root=None, version='auto',
            fallback_filter=None, grid_n=32):
    """GDC-corrected SkyCoord for raw 0-based pixel positions in ``cal_file``.

    Convenience one-shot: loads the frame's WCS, resolves (detector, filter)
    from the primary header, loads the STDGDC and anchors it.  Returns
    ``(SkyCoord, GDCSkySolution)``.
    """
    wcs_like, header = load_frame_wcs(cal_file)
    detector, filt = detector_filter_from_header(header)
    gdc = STDGDC.load(detector, filt, root=root, version=version,
                      fallback_filter=fallback_filter)
    sol = GDCSkySolution(wcs_like, gdc, grid_n=grid_n)
    return sol.gdc_sky(x_fit, y_fit), sol
