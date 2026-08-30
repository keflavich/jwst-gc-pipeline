"""The in-driver check that ``fix_alignment`` moved the WCS it was asked to.

One implementation for all three instrument drivers.  There were three, and
they had diverged in a way that made two of them report nothing at all.

Every driver printed the sky position of pixel ``(1024, 1024)`` -- the NIRCam
detector centre -- from the new GWCS, from ``meta.oldwcs`` when present, and
from the FITS header, plus the separations between them.  Those separations
are the only in-driver evidence that the shift the offsets table asked for is
the shift the frame received.

``(1024, 1024)`` is not on a MIRI detector.  MIRI imaging arrays are
``(1024, 1032)`` and the GWCS bounding box stops at ``x1 = 1023.5``, so the
GWCS returns ``NaN`` there and every separation the check prints is ``nan``
on every MIRI frame::

    data.shape (ny, nx) = (1024, 1032)
    pix(1024,1024) ->  (nan, nan)
    array center   ->  (290.94196378, 14.51179787)

A MIRI reduction that applied the wrong shift, applied none, or wrote a SIP
header disagreeing with its GWCS printed exactly what a correct one printed.
NIRCam's copy was fixed to use the array centre from ``data.shape`` (which is
also subarray-safe) and gained the FITS-vs-GWCS discrepancy check that
ASTROMETRY RULE #2 rests on; neither travelled to the siblings.

The check WARNS and does not raise: frames written before the tight-SIP fix
legitimately trip the discrepancy tolerance, and this runs inside a reduction
whose products are still wanted.
"""
import os
import warnings

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from stdatamodels.jwst.datamodels import ImageModel

__all__ = ['check_wcs']


def check_wcs(fn):
    """Print the GWCS / old-GWCS / FITS positions of ``fn``'s array centre.

    The array centre comes from ``data.shape``, so it is on the detector for
    every instrument and every subarray.  A full-array MIRI frame evaluated at
    the hardcoded NIRCam centre returns ``NaN``, which reads exactly like a
    clean result.
    """
    if not os.path.exists(fn):
        print(f"COULD NOT CHECK WCS FOR {fn}: does not exist")
        return

    print(f"Checking WCS of {fn}")
    fa = ImageModel(fn)
    wcsobj = fa.meta.wcs
    ny, nx = fa.data.shape[0], fa.data.shape[1]
    _fy, _fx = ny / 2.0, nx / 2.0   # array center (subarray- and MIRI-safe)
    print(f"fa['meta']['wcs'] crval={wcsobj.to_fits()[0]['CRVAL1']}, "
          f"{wcsobj.to_fits()[0]['CRVAL2']}, "
          f"{wcsobj.forward_transform.param_sets[-1]}")
    new_center = wcsobj.pixel_to_world(_fx, _fy)
    print(f"new pixel_to_world({_fx},{_fy}) = {new_center}")
    if 'oldwcs' in fa.meta:
        oldwcsobj = fa.meta.oldwcs
        print(f"fa['meta']['oldwcs'] crval={oldwcsobj.to_fits()[0]['CRVAL1']}, "
              f"{oldwcsobj.to_fits()[0]['CRVAL2']}, "
              f"{oldwcsobj.forward_transform.param_sets[-1]}")
        old_center = oldwcsobj.pixel_to_world(_fx, _fy)
        print(f"old pixel_to_world({_fx},{_fy}) = {old_center}, sep from new "
              f"GWCS={old_center.separation(new_center).to(u.arcsec)}")
    fa.close()

    # FITS header
    fh = fits.open(fn)
    print(f"CRVAL1={fh[1].header['CRVAL1']}, CRVAL2={fh[1].header['CRVAL2']}")
    if 'OLCRVAL1' in fh[1].header:
        print(f"OLCRVAL1={fh[1].header['OLCRVAL1']}, "
              f"OLCRVAL2={fh[1].header['OLCRVAL2']}")
    if 'RAOFFSET' in fh[1].header:
        print("RA, DE offset: ", fh[1].header['RAOFFSET'],
              fh[1].header['DEOFFSET'])
    # relax=True: a header whose CTYPE lost the '-SIP' suffix still carries
    # A_*/B_*; without relax the distortion is silently dropped.
    ww = WCS(fh[1].header, relax=True)
    fits_center = ww.pixel_to_world(_fx, _fy)
    print(f"FITS pixel_to_world({_fx},{_fy}) = {fits_center}, sep from new "
          f"GWCS={fits_center.separation(new_center).to(u.arcsec)}")
    # The center agrees by construction even when the SIP fit is poor -- the
    # distortion residual lives at the corners.  Measure the whole array (this
    # is what caught the 0.25 px to_fits() default).
    from jwst_gc_pipeline.reduction.fits_wcs_sync import (
        fits_gwcs_discrepancy_mas, FITS_GWCS_TOL_MAS)
    _max_mas, _med_mas = fits_gwcs_discrepancy_mas(
        fh[1].header, wcsobj, (ny, nx))
    print(f"FITS/SIP vs GWCS over the array: max {_max_mas:.4f} mas, "
          f"median {_med_mas:.4f} mas")
    if _max_mas > FITS_GWCS_TOL_MAS:
        warnings.warn(
            f"{fn}: the FITS/SIP header disagrees with the GWCS by up to "
            f"{_max_mas:.3f} mas (> {FITS_GWCS_TOL_MAS} mas). This frame "
            f"predates the tight-SIP fix; anything reading the FITS header "
            f"instead of the GWCS carries that position-dependent error. "
            f"Regenerate it, or read the GWCS.")
    fh.close()
