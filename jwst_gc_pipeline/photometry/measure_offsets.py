
import numpy as np
from astropy import units as u
from astropy import stats
from astropy.coordinates import SkyCoord
import os


class DenseNNMedianAstrometryError(RuntimeError):
    """Raised when the FORBIDDEN dense-nearest-neighbour-median astrometry method
    is attempted against a dense reference catalogue.  See
    ``assert_sparse_reference_for_nn_median`` and ASTROMETRY_WCS_CORRECTION_FLOW.md."""


_DENSE_NN_MEDIAN_MESSAGE = (
    "\n"
    "*************************************************************************\n"
    "*** FORBIDDEN: dense-nearest-neighbour-median astrometry ***\n"
    "*************************************************************************\n"
    "You tried to compute/apply a bulk astrometric offset as the MEDIAN (or mean)\n"
    "of nearest-neighbour matches (match_to_catalog_sky / search_around_sky)\n"
    "against a DENSE reference catalogue.\n"
    "  context           : {context}\n"
    "  reference sources : {n}\n"
    "  reference NN spacing (median): {med_nn:.3f}\"  <  dense threshold {mr:.3f}\"\n"
    "\n"
    "THIS METHOD IS KNOWN-BROKEN.  When the true shift exceeds the reference's\n"
    "nearest-neighbour spacing, NN pairs the WRONG stars and the median COLLAPSES\n"
    "toward ~0 (or a spurious value).  It has SILENTLY CORRUPTED brick-1182\n"
    "astrometry TWICE (mosaic left ~1.6-4\" off the absolute frame).\n"
    "\n"
    "USE INSTEAD:\n"
    "  - 2D offset-HISTOGRAM stacking: histogram ALL pairwise offsets within ~3\",\n"
    "    take the peak (robust no matter how large the shift), OR\n"
    "  - a SPARSE reference: the Gaia-only subset (source==b'GaiaDR3'), never the\n"
    "    full dense VIRAC2/VVV/GNS catalogue.\n"
    "See ASTROMETRY_WCS_CORRECTION_FLOW.md and the brick-1182 memory.\n"
    "*************************************************************************\n"
)


# A reference whose median nearest-neighbour spacing is below this is "dense": a
# plausible uncorrected JWST frame error (up to ~2"; brick-1182 was 1.9") can
# exceed the spacing, so NN matching pairs the WRONG star and the median offset is
# spurious.  Calibration (brick gaia_virac2 refcat): full/VIRAC2 medNN ~1.15"
# (DENSE, the catalog that corrupted 1182); the Gaia-only subset medNN ~5.72"
# (SPARSE, safe).  3" cleanly separates them.  Override via env for diagnostics.
DENSE_NN_MEDIAN_MIN_SPACING_ARCSEC = float(
    os.environ.get('DENSE_NN_MEDIAN_MIN_SPACING_ARCSEC', 3.0))


def assert_sparse_reference_for_nn_median(reference_coordinates, match_radius, *,
                                          context="", min_spacing_factor=3.0,
                                          min_nn_spacing=None, sample=3000):
    """Guard the FORBIDDEN dense-NN-median astrometry method.

    Raise ``DenseNNMedianAstrometryError`` when ``reference_coordinates`` is DENSE:
    its own median nearest-neighbour spacing is below ``min_nn_spacing`` (default
    ``DENSE_NN_MEDIAN_MIN_SPACING_ARCSEC`` = 3"), or below ``min_spacing_factor`` x
    ``match_radius`` (whichever is larger).  In that regime nearest-neighbour /
    search-around-sky median offset estimation returns a SPURIOUS shift (it
    collapses toward ~0 when the true offset exceeds the NN spacing).  Sparse
    references (e.g. a Gaia-only subset, medNN ~5.7") pass through unchanged, so the
    only thing this forbids is exactly the method that keeps corrupting the GC
    mosaics.  Never silently downgrade -- callers must switch to offset-histogram
    stacking or a sparse reference.
    """
    rc = reference_coordinates
    try:
        n = int(len(rc))
    except TypeError:
        return
    if n < 3:
        return
    step = max(1, n // int(sample))
    rc_sample = rc[::step]
    # nthneighbor=2: the reference point's own nearest OTHER reference source.
    _, sep2, _ = rc_sample.match_to_catalog_sky(rc, nthneighbor=2)
    med_nn = float(np.nanmedian(sep2.to(u.arcsec).value))
    mr = match_radius.to(u.arcsec).value if hasattr(match_radius, 'to') else float(match_radius)
    floor = DENSE_NN_MEDIAN_MIN_SPACING_ARCSEC if min_nn_spacing is None else (
        min_nn_spacing.to(u.arcsec).value if hasattr(min_nn_spacing, 'to') else float(min_nn_spacing))
    threshold = max(floor, min_spacing_factor * mr)
    if med_nn < threshold:
        raise DenseNNMedianAstrometryError(
            _DENSE_NN_MEDIAN_MESSAGE.format(context=context or "(unspecified)",
                                            n=n, med_nn=med_nn, mr=threshold,
                                            factor=min_spacing_factor))


def measure_offsets(reference_coordinates, skycrds_cat, refflux, skyflux, total_dra=0*u.arcsec,
                    total_ddec=0*u.arcsec, max_offset=0.2*u.arcsec, threshold=0.01*u.arcsec,
                    sel=slice(None),
                    verbose=False,
                    ratio_match=True,
                    nsigma_reject=5,
                    reject_niter=7,
                    filtername='', ab='', expno=''):
    # FORBIDDEN-METHOD GUARD: this routine estimates a bulk shift as the median of
    # nearest-neighbour matches, which is invalid against a dense reference (see
    # assert_sparse_reference_for_nn_median).  Refuse dense references outright.
    assert_sparse_reference_for_nn_median(
        reference_coordinates, max_offset,
        context=f"measure_offsets(filter={filtername!r}, ab={ab!r}, expno={expno!r})")
    med_dra = 100*u.arcsec
    med_ddec = 100*u.arcsec

    success = False

    iteration = 0
    while np.abs(med_dra) > threshold or np.abs(med_ddec) > threshold:

        idx, offset, _ = reference_coordinates.match_to_catalog_sky(skycrds_cat[sel], nthneighbor=1)
        reverse_idx, reverse_sep, _ = skycrds_cat[sel].match_to_catalog_sky(reference_coordinates, nthneighbor=1)

        reverse_mutual_matches = (idx[reverse_idx] == np.arange(len(reverse_idx))) & (reverse_sep < max_offset)
        mutual_matches = (reverse_idx[idx] == np.arange(len(idx)))

        keep = (offset < max_offset) & mutual_matches
        skykeep = (reverse_sep < max_offset) & reverse_mutual_matches
        if keep.sum() < 5:
            print(f"Only {keep.sum()} sources matched - this is too few to be useful")
            print(f"{filtername:5s}, {ab:3s}, {expno:5s}, {keep.sum():6d}, {iteration:5d}", flush=True)
            # same types as the normal-path return: Quantities for the offsets/
            # scatters, boolean arrays for keep/skykeep/reject, int iteration
            zero = 0 * u.arcsec
            return (zero, zero, zero, zero, zero, zero,
                    np.asarray(keep, dtype=bool), np.asarray(skykeep, dtype=bool),
                    np.zeros(int(keep.sum()), dtype=bool), iteration)

        # ratio = skyflux[idx[keep]] / refflux[keep]
        # magnitude-style
        ratio = np.log(skyflux[idx[keep]]) - np.log(refflux[keep])

        if hasattr(ratio, 'mask'):
            # masks break all the math below?
            ratio = np.where(ratio.mask, np.nan, ratio.data)

        reject = np.zeros(ratio.size, dtype='bool')
        if ratio_match:
            rejection_data = []
            for ii in range(reject_niter):
                madstd = stats.mad_std(ratio[~reject], ignore_nan=True)
                med = np.nanmedian(ratio[~reject])
                reject = (ratio < (med - nsigma_reject * madstd)) | (ratio > (med + nsigma_reject * madstd)) | reject
                rejection_data.append([med, madstd, reject.sum()])
                if np.all(reject):
                    print(f"ALL SOURCES WERE REJECTED - this isn't really possible so it indicates an error {ii}")
                    print(f"Iterations were: {rejection_data}")
                    reject = np.zeros(ratio.size, dtype='bool')

        # dra and ddec should be the vector added to CRVAL to put the image in the right place.
        # NOTE: dra is RAW RA (no cos(dec)); it is applied raw to .ra below, so the routine is
        # internally self-consistent. Caveat: the convergence `threshold` and reported std are in
        # raw RA, i.e. ~cos(dec) smaller in true angle (~12% at the GC); negligible vs the threshold
        # itself. For <10 mas absolute astrometry tie to GSC3.2/Gaia at the observation epoch and use
        # a great-circle affine fit (see brick2221/analysis/retie_to_gsc.py).
        dra = -(skycrds_cat[sel][idx[keep][~reject]].ra - reference_coordinates[keep][~reject].ra).to(u.arcsec)
        ddec = -(skycrds_cat[sel][idx[keep][~reject]].dec - reference_coordinates[keep][~reject].dec).to(u.arcsec)

        med_dra = np.median(dra)
        med_ddec = np.median(ddec)
        std_dra = stats.mad_std(dra)
        std_ddec = stats.mad_std(ddec)

        if np.isnan(med_dra):
            print(f'len(refcoords) = {len(reference_coordinates)}')
            print(f'len(idx) = {len(idx)}')
            print(f'keep.sum() = {keep.sum()}')
            print(f'reject.size: {reject.size}')
            print(f'reject.sum() = {reject.sum()}')
            print(f'~reject.sum() = {(~reject).sum()}')
            # print(f'len(sidx) = {len(sidx)}')
            raise ValueError(f"median(dra) = {med_dra}.  np.nanmedian(dra) = {np.nanmedian(dra)}")

        total_dra = total_dra + med_dra.to(u.arcsec)
        total_ddec = total_ddec + med_ddec.to(u.arcsec)

        skycrds_cat = SkyCoord(ra=skycrds_cat.ra + med_dra, dec=skycrds_cat.dec + med_ddec, frame=skycrds_cat.frame)

        success = True

        iteration += 1
        if iteration > 50:
            # There is at least one known case in which the loop converges to an
            # oscillator rather than a fixed point; accept the current solution
            # (loudly) instead of raising.
            print(f"WARNING: measure_offsets did not converge after {iteration} "
                  f"iterations (likely oscillating); accepting current solution "
                  f"med_dra={med_dra}, med_ddec={med_ddec} "
                  f"(filter={filtername!r}, ab={ab!r}, expno={expno!r})",
                  flush=True)
            break

    if verbose and success:
        print(f"{filtername:5s}, {ab:3s}, {expno:5s}, {total_dra.value:8.3f}, {total_ddec.value:8.3f}, {med_dra.value:8.3f}, {med_ddec.value:8.3f}, {std_dra.value:8.3f}, {std_ddec.value:8.3f}, {keep.sum():6d}, {reject.sum():7d}, {iteration:5d}", flush=True)

    return total_dra, total_ddec, med_dra, med_ddec, std_dra, std_ddec, keep, skykeep, reject, iteration
