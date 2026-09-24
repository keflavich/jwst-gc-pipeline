"""Exclude reference stars that JWST resolves into a binary or a group (#957).

VIRAC2's ~0.3" resolution merges stars that JWST separates.  Its position for
such a source is a flux-weighted blend that matches none of the JWST stars, so
the same-star tie pairs it with one of them at a 100-300 mas residual, and
enough such pairs refuse the region map.  The maintainer's criterion, applied
to the JWST catalog around each reference position:

* **binary** -- another JWST star within ``BLEND_BINARY_RADIUS_ARCSEC`` of the
  reference's primary (the brightest JWST star within
  ``BLEND_PRIMARY_RADIUS_ARCSEC`` of the reference) with at least
  ``BLEND_BINARY_FLUX_RATIO`` of the primary's flux;
* **group** -- the other JWST stars within ``BLEND_GROUP_RADIUS_ARCSEC`` of
  the primary sum to more than ``BLEND_GROUP_FLUX_FRACTION`` of its flux.

The test is run on every exposure's own catalog (all rows, no quality cut: a
faint companion is exactly what it looks for) and a reference is excluded when
the majority of the exposures that see a primary for it flag it.  Voting over
exposures keeps one exposure's spurious detection from removing a star, and
means the test never mixes positions from different frames.
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

BLEND_PRIMARY_RADIUS_ARCSEC = 0.3
BLEND_BINARY_RADIUS_ARCSEC = 0.6
BLEND_BINARY_FLUX_RATIO = 0.5
BLEND_GROUP_RADIUS_ARCSEC = 0.3
BLEND_GROUP_FLUX_FRACTION = 0.25


def _blend_flags_one_exposure(ref, coords, flux):
    """(has_primary, blended) per reference for one exposure's catalog."""
    nref = len(ref)
    has = np.zeros(nref, bool)
    blended = np.zeros(nref, bool)
    if len(coords) == 0 or nref == 0:
        return has, blended
    i_ref, i_src, _, _ = search_around_sky(
        ref, coords, BLEND_PRIMARY_RADIUS_ARCSEC * u.arcsec)
    if len(i_ref) == 0:
        return has, blended
    # primary = brightest source within the primary radius of each reference
    order = np.lexsort((-flux[i_src], i_ref))
    first = np.r_[True, i_ref[order][1:] != i_ref[order][:-1]]
    refs = i_ref[order][first]
    prim = i_src[order][first]
    has[refs] = True

    j_p, j_s, sep, _ = search_around_sky(
        coords[prim], coords, BLEND_BINARY_RADIUS_ARCSEC * u.arcsec)
    other = j_s != prim[j_p]
    comp_max = np.zeros(len(prim))
    np.maximum.at(comp_max, j_p[other], flux[j_s[other]])
    near = other & (sep < BLEND_GROUP_RADIUS_ARCSEC * u.arcsec)
    group_sum = np.bincount(j_p[near], weights=flux[j_s[near]], minlength=len(prim))
    pf = flux[prim]
    blended[refs] = ((comp_max >= BLEND_BINARY_FLUX_RATIO * pf)
                     | (group_sum > BLEND_GROUP_FLUX_FRACTION * pf))
    return has, blended


def blended_reference_mask(ref_coords, exposure_catalogs, dra_mas=0.0, ddec_mas=0.0):
    """Boolean mask over ``ref_coords``: True = JWST resolves it into a blend.

    Parameters
    ----------
    ref_coords : SkyCoord
        Reference positions.
    exposure_catalogs : iterable of (SkyCoord, ndarray)
        Each exposure's source positions and fluxes, all rows.  Rows with a
        non-finite position or a non-positive / non-finite flux are ignored.
    dra_mas, ddec_mas : float
        Verified reference-minus-JWST tie (on-sky mas); removed from the
        reference positions so the primary search is centred on the star.

    Returns
    -------
    (mask, info) -- ``info`` counts references seen by any exposure and the
    number excluded, for the checkpoint record.
    """
    cosd = np.cos(np.radians(ref_coords.dec.deg))
    ref = SkyCoord(ref_coords.ra.deg - dra_mas / 3.6e6 / cosd,
                   ref_coords.dec.deg - ddec_mas / 3.6e6, unit="deg")
    seen = np.zeros(len(ref), int)
    votes = np.zeros(len(ref), int)
    for coords, flux in exposure_catalogs:
        flux = np.asarray(flux, float)
        ok = (np.isfinite(coords.ra.deg) & np.isfinite(coords.dec.deg)
              & np.isfinite(flux) & (flux > 0))
        has, blended = _blend_flags_one_exposure(ref, coords[ok], flux[ok])
        seen += has
        votes += blended
    mask = (seen > 0) & (2 * votes > seen)
    return mask, dict(n_seen=int((seen > 0).sum()), n_excluded=int(mask.sum()),
                      primary_arcsec=BLEND_PRIMARY_RADIUS_ARCSEC,
                      binary_arcsec=BLEND_BINARY_RADIUS_ARCSEC,
                      binary_flux_ratio=BLEND_BINARY_FLUX_RATIO,
                      group_arcsec=BLEND_GROUP_RADIUS_ARCSEC,
                      group_flux_fraction=BLEND_GROUP_FLUX_FRACTION)
