"""The ZEROFRAME deblender seeds only stars of the component it is deblending.

``deblend_blob_zeroframe`` searches the component dilated by r_sat + 2 FWHM.  That
region reaches saturated neighbours a few px away, and each neighbour already
seeds its own star.  Before this fix the neighbour's ZEROFRAME peak (or its
ZEROFRAME-saturated core) was returned as a second star of the component being
deblended, so the neighbour got two seeds.  ``get_saturated_stars`` fits them one
after the other with iterative subtraction, which splits that star's flux between
two rows at the same position.  On wd2 F150W nrcb3 (``--deblend-satstars``) 228
of 1865 seeds sat inside another component, and 42 pairs of accepted rows lay
within 1 px of each other.
"""
import numpy as np
from scipy import ndimage

from jwst_gc_pipeline.reduction.satstar_deblend import (
    build_deblended_source_records, deblend_blob_zeroframe)

SHAPE = (80, 80)
FWHM = 1.61   # NIRCam F150W, px


def _disk(x, y, r=2):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _gauss(x, y, amp, sig=0.8):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _labels(sat):
    sources, n = ndimage.label(sat)
    return sources, n, ndimage.find_objects(sources)


def _deblend(zf, sat, label_id, **kw):
    sources, _, slices = _labels(sat)
    centers, _ = deblend_blob_zeroframe(
        zf, np.zeros(SHAPE), sources, label_id, slices[label_id - 1], FWHM,
        sat_ceiling=kw.pop('sat_ceiling', np.inf), **kw)
    return centers


def _near(centers, x, y, tol=1.0):
    return [c for c in centers if np.hypot(c[1] - x, c[0] - y) <= tol]


# two separate saturated stars 7 px apart; the dilated search region of each
# one covers the other's core
STAR1, STAR2 = (30, 40), (37, 40)


def _two_neighbours():
    sat = _disk(*STAR1) | _disk(*STAR2)
    zf = 100.0 + _gauss(*STAR1, 5000.0) + _gauss(*STAR2, 3000.0)
    return zf, sat


def test_scene_neighbour_inside_search_region():
    """Guard: the neighbour's core lies inside the dilated search region."""
    _, sat = _two_neighbours()
    sources, n, _ = _labels(sat)
    assert n == 2
    blob = sources == sources[STAR1[1], STAR1[0]]
    r_sat = np.sqrt(blob.sum() / np.pi)
    dil = ndimage.binary_dilation(blob, iterations=int(np.ceil(r_sat + 2 * FWHM)))
    assert dil[STAR2[1], STAR2[0]]


def test_neighbour_peak_is_not_a_second_star():
    zf, sat = _two_neighbours()
    sources, _, _ = _labels(sat)
    for (x, y), (ox, oy) in ((STAR1, STAR2), (STAR2, STAR1)):
        centers = _deblend(zf, sat, sources[y, x])
        assert len(centers) == 1, centers
        assert _near(centers, x, y)
        assert not _near(centers, ox, oy)


def test_neighbour_zf_saturated_core_is_not_a_second_star():
    """A neighbour whose ZEROFRAME core also saturates (an invalid hole) used to
    come back as a core centre of the component being deblended."""
    zf, sat = _two_neighbours()
    zf[_disk(*STAR2, r=1)] = 0.0          # zf == 0 -> invalid core pixels
    sources, _, _ = _labels(sat)
    centers = _deblend(zf, sat, sources[STAR1[1], STAR1[0]])
    assert len(centers) == 1, centers
    assert _near(centers, *STAR1)


def test_records_one_seed_per_isolated_component():
    zf, sat = _two_neighbours()
    sources, n, _ = _labels(sat)
    idx = np.arange(n) + 1
    coms = ndimage.center_of_mass(sat, sources, idx)
    sizes = ndimage.sum_labels(sat, sources, idx)
    recs = build_deblended_source_records(sat, sources, coms, sizes, zf,
                                          np.zeros(SHAPE), FWHM)
    assert len(recs) == 2
    for r in recs:
        cy, cx = r['com']
        assert sources[int(round(cy)), int(round(cx))] == r['label']


def test_merged_double_still_deblends():
    """Two stars whose saturated cores touch form ONE component; the fix leaves
    their split into two seeds unchanged."""
    a, b = (30, 40), (34, 40)
    sat = _disk(*a) | _disk(*b)
    sources, n, _ = _labels(sat)
    assert n == 1
    zf = 100.0 + _gauss(*a, 5000.0) + _gauss(*b, 3000.0)
    centers = _deblend(zf, sat, 1)
    assert len(centers) == 2, centers
    assert _near(centers, *a) and _near(centers, *b)


def test_unsaturated_companion_still_found():
    """A companion outside every saturated component (no DQ SATURATED pixel)
    belongs to no other seed, so the deblender still returns it."""
    star, comp = (30, 40), (35, 40)
    sat = _disk(*star)
    zf = 100.0 + _gauss(*star, 5000.0) + _gauss(*comp, 2000.0)
    sources, _, _ = _labels(sat)
    assert sources[comp[1], comp[0]] == 0
    centers = _deblend(zf, sat, 1)
    assert len(centers) == 2, centers
    assert _near(centers, *star) and _near(centers, *comp)


def test_fallback_peak_inside_a_neighbour_is_not_seeded():
    """A component with no ZEROFRAME star of its own (flat first read) finds no
    core and no peak; the brightest-pixel fallback inside its search region then
    lands on the neighbour's peak.  That pixel belongs to the neighbour, so the
    component falls back to its own bbox centre instead."""
    own, nb = (30, 40), (35, 40)
    sat = _disk(*own, r=1) | _disk(*nb)
    sources, n, _ = _labels(sat)
    assert n == 2
    zf = 100.0 + _gauss(*nb, 3000.0)
    blob = sources == sources[own[1], own[0]]
    r_sat = np.sqrt(blob.sum() / np.pi)
    dil = ndimage.binary_dilation(blob, iterations=int(np.ceil(r_sat + 2 * FWHM)))
    assert dil[nb[1], nb[0]]                 # the fallback can reach it
    centers = _deblend(zf, sat, sources[own[1], own[0]])
    assert not _near(centers, *nb, tol=2.0), centers
    assert _near(centers, *own), centers


def test_neighbour_zf_core_claims_its_ring():
    """A neighbour's ZEROFRAME-saturated core claims the ring around it, as the
    component's own core would: a maximum there is the neighbour's flux, also
    where it lies just outside the neighbour's SATURATED footprint (so the
    ``other`` test cannot drop it), and it is not seeded as a second star even
    when a catalog position confirms it."""
    own, nb, bump = (28, 40), (35, 40), (32, 40)
    sat = _disk(*own) | _disk(*nb)
    sources, n, _ = _labels(sat)
    assert n == 2 and sources[bump[1], bump[0]] == 0
    zf = 100.0 + _gauss(*own, 5000.0) + _gauss(*nb, 3000.0) + _gauss(*bump, 1000.0)
    nb_core = _disk(*nb, r=1)
    zf[nb_core] = 0.0                        # the neighbour's invalid ZF core
    blob = sources == sources[own[1], own[0]]
    r_sat = np.sqrt(blob.sum() / np.pi)
    dil = ndimage.binary_dilation(blob, iterations=int(np.ceil(r_sat + 2 * FWHM)))
    assert dil[nb_core].all()                # the core is seen whole
    assert dil[bump[1], bump[0]]             # the bump is searched
    # the bump lies within min_sep (2 px) of the neighbour's core
    assert ndimage.binary_dilation(nb_core, iterations=2)[bump[1], bump[0]]
    centers = _deblend(zf, sat, sources[own[1], own[0]],
                       confirm_xy=np.array([[bump[0], bump[1]]], float))
    assert len(centers) == 1, centers
    assert _near(centers, *own)


# a bright star and a small saturated neighbour 10 px away.  The neighbour's
# first-read peak lies off its own SATURATED pixels, in the gap between the two
# components, 2 px from the neighbour and 3 px from the bright star's footprint.
BIG, SMALL, GAP_PEAK = (30, 40), (40, 40), (37, 40)


def _gap_scene():
    sat = _disk(*BIG, r=4) | _disk(*SMALL, r=1)
    zf = 100.0 + _gauss(*BIG, 5000.0) + _gauss(*GAP_PEAK, 2000.0)
    return zf, sat


def test_gap_scene_geometry():
    """Guard: the gap peak is off every SATURATED pixel, nearer to the small
    neighbour than to the bright star, and inside both search regions."""
    zf, sat = _gap_scene()
    sources, n, _ = _labels(sat)
    assert n == 2 and sources[GAP_PEAK[1], GAP_PEAK[0]] == 0
    big = sources == sources[BIG[1], BIG[0]]
    small = sources == sources[SMALL[1], SMALL[0]]
    d_big = ndimage.distance_transform_edt(~big)[GAP_PEAK[1], GAP_PEAK[0]]
    d_small = ndimage.distance_transform_edt(~small)[GAP_PEAK[1], GAP_PEAK[0]]
    assert d_small < d_big
    for blob in (big, small):
        r_sat = np.sqrt(blob.sum() / np.pi)
        dil = ndimage.binary_dilation(blob, iterations=int(np.ceil(r_sat + 2 * FWHM)))
        assert dil[GAP_PEAK[1], GAP_PEAK[0]]


def test_gap_peak_nearer_the_neighbour_is_seeded_once():
    """The peak between the two components goes to the nearer one only.  The
    bright star's search used to return it as a second star; the unlocked fit
    then drifted onto the neighbour and split its flux between two rows."""
    zf, sat = _gap_scene()
    sources, _, _ = _labels(sat)
    big = _deblend(zf, sat, sources[BIG[1], BIG[0]])
    assert len(big) == 1, big
    assert _near(big, *BIG)
    small = _deblend(zf, sat, sources[SMALL[1], SMALL[0]])
    assert len(small) == 1, small
    assert _near(small, *GAP_PEAK)


def test_gap_peak_out_of_the_neighbours_reach_is_kept():
    """A peak nearer to a neighbour but beyond the smallest search radius of
    any component (1 + 2 FWHM, in the L1 metric of the dilation) stays with
    the component that found it, because the neighbour might not search
    that far."""
    big_c, small_c, peak = (20, 40), (38, 47), (38, 40)
    sat = _disk(*big_c, r=8) | _disk(*small_c, r=0)
    sources, n, _ = _labels(sat)
    assert n == 2
    small = sources == sources[small_c[1], small_c[0]]
    big = sources == sources[big_c[1], big_c[0]]
    d_small = ndimage.distance_transform_edt(~small)[peak[1], peak[0]]
    d_big = ndimage.distance_transform_edt(~big)[peak[1], peak[0]]
    assert d_small < d_big
    assert d_small > np.ceil(1.0 + 2 * FWHM)
    zf = 100.0 + _gauss(*big_c, 5000.0) + _gauss(*peak, 2000.0)
    # pad=12 puts the peak (10 px from the bright star's footprint) in the crop
    centers = _deblend(zf, sat, sources[big_c[1], big_c[0]], pad=12)
    assert _near(centers, *peak), centers
