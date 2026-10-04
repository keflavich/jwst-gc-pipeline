"""The ZEROFRAME deblender resolves stars inside a large blended component.

Two limits in ``deblend_blob_zeroframe`` collapsed big blends to a single seed:

1. The final friends-of-friends guard linked centres within 0.6 r_sat, where
   r_sat is the equivalent radius of the whole component.  For a blend of many
   stars that link exceeds the star spacing and chains every centre together.
2. ``peak_local_max`` returned at most ``max_stars`` (6) peaks before the
   claimed-region filter.  In a component with 6 or more ZF-saturated cores
   those slots go to the filled cores, which the filter then discards, so no
   secondary peak survives.

On wd2 F277W nrcblong the cluster-core component (34,700 saturated px, about 150
dolphot stars) gave one seed.  The guard now uses the largest ZF-core radius and
the peak limit grows with the saturated area.
"""
import numpy as np
from scipy import ndimage

from jwst_gc_pipeline.reduction.satstar_deblend import deblend_blob_zeroframe

SHAPE = (90, 90)
FWHM = 1.61   # NIRCam F150W, px


def _gauss(x, y, amp, sig=0.8):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _disk(x, y, r):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _deblend(zf, sat, **kw):
    sources, n = ndimage.label(sat)
    assert n == 1
    sl = ndimage.find_objects(sources)[0]
    centers, info = deblend_blob_zeroframe(
        zf, np.zeros(SHAPE), sources, 1, sl, FWHM, sat_ceiling=np.inf, **kw)
    return centers, info


def _matched(centers, stars, tol=1.0):
    return [s for s in stars
            if any(np.hypot(c[1] - s[0], c[0] - s[1]) <= tol for c in centers)]


# 4 x 3 grid of stars 9 px apart inside one filled saturated rectangle
# (area ~850 px, r_sat ~16.5 px, old link ~9.9 px > spacing)
GRID = [(25 + 9 * i, 30 + 9 * j) for i in range(4) for j in range(3)]


def _grid_sat():
    sat = np.zeros(SHAPE, bool)
    sat[25:50, 20:55] = True
    return sat


def _grid_zf(n_cores):
    zf = np.full(SHAPE, 100.0)
    for k, (x, y) in enumerate(GRID):
        zf += _gauss(x, y, 5000.0 if k < n_cores else 3000.0)
    for (x, y) in GRID[:n_cores]:
        zf[_disk(x, y, 1)] = 0.0       # ZF-saturated core -> invalid hole
    return zf


def test_grid_scene_old_link_exceeds_spacing():
    """Guard: 0.6 r_sat of the rectangle is larger than the 9 px spacing."""
    sat = _grid_sat()
    r_sat = np.sqrt(sat.sum() / np.pi)
    assert 0.6 * r_sat > 9


def test_many_core_blend_keeps_every_star():
    """6 ZF-saturated cores + 6 unsaturated stars in one component: all 12."""
    centers, _ = _deblend(_grid_zf(6), _grid_sat())
    assert len(_matched(centers, GRID)) == 12, centers
    assert len(centers) == 12, centers


def test_many_core_blend_old_limits_collapse(monkeypatch):
    """With the previous limits (fixed 6 peaks, link 0.6 r_sat) the same blend
    collapses; this pins the failure the change addresses."""
    from jwst_gc_pipeline.reduction import satstar_deblend as sd
    zf, sat = _grid_zf(6), _grid_sat()
    r_sat = np.sqrt(sat.sum() / np.pi)
    orig = sd._fof_merge
    monkeypatch.setattr(sd, '_fof_merge',
                        lambda centers, link: orig(centers, link=max(link, 0.6 * r_sat)))
    centers, _ = _deblend(zf, sat, area_per_peak=None)
    assert len(centers) <= 2, centers


def test_peak_limit_scales_with_area():
    """No ZF-saturated core, 12 unsaturated stars: a fixed limit of 6 peaks
    drops half of them; the area-scaled limit keeps all."""
    zf, sat = _grid_zf(0), _grid_sat()
    fixed, _ = _deblend(zf, sat, area_per_peak=None)
    scaled, _ = _deblend(zf, sat)
    assert len(_matched(fixed, GRID)) <= 6
    assert len(_matched(scaled, GRID)) == 12, scaled


def test_single_star_with_wide_saturation_stays_one_seed():
    """One bright star: ZF-saturated core, DQ saturation out to 10 px and a
    smooth ring at 5 px.  The ring maxima are not compact, so the smaller
    guard link still returns a single seed."""
    x, y = 45, 45
    sat = _disk(x, y, 10)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    rr = np.hypot(xx - x, yy - y)
    zf = 100.0 + _gauss(x, y, 20000.0, sig=1.2) + 800.0 * np.exp(-(rr - 5.0) ** 2 / 2.0)
    zf[_disk(x, y, 2.5)] = 0.0
    centers, _ = _deblend(zf, sat)
    assert len(centers) == 1, centers
    assert np.hypot(centers[0][1] - x, centers[0][0] - y) <= 1.0


def test_small_component_keeps_fixed_limit():
    """Below max_stars * area_per_peak saturated px the limit stays max_stars."""
    a, b = (40, 45), (44, 45)
    sat = _disk(*a, 2) | _disk(*b, 2)
    zf = 100.0 + _gauss(*a, 5000.0) + _gauss(*b, 3000.0)
    assert sat.sum() < 6 * 50
    c_default, _ = _deblend(zf, sat)
    c_fixed, _ = _deblend(zf, sat, area_per_peak=None)
    assert sorted(c_default) == sorted(c_fixed)
    assert len(c_default) == 2
