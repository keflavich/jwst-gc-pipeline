"""Size-gated NIRCam satstar position lock (NIRCAM_SATSTAR_LOCK_MIN_AREA_ARCSEC2).

Under ``NIRCAM_SATSTAR_LOCK_POS`` every satstar was fitted flux-only at the raw
saturated-mask centre of mass.  That seed is stable across frames for the large
variable-extent cores the lock was written for (W51 darkfil, 1.5-2.6 arcsec^2),
but sits 0.2-0.3 px off a compact stellar core (wd2 F150W against dolphot and
Gaia).  The gate keeps the lock for cores at or above the threshold and gives
smaller ones the refined centroid and the bounded fit.
"""
import ast
import inspect
import textwrap

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import center_of_mass, label

from jwst_gc_pipeline.reduction import saturated_star_finding as ssf


def _hdul(pixar=None, cdelt_arcsec=0.031):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [156.0, -57.75]
    w.wcs.crpix = [50, 50]
    w.wcs.cdelt = [-cdelt_arcsec / 3600.0, cdelt_arcsec / 3600.0]
    sci = fits.ImageHDU(np.zeros((100, 100)), header=w.to_header(), name='SCI')
    if pixar is not None:
        sci.header['PIXAR_A2'] = pixar
    return fits.HDUList([fits.PrimaryHDU(), sci])


def test_default_is_lock_everything():
    assert ssf.nircam_lock_min_area_px(_hdul(pixar=0.00093), env={}) == 0.0
    assert ssf.nircam_lock_min_area_px(
        _hdul(pixar=0.00093), env={'NIRCAM_SATSTAR_LOCK_MIN_AREA_ARCSEC2': ' '}) == 0.0
    assert ssf.nircam_lock_min_area_px(
        _hdul(pixar=0.00093), env={'NIRCAM_SATSTAR_LOCK_MIN_AREA_ARCSEC2': '0'}) == 0.0


def test_threshold_uses_pixar_a2():
    env = {'NIRCAM_SATSTAR_LOCK_MIN_AREA_ARCSEC2': '0.5'}
    # SW (0.031"/px) and LW (0.063"/px) give the same area on the sky
    assert ssf.nircam_lock_min_area_px(_hdul(pixar=0.031 ** 2), env=env) == pytest.approx(520.3, rel=1e-3)
    assert ssf.nircam_lock_min_area_px(_hdul(pixar=0.063 ** 2), env=env) == pytest.approx(126.0, rel=1e-3)


def test_threshold_falls_back_to_wcs_pixel_scale():
    env = {'NIRCAM_SATSTAR_LOCK_MIN_AREA_ARCSEC2': '0.5'}
    assert ssf.nircam_lock_min_area_px(_hdul(pixar=None, cdelt_arcsec=0.063), env=env) == \
        pytest.approx(0.5 / 0.063 ** 2, rel=1e-6)


def _two_components():
    """A small core with a one-sided spike (raw centre of mass pulled off the
    core) and a large round core."""
    sat = np.zeros((120, 120), bool)
    yy, xx = np.mgrid[:120, :120]
    sat |= np.hypot(xx - 30.0, yy - 30.0) <= 3.0          # 29 px core
    sat[30, 34:44] = True                                  # 1-px spike to +x
    sat |= np.hypot(xx - 85.0, yy - 85.0) <= 14.0         # ~620 px core
    sources, n = label(sat)
    coms = list(center_of_mass(sat, labels=sources, index=np.arange(n) + 1))
    return sat, sources, coms


def test_small_core_gets_refined_seed_large_core_keeps_raw():
    sat, sources, coms = _two_components()
    seeds, n_locked = ssf._lock_gated_coms(coms, np.zeros(sat.shape), sources, sat, 200)
    assert n_locked == 1
    # raw centre of mass of the small core is pulled ~1.5 px toward the spike;
    # the refined seed (eroded core) is back on the core
    assert coms[0][1] > 31.0
    assert abs(seeds[0][1] - 30.0) < 0.3 and abs(seeds[0][0] - 30.0) < 0.3
    assert seeds[1] == coms[1]


def test_threshold_above_every_core_refines_all():
    sat, sources, coms = _two_components()
    refined = ssf._refine_coms_by_data(coms, np.zeros(sat.shape), sources)
    seeds, n_locked = ssf._lock_gated_coms(coms, np.zeros(sat.shape), sources, sat, 1e6)
    assert n_locked == 0
    assert seeds == refined


@pytest.fixture(scope='module')
def tree():
    return ast.parse(textwrap.dedent(inspect.getsource(ssf.get_saturated_stars)))


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_lock_decision_reads_the_gate(tree):
    """The flux-only lock applies only at sat_area >= the gate (or when the
    gate is off, or for forced sources with no sat_area)."""
    vals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(getattr(t, 'id', None) == '_nc_lock' for t in n.targets)]
    assert len(vals) == 1
    assert {'_lock_pos', '_lock_min_px', 'src_sat_area'} <= _names(vals[0])


def test_lock_on_shared_model_is_cleared_for_the_next_source():
    """The loop reuses one PSF model object; a lock set for a large core must
    not carry into the next (unlocked) source."""
    from photutils.psf import CircularGaussianPRF
    model = CircularGaussianPRF(fwhm=2.0)
    ssf._set_position_fixed(model, True)
    assert model.x_0.fixed and model.y_0.fixed
    ssf._set_position_fixed(model, False)
    assert not model.x_0.fixed and not model.y_0.fixed


def test_both_fit_branches_set_the_position_freedom(tree):
    """Lock branch fixes, bounded branch frees: each calls _set_position_fixed."""
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == '_set_position_fixed']
    flags = sorted(bool(c.args[1].value) for c in calls)
    assert flags == [False, True]


def test_driver_defaults_the_gate_for_extended_emission_nircam():
    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(cataloging._prepare_frame_for_photometry)
    assert "os.environ['NIRCAM_SATSTAR_LOCK_MIN_AREA_ARCSEC2']" in src
    assert "'0.5' if _sat_ext_nircam else '0'" in src


if __name__ == '__main__':
    pytest.main([__file__])
