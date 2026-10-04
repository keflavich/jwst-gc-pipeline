"""NIRCam recovered-core cap (NIRCAM_SATSTAR_RECOVERED_CAP) reads only the
fitted star's own pixels.

Before the fix the cap read ``satmask_combined``: this star's dilated deep
core plus every OTHER saturated source's deep core in the cutout.  A weakly
saturated star whose own deep core was empty (ZEROFRAME anchor recovered all
its SATURATED pixels, or SATSTAR_ZF_KEEP_FINITE kept them) was then capped by
a neighbour's model-subtracted core, ~1% of its flux, and rejected by the
fit-quality gate (wd2 F150W nrcb1: 245 cap firings, median capped/fitted flux
0.011).

When the model's peak pixel is measured, the brightest measured pixel of that
region bounds the model peak.  Otherwise the cap compares the model with the
data at the measured pixel nearest the model peak (``recovered_cap_flux``).
Dividing the brightest measured pixel by the PSF PEAK in that case under-read
stars whose region is a recovered ring around an unmeasured core.
"""
import numpy as np
import pytest
from astropy.io import fits
from jwst.datamodels import dqflags
from photutils.psf import CircularGaussianPRF
from scipy.ndimage import binary_dilation

import jwst_gc_pipeline.reduction.satstar_deblend as SD
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    nearest_seed_cell, recovered_cap_flux, recovered_cap_region,
    recovered_core_peak)


def _disk(shape, x, y, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _scene():
    """Cutout with the fitted star A (weakly saturated: SAT component with
    finite rates, empty deep core) and a neighbour B whose deep core holds a
    small model-subtracted residual."""
    shape = (41, 41)
    cutout = np.full(shape, 2.0)
    comp_a = _disk(shape, 20, 20, 2)
    cutout[comp_a] = 1200.0
    cutout[20, 20] = 1800.0                     # A's measured peak
    deep_b = _disk(shape, 33, 8, 1)
    cutout[deep_b] = 5.0                        # B's core after subtraction
    unrec = np.zeros(shape, bool)
    return cutout, comp_a, deep_b, unrec


def test_region_is_own_component_when_own_deep_core_empty():
    cutout, comp_a, deep_b, unrec = _scene()
    own_deep = np.zeros_like(comp_a)
    region = recovered_cap_region(comp_a, own_deep,
                                  binary_dilation(own_deep, iterations=2))
    assert np.array_equal(region, comp_a)
    assert not (region & deep_b).any()


def test_region_is_dilated_own_deep_core_when_present():
    _, comp_a, deep_b, _ = _scene()
    own_deep = _disk(comp_a.shape, 20, 20, 1)
    expanded = binary_dilation(own_deep, iterations=2)
    region = recovered_cap_region(comp_a, own_deep, expanded)
    assert np.array_equal(region, expanded)
    assert not (region & deep_b).any()


def test_cap_reads_own_peak_not_neighbour_core():
    """The new region gives A's peak; the old satmask_combined region
    (empty own deep core | B's deep core) gave B's residual."""
    cutout, comp_a, deep_b, unrec = _scene()
    own_deep = np.zeros_like(comp_a)
    expanded = binary_dilation(own_deep, iterations=2)
    region = recovered_cap_region(comp_a, own_deep, expanded)
    peak, lost, nrec = recovered_core_peak(cutout, region, unrec)
    assert peak == 1800.0
    assert lost == 0.0 and nrec == int(comp_a.sum())
    old_peak, _, _ = recovered_core_peak(cutout, expanded | deep_b, unrec)
    assert old_peak == 5.0


def test_cap_skipped_when_core_mostly_lost():
    cutout, comp_a, _, _ = _scene()
    unrec = comp_a.copy()
    unrec[20, 18] = False                       # 1 of 13 pixels recoverable
    peak, lost, nrec = recovered_core_peak(cutout, comp_a, unrec, max_lost=0.2)
    assert np.isnan(peak)
    assert lost > 0.9 and nrec == 1


def test_cap_skipped_with_too_few_measured_pixels():
    cutout, comp_a, _, unrec = _scene()
    region = np.zeros_like(comp_a)
    region[20, 20] = region[20, 21] = True
    peak, lost, nrec = recovered_core_peak(cutout, region, unrec)
    assert np.isnan(peak) and nrec == 2


def test_empty_region_skips_cap():
    cutout, comp_a, _, unrec = _scene()
    peak, lost, nrec = recovered_core_peak(cutout, np.zeros_like(comp_a), unrec)
    assert np.isnan(peak) and lost == 1.0 and nrec == 0


# --------------------------------------------------------------------------
# recovered_cap_flux: the model is compared with the data at one pixel
# --------------------------------------------------------------------------

def _unit_psf(shape=(41, 41), x=20.3, y=19.6, sigma=1.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def test_cap_at_the_model_peak_pixel_is_peak_over_psf_peak():
    psf = _unit_psf()
    flux = 5.0e4
    cutout = flux * psf
    region = _disk(psf.shape, 20, 20, 3)
    cap, pfrac = recovered_cap_flux(cutout, region,
                                    np.zeros(psf.shape, bool), psf)
    assert pfrac == 1.0
    assert cap == np.max(cutout[region]) / psf.max()
    assert cap == pytest.approx(flux)


def test_cap_from_a_recovered_ring_compares_model_and_data_on_the_ring():
    """The core (r <= 1.5) is unmeasured; the ring holds the star's true
    profile.  The cap reads the brightest ring pixel against the PSF at that
    pixel and returns the star's flux.  Dividing the ring value by the PSF
    peak gives a small fraction of it."""
    psf = _unit_psf()
    flux = 5.0e4
    cutout = flux * psf
    core = _disk(psf.shape, 20, 20, 1.5)
    region = binary_dilation(core, iterations=2)
    cutout[core] = np.nan
    cap, pfrac = recovered_cap_flux(cutout, region, core, psf)
    assert cap == pytest.approx(flux)
    assert pfrac < 0.5
    peak, _, _ = recovered_core_peak(cutout, region, core, max_lost=1.0)
    assert peak / psf.max() < 0.5 * flux


def test_cap_around_a_dead_pixel_beside_the_star():
    """wd2 F150W nrcb3 (1570, 1666): the deep core was one dead pixel 3 px
    from the star's peak.  Its dilation reaches 1 px from the peak."""
    psf = _unit_psf(x=20.0, y=20.0, sigma=0.8)
    flux = 3.7e4
    cutout = flux * psf
    dead = np.zeros(psf.shape, bool)
    dead[23, 20] = True
    cutout[dead] = np.nan
    region = binary_dilation(dead, iterations=2)
    cap, pfrac = recovered_cap_flux(cutout, region, dead, psf)
    assert cap == pytest.approx(flux)
    peak, _, _ = recovered_core_peak(cutout, region, dead)
    assert peak / psf.max() < 0.5 * flux


def test_cap_tolerates_a_fit_a_fraction_of_a_pixel_off_the_star():
    """The star sits at x = 20.4 and the fit at x = 20.6: the brightest data
    pixel (20) and the model's peak pixel (21) differ, and both are measured.
    The brightest pixel bounds the model peak.  Reading the data at pixel 20
    against the model there would loosen the cap by 10%."""
    flux = 5.0e4
    cutout = flux * _unit_psf(x=20.4, y=20.0)
    psf = _unit_psf(x=20.6, y=20.0)
    region = _disk(psf.shape, 20, 20, 3)
    cap, pfrac = recovered_cap_flux(cutout, region,
                                    np.zeros(psf.shape, bool), psf)
    assert pfrac == 1.0
    assert cap == pytest.approx(flux, rel=1e-6)
    assert cutout[20, 20] / psf[20, 20] > 1.05 * flux


def test_cap_of_a_blended_star_reads_the_pixel_nearest_its_own_peak():
    """wd2 F150W nrcb3 (1891, 1547): a star with an unmeasured core shares its
    region with a 14.75 mag neighbour 6.7 px away whose measured peak is the
    brightest pixel there.  The model of the first star is faint at that
    pixel, so it bounds nothing (the fit came out 3x dolphot's other bands and
    took the neighbour's flux).  The pixel next to the star's own core bounds
    it."""
    flux_a, flux_b = 1.5e5, 6.5e4
    star_a = flux_a * _unit_psf(x=14.0, y=20.0)
    cutout = star_a + flux_b * _unit_psf(x=20.0, y=20.0)
    core_a = _disk(cutout.shape, 14, 20, 1.5)
    cutout[core_a] = np.nan
    region = _disk(cutout.shape, 14, 20, 7)
    psf = _unit_psf(x=14.0, y=20.0)
    assert np.nanargmax(np.where(region, cutout, np.nan)) == 20 * 41 + 20
    cap, pfrac = recovered_cap_flux(cutout, region, core_a, psf)
    assert cap == pytest.approx(flux_a, rel=0.01)
    assert pfrac < 0.5
    assert cutout[20, 20] / psf[20, 20] > 100 * flux_a


def test_cap_of_a_refit_after_subtraction_reads_the_brightest_pixel():
    """wd2 F150W nrcb3 (1926, 819): a second seed on a star whose model was
    already subtracted.  The core residual is negative and measured, so the
    brightest measured pixel (a residual 4 px out) bounds the model peak,
    and the refit is capped to the residual.  Reading only the positive
    pixels found the model's peak unmeasured and read that residual where
    the PSF is near zero, which left the refit at 85% of its flux and split
    the star between two rows."""
    psf = _unit_psf()
    flux = 8.0e3
    cutout = -0.1 * flux * psf
    cutout[20, 24] = 30.0
    region = _disk(psf.shape, 20, 20, 4)
    cap, pfrac = recovered_cap_flux(cutout, region,
                                    np.zeros(psf.shape, bool), psf)
    assert pfrac == 1.0
    assert cap == 30.0 / psf.max()
    assert cap < 0.05 * flux
    assert cutout[20, 24] / psf[20, 24] > 10 * flux


def test_cap_of_a_refit_with_an_unmeasured_core_reads_the_brightest_pixel():
    """wd2 F150W nrcb1 (955, 1759): a second seed 1.4 px from a star whose
    model was already subtracted.  The refit's peak pixel is in an unmeasured
    core, and the measured ring around it is a negative residual.  The
    brightest measured pixel bounds the model peak.  Skipping the negative
    ring read a residual 4 px out where the PSF is near zero, which left the
    refit at 2.0e3 and split the star between two rows."""
    psf = _unit_psf()
    flux = 8.0e3
    cutout = -0.05 * flux * psf
    core = _disk(psf.shape, 20, 20, 1.5)
    cutout[core] = np.nan
    cutout[20, 24] = 30.0
    region = _disk(psf.shape, 20, 20, 5)
    cap, pfrac = recovered_cap_flux(cutout, region, core, psf)
    assert pfrac == 1.0
    assert cap == 30.0 / psf.max()
    assert cap < 0.05 * flux
    assert cutout[20, 24] / psf[20, 24] > 10 * flux


def test_cap_still_binds_when_the_data_lie_below_the_model():
    """Extended emission: the measured core is half the fitted model."""
    psf = _unit_psf()
    cutout = 1.0e4 * psf
    region = _disk(psf.shape, 20, 20, 3)
    cap, _ = recovered_cap_flux(cutout, region, np.zeros(psf.shape, bool),
                                psf)
    assert cap == pytest.approx(1.0e4)
    assert cap < 2.0e4


def test_cap_is_nan_without_a_measured_pixel_or_psf():
    psf = _unit_psf()
    region = _disk(psf.shape, 20, 20, 2)
    cutout = 1.0e4 * psf
    cap, pfrac = recovered_cap_flux(cutout, region, region.copy(), psf)
    assert np.isnan(cap) and np.isnan(pfrac)
    cap, pfrac = recovered_cap_flux(cutout, region,
                                    np.zeros(psf.shape, bool),
                                    np.zeros(psf.shape))
    assert np.isnan(cap) and np.isnan(pfrac)
    far = _disk(psf.shape, 2, 2, 1)
    psf_far = psf.copy()
    psf_far[far] = 0.0
    cap, pfrac = recovered_cap_flux(np.where(far, 50.0, cutout), far,
                                    np.zeros(psf.shape, bool), psf_far)
    assert np.isnan(cap)


# --------------------------------------------------------------------------
# Blended components: the deblend gives every seed of a SAT component the
# component's label, so the cap must read only this seed's share
# --------------------------------------------------------------------------

def test_nearest_seed_cell_splits_at_the_midline():
    cell = nearest_seed_cell((21, 41), (10, 10), [(10, 30)])
    assert cell[10, :20].all() and not cell[10, 21:].any()
    assert cell[10, 20]                       # a tie stays with both seeds
    assert nearest_seed_cell((5, 5), (2, 2), []).all()


def _moffat_unit_psf(shape=(41, 41), x=12.0, y=20.0, a=1.5, beta=2.5):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    p = (1 + ((xx - x) ** 2 + (yy - y) ** 2) / a ** 2) ** -beta
    return p / p.sum()


def _blend_scene():
    """wd2 F150W nrcb3 (1893.7, 1338.2), arm with the first-frame anchor: a
    16.3 mag star A shares SAT component 521 with a 12.9 mag sibling B
    10.6 px away.  A's peak is unmeasured and its own ring lost; B's
    recovered ring holds B's model-subtracted residual, here 10% of A's
    model.  The cap read that residual where A's PSF is 0.0044 of its peak
    and cut A from 1.5e4 to 1.6e3."""
    shape = (41, 41)
    flux_a = 1.5e4
    psf_a = _moffat_unit_psf(shape, x=12.0, y=20.0)
    a_core = _disk(shape, 12, 20, 1)
    b_core = _disk(shape, 23, 20, 4)
    comp = _disk(shape, 12, 20, 2) | _disk(shape, 23, 20, 6)
    comp[19:22, 12:24] = True
    deep = a_core | b_core
    deep_exp = binary_dilation(deep, iterations=2)
    unrec = _disk(shape, 12, 20, 3) & ~a_core
    cutout = flux_a * psf_a
    b_ring = deep_exp & _disk(shape, 23, 20, 6) & ~b_core
    cutout[b_ring] *= 0.1
    cutout[deep] = np.nan
    return cutout, psf_a, flux_a, comp, deep, deep_exp, unrec, b_core


def test_blend_whole_component_region_reads_the_sibling_ring():
    """The pre-fix behaviour: the component-wide region and no PSF floor
    cap A to a tenth of its flux."""
    cutout, psf_a, flux_a, comp, deep, deep_exp, unrec, _ = _blend_scene()
    region = recovered_cap_region(comp, deep, deep_exp)
    peak, lost, _ = recovered_core_peak(cutout, region, unrec)
    assert np.isfinite(peak) and lost < 0.2
    cap, pfrac = recovered_cap_flux(cutout, region, unrec, psf_a,
                                    min_psf_frac=0.0)
    assert pfrac < 0.005
    assert cap == pytest.approx(0.1 * flux_a, rel=0.01)


def test_blend_cap_skipped_below_the_psf_floor():
    cutout, psf_a, _, comp, deep, deep_exp, unrec, _ = _blend_scene()
    region = recovered_cap_region(comp, deep, deep_exp)
    cap, pfrac = recovered_cap_flux(cutout, region, unrec, psf_a)
    assert np.isnan(cap)
    assert 0 < pfrac < 0.005


def test_blend_own_seed_region_leaves_out_the_sibling_core():
    """With A's share of the component only, B's core and most of its ring
    leave the region, A's lost ring dominates it, and the cap is skipped."""
    cutout, psf_a, _, comp, deep, deep_exp, unrec, b_core = _blend_scene()
    cell = nearest_seed_cell(comp.shape, (20, 12), [(20, 23)])
    region = recovered_cap_region(comp, deep, deep_exp, own_cell=cell)
    assert not (region & b_core).any()
    assert region[20, 12]
    peak, lost, _ = recovered_core_peak(cutout, region, unrec)
    assert np.isnan(peak) and lost >= 0.2


def test_blend_without_own_deep_core_keeps_own_share_of_component():
    """No deep core (the anchor recovered every SATURATED pixel): the region
    is this seed's share of the component, without the sibling's half."""
    _, _, _, comp, _, _, _, b_core = _blend_scene()
    none = np.zeros_like(comp)
    cell = nearest_seed_cell(comp.shape, (20, 12), [(20, 23)])
    region = recovered_cap_region(comp, none, none, own_cell=cell)
    assert np.array_equal(region, comp & cell)
    assert region[20, 12] and not region[20, 23]
    assert not (region & b_core).any()
    assert region.sum() < comp.sum()


# --------------------------------------------------------------------------
# wiring in get_saturated_stars
# --------------------------------------------------------------------------

class _CapRegionReached(Exception):
    pass


def _two_star_frame(n=200):
    """One SATURATED component holding two stars 14 px apart."""
    sat = dqflags.pixel['SATURATED']
    sci = np.ones((n, n))
    dq = np.zeros((n, n), dtype=np.uint32)
    dq[96:105, 88:113] = sat
    sci[dq != 0] = np.nan
    wcs_hdr = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                           'CRPIX1': 100, 'CRPIX2': 100, 'CRVAL1': 150.0,
                           'CRVAL2': 2.0, 'CDELT1': -1.7e-5, 'CDELT2': 1.7e-5,
                           'BUNIT': 'MJy/sr'})
    fh = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({
            'TELESCOP': 'JWST', 'INSTRUME': 'NIRCAM', 'FILTER': 'F150W',
            'PUPIL': 'CLEAR',
            'DETECTOR': 'NRCB1', 'MODULE': 'B', 'CHANNEL': 'SHORT'})),
        fits.ImageHDU(sci, header=wcs_hdr, name='SCI'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.1), name='ERR'),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.01), name='VAR_POISSON')])
    return fh, dq != 0


@pytest.mark.parametrize('seeds', [[(100.0, 93.0), (100.0, 107.0)],
                                   [(100.0, 100.0)]])
def test_get_saturated_stars_hands_the_cap_this_seeds_cell(monkeypatch,
                                                            seeds):
    """The deblend gives both seeds of the component its label.  The cap
    region of the first seed fitted reaches ``recovered_cap_region`` with a
    cell that holds that seed and leaves out its sibling; a lone seed gets
    no cell.  The deblend, the anchor and the PSF grid are stubbed, and the
    region helper stops the fit."""
    monkeypatch.setenv('NIRCAM_SATSTAR_RECOVERED_CAP', '1')
    fh, sat = _two_star_frame()
    n = sat.shape[0]

    def _records(saturated, sources, coms, sizes, zeroframe, data, fwhm,
                 **kw):
        return [{'com': c, 'label': 1, 'forced': False,
                 'sat_area': int(sat.sum())} for c in seeds]

    def _anchor(data, dq, zeroframe, **kw):
        return data, sat.copy(), np.zeros_like(sat), None

    seen = {}
    _cell = SSF.nearest_seed_cell

    def _nearest(shape, own_yx, sibling_yx):
        seen.update(own=own_yx, sibs=list(sibling_yx))
        return _cell(shape, own_yx, sibling_yx)

    def _region(own_component, own_deep_core, own_deep_core_expanded,
                own_cell=None):
        seen.update(component=own_component, cell=own_cell)
        raise _CapRegionReached
    monkeypatch.setattr(SD, 'build_deblended_source_records', _records)
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    monkeypatch.setattr(SSF, 'get_psf',
                        lambda header, **kw: CircularGaussianPRF(fwhm=1.6))
    monkeypatch.setattr(SSF, 'nearest_seed_cell', _nearest)
    monkeypatch.setattr(SSF, 'recovered_cap_region', _region)
    with pytest.raises(_CapRegionReached):
        SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)),
                                zeroframe_deblend=True, plot=False)
    if len(seeds) == 1:
        assert seen['cell'] is None and 'own' not in seen
        return
    cell = seen['cell']
    assert cell is not None and cell.shape == seen['component'].shape
    (oy, ox), [(sy, sx)] = seen['own'], seen['sibs']
    assert (oy, abs(ox - sx)) == (sy, 14.0)
    assert cell[int(oy), int(ox)] and not cell[int(sy), int(sx)]
    comp = seen['component']
    assert (comp & cell).any() and (comp & ~cell).any()
