"""single_exposure_events, ramp test: counts in one integration of an exposure and
not in the others are an event; a star, the same in every integration, is not."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import single_exposure_events as se

pytestmark = pytest.mark.filterwarnings(
    'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')

NY = NX = 96
NGROUP = 4
READ_NOISE = 3.0          # DN per read
CRF_PER_DN = 0.1          # MJy/sr per DN of the group-mean level
CRF_PER_RATE = 2.5 * CRF_PER_DN    # MJy/sr per DN per group (the group mean of 4 groups of a ramp is 2.5 rates)


def _profile(x, y, sigma=1.1):
    yy, xx = np.mgrid[0:NY, 0:NX]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


CALIB = (15, 75, 20000.0)    # the bright star every frame is calibrated on, away from the hits


def _cube(nint=2, seed=0, stars=None, hits=(), noise=READ_NOISE):
    """Ramp cube (integration, group, y, x) in DN.  ``stars`` = (x, y, rate): rate DN per
    group in every integration (default: the calibration star ``CALIB``).  ``hits`` =
    (integration, x, y, rate[, first_group]): the same ramp in one integration only,
    from ``first_group`` on."""
    stars = (CALIB,) if stars is None else stars
    rng = np.random.default_rng(seed)
    cube = rng.normal(0.0, noise, (nint, NGROUP, NY, NX)) if noise else np.zeros((nint, NGROUP, NY, NX))
    for x, y, rate in stars:
        for g in range(NGROUP):
            cube[:, g] += rate * _profile(x, y) * (g + 1)
    for hit in hits:
        i, x, y, rate = hit[:4]
        first = hit[4] if len(hit) > 4 else 0
        for g in range(first, NGROUP):
            cube[i, g] += rate * _profile(x, y) * (g + 1 - first)
    return cube.astype(np.float32)


def _crf(cube, seed=1):
    """The crf of an exposure that kept everything its ramp holds: sky noise plus the
    mean over the integrations."""
    rng = np.random.default_rng(seed)
    w = np.arange(cube.shape[1]) - (cube.shape[1] - 1) / 2.0
    rate = np.tensordot(cube.astype(float), w, axes=([1], [0])) / (w * w).sum()      # least-squares slope
    return 10.0 + rng.normal(0.0, 2.0, (NY, NX)) + CRF_PER_RATE * rate.mean(axis=0)


class FakeFrame:
    """A frame with a crf image derived from its own ramp (the mean over integrations)."""

    def __init__(self, name, cube, seed=1, sci=None, dq=None):
        self.name = name
        self.cube = cube
        self.shape = (NY, NX)
        if sci is None:
            sci = _crf(cube, seed)
        self.sci = sci
        self.dq = np.zeros(self.shape, dtype=np.int64) if dq is None else dq

    def load(self):
        return self.sci.astype(float).copy(), self.dq.copy()

    def load_ramp(self):
        return self.cube


STAR = (30, 30, 20000.0)
HIT = (1, 60, 50, 20000.0)


def _xy(events):
    return sorted((e.frame, round(e.x), round(e.y)) for e in events)


def test_hit_in_one_integration_is_an_event_and_a_star_is_not():
    fr = FakeFrame('f0', _cube(stars=[STAR], hits=[HIT]))
    events, n_tested = se.find_ramp_events([fr])
    assert n_tested == 1
    assert _xy(events) == [('f0', 60, 50)]
    ev = events[0]
    assert ev.excess > 1000.0 and ev.others == []
    strongest, weakest = ev.ramp
    assert strongest > 10000.0 and abs(weakest) < 100.0
    assert ev.ys.min() <= 49 and ev.ys.max() >= 51 and ev.xs.min() <= 59 and ev.xs.max() >= 61
    # the star of the same frame is bright in both integrations: no footprint pixel on it
    assert not ((np.abs(ev.ys - 30) < 6) & (np.abs(ev.xs - 30) < 6)).any()


def test_one_frame_is_enough_where_the_cross_frame_test_needs_two():
    fr = FakeFrame('only', _cube(hits=[HIT]))
    events, n_tested = se.find_ramp_events([fr])
    assert n_tested == 1 and _xy(events) == [('only', 60, 50)]


def test_hit_in_the_first_integration_and_a_hit_that_starts_in_the_third_group():
    fr = FakeFrame('f0', _cube(hits=[(0, 20, 20, 20000.0), (1, 70, 70, 40000.0, 2)]))
    events, _n = se.find_ramp_events([fr])
    assert _xy(events) == [('f0', 20, 20), ('f0', 70, 70)]


def test_weaker_integration_above_the_fraction_is_not_an_event():
    # a 30% hit on a star: the integrations differ by far more than the noise,
    # but the weaker one holds 77% of the stronger (the frame is calibrated on a second star)
    fr = FakeFrame('f0', _cube(stars=[STAR, CALIB], hits=[(0, 30, 30, 0.3 * STAR[2])]))
    assert se.find_ramp_events([fr]) == ([], 1)
    events, _n = se.find_ramp_events([fr], params=se.RampParams(min_diff_frac=0.1))
    assert _xy(events) == [('f0', 30, 30)]


def test_difference_must_exceed_nsigma_times_the_scatter():
    # a faint hit: its box sum differs by about 19 times the scatter of the integration difference
    lev = se.box_sums(se.ramp_slopes(_cube(hits=[(1, 50, 50, 20.0)])), 3)
    loose, _strong, _weak, sigma = se.ramp_event_pixels(lev, se.RampParams(nsigma=10.0))
    strict, *_rest = se.ramp_event_pixels(lev, se.RampParams(nsigma=40.0))
    assert 4.0 < sigma < 9.0                      # 3 * 1.34 DN per group of an integration, times sqrt(2)
    assert loose[50, 50] and not strict[50, 50]
    assert (strict & ~loose).sum() == 0


def test_event_below_min_excess_is_ignored():
    # a hit of 800 DN per group: crf excess 100 MJy/sr, below the default 200, above 50
    fr = FakeFrame('f0', _cube(hits=[(1, 50, 50, 800.0)]))
    assert se.find_ramp_events([fr])[0] == []
    events, _n = se.find_ramp_events([fr], params=se.RampParams(min_excess=50.0))
    assert _xy(events) == [('f0', 50, 50)]


def _cleaned_crf(clean, hit, carried, seed=1):
    """The crf of an exposure whose jump step left the fraction ``carried`` of the hit."""
    base = _crf(clean, seed)
    return base + carried * (_crf(hit, seed) - base)


def test_hit_the_crf_does_not_carry_is_not_an_event():
    # the jump step removed the hit from the rate image: the ramp still shows it, the mosaic does not
    clean = _cube(seed=0)
    hit = _cube(seed=0, hits=[HIT])
    fr = FakeFrame('f0', hit, sci=_crf(clean))
    assert se.find_ramp_events([fr]) == ([], 1)


@pytest.mark.parametrize('carried, found', [(0.05, False), (0.1, True), (1.0, True)])
def test_what_a_cleaned_hit_leaves_is_an_event_when_it_reaches_min_excess(carried, found):
    # the hit adds CRF_PER_RATE * 20000 / 2 = 2500 to the crf; 5 % of it is 125, below the default 200
    clean = _cube(seed=0)
    hit = _cube(seed=0, hits=[HIT])
    fr = FakeFrame('f0', hit, sci=_cleaned_crf(clean, hit, carried))
    events, n_tested = se.find_ramp_events([fr])
    assert n_tested == 1
    assert _xy(events) == ([('f0', 60, 50)] if found else [])
    if found:
        assert events[0].excess == pytest.approx(carried * 2500.0, rel=0.1)
        assert events[0].peak == pytest.approx(events[0].bkg + events[0].excess)


def test_pixels_beside_a_one_pixel_hit_pass_the_box_test_but_are_not_event_pixels():
    # the 3x3 box sum carries the hit onto its eight neighbours, among them pixels of a star whose
    # crf does not hold the hit
    cube = _cube(stars=[CALIB, (40, 40, 3000.0)])
    for g in range(NGROUP):
        cube[1, g, 40, 42] += 30000.0 * (g + 1)
    box = se.ramp_event_pixels(se.box_sums(se.ramp_slopes(cube), 3), se.RampParams())[0]
    assert box[39:42, 41:44].all()
    bare, _n = se.find_ramp_events([FakeFrame('f0', cube)], params=se.RampParams(dilate=0))
    assert [(round(e.x), round(e.y)) for e in bare] == [(42, 40)]
    assert set(zip(bare[0].xs.tolist(), bare[0].ys.tolist())) == {(42, 40)}
    dilated, _n = se.find_ramp_events([FakeFrame('f0', cube)])
    assert set(zip(dilated[0].xs.tolist(), dilated[0].ys.tolist())) == {
        (x, y) for x in (41, 42, 43) for y in (39, 40, 41)}


def test_pixels_already_flagged_do_not_use_are_not_event_pixels():
    dq = np.zeros((NY, NX), dtype=np.int64)
    dq[40:61, 50:71] |= se.DO_NOT_USE
    assert se.find_ramp_events([FakeFrame('f0', _cube(hits=[HIT]), dq=dq)]) == ([], 1)


def test_frame_without_bright_pixels_to_calibrate_on_is_not_tested():
    assert se.find_ramp_events([FakeFrame('dark', _cube(stars=[], hits=[HIT]))]) == ([], 0)


def test_calibration_is_the_median_crf_per_rate_of_the_bright_agreeing_pixels():
    cube = _cube(stars=[STAR, CALIB], hits=[HIT])
    sci, dq = FakeFrame('f0', cube).load()
    rate = se.ramp_slopes(cube)
    params = se.RampParams()
    assert se.ramp_calibration(sci, dq, rate, params) == pytest.approx(CRF_PER_RATE, rel=0.05)
    assert np.isnan(se.ramp_calibration(sci, np.full((NY, NX), se.SATURATED), rate, params))
    assert np.isnan(se.ramp_calibration(sci, np.full((NY, NX), se.DO_NOT_USE), rate, params))
    assert np.isnan(se.ramp_calibration(sci, dq, rate, se.RampParams(calib_min_pixels=10 ** 6)))
    # a pixel the integrations disagree on is no calibrator: the only bright pixels left are the hit's
    only_hit = se.ramp_slopes(_cube(stars=[], hits=[HIT]))
    assert np.isnan(se.ramp_calibration(sci, dq, only_hit, se.RampParams(calib_rate=10.0)))


def test_offset_of_one_integration_is_not_an_event():
    # a pedestal that differs between the integrations moves no rate: the box test does not see it
    # and the crf, which is made from the rates, does not hold it
    cube = _cube()
    cube[1, :, 50:56, 50:56] += 5000.0
    assert se.find_ramp_events([FakeFrame('f0', cube)]) == ([], 1)


@pytest.mark.parametrize('cleaned', [False, True])
def test_integration_whose_ramp_falls_is_not_an_event(cleaned):
    # a ramp that falls in one integration (a negative rate): the box test sees the difference, and the
    # crf, which is above that negative rate, would seem to hold hundreds of MJy/sr beyond it; the crf
    # of the other integration holds nothing (``cleaned``: the crf did not follow the falling one)
    fall = _cube(hits=[(1, 50, 50, -20000.0)])
    box = se.ramp_event_pixels(se.box_sums(se.ramp_slopes(fall), 3), se.RampParams())[0]
    assert box[50, 50]
    fr = FakeFrame('f0', fall, sci=_crf(_cube()) if cleaned else None)
    assert se.find_ramp_events([fr]) == ([], 1)


def test_hit_on_a_pixel_whose_other_integration_falls_slightly_is_still_an_event():
    # the floor at zero moves the prediction of a noisy sky pixel by a few MJy/sr and no more
    cube = _cube(hits=[HIT])
    cube[0, :, 60, 50] -= 20.0 * np.arange(1, NGROUP + 1)
    events, _n = se.find_ramp_events([FakeFrame('f0', cube)])
    assert _xy(events) == [('f0', 60, 50)]


def test_three_integrations_compare_with_the_lower_median():
    one = FakeFrame('one', _cube(nint=3, hits=[(2, 50, 50, 20000.0)]))
    two = FakeFrame('two', _cube(nint=3, hits=[(1, 50, 50, 20000.0), (2, 50, 50, 20000.0)]))
    events, n_tested = se.find_ramp_events([one, two])
    assert n_tested == 2
    assert _xy(events) == [('one', 50, 50)]      # bright in two of three: counted as the star


def test_frames_without_a_usable_ramp_are_not_tested():
    class NoRamp(FakeFrame):
        def load_ramp(self):
            return None

    class Cutout(FakeFrame):
        def load_ramp(self):
            return self.cube[:, :, :60, :60]

    one_integration = FakeFrame('one', _cube(nint=1))
    frames = [NoRamp('none', _cube(hits=[HIT])), Cutout('crop', _cube(hits=[HIT])),
              one_integration, FakeFrame('ok', _cube(hits=[HIT]))]
    events, n_tested = se.find_ramp_events(frames)
    assert n_tested == 1 and _xy(events) == [('ok', 60, 50)]


def test_frame_with_identical_integrations_has_no_scatter_and_is_not_tested():
    cube = _cube(noise=0.0, stars=[STAR])
    events, n_tested = se.find_ramp_events([FakeFrame('flat', cube)])
    assert (events, n_tested) == ([], 0)


def test_footprint_is_the_event_pixels_dilated_and_stays_on_the_array():
    fr = FakeFrame('f0', _cube(hits=[(1, 1, 1, 20000.0)]))
    small, _n = se.find_ramp_events([fr], params=se.RampParams(dilate=0))
    big, _n = se.find_ramp_events([fr], params=se.RampParams(dilate=2))
    assert len(small) == len(big) == 1
    assert len(big[0].ys) > len(small[0].ys)
    for ev in (small[0], big[0]):
        assert ev.ys.min() >= 0 and ev.xs.min() >= 0 and ev.ys.max() < NY and ev.xs.max() < NX


def test_component_larger_than_max_area_is_left_alone():
    fr = FakeFrame('f0', _cube(hits=[HIT]))
    assert se.find_ramp_events([fr], params=se.RampParams(max_area=3))[0] == []


def test_event_whose_crf_pixels_are_all_nan_is_ignored():
    cube = _cube(hits=[HIT])
    fr = FakeFrame('f0', cube)
    fr.sci[40:61, 50:71] = np.nan
    assert se.find_ramp_events([fr])[0] == []


def test_duplicate_frame_names_are_refused():
    frames = [FakeFrame('a', _cube()), FakeFrame('a', _cube(seed=3))]
    with pytest.raises(ValueError, match='not unique'):
        se.find_ramp_events(frames)


def test_ramp_slopes_are_least_squares_rates_without_the_offset():
    cube = np.zeros((2, NGROUP, NY, NX), dtype=np.float32)
    cube[0] = 3.0 * np.arange(NGROUP, dtype=np.float32)[:, None, None] + 700.0     # rate 3 on a pedestal of 700
    cube[1] = -250.0                                                               # no signal, a negative offset
    rate = se.ramp_slopes(cube)
    assert rate.shape == (2, NY, NX) and rate.dtype == np.float32
    assert np.allclose(rate[0], 3.0) and np.allclose(rate[1], 0.0)
    # a ramp that is not linear: the slope of [0, 0, 0, 4] is 1.5 * 4 / 5
    step = np.zeros((1, NGROUP, 2, 2), dtype=np.float32)
    step[0, 3] = 4.0
    assert np.allclose(se.ramp_slopes(step), 1.2)


def test_ramp_slopes_need_two_groups():
    with pytest.raises(ValueError, match='two or more groups'):
        se.ramp_slopes(np.zeros((2, 1, 4, 4), dtype=np.float32))


def test_box_sums_add_the_box():
    values = np.zeros((2, NY, NX), dtype=np.float32)
    values[0] = 2.0
    values[1, 10, 10] = 9.0
    sums = se.box_sums(values, 3)
    assert sums.shape == (2, NY, NX)
    assert np.allclose(sums[0], 18.0) and sums[1, 10, 10] == pytest.approx(9.0)
    assert sums[1, 9:12, 9:12].sum() == pytest.approx(81.0) and sums[1, 12, 12] == 0.0


def test_ramp_event_pixels_needs_two_integrations():
    with pytest.raises(ValueError, match='two or more'):
        se.ramp_event_pixels(np.zeros((1, 4, 4), dtype=np.float32), se.RampParams())


# --- the crf file and its sibling ramp ---------------------------------------------

def _crf_file(path, shape=(NY, NX)):
    ww = WCS(naxis=2)
    ww.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    ww.wcs.crval = [266.5, -28.8]
    ww.wcs.cdelt = [-1e-5, 1e-5]
    ww.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros(shape, dtype='float32'), header=ww.to_header(), name='SCI'),
                  fits.ImageHDU(np.zeros(shape, dtype='uint32'), name='DQ')]).writeto(path)
    return str(path)


@pytest.mark.parametrize('crf, ramp', [
    ('jw10678045001_02101_00004_nrcb3_destreak_o045_crf.fits',
     'jw10678045001_02101_00004_nrcb3_ramp.fits'),
    ('jw10678045001_02101_00004_nrcalong_align_o045_crf.fits',
     'jw10678045001_02101_00004_nrcalong_ramp.fits'),
])
def test_ramp_path_is_the_detector_ramp_beside_the_crf(tmp_path, crf, ramp):
    frame = se.CrfFrame(_crf_file(tmp_path / crf))
    assert frame.ramp_path() == str(tmp_path / ramp)


def test_ramp_path_is_none_without_a_detector_in_the_name(tmp_path):
    assert se.CrfFrame(_crf_file(tmp_path / 'mosaic_crf.fits')).ramp_path() is None


def test_load_ramp_reads_the_sibling_cube_as_float32(tmp_path):
    crf = _crf_file(tmp_path / 'jw_a_nrca1_destreak_o001_crf.fits')
    frame = se.CrfFrame(crf)
    assert frame.load_ramp() is None                              # no sibling file yet
    cube = _cube(hits=[HIT])
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube, name='SCI')]).writeto(
        tmp_path / 'jw_a_nrca1_ramp.fits')
    got = frame.load_ramp()
    assert got.dtype == np.float32 and got.shape == cube.shape and np.array_equal(got, cube)


@pytest.mark.parametrize('cube, hdu_name', [
    (np.zeros((4, NY, NX), dtype='float32'), 'SCI'),            # not (integration, group, y, x)
    (np.zeros((2, 4, 60, 60), dtype='float32'), 'SCI'),         # a cutout of the detector
    (np.zeros((2, 4, NY, NX), dtype='float32'), 'DATA'),        # no SCI cube
])
def test_load_ramp_refuses_a_cube_that_is_not_on_the_frames_grid(tmp_path, cube, hdu_name):
    frame = se.CrfFrame(_crf_file(tmp_path / 'jw_b_nrcb1_destreak_o001_crf.fits'))
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube, name=hdu_name)]).writeto(
        tmp_path / 'jw_b_nrcb1_ramp.fits')
    assert frame.load_ramp() is None
