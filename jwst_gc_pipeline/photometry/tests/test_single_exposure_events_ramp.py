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


def _profile(x, y, sigma=1.1):
    yy, xx = np.mgrid[0:NY, 0:NX]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def _cube(nint=2, seed=0, stars=(), hits=(), noise=READ_NOISE):
    """Ramp cube (integration, group, y, x) in DN.  ``stars`` = (x, y, rate): rate DN per
    group in every integration.  ``hits`` = (integration, x, y, rate[, first_group]): the
    same ramp in one integration only, from ``first_group`` on."""
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


class FakeFrame:
    """A frame with a crf image derived from its own ramp (the mean over integrations)."""

    def __init__(self, name, cube, seed=1, sci=None):
        self.name = name
        self.cube = cube
        self.shape = (NY, NX)
        rng = np.random.default_rng(seed)
        if sci is None:
            sci = 10.0 + rng.normal(0.0, 2.0, (NY, NX)) + CRF_PER_DN * cube.mean(axis=1).mean(axis=0)
        self.sci = sci

    def load(self):
        return self.sci.astype(float).copy(), np.zeros(self.shape, dtype=np.int64)

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
    # but the weaker one holds 77% of the stronger
    fr = FakeFrame('f0', _cube(stars=[STAR], hits=[(0, 30, 30, 0.3 * STAR[2])]))
    assert se.find_ramp_events([fr])[0] == []
    events, _n = se.find_ramp_events([fr], params=se.RampParams(min_diff_frac=0.1))
    assert _xy(events) == [('f0', 30, 30)]


def test_difference_must_exceed_nsigma_times_the_scatter():
    # a faint hit: its box sum differs by about 17 times the scatter of the integration difference
    lev = se.ramp_levels(_cube(hits=[(1, 50, 50, 8.0)]), 3)
    loose, _strong, _weak, sigma = se.ramp_event_pixels(lev, se.RampParams(nsigma=10.0))
    strict, *_rest = se.ramp_event_pixels(lev, se.RampParams(nsigma=40.0))
    assert 4.0 < sigma < 9.0                      # 3 * 1.5 DN per integration, times sqrt(2)
    assert loose[50, 50] and not strict[50, 50]
    assert (strict & ~loose).sum() == 0


def test_event_below_min_excess_is_ignored():
    # a hit of 800 DN per group: crf excess 100 MJy/sr, below the default 200, above 50
    fr = FakeFrame('f0', _cube(hits=[(1, 50, 50, 800.0)]))
    assert se.find_ramp_events([fr])[0] == []
    events, _n = se.find_ramp_events([fr], params=se.RampParams(min_excess=50.0))
    assert _xy(events) == [('f0', 50, 50)]


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


def test_ramp_levels_sum_the_group_mean_over_the_box():
    cube = np.zeros((2, NGROUP, NY, NX), dtype=np.float32)
    cube[0] = 5.0
    cube[1, :, :, :] = np.arange(NGROUP, dtype=np.float32)[:, None, None]
    lev = se.ramp_levels(cube, 3)
    assert lev.shape == (2, NY, NX)
    assert np.allclose(lev[0], 5.0 * 9) and np.allclose(lev[1], 1.5 * 9)


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
