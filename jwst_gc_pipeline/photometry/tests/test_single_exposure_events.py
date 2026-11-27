"""single_exposure_events: a bright compact SATURATED core carried by one frame
of several is an event; a star, bright in every covering frame, is not."""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry import single_exposure_events as se

NY = NX = 160
SCALE = 1e-5     # degrees per pixel


class FakeWCS:
    """Pixel -> sky as a pure shift: frame offset (ox, oy) pixels from frame 0."""

    def __init__(self, ox, oy):
        self.ox, self.oy = ox, oy

    def pixel_to_world_values(self, x, y):
        return (np.asarray(x, float) + self.ox) * SCALE, (np.asarray(y, float) + self.oy) * SCALE

    def world_to_pixel_values(self, ra, dec):
        return np.asarray(ra, float) / SCALE - self.ox, np.asarray(dec, float) / SCALE - self.oy


class FakeFrame:
    def __init__(self, name, sci, dq, offset=(0, 0)):
        self.name = name
        self.sci = sci
        self.dq = dq
        self.shape = sci.shape
        self.wcs = FakeWCS(*offset)

    def load(self):
        return self.sci.astype(float).copy(), self.dq.astype(np.int64).copy()


def _sky(seed):
    rng = np.random.default_rng(seed)
    return 10.0 + rng.normal(0.0, 2.0, (NY, NX)), np.zeros((NY, NX), np.int64)


def _blob(sci, x, y, peak, sigma=1.1):
    yy, xx = np.mgrid[0:NY, 0:NX]
    sci += peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def _core(dq, x, y, half=2):
    dq[y - half:y + half + 1, x - half:x + half + 1] |= se.SATURATED


def _frames(n=3, offsets=None):
    out = []
    for i in range(n):
        sci, dq = _sky(i)
        out.append(FakeFrame(f'f{i}', sci, dq, offsets[i] if offsets else (0, 0)))
    return out


def _by_frame(events):
    return sorted((e.frame, round(e.x), round(e.y)) for e in events)


def test_event_in_one_frame_is_flagged_and_a_saturated_star_is_not():
    frames = _frames()
    for f in frames:            # a real saturated star: bright, SATURATED, in all three
        _blob(f.sci, 40, 40, 30000.0)
        _core(f.dq, 40, 40)
    _blob(frames[1].sci, 100, 90, 20000.0)       # the event: frame 1 only
    _core(frames[1].dq, 100, 90)
    events = se.find_events(frames)
    assert _by_frame(events) == [('f1', 100, 90)]
    ev = events[0]
    assert ev.excess > 15000.0 and len(ev.others) == 2 and max(e for _n, e in ev.others) < 30.0


def test_star_saturated_in_one_frame_only_is_not_an_event():
    # frame 0 saturates the star, frames 1 and 2 see 90% of the peak without a flag
    frames = _frames()
    _blob(frames[0].sci, 60, 60, 30000.0)
    _core(frames[0].dq, 60, 60)
    for f in frames[1:]:
        _blob(f.sci, 60, 60, 27000.0)
    assert se.find_events(frames) == []


def test_position_covered_by_one_frame_cannot_be_tested():
    # frame 1 is shifted 120 px in x, so the event at x=100 of frame 0 is off its array
    frames = _frames(2, offsets=[(0, 0), (120, 0)])
    _blob(frames[0].sci, 100, 80, 20000.0)
    _core(frames[0].dq, 100, 80)
    assert se.find_events(frames) == []


def test_other_frame_without_finite_pixels_does_not_count():
    frames = _frames()
    _blob(frames[0].sci, 80, 80, 20000.0)
    _core(frames[0].dq, 80, 80)
    for f in frames[1:]:
        f.sci[70:91, 70:91] = np.nan
    assert se.find_events(frames) == []
    # one clean frame is enough
    frames[1].sci, _ = _sky(1)
    assert _by_frame(se.find_events(frames)) == [('f0', 80, 80)]
    assert len(se.find_events(frames, se.EventParams(min_other_frames=2))) == 0


def test_saturated_component_larger_than_max_core_area_is_a_star():
    frames = _frames()
    _blob(frames[0].sci, 80, 80, 20000.0, sigma=6.0)
    frames[0].dq[70:90, 70:90] |= se.SATURATED          # 400 px
    assert se.find_events(frames) == []
    assert len(se.find_events(frames, se.EventParams(max_core_area=500))) == 1


def test_event_below_min_excess_is_ignored():
    frames = _frames()
    _blob(frames[0].sci, 80, 80, 200.0)
    _core(frames[0].dq, 80, 80)
    assert se.find_events(frames) == []
    assert len(se.find_events(frames, se.EventParams(min_excess=100.0))) == 1


def test_all_nan_core_is_a_star_core():
    frames = _frames()
    for f in frames:
        _blob(f.sci, 80, 80, 20000.0)
        _core(f.dq, 80, 80)
    frames[0].sci[78:83, 78:83] = np.nan
    assert se.find_events(frames) == []


def test_dithered_frames_compare_the_same_sky_position():
    frames = _frames(3, offsets=[(0, 0), (5, -3), (-4, 7)])
    # the event's sky position is (100, 90) of frame 0; a star at the same sky position is
    # at (95, 93) of frame 1 and (104, 83) of frame 2
    _blob(frames[0].sci, 100, 90, 20000.0)
    _core(frames[0].dq, 100, 90)
    ev = se.find_events(frames)
    assert _by_frame(ev) == [('f0', 100, 90)]
    for f, (x, y) in zip(frames[1:], [(95, 93), (104, 83)]):
        _blob(f.sci, x, y, 19000.0)
    assert se.find_events(frames) == []


def test_footprint_covers_the_core_and_wings_within_max_radius():
    frames = _frames()
    _blob(frames[0].sci, 80, 80, 20000.0, sigma=1.6)
    _core(frames[0].dq, 80, 80)
    # a neighbouring real star beyond max_radius must stay out of the footprint
    for f in frames:
        _blob(f.sci, 80, 104, 3000.0)
    (ev,) = se.find_events(frames)
    mask = se.event_masks([ev], {'f0': (NY, NX)})['f0']
    assert mask[80, 80] and mask[78:83, 78:83].all()
    # wings above 0.2% of 20000 = 40 MJy/sr reach r ~ 5 px of a sigma = 1.6 px blob
    assert mask[80, 85] and not mask[80, 95]
    assert not mask[104, 80]
    yy, xx = np.nonzero(mask)
    assert np.hypot(xx - ev.x, yy - ev.y).max() <= se.EventParams().max_radius


def test_apply_mask_to_dq_sets_do_not_use_in_the_mask_only():
    dq = np.zeros((6, 6), np.int64)
    dq[1, 1] = se.SATURATED
    mask = np.zeros((6, 6), bool)
    mask[1:3, 1:3] = True
    out = se.apply_mask_to_dq(dq, mask)
    assert out[1, 1] == se.SATURATED | se.DO_NOT_USE and out[2, 2] == se.DO_NOT_USE
    assert out[0, 0] == 0 and out.sum() == out[mask].sum()
    assert dq[2, 2] == 0           # the input is not changed


def test_duplicate_frame_names_are_refused():
    a, b = _frames(2)
    b.name = a.name
    with pytest.raises(ValueError, match='not unique'):
        se.find_events([a, b])


def test_no_candidates_gives_no_events():
    assert se.find_events(_frames()) == []


# --- extreme negative pixels -------------------------------------------------------

def _neg_arrays():
    sci = np.full((6, 6), 10.0)
    err = np.full((6, 6), 3.0)
    dq = np.zeros((6, 6), np.int64)
    return sci, err, dq


def test_negative_pixel_mask_marks_the_unflagged_pixels_far_below_zero_only():
    sci, err, dq = _neg_arrays()
    sci[1, 1] = -19000.0                  # the defect: -6333 ERR
    sci[2, 2] = -100.0                    # -33 ERR: inside a 50 ERR limit
    sci[3, 3] = -19000.0
    dq[3, 3] = se.SATURATED               # left to the satstar handling
    sci[4, 4] = -19000.0
    dq[4, 4] = se.DO_NOT_USE              # already dropped
    sci[0, 5] = np.nan
    sci[5, 0] = -19000.0
    err[5, 0] = 0.0                       # no usable ERR
    sci[5, 5] = -19000.0
    err[5, 5] = np.nan
    sci[0, 0] = 1e5                       # a huge positive pixel is not this mask's business
    before = sci.copy()
    mask = se.negative_pixel_mask(sci, err, dq, 50.0)
    assert mask.dtype == bool and mask.shape == sci.shape
    assert list(zip(*np.nonzero(mask))) == [(1, 1)]
    assert np.array_equal(sci, before, equal_nan=True)       # the input is not changed
    assert se.negative_pixel_mask(sci, err, dq, 30.0)[2, 2]  # the limit is the argument
    assert not se.negative_pixel_mask(sci, err, dq, 40000.0).any()


def test_negative_pixel_mask_keeps_other_dq_bits_unmasked():
    sci, err, dq = _neg_arrays()
    sci[2, 3] = -5000.0
    dq[2, 3] = 4                          # JUMP_DET only: the o105 nrca4 pixel
    assert se.negative_pixel_mask(sci, err, dq, 50.0)[2, 3]


@pytest.mark.parametrize('value, expected', [
    (None, None), ('', None), ('  ', None), ('0', None), ('0.0', None),
    ('50', 50.0), (' 2.5 ', 2.5), ('1e2', 100.0),
])
def test_negative_pixel_nsigma_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(se.ENV_NEGATIVE, raising=False)
    else:
        monkeypatch.setenv(se.ENV_NEGATIVE, value)
    assert se.negative_pixel_nsigma() == expected


@pytest.mark.parametrize('value', ['abc', 'nan', 'inf', '-1', '5 sigma'])
def test_negative_pixel_nsigma_typo_raises_naming_the_variable(monkeypatch, value):
    monkeypatch.setenv(se.ENV_NEGATIVE, value)
    with pytest.raises(ValueError, match=se.ENV_NEGATIVE):
        se.negative_pixel_nsigma()


# --- isolated one-pixel positive spikes ------------------------------------------------

def _flat(bkg=10.0, shape=(40, 40)):
    return np.full(shape, bkg), np.zeros(shape, np.int64)


def _gauss(x0, y0, peak, sigma=0.95, shape=(40, 40)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return peak * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma ** 2))


def test_find_positive_spikes_marks_an_isolated_pixel_and_not_a_star():
    sci, dq = _flat()
    sci[20, 20] = 2010.0                         # 2000 MJy/sr over the background
    sci += _gauss(8, 30, 30000.0)                # a star on a pixel: its neighbours are at 0.58
    sci += _gauss(30.5, 30.5, 30000.0)           # a star on a pixel corner: four equal pixels
    before = sci.copy()
    spikes = se.find_positive_spikes(sci, dq, 300.0)
    assert [(s['x'], s['y']) for s in spikes] == [(20.0, 20.0)]
    s = spikes[0]
    assert s['peak'] == 2010.0 and s['bkg'] == pytest.approx(10.0, abs=1e-3)
    assert s['excess'] == pytest.approx(2000.0, abs=1e-3) and s['neighbour'] == pytest.approx(0.0, abs=1e-3)
    assert np.array_equal(sci, before)           # the input is not changed


def test_find_positive_spikes_does_not_judge_a_pixel_beside_a_flagged_missing_or_edge_pixel():
    sci, dq = _flat()
    sci[20, 20] = 2010.0                         # the control
    for x, y in [(5, 5), (5, 15), (5, 25), (30, 10), (33, 10), (30, 0), (39, 25), (30, 39)]:
        sci[y, x] = 2010.0
    dq[5, 6] = se.SATURATED                      # beside a saturated core
    sci[15, 6] = np.nan                          # beside a missing pixel
    dq[24, 5] = se.DO_NOT_USE                    # beside a pixel the pipeline drops
    dq[10, 30] = se.SATURATED                    # the spike pixel itself is saturated
    dq[10, 33] = se.DO_NOT_USE                   # ... or dropped
    spikes = se.find_positive_spikes(sci, dq, 300.0)
    assert [(s['x'], s['y']) for s in spikes] == [(20.0, 20.0)]   # (30, 0), (39, 25), (30, 39): array edge


def test_find_positive_spikes_keeps_other_dq_bits_and_follows_the_background():
    sci, dq = _flat(bkg=1000.0)
    sci[20, 20] = 2900.0                         # on a bright background: 1900 MJy/sr over it
    dq[20, 20] = 4                               # JUMP_DET only
    (s,) = se.find_positive_spikes(sci, dq, 300.0)
    assert (s['x'], s['y']) == (20.0, 20.0)
    assert s['excess'] == pytest.approx(1900.0, abs=1e-3) and s['bkg'] == pytest.approx(1000.0, abs=1e-3)


def test_find_positive_spikes_limits_are_the_arguments():
    sci, dq = _flat()
    sci[10, 10] = 160.0                          # excess 150
    sci[20, 20] = 2010.0
    sci[20, 21] = 10.0 + 0.10 * 2000.0           # a neighbour at 10 % of the excess: isolated
    sci[30, 30] = 2010.0
    sci[30, 31] = 10.0 + 0.20 * 2000.0           # a neighbour at 20 %: not isolated at 0.15
    found = lambda **kw: sorted((s['x'], s['y']) for s in se.find_positive_spikes(sci, dq, **kw))
    assert found(min_excess=300.0) == [(20.0, 20.0)]
    assert found(min_excess=100.0) == [(10.0, 10.0), (20.0, 20.0)]
    assert found(min_excess=300.0, isolation=0.25) == [(20.0, 20.0), (30.0, 30.0)]
    assert found(min_excess=3000.0) == []


def test_find_positive_spikes_needs_min_sigma_over_the_noise_of_its_annulus():
    sci, dq = _flat()
    # the columns step through -600 ... +600 around the background: a noisy halo whose
    # annulus MAD-sigma is hundreds of MJy/sr, and no pixel stands over the one above it
    sci += np.array([-600.0, -300.0, 0.0, 300.0, 600.0])[np.arange(40) % 5][None, :]
    sci[19:22, 19:22] = 10.0                     # calm neighbours
    sci[20, 20] = 5000.0
    (probe,) = se.find_positive_spikes(sci, dq, 300.0, min_sigma=0.0)
    bkg, sigma = probe['bkg'], probe['sigma']
    assert sigma > 300.0
    found = lambda **kw: [(s['x'], s['y']) for s in se.find_positive_spikes(sci, dq, 300.0, **kw)]
    sci[20, 20] = bkg + 3.0 * sigma
    assert found() == []                         # three sigma of the halo is noise at the default
    assert found(min_sigma=2.0) == [(20.0, 20.0)]
    assert found(min_sigma=0.0) == [(20.0, 20.0)]
    sci[20, 20] = bkg + 8.0 * sigma
    assert found() == [(20.0, 20.0)]
    assert se.SPIKE_MIN_SIGMA == 5.0


def test_find_positive_spikes_of_a_frame_without_one():
    sci, dq = _flat()
    assert se.find_positive_spikes(sci, dq, 300.0) == []
    sci[:] = np.nan
    assert se.find_positive_spikes(sci, dq, 300.0) == []


# --- spikes beside a missing pixel (max_unusable) ---------------------------------------

#: SCI of the 17x17 pixels around the spike of the F480M o130 nrcalong exposure 3 crf frame at
#: (x, y) = (1525, 141), which is [8, 8] here (1572 MJy/sr over 5-15 MJy/sr in its neighbours).
#: The nan pixels are the ones with DO_NOT_USE in that frame, among them the lower left
#: neighbour of the spike (DQ 5 = DO_NOT_USE + JUMP_DET); no pixel is SATURATED.
O130_F480M_SPIKE = np.array([
    [5.054, 4.861, 4.466, 4.693, 5.746, 7.243, 6.275, 6.595, 8.555, 9.734, 8.930, 7.128, 6.147, 7.981, 8.172, 5.953, 5.539],
    [4.580, 4.658, 4.585, 5.018, 5.403, 5.746, 4.964, 4.379, 6.399, 8.927, 11.319, 10.064, 11.088, 13.363, 13.211, 9.901, 7.821],
    [4.532, 4.850, 5.045, 5.280, 5.968, 5.233, 4.943, 4.800, 6.135, 10.450, 13.526, 12.026, 12.370, 15.004, 14.087, 10.546, 9.946],
    [4.737, 4.549, 4.928, 6.344, 7.170, 5.912, 5.066, 4.840, 5.902, 11.467, 14.241, 10.080, 15.724, 30.813, 29.659, 13.383, 8.880],
    [np.nan, 5.687, 5.506, 6.875, 7.432, 6.311, 4.707, 5.278, 6.054, 11.784, 12.955, 11.920, 39.147, 87.632, 88.504, 40.764, 10.220],
    [6.112, 5.863, 6.154, 6.775, 6.806, 5.622, np.nan, np.nan, np.nan, 11.251, 12.310, 15.793, 63.290, 140.646, 146.739, 70.415, 15.366],
    [5.607, 5.857, 5.353, 5.876, 6.112, 5.923, np.nan, np.nan, 6.169, 9.838, 10.668, 11.556, 50.879, 112.942, 118.248, 56.169, 13.143],
    [5.454, 5.098, 5.267, 5.868, 5.977, 5.634, 6.106, np.nan, 8.043, 10.075, 10.415, 7.062, 18.708, 42.089, 45.502, 20.768, 7.483],
    [5.259, 4.609, 5.178, 6.007, 6.112, 5.516, 5.246, 6.265, 1572.045, 15.033, 10.121, 7.703, 6.424, 8.481, 8.353, 7.022, 8.356],
    [5.702, 5.215, 5.216, 5.694, np.nan, 5.041, 4.993, 4.769, 5.161, 4.818, 5.974, 6.256, 6.958, 8.013, 8.706, 7.955, 7.733],
    [6.609, 5.868, 5.339, 4.937, 4.773, 4.735, 5.074, 5.010, 4.577, 4.477, np.nan, 4.694, 6.251, 8.980, 9.111, 7.277, 5.097],
    [6.525, 4.576, 5.886, 6.595, 6.628, 5.183, 5.105, 5.995, 5.036, 4.886, 5.420, 4.425, 4.835, 6.702, 7.007, 5.685, 4.900],
    [5.340, 5.281, 10.463, 18.240, 16.116, 8.743, 5.811, 6.465, 6.148, 4.758, 4.755, 4.520, 5.309, 5.705, 5.583, 5.044, 4.956],
    [4.596, 6.285, 16.082, 26.880, 25.934, 13.169, 7.179, 7.279, 6.386, 5.208, 5.119, 4.547, 4.743, 5.056, 4.913, 4.756, 4.766],
    [4.900, 5.474, 13.923, 23.015, 21.781, 11.753, 8.148, 8.742, 7.727, 5.584, 5.737, 5.191, 3.830, 3.897, 5.049, 6.218, np.nan],
    [4.921, 4.824, 7.491, 10.928, 10.284, 8.043, 7.443, 9.515, 7.931, 5.560, 4.931, 4.774, 4.330, 4.966, 5.773, 7.196, 7.086],
    [4.930, 4.653, 4.998, 5.356, 6.345, 7.645, 8.442, 8.194, 6.927, 5.165, 4.533, 4.487, 4.753, 4.920, 5.874, 6.258, 5.404],
])


def _o130_cutout():
    sci = O130_F480M_SPIKE.copy()
    dq = np.zeros(sci.shape, np.int64)
    dq[np.isnan(sci)] = se.DO_NOT_USE
    return sci, dq


def _shipped_find_positive_spikes(sci, dq, min_excess, isolation=se.SPIKE_ISOLATION,
                                  bkg_r_in=se.SPIKE_BKG_R_IN, bkg_r_out=se.SPIKE_BKG_R_OUT,
                                  min_sigma=se.SPIKE_MIN_SIGMA):
    """``find_positive_spikes`` as it was before ``max_unusable``: all nine pixels of the
    3x3 box usable.  It pins the default of the tolerant finder."""
    from scipy import ndimage
    sci = np.asarray(sci, dtype=float)
    dq = np.asarray(dq)
    usable = np.isfinite(sci) & ((dq & (se.DO_NOT_USE | se.SATURATED)) == 0)
    complete = ndimage.minimum_filter(usable.astype(np.uint8), size=3,
                                      mode='constant', cval=0) > 0
    filled = np.where(usable, sci, -np.inf)
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    neighbour_max = ndimage.maximum_filter(filled, footprint=ring, mode='constant',
                                           cval=-np.inf)
    with np.errstate(invalid='ignore'):
        step = filled - neighbour_max
        pre = complete & (step >= (1.0 - float(isolation)) * float(min_excess))
    out = []
    for y, x in zip(*np.nonzero(pre)):
        bkg, sig = se._annulus_stats(sci, x, y, bkg_r_in, bkg_r_out)
        if not np.isfinite(bkg):
            continue
        excess = float(sci[y, x] - bkg)
        neighbour = float(neighbour_max[y, x] - bkg)
        if (excess < min_excess or neighbour > isolation * excess
                or excess < min_sigma * sig):
            continue
        out.append(dict(x=float(x), y=float(y), peak=float(sci[y, x]), excess=excess,
                        bkg=bkg, sigma=sig, neighbour=neighbour))
    return out


def _messy_frame(seed):
    """Noise, defects of one pixel (some with a neighbour at 5-30 % of the excess), stars on
    pixel centres and corners, and random NaN, DO_NOT_USE and SATURATED pixels."""
    rng = np.random.default_rng(seed)
    shape = (90, 90)
    sci = 10.0 + rng.normal(0.0, 2.0, shape)
    dq = np.zeros(shape, np.int64)
    for _ in range(14):
        x, y = rng.integers(4, 86, 2)
        sci[y, x] += rng.uniform(500.0, 3000.0)
        if rng.random() < 0.4:
            sci[y + rng.integers(-1, 2), x + rng.integers(0, 2)] += rng.uniform(0.05, 0.3) * (sci[y, x] - 10.0)
    for _ in range(6):
        x0, y0 = rng.integers(8, 82, 2) + rng.choice([0.0, 0.5], 2)
        yy, xx = np.mgrid[0:90, 0:90]
        sci += rng.uniform(800.0, 20000.0) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 0.95 ** 2))
    sci[rng.random(shape) < 0.02] = np.nan
    dq[rng.random(shape) < 0.01] |= se.DO_NOT_USE
    dq[rng.random(shape) < 0.005] |= se.SATURATED
    dq[rng.random(shape) < 0.02] |= 4                 # JUMP_DET alone changes nothing
    return sci, dq


@pytest.mark.parametrize('seed', range(8))
def test_find_positive_spikes_default_is_the_finder_as_shipped(seed):
    sci, dq = _messy_frame(seed)
    shipped = _shipped_find_positive_spikes(sci, dq, 300.0)
    assert se.find_positive_spikes(sci, dq, 300.0) == shipped
    assert se.find_positive_spikes(sci, dq, 300.0, max_unusable=0) == shipped
    assert se.find_positive_spikes(sci, dq, 300.0, isolation=0.25, min_sigma=0.0, max_unusable=0) == \
        _shipped_find_positive_spikes(sci, dq, 300.0, isolation=0.25, min_sigma=0.0)


def test_find_positive_spikes_default_pin_is_not_empty():
    found = sum(len(_shipped_find_positive_spikes(*_messy_frame(seed), 300.0)) for seed in range(8))
    assert found >= 20


@pytest.mark.parametrize('seed', range(8))
def test_find_positive_spikes_tolerance_only_adds_pixels_beside_missing_ones(seed):
    sci, dq = _messy_frame(seed)
    shipped = {(s['x'], s['y']): s for s in se.find_positive_spikes(sci, dq, 300.0)}
    for k in (1, 2, 3):
        tolerant = {(s['x'], s['y']): s for s in se.find_positive_spikes(sci, dq, 300.0, max_unusable=k)}
        assert set(shipped) <= set(tolerant)
        for key, s in tolerant.items():
            assert 0 <= s['unusable'] <= k
            assert (s['unusable'] > 0) == (key not in shipped)
            if key in shipped:
                assert all(s[f] == shipped[key][f] for f in ('peak', 'excess', 'bkg', 'sigma'))
                assert s['neighbour'] == shipped[key]['neighbour']
            ix, iy = int(key[0]), int(key[1])
            box_dq = dq[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2]
            assert not (box_dq & se.SATURATED).any()          # a saturated neighbour is never tolerated


def test_find_positive_spikes_tolerates_the_missing_neighbour_of_the_o130_f480m_spike():
    sci, dq = _o130_cutout()
    assert sci[8, 8] == pytest.approx(1572.045, abs=1e-3)
    assert np.isnan(sci[7, 7]) and dq[7, 7] == se.DO_NOT_USE      # the lower left neighbour
    assert se.find_positive_spikes(sci, dq, 300.0) == []           # a 3x3 box with a missing pixel is not judged
    assert se.find_positive_spikes(sci, dq, 300.0, max_unusable=0) == []
    (s,) = se.find_positive_spikes(sci, dq, 300.0, max_unusable=1)
    assert (s['x'], s['y'], s['unusable']) == (8.0, 8.0, 1)
    assert s['peak'] == pytest.approx(1572.045, abs=1e-3)
    assert s['excess'] == pytest.approx(1565.68, abs=0.05) and s['bkg'] == pytest.approx(6.365, abs=0.005)
    assert s['excess'] / s['sigma'] == pytest.approx(738.0, abs=1.0)
    assert s['neighbour'] == pytest.approx(8.668, abs=0.005)       # the brightest usable neighbour: 0.0055 of the excess
    assert s['neighbour'] / s['excess'] < 0.006
    assert [(t['x'], t['y']) for t in se.find_positive_spikes(sci, dq, 300.0, max_unusable=2)] == [(8.0, 8.0)]
    assert se.find_positive_spikes(sci, dq, 3000.0, max_unusable=1) == []


def test_find_positive_spikes_does_not_tolerate_a_saturated_neighbour():
    sci, dq = _flat()
    sci[20, 20] = 2010.0
    dq[20, 21] = se.SATURATED
    for k in (0, 1, 6):
        assert se.find_positive_spikes(sci, dq, 300.0, max_unusable=k) == []
    dq[20, 21] = se.SATURATED | se.DO_NOT_USE
    assert se.find_positive_spikes(sci, dq, 300.0, max_unusable=6) == []
    dq[20, 21] = se.DO_NOT_USE                   # dropped, not saturated: tolerated
    assert [(s['x'], s['y']) for s in se.find_positive_spikes(sci, dq, 300.0, max_unusable=1)] == [(20.0, 20.0)]


def test_find_positive_spikes_counts_the_missing_neighbours_of_each_pixel():
    sci, dq = _flat()
    sci[20, 20] = 2010.0
    sci[10, 10] = 2010.0
    sci[10, 9] = np.nan
    sci[11, 11] = np.nan
    dq[20, 21] = se.DO_NOT_USE
    found = lambda k: {(s['x'], s['y']): s['unusable'] for s in se.find_positive_spikes(sci, dq, 300.0, max_unusable=k)}
    assert found(0) == {}
    assert found(1) == {(20.0, 20.0): 1}
    assert found(2) == {(20.0, 20.0): 1, (10.0, 10.0): 2}


def test_find_positive_spikes_counts_the_array_edge_as_missing():
    sci, dq = _flat()
    sci[0, 20] = 2010.0                          # top row: three neighbours are outside the array
    sci[39, 10] = 2010.0                         # bottom row, three
    sci[20, 0] = 2010.0                          # first column, three
    sci[0, 0] = 2010.0                           # a corner: five
    found = lambda k: sorted((s['x'], s['y'], s['unusable']) for s in se.find_positive_spikes(sci, dq, 300.0, max_unusable=k))
    assert found(2) == []
    assert found(3) == [(0.0, 20.0, 3), (10.0, 39.0, 3), (20.0, 0.0, 3)]
    assert (0.0, 0.0, 5) in found(5)
    assert (0.0, 0.0, 5) not in found(4)


def _star_and_neighbours(offset, sigma):
    """A Gaussian star of peak 30000 on a flat 10 MJy/sr, its brightest pixel and the eight
    neighbours of that pixel from the brightest to the faintest."""
    sci, dq = _flat()
    sci = sci + _gauss(20.0 + offset[0], 20.0 + offset[1], 30000.0, sigma=sigma)
    iy, ix = np.unravel_index(np.argmax(sci), sci.shape)
    nbrs = sorted(((sci[iy + dy, ix + dx], (iy + dy, ix + dx))
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)), reverse=True)
    return sci, dq, (iy, ix), nbrs


@pytest.mark.parametrize('sigma', [0.95, 1.1])
@pytest.mark.parametrize('offset', [(0.0, 0.0), (0.25, 0.1), (0.5, 0.0), (0.3, 0.3), (0.45, 0.45),
                                    (0.5, 0.5)])
def test_a_star_whose_brightest_neighbours_are_missing_is_not_a_spike(offset, sigma):
    """The worst case of the tolerance: the n brightest neighbours of the brightest pixel of a
    Gaussian star (PSF sigma 0.95 pixel is F212N, 1.1 pixel F480M) are the missing ones, for
    every n up to the largest tolerance."""
    sci, dq0, (iy, ix), nbrs = _star_and_neighbours(offset, sigma)
    for n_missing in range(0, se.SPIKE_MAX_UNUSABLE_LIMIT + 1):
        dq = dq0.copy()
        for _, (yy, xx) in nbrs[:n_missing]:
            dq[yy, xx] = se.DO_NOT_USE
        found = se.find_positive_spikes(sci, dq, 300.0, max_unusable=n_missing)
        assert (float(ix), float(iy)) not in [(f['x'], f['y']) for f in found], (offset, sigma, n_missing)


def test_the_tolerance_stops_where_a_star_corner_pixel_would_look_isolated():
    """A star of PSF sigma 0.95 pixel on a pixel corner: its brightest pixel has three
    neighbours at 1.0 of its value, four at 0.33 and the corner pixel at 0.109.  With six
    neighbours missing the brightest one left is at 0.33; with seven it is the corner pixel,
    below the isolation of 0.15.  The tolerance therefore ends at six."""
    sci, _, (iy, ix), nbrs = _star_and_neighbours((0.5, 0.5), 0.95)
    ratios = [(v - 10.0) / (sci[iy, ix] - 10.0) for v, _ in nbrs]
    assert ratios[6] == pytest.approx(0.33, abs=0.01)
    assert ratios[7] == pytest.approx(0.109, abs=0.005)
    assert ratios[6] > se.SPIKE_ISOLATION > ratios[7]
    assert se.SPIKE_MAX_UNUSABLE_LIMIT == 6


@pytest.mark.parametrize('k', [-1, 7, 8, 100])
def test_find_positive_spikes_max_unusable_out_of_range_raises(k):
    sci, dq = _flat()
    with pytest.raises(ValueError, match='max_unusable'):
        se.find_positive_spikes(sci, dq, 300.0, max_unusable=k)


@pytest.mark.parametrize('value, expected', [
    (None, 0), ('', 0), ('  ', 0), ('0', 0), ('1', 1), (' 2 ', 2), ('6', 6),
])
def test_positive_spike_max_unusable_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(se.ENV_SPIKE_UNUSABLE, raising=False)
    else:
        monkeypatch.setenv(se.ENV_SPIKE_UNUSABLE, value)
    assert se.positive_spike_max_unusable() == expected


@pytest.mark.parametrize('value', ['abc', 'nan', '-1', '7', '8', '1.5', '1 pixel', '1e0'])
def test_positive_spike_max_unusable_typo_raises_naming_the_variable(monkeypatch, value):
    monkeypatch.setenv(se.ENV_SPIKE_UNUSABLE, value)
    with pytest.raises(ValueError, match=se.ENV_SPIKE_UNUSABLE):
        se.positive_spike_max_unusable()


def test_positive_spike_mask_is_the_three_by_three_box_around_each_spike():
    spikes = [dict(x=20.0, y=20.0), dict(x=21.0, y=22.0), dict(x=5.0, y=30.0)]
    mask = se.positive_spike_mask((40, 40), spikes)
    assert mask.dtype == bool and mask.shape == (40, 40)
    assert mask[19:22, 19:22].all() and mask[21:24, 20:23].all() and mask[29:32, 4:7].all()
    assert int(mask.sum()) == 9 + 9 + 9 - 2          # the first two boxes overlap in x 20-21 at y 21
    assert not se.positive_spike_mask((40, 40), []).any()


@pytest.mark.parametrize('value, expected', [
    (None, None), ('', None), ('  ', None), ('0', None), ('0.0', None),
    ('300', 300.0), (' 2.5 ', 2.5), ('1e3', 1000.0),
])
def test_positive_spike_min_excess_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(se.ENV_SPIKE, raising=False)
    else:
        monkeypatch.setenv(se.ENV_SPIKE, value)
    assert se.positive_spike_min_excess() == expected


@pytest.mark.parametrize('value', ['abc', 'nan', 'inf', '-1', '300 MJy/sr'])
def test_positive_spike_min_excess_typo_raises_naming_the_variable(monkeypatch, value):
    monkeypatch.setenv(se.ENV_SPIKE, value)
    with pytest.raises(ValueError, match=se.ENV_SPIKE):
        se.positive_spike_min_excess()
