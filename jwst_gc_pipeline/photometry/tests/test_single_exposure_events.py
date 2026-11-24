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
