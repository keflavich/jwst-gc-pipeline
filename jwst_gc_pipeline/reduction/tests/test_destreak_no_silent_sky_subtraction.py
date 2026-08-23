"""The destreaker must not delete the sky when it has no map to put back.

``destreak_data(add_smoothed=False)`` subtracts a per-row, per-amplifier
percentile and adds nothing back.  That is the correct first half of
"subtract everything, then add the mosaic background", and it is a plain sky
subtraction in every other context: it drives each chunk's row percentile to
exactly zero, so the pedestal goes and ``percentile``% of the pixels go
negative.

``destreak()`` used to ask for it via ``add_smoothed = not
use_background_map`` -- decided from the CALLER'S INTENT to use a map, before
anyone checked whether a map existed.  Both production call sites pass
``use_background_map=True``, and only seven (field, observation, filter)
combinations in the archive have a map, so the other ~95 got the subtraction
with no restore.  Measured consequences on those products: Cloud E/F F210M
frame median 6.37 -> 1.45 MJy/sr, frame p10 4.36 -> -0.001, 10.06% of pixels
negative, mosaic diffuse-sky transfer slope 0.17 (vs 0.95 where a map exists).

These tests pin the things that make that unrepresentable:
  1. the bare branch raises unless the caller declares it restores the scales;
  2. ``destreak()`` resolves the map first and picks the mode from the result;
  3. with a map, the output is bit-identical to the old code (so the seven
     good combinations are not silently re-reduced into something new);
  4. the chunk width comes from NOUTPUTS, so a subarray is not split at a
     boundary its readout does not have;
  5. the product records which of the two sky conventions it carries.
"""
import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import destreak as D


def _sky_frame(nrows=2048, ncols=2048, pedestal=5.0, seed=0):
    """A frame with a sky pedestal, large-scale structure in both axes, and 1/f."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:nrows, :ncols]
    sky = pedestal + 2.0 * np.sin(2 * np.pi * y / nrows) + 1.5 * np.cos(2 * np.pi * x / ncols)
    streaks = rng.normal(0, 0.4, (nrows, 1)) * np.ones((1, ncols))
    return (sky + streaks + rng.normal(0, 0.05, (nrows, ncols))).astype('float32')


def _frame_hdul(program='02092', obs='002', visit='001',
                filtername='F210M', pupil='CLEAR', data=None):
    primary = fits.PrimaryHDU()
    primary.header['PROGRAM'] = program
    primary.header['OBSERVTN'] = obs
    primary.header['VISIT'] = visit
    primary.header['FILTER'] = filtername
    primary.header['PUPIL'] = pupil
    sci = fits.ImageHDU(_sky_frame() if data is None else data, name='SCI')
    sci.header['CTYPE1'] = 'RA---TAN'
    sci.header['CTYPE2'] = 'DEC--TAN'
    sci.header['CRVAL1'] = 266.5
    sci.header['CRVAL2'] = -28.7
    sci.header['CRPIX1'] = 1024
    sci.header['CRPIX2'] = 1024
    sci.header['CDELT1'] = -8.6e-6
    sci.header['CDELT2'] = 8.6e-6
    return fits.HDUList([primary, sci])


# --------------------------------------------------------------------------
# 1. the bare-subtraction branch
# --------------------------------------------------------------------------

def test_bare_subtraction_raises_instead_of_deleting_the_sky():
    with pytest.raises(D.DestreakWouldDeleteSky) as err:
        D.destreak_data(_sky_frame(), add_smoothed=False)
    # the message has to name the way out, or the next person just flips it back
    assert 'add_smoothed=True' in str(err.value)
    assert 'caller_restores_large_scales=True' in str(err.value)


def test_a_caller_that_restores_the_scales_may_still_ask_for_it():
    """The map path needs the full subtraction; declaring it keeps it available."""
    out = D.destreak_data(_sky_frame(), add_smoothed=False,
                          caller_restores_large_scales=True)
    assert np.isfinite(out).all()


def test_the_declared_bare_subtraction_is_bit_identical_to_the_old_code():
    """brick 2221/001 and Cloud C F405N must not change.

    Re-implement the pre-change expression literally and compare.  If this
    fails, the seven combinations that DO have a background map would come out
    of a re-reduction different from the products on disk.
    """
    data = _sky_frame()
    expected = data.copy()
    for start in range(0, 2048, 512):
        chunk = expected[:, start:start + 512]
        pct = D.nozero_percentile(chunk, 10, axis=1)
        expected[:, start:start + 512] = chunk - pct[:, None]

    got = D.destreak_data(data.copy(), percentile=10, median_filter_size=2048,
                          add_smoothed=False, caller_restores_large_scales=True)
    assert np.array_equal(got, expected)


def test_the_bare_subtraction_is_what_zeroes_the_sky():
    """Pins the behaviour the guard exists to prevent, so the numbers in the
    docstrings above stay checkable.

    The frame here is noise-dominated WITHIN a row-chunk, which is the real
    regime: the sky varies slowly across a detector, so the p10 of a 512-pixel
    row segment is set by the noise.  That is why the archive's affected frames
    all read exactly 10.06% negative -- it is the ``percentile`` argument
    showing through, not a property of any particular field.
    """
    rng = np.random.default_rng(7)
    data = (5.0 + rng.normal(0, 0.5, (2048, 2048))).astype('float32')
    before_p10 = np.percentile(data, 10)
    out = D.destreak_data(data.copy(), percentile=10, add_smoothed=False,
                          caller_restores_large_scales=True)
    assert before_p10 > 4.0
    assert abs(np.percentile(out, 10)) < 0.05          # pedestal gone
    assert 0.09 < np.mean(out < 0) < 0.11              # ~percentile% negative


# --------------------------------------------------------------------------
# 2. the safe branch actually smooths
# --------------------------------------------------------------------------

@pytest.mark.parametrize('median_filter_size', [256, 512, 2048])
def test_add_smoothed_keeps_the_pedestal_and_stays_positive(median_filter_size):
    data = _sky_frame(pedestal=5.0)
    out = D.destreak_data(data.copy(), percentile=10,
                          median_filter_size=median_filter_size, add_smoothed=True)
    assert abs(np.median(out) - np.median(data)) < 0.5
    assert np.mean(out < 0) < 0.01


def test_a_sub_detector_window_no_longer_collapses_to_a_flat_scalar():
    """The old branch read ``if median_filter_size >= 2048: median_filter(...)
    else: np.ones(2048) * np.median(pct)``, so every window smaller than the
    whole detector -- including the function's own default of 256 -- added back
    a single number and destroyed all row structure.  A 256-row window must now
    follow the row profile."""
    data = _sky_frame()
    out = D.destreak_data(data.copy(), median_filter_size=256, add_smoothed=True)
    # what was added back, recovered per chunk
    diff = out - data
    added = np.nanmedian(diff[:, 0:512], axis=1) + D.nozero_percentile(
        data[:, 0:512], 10, axis=1)
    assert np.std(added) > 0.1, 'add-back is flat: the old scalar branch is back'


def test_add_smoothed_removes_the_streaks_it_is_for():
    """Sanity: the safe mode is still a destreaker, not a no-op."""
    rng = np.random.default_rng(3)
    base = _sky_frame(seed=1)
    streaks = rng.normal(0, 1.0, (2048, 1)) * np.ones((1, 2048))
    out = D.destreak_data((base + streaks).astype('float32'),
                          median_filter_size=2048, add_smoothed=True)
    # row-to-row scatter of the sky floor, before vs after
    def rowfloor(a):
        return np.percentile(a, 10, axis=1)
    hp = lambda p: np.std(p - np.convolve(p, np.ones(65) / 65, mode='same'))
    assert hp(rowfloor(out)) < 0.35 * hp(rowfloor(base + streaks))


# --------------------------------------------------------------------------
# 3. the chunk width is the readout structure, not a constant
# --------------------------------------------------------------------------

def test_a_full_frame_still_chunks_at_512():
    """NOUTPUTS=4 over 2048 columns reproduces the old hardcode exactly."""
    assert D.amplifier_width(np.zeros((2048, 2048)), 4) == 512


def test_a_single_output_subarray_is_one_chunk():
    """sickle 3958/007 is NIRCam SUB640: 640x640, NOUTPUTS=1.  The old
    `range(0, 2048, 512)` gave it a 512-column chunk, a 128-column remainder
    estimated from a quarter as many pixels, and two empty slices."""
    assert D.amplifier_width(np.zeros((640, 640)), 1) == 640


def test_a_subarray_gets_no_step_at_column_512():
    """The defect the width fix removes.  Build a SUB640-shaped frame with a
    flat sky and per-row streaks; destreaking must not leave a discontinuity
    at column 512, which is where the hardcoded chunk boundary used to fall."""
    rng = np.random.default_rng(11)
    # an x-gradient is what makes the two chunks disagree: chunk 0 sees columns
    # 0-511 and the remainder sees 512-639, so their percentiles differ by the
    # sky difference between those column ranges, and that difference is what
    # gets stamped in as a step.  A spatially flat sky hides the defect.
    x = np.arange(640)[None, :]
    sky = 20.0 + 4.0 * (x / 640.0) + rng.normal(0, 0.3, (640, 640))
    streaks = rng.normal(0, 1.5, (640, 1)) * np.ones((1, 640))
    frame = (sky + streaks).astype('float32')

    good = D.destreak_data(frame.copy(), median_filter_size=2048,
                           add_smoothed=True, noutputs=1)
    colmed = np.median(good, axis=0)
    step = abs(np.median(colmed[512:524]) - np.median(colmed[500:512]))
    assert step < 0.1, f'{step:.3f} MJy/sr step at the old chunk boundary'

    # and the old chunking on the same frame does leave one
    bad = frame.copy()
    for start in (0, 512):
        chunk = bad[:, start:start + 512]
        pct = D.nozero_percentile(chunk, 10, axis=1)
        bad[:, start:start + 512] = chunk - pct[:, None]
    colmed = np.median(bad, axis=0)
    assert abs(np.median(colmed[512:524]) - np.median(colmed[500:512])) > 0.1


@pytest.mark.parametrize('mode', ['insitu', 'bare'])
def test_one_output_gives_one_correction_per_row(mode):
    """The property `test_a_subarray_gets_no_step_at_column_512` is after, on
    the CORRECTION rather than on the output image.

    Inside a chunk the correction is ``-pct[:, None] (+ smoothed_pct[:, None])``
    -- a per-row SCALAR.  So a single-output frame, which is one chunk, must
    have a correction that is constant along every row, no matter what its sky
    looks like.  Split it at 512 and columns 512+ get a second chunk's
    percentile instead, which is exactly the defect.

    Asserting on the output image instead does not pin it: in ``add_smoothed``
    mode most of the two chunks' pedestal difference is added straight back, so
    the surviving step in a synthetic frame sits under a 0.1 MJy/sr threshold
    even when the frame IS split at 512.  Reverting ``width =
    amplifier_width(data, noutputs)`` to the old literal 512 left the whole
    suite green (20 passed) before this test existed; it fails here in both
    modes.
    """
    rng = np.random.default_rng(11)
    # An x-gradient is what makes two chunks disagree: columns 0-511 and 512-639
    # have different sky, so their percentiles differ.  A spatially flat sky
    # hides the defect.
    x = np.arange(640)[None, :]
    sky = 20.0 + 4.0 * (x / 640.0) + rng.normal(0, 0.3, (640, 640))
    streaks = rng.normal(0, 1.5, (640, 1)) * np.ones((1, 640))
    frame = (sky + streaks).astype('float32')

    kwargs = ({'add_smoothed': True} if mode == 'insitu'
              else {'add_smoothed': False, 'caller_restores_large_scales': True})
    corr = D.destreak_data(frame.copy(), median_filter_size=2048,
                           noutputs=1, **kwargs) - frame
    spread = float(np.max(np.abs(corr - corr[:, :1])))
    # 1e-3 MJy/sr is float32 round-off headroom on a ~20 MJy/sr frame and far
    # below what a 512 split produces (0.33 insitu / 3.77 bare on sickle o007).
    assert spread < 1e-3, (
        f'{mode}: correction varies by {spread:.4f} across a single-output '
        f'frame -- it was split at a boundary its readout does not have')


def test_a_width_the_outputs_do_not_divide_is_refused():
    with pytest.raises(ValueError, match='NOUTPUTS must divide'):
        D.amplifier_width(np.zeros((1024, 1030)), 4)   # 1030 % 4 == 2


def test_the_background_is_sampled_at_the_centre_of_each_chunk(tmp_path):
    """`add_background_map` puts back what `destreak_data` took off, so it
    chunks the same way and samples each chunk at ITS centre,
    ``start + width / 2``.

    A FULL frame makes that indistinguishable from a hardcoded ``start + 256``
    (2048 // 4 // 2 == 256), so only a subarray can pin it: a single-output
    640-wide frame is one chunk whose centre is column 320.  Nothing on disk is
    that combination today -- a SUB640 frame with a registered background map
    -- which is why this is regression insurance ahead of the case rather than
    a live gap.

    The background map here is a column ramp on the frame's own WCS, so the
    value sampled at ``pxx`` is ``pxx`` itself and the assertion reads the
    sampling column directly.
    """
    hdul = _frame_hdul(filtername='F187N',
                       data=np.zeros((640, 640), dtype='float32'))
    hdul[0].header['NOUTPUTS'] = 1
    hdul[0].header['SUBARRAY'] = 'SUB640'

    bgpath = tmp_path / 'bg_column_ramp.fits'
    ramp = np.repeat(np.arange(640, dtype='float32')[None, :], 640, axis=0)
    bghdu = fits.PrimaryHDU(ramp)
    for key in ('CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2'):
        bghdu.header[key] = hdul[('SCI', 1)].header[key]
    bghdu.writeto(bgpath)

    out = D.add_background_map(np.zeros((640, 640), dtype='float32'), hdul,
                               bgfile=str(bgpath))
    sampled = float(np.median(out))
    assert abs(sampled - 320.0) < 0.5, (
        f'background sampled at column {sampled:.1f}; the single 640-wide '
        f'chunk is centred on 320, and 256 is the full-frame constant')


def test_destreak_takes_the_chunk_width_from_the_frame(tmp_path):
    """NOUTPUTS comes off the header, so a subarray does not need a caller to
    know it is one."""
    frame = tmp_path / 'jw03958007001_03102_00001_nrcb1_cal.fits'
    hdul = _frame_hdul(program='03958', obs='007', filtername='F187N')
    hdul[0].header['NOUTPUTS'] = 1
    hdul[0].header['SUBARRAY'] = 'SUB640'
    rng = np.random.default_rng(5)
    hdul[('SCI', 1)].data = (8.0 + rng.normal(0, 0.4, (640, 640))).astype('float32')
    hdul.writeto(frame)

    out = D.destreak(str(frame), median_filter_size=2048, use_background_map=True)
    hdr = fits.getheader(out)
    assert hdr['DESTRKAW'] == 640
    assert hdr['DESTRKMD'] == 'insitu'


# --------------------------------------------------------------------------
# 3b. the two sky conventions are distinguishable in the product
# --------------------------------------------------------------------------

def test_the_output_records_which_sky_convention_it_carries(tmp_path):
    """Both generations are named `*_destreak.fits`.  ~6,200 products on disk
    were made under the bare subtraction and nothing in them says so."""
    frame = tmp_path / 'jw02092002001_02101_00001_nrca1_cal.fits'
    _frame_hdul(program='02092').writeto(frame)
    out = D.destreak(str(frame), median_filter_size=2048, use_background_map=True)
    hdr = fits.getheader(out)
    assert hdr['DESTRKMD'] == 'insitu'
    assert hdr['DESTRKAW'] == 512
    assert 'DESTRKBG' not in hdr


# --------------------------------------------------------------------------
# 4. destreak() picks the mode from the resolved map, not from the flag
# --------------------------------------------------------------------------

def test_destreak_without_a_registered_map_keeps_the_sky(tmp_path, capsys):
    """The regression that motivated all of this: a Cloud E/F-shaped frame
    (proposal 2092, absent from background_mapping) asked for a background map
    and got a sky subtraction.  It must now come out with its pedestal."""
    frame = tmp_path / 'jw02092002001_02101_00001_nrca1_cal.fits'
    _frame_hdul(program='02092').writeto(frame)
    before = fits.getdata(frame, ext=('SCI', 1)).astype('float64')

    D.destreak(str(frame), median_filter_size=2048, use_background_map=True)

    after = fits.getdata(str(frame).replace('_cal.fits', '_destreak.fits'),
                         ext=('SCI', 1)).astype('float64')
    assert abs(np.median(after) - np.median(before)) < 0.5
    assert np.mean(after < 0) < 0.01
    assert 'no background map registered' in capsys.readouterr().out


def test_destreak_with_a_registered_map_subtracts_in_full_then_restores(tmp_path):
    """The other branch still runs, and still runs in the full-subtraction mode."""
    bg = tmp_path / 'bgmap.fits'
    hdul = _frame_hdul(program='02221', obs='001', filtername='F212N')
    bgdata = np.full((2048, 2048), 7.0, dtype='float32')
    fits.PrimaryHDU(bgdata, header=hdul[('SCI', 1)].header).writeto(bg)

    mapping = {'2221': {'001': {'regionname': 'brick', 'f212n': 'bgmap.fits'}}}
    frame = tmp_path / 'jw02221001001_02101_00001_nrca1_cal.fits'
    hdul.writeto(frame)

    calls = {}
    real = D.destreak_data

    def spy(data, **kwargs):
        calls.update(kwargs)
        return real(data, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(D, 'destreak_data', spy):
        D.destreak(str(frame), use_background_map=True,
                   background_mapping=mapping,
                   background_folder=str(tmp_path))

    assert calls['add_smoothed'] is False
    assert calls['caller_restores_large_scales'] is True
    out = fits.getdata(str(frame).replace('_cal.fits', '_destreak.fits'),
                       ext=('SCI', 1))
    # subtracted to zero, then the constant map added on top
    assert abs(np.percentile(out, 10) - 7.0) < 0.2


# --------------------------------------------------------------------------
# 5. background_map_path: absent vs stale
# --------------------------------------------------------------------------

def test_no_registered_map_is_none_not_an_error(tmp_path):
    assert D.background_map_path(_frame_hdul(program='02092')[0].header,
                                 background_mapping={'2221': {'001': {}}},
                                 bgmap_path=str(tmp_path)) is None


def test_a_registered_filter_with_no_file_raises(tmp_path):
    """A stale entry is a configuration error.  The brick-1182 entries pointed
    at names renamed `.fits_stale` in 2023 and nobody noticed for two years
    because the old code warned and carried on."""
    mapping = {'2221': {'001': {'regionname': 'brick', 'f212n': 'gone.fits'}}}
    with pytest.raises(FileNotFoundError, match='gone.fits'):
        D.background_map_path(_frame_hdul(program='02221', obs='001',
                                          filtername='F212N')[0].header,
                              background_mapping=mapping,
                              bgmap_path=str(tmp_path))


def test_a_filter_with_no_entry_in_a_mapped_observation_is_none(tmp_path):
    """Cloud C's five unmapped filters: the observation is mapped, the filter
    is not.  That is 'no map', not 'stale map'."""
    mapping = {'2221': {'002': {'regionname': 'cloudc', 'f405n': 'x.fits'}}}
    assert D.background_map_path(_frame_hdul(program='02221', obs='002',
                                             filtername='F212N')[0].header,
                                 background_mapping=mapping,
                                 bgmap_path=str(tmp_path)) is None
