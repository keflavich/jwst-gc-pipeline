"""The integration-consistency (ramp) event mask in the mosaic builders (env
``SINGLE_EXPOSURE_EVENT_RAMP``, off by default): the switch, the masks over crf
files with a sibling ``_ramp.fits``, and its union with the cross-frame test.  The
detector itself is tested in ``test_single_exposure_events_ramp``."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as L
from jwst_gc_pipeline.photometry import single_exposure_events as se

pytestmark = pytest.mark.filterwarnings(
    'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')

ENV = se.ENV_RAMP
CROSS = se.ENV_FLAG
NPIX = 96
NGROUP = 4


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in (ENV, CROSS, se.ENV_NEGATIVE):
        monkeypatch.delenv(var, raising=False)
    L._EVENT_MASK_CACHE.clear()
    yield
    L._EVENT_MASK_CACHE.clear()


# --- the switch ----------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    (None, False), ('', False), ('  ', False),
    ('1', True), ('true', True), ('TRUE', True), (' Yes ', True), ('on', True),
    ('0', False), ('false', False), ('No', False), ('OFF', False),
])
def test_switch_values(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert se.ramp_event_enabled() is expected


def test_switch_typo_raises_naming_the_variable(monkeypatch):
    monkeypatch.setenv(ENV, 'maybe')
    with pytest.raises(ValueError, match=ENV):
        se.ramp_event_enabled()


def test_the_two_switches_are_independent(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    assert se.ramp_event_enabled() and not se.event_mask_enabled()
    monkeypatch.setenv(CROSS, 'yes')
    assert se.ramp_event_enabled() and se.event_mask_enabled()


# --- masks over crf files with sibling ramps -----------------------------------------

def _profile(x, y, sigma=1.1):
    yy, xx = np.mgrid[0:NPIX, 0:NPIX]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def _exposure(tmp_path, k, hit=None, saturated_event=None, ramp=True, detector='nrca1'):
    """crf + sibling ramp of one exposure: sky noise, a star at the same sky position in
    every exposure (the frames are dithered by (3k, -2k) pixels), and optionally
    ``hit`` = (integration, x, y): counts in one integration only (the crf shows
    their mean) and ``saturated_event`` = (x, y): the same in the crf with a
    SATURATED core, for the cross-frame test."""
    rng = np.random.default_rng(k)
    cube = rng.normal(0.0, 3.0, (2, NGROUP, NPIX, NPIX))
    star = _profile(40 + 3.0 * k, 40 - 2.0 * k)
    for g in range(NGROUP):
        cube[:, g] += 20000.0 * star * (g + 1)
    if hit is not None:
        i, x, y = hit
        for g in range(NGROUP):
            cube[i, g] += 20000.0 * _profile(x, y) * (g + 1)
    sci = 10.0 + rng.normal(0.0, 2.0, (NPIX, NPIX)) + 0.1 * cube.mean(axis=1).mean(axis=0)
    dq = np.zeros((NPIX, NPIX), dtype=np.uint32)
    if saturated_event is not None:
        x, y = saturated_event
        sci += 2000.0 * _profile(x, y)
        dq[y - 2:y + 3, x - 2:x + 3] |= se.SATURATED
    ww = WCS(naxis=2)
    ww.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    ww.wcs.crval = [266.5, -28.8]
    ww.wcs.cdelt = [-1e-5, 1e-5]
    ww.wcs.crpix = [NPIX / 2 + 0.5 + 3.0 * k, NPIX / 2 + 0.5 - 2.0 * k]
    crf = tmp_path / f'jw_e{k}_{detector}_destreak_o001_crf.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(sci.astype('float32'), header=ww.to_header(), name='SCI'),
                  fits.ImageHDU(dq, name='DQ')]).writeto(crf)
    if ramp:
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cube.astype('float32'), name='SCI')]
                     ).writeto(tmp_path / f'jw_e{k}_{detector}_ramp.fits')
    return str(crf)


def _three(tmp_path, **kw):
    return [_exposure(tmp_path, 0), _exposure(tmp_path, 1, hit=(1, 70, 60), **kw),
            _exposure(tmp_path, 2)]


def test_switch_off_reads_no_file(monkeypatch):
    monkeypatch.setattr(se, 'CrfFrame', lambda p: pytest.fail('read a frame'))
    assert L.single_exposure_event_masks(['nowhere_a.fits', 'nowhere_b.fits']) == {}


def test_mask_is_keyed_by_the_frame_with_the_hit_and_covers_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV, '1')
    files = _three(tmp_path)
    masks = L.single_exposure_event_masks(files)
    assert list(masks) == [files[1]]
    mask = masks[files[1]]
    assert mask.shape == (NPIX, NPIX) and mask.dtype == bool
    assert mask[60, 70] and mask[58:63, 68:73].all()
    assert not mask[40 - 2, 40 + 3]                   # the star of frame 1 is not masked
    out = capsys.readouterr().out
    assert 'single-exposure ramp events: 1 in 1 of 3 frames tested (0 of 3 without a usable ramp)' in out
    assert 'single-exposure events:' not in out       # the cross-frame test is off


def test_one_file_is_enough_for_the_ramp_test(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, '1')
    files = _three(tmp_path)
    masks = L.single_exposure_event_masks(files[1:2])
    assert list(masks) == [files[1]] and masks[files[1]][60, 70]


def test_frames_without_a_sibling_ramp_are_counted_and_not_masked(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV, '1')
    files = [_exposure(tmp_path, k, hit=(1, 70, 60) if k == 1 else None, ramp=(k != 1))
             for k in range(3)]
    assert L.single_exposure_event_masks(files) == {}
    out = capsys.readouterr().out
    assert 'single-exposure ramp events: 0 in 0 of 2 frames tested (1 of 3 without a usable ramp)' in out


def test_both_switches_give_the_union_per_frame(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV, '1')
    monkeypatch.setenv(CROSS, '1')
    files = [_exposure(tmp_path, 0, saturated_event=(20, 80)),
             _exposure(tmp_path, 1, hit=(1, 70, 60)),
             _exposure(tmp_path, 2)]
    masks = L.single_exposure_event_masks(files)
    assert set(masks) == {files[0], files[1]}
    assert masks[files[0]][80, 20] and masks[files[1]][60, 70]
    out = capsys.readouterr().out
    assert 'single-exposure events: 1 in 1 of 3 frames' in out
    assert 'single-exposure ramp events: 1 in 1 of 3 frames tested' in out


def test_the_same_event_found_by_both_tests_is_one_mask(tmp_path, monkeypatch):
    # frame 0 carries a SATURATED core that holds counts in one integration only
    monkeypatch.setenv(CROSS, '1')
    files = [_exposure(tmp_path, 0, hit=(0, 20, 80), saturated_event=(20, 80)),
             _exposure(tmp_path, 1), _exposure(tmp_path, 2)]
    cross_only = L.single_exposure_event_masks(files)
    L._EVENT_MASK_CACHE.clear()
    monkeypatch.setenv(ENV, '1')
    both = L.single_exposure_event_masks(files)
    assert list(both) == [files[0]]
    assert both[files[0]][80, 20] and both[files[0]].sum() >= cross_only[files[0]].sum()
    assert (both[files[0]] | cross_only[files[0]]).sum() == both[files[0]].sum()


def test_masks_follow_the_ramp_switch_in_the_cache(tmp_path, monkeypatch):
    files = _three(tmp_path)
    monkeypatch.setenv(CROSS, '1')
    cross_only = L.single_exposure_event_masks(files)
    assert cross_only == {}                            # no SATURATED core, nothing to find
    monkeypatch.setenv(ENV, '1')
    with_ramp = L.single_exposure_event_masks(files)
    assert list(with_ramp) == [files[1]]


def test_cutout_crops_cannot_be_tested(tmp_path, monkeypatch, capsys):
    # a crop of the exposure: the sibling ramp of the full detector is not on its grid
    monkeypatch.setenv(ENV, '1')
    files = _three(tmp_path)
    crops = []
    for f in files:
        with fits.open(f) as hdul:
            sci = hdul['SCI'].data[:60, :60]
            dq = hdul['DQ'].data[:60, :60]
            header = hdul['SCI'].header.copy()
        crop = f.replace('_destreak_o001_crf.fits', '_destreak_o001_crf_cutout_lab.fits')
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(sci, header=header, name='SCI'),
                      fits.ImageHDU(dq, name='DQ')]).writeto(crop)
        crops.append(crop)
    assert L.single_exposure_event_masks(crops) == {}
    assert '(3 of 3 without a usable ramp)' in capsys.readouterr().out
