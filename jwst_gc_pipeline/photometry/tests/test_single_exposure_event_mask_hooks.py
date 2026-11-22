"""The single-exposure event mask in the mosaic builders (env
``SINGLE_EXPOSURE_EVENT_MASK``, off by default): the switch, the masks over crf
files, the residual DQ, the resample of masked copies, and the hand-over from
``build_mergedcat_residuals`` to the per-frame residual writer.  The detector
itself is tested in ``test_single_exposure_events``."""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as L
from jwst_gc_pipeline.photometry import single_exposure_events as se
from jwst_gc_pipeline.photometry.tests.test_cutout_mosaic_wcs import SHAPE, _write_frame
from jwst_gc_pipeline.photometry.tests.test_mergedcat_render_procs import _setup
from jwst_gc_pipeline.photometry.tests.test_mergedcat_satstar_cover import (
    E2E_FIELD, E2E_FWHM_PIX, E2E_ITER, E2E_PID, E2E_SHAPE,
    _e2e_grid, _e2e_options,
)
import jwst_gc_pipeline.reduction.filtering as filtering

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning'),
    pytest.mark.filterwarnings(
        'ignore:This process .* is multi-threaded:DeprecationWarning'),
]

ENV = 'SINGLE_EXPOSURE_EVENT_MASK'
NPIX = 160


@pytest.fixture(autouse=True)
def _no_cached_masks(monkeypatch):
    monkeypatch.delenv(se.ENV_NEGATIVE, raising=False)
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
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert se.event_mask_enabled() is expected


def test_switch_typo_raises_naming_the_variable(monkeypatch):
    monkeypatch.setenv(ENV, 'maybe')
    with pytest.raises(ValueError, match=ENV):
        se.event_mask_enabled()


# --- masks over crf files --------------------------------------------------------

def _crf(path, k, event=None):
    """A crf-like file (SCI + DQ, FITS WCS only) of three dithered exposures: sky
    noise, a real saturated star at the same sky position in all of them, and
    optionally ``event`` = (x, y): a SATURATED core in this frame only."""
    rng = np.random.default_rng(k)
    sci = 10.0 + rng.normal(0.0, 2.0, (NPIX, NPIX))
    dq = np.zeros((NPIX, NPIX), dtype=np.uint32)
    ww = WCS(naxis=2)
    ww.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    ww.wcs.crval = [266.5, -28.8]
    ww.wcs.cdelt = [-1e-5, 1e-5]
    ww.wcs.crpix = [80.5 + 3.0 * k, 80.5 - 2.0 * k]
    yy, xx = np.mgrid[0:NPIX, 0:NPIX]

    def add(x, y, peak):
        sci[:] += peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.1 ** 2))
        dq[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] |= 2

    add(40 + 3.0 * k, 40 - 2.0 * k, 30000.0)          # a star: the same sky spot
    if event is not None:
        add(*event, 20000.0)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(sci.astype('float32'), header=ww.to_header(), name='SCI'),
                  fits.ImageHDU(dq, name='DQ')]).writeto(path)
    return str(path)


def _three_frames(tmp_path):
    return [_crf(tmp_path / f'f{k}_crf.fits', k, event=(100, 90) if k == 1 else None)
            for k in range(3)]


def test_switch_off_reads_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(se, 'CrfFrame', lambda p: pytest.fail('read a frame'))
    assert L.single_exposure_event_masks(['nowhere_a.fits', 'nowhere_b.fits']) == {}


def test_masks_are_keyed_by_the_frame_that_carries_the_event(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV, '1')
    files = _three_frames(tmp_path)
    masks = L.single_exposure_event_masks(files)
    assert list(masks) == [files[1]]
    mask = masks[files[1]]
    assert mask.shape == (NPIX, NPIX) and mask.dtype == bool
    assert mask[90, 100] and mask[88:93, 98:103].all()
    assert not mask[40 - 2, 40 + 3]          # the star of frame 1 is not masked
    assert 'single-exposure events: 1 in 1 of 3 frames' in capsys.readouterr().out


def test_one_frame_cannot_test_anything(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, '1')
    assert L.single_exposure_event_masks(_three_frames(tmp_path)[:1]) == {}


def test_masks_are_kept_per_file_list_and_recomputed_when_a_file_changes(tmp_path,
                                                                          monkeypatch):
    monkeypatch.setenv(ENV, '1')
    files = _three_frames(tmp_path)
    first = L.single_exposure_event_masks(files)
    assert L.single_exposure_event_masks(list(reversed(files))) is first
    os.utime(files[0], ns=(1, 1))
    assert L.single_exposure_event_masks(files) is not first


# --- extreme negative pixels over crf files ----------------------------------------

def _crf_defects(path, k, defects=(), event=None):
    """``_crf`` with an ERR extension (2 MJy/sr everywhere) and ``defects``: a list
    of (x, y, value, dq) pixels written over the frame."""
    _crf(path, k, event=event)
    with fits.open(path, mode='update', memmap=False) as hdul:
        sci = hdul['SCI'].data.copy()
        dq = hdul['DQ'].data.copy()
        for x, y, value, bits in defects:
            sci[y, x] = value
            dq[y, x] = bits
        hdul['SCI'].data = sci
        hdul['DQ'].data = dq
        hdul.append(fits.ImageHDU(np.full((NPIX, NPIX), 2.0, dtype='float32'), name='ERR'))
    return str(path)


def test_negative_pixels_are_masked_in_their_own_frame_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(se.ENV_NEGATIVE, '50')
    files = [_crf_defects(tmp_path / 'f0_crf.fits', 0),
             _crf_defects(tmp_path / 'f1_crf.fits', 1, [(20, 30, -19000.0, 4)]),
             _crf_defects(tmp_path / 'f2_crf.fits', 2)]
    masks = L.single_exposure_event_masks(files)
    assert list(masks) == [files[1]]
    assert masks[files[1]].sum() == 1 and masks[files[1]][30, 20]
    assert ('negative pixels: 1 px below -50 ERR in 1 of 3 frames set to DO_NOT_USE'
            in capsys.readouterr().out)


def test_negative_pixels_are_tested_with_one_file_where_events_cannot_be(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, '1')
    monkeypatch.setenv(se.ENV_NEGATIVE, '50')
    (f,) = [_crf_defects(tmp_path / 'f0_crf.fits', 0, [(5, 6, -9000.0, 0)])]
    masks = L.single_exposure_event_masks([f])
    assert list(masks) == [f] and masks[f].sum() == 1 and masks[f][6, 5]


def test_event_and_negative_pixel_masks_are_one_mask_per_frame(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, '1')
    monkeypatch.setenv(se.ENV_NEGATIVE, '50')
    files = [_crf_defects(tmp_path / f'f{k}_crf.fits', k,
                          [(20, 30, -19000.0, 4)] if k == 1 else [],
                          event=(100, 90) if k == 1 else None)
             for k in range(3)]
    masks = L.single_exposure_event_masks(files)
    assert list(masks) == [files[1]]
    mask = masks[files[1]]
    assert mask[30, 20] and mask[90, 100] and mask[88:93, 98:103].all()
    ev_only = int(mask.sum()) - 1
    monkeypatch.delenv(se.ENV_NEGATIVE)
    L._EVENT_MASK_CACHE.clear()
    assert int(L.single_exposure_event_masks(files)[files[1]].sum()) == ev_only


def test_negative_pixel_masks_follow_the_switch_in_the_cache(tmp_path, monkeypatch):
    files = [_crf_defects(tmp_path / f'f{k}_crf.fits', k,
                          [(20, 30, -19000.0, 4)] if k == 0 else []) for k in range(2)]
    assert L.single_exposure_event_masks(files) == {}
    monkeypatch.setenv(se.ENV_NEGATIVE, '50')
    first = L.single_exposure_event_masks(files)
    assert list(first) == [files[0]]
    assert L.single_exposure_event_masks(files) is first
    monkeypatch.setenv(se.ENV_NEGATIVE, '1e6')
    assert L.single_exposure_event_masks(files) == {}


def test_negative_pixel_switch_typo_raises_naming_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(se.ENV_NEGATIVE, 'many')
    with pytest.raises(ValueError, match=se.ENV_NEGATIVE):
        L.single_exposure_event_masks([str(tmp_path / 'x.fits')])


# --- the residual DQ ---------------------------------------------------------------

def _model_file(tmp_path, name='src.fits', shape=(16, 16)):
    from astropy.modeling import models
    from gwcs import WCS as GWCS
    from stdatamodels.jwst.datamodels import ImageModel
    src = str(tmp_path / name)
    with ImageModel(np.zeros(shape, dtype='float32')) as m:
        m.meta.wcs = GWCS(models.Shift(0) & models.Shift(0),
                          input_frame='detector', output_frame='world')
        m.dq = np.zeros(shape, dtype='uint32')
        m.dq[3, 3] = 2                                  # a SATURATED pixel
        m.save(src)
    return src


def test_bad_mask_sets_do_not_use_in_the_written_dq(tmp_path):
    from stdatamodels.jwst.datamodels import ImageModel
    src = _model_file(tmp_path)
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:5, 2:5] = True
    out = str(tmp_path / 'resid.fits')
    L.save_residual_datamodel(src, out, np.full((16, 16), 7.0, dtype='float32'),
                              bad_mask=mask)
    with ImageModel(out) as m:
        dq = np.asarray(m.dq)
        assert float(m.data[0, 0]) == 7.0                # the data is not touched
    assert (dq[mask] & 1).all() and not (dq[~mask] & 1).any()
    assert dq[3, 3] == 3                                 # SATURATED kept, DO_NOT_USE added


def test_no_bad_mask_leaves_the_dq_as_it_was(tmp_path):
    from stdatamodels.jwst.datamodels import ImageModel
    src = _model_file(tmp_path)
    out = str(tmp_path / 'resid.fits')
    L.save_residual_datamodel(src, out, np.zeros((16, 16), dtype='float32'))
    with ImageModel(out) as m:
        assert np.asarray(m.dq).sum() == 2


def test_bad_mask_of_the_wrong_shape_and_with_clear_dq_are_refused(tmp_path):
    src = _model_file(tmp_path)
    out = str(tmp_path / 'resid.fits')
    data = np.zeros((16, 16), dtype='float32')
    with pytest.raises(ValueError, match='bad_mask'):
        L.save_residual_datamodel(src, out, data, bad_mask=np.zeros((8, 8), bool))
    with pytest.raises(ValueError, match='exclusive'):
        L.save_residual_datamodel(src, out, data, clear_dq=True,
                                  bad_mask=np.zeros((16, 16), bool))
    assert not os.path.exists(out)


# --- the resample --------------------------------------------------------------------

def _spiked_pair(tmp_path):
    """Two frames of one sky; frame 0 has a SATURATED-flagged spike (no DO_NOT_USE)
    at (40, 40), the case ResampleStep keeps."""
    paths = [_write_frame(str(tmp_path / f'f{k}_crf.fits'), dither_x=0.0, seed=k)
             for k in range(2)]
    with fits.open(paths[0], mode='update', memmap=False) as hdul:
        hdul['SCI'].data[40, 40] = 50000.0
        hdul['DQ'].data[40, 40] |= 2
    mask = np.zeros(SHAPE, dtype=bool)
    mask[38:43, 38:43] = True
    return paths, mask


def test_masked_copies_change_the_copy_only(tmp_path):
    paths, mask = _spiked_pair(tmp_path)
    work = tmp_path / 'work'
    work.mkdir()
    out = L._event_masked_copies(paths, {paths[0]: mask}, str(work))
    assert out[1] == paths[1] and out[0] == str(work / os.path.basename(paths[0]))
    with fits.open(out[0], memmap=False) as hdul:
        dq = np.asarray(hdul['DQ'].data)
        assert (dq[mask] & 1).all() and dq[40, 40] == 3 and not (dq[~mask] & 1).any()
        assert hdul['SCI'].data[40, 40] == 50000.0
    with fits.open(paths[0], memmap=False) as hdul:
        assert not (np.asarray(hdul['DQ'].data) & 1).any()        # the input is as it was


def test_masked_resample_drops_the_event_and_cleans_up(tmp_path):
    paths, mask = _spiked_pair(tmp_path)
    d = str(tmp_path)

    def peak(product):
        return float(np.nanmax(fits.getdata(os.path.join(d, f'{product}_i2d.fits'), 'SCI')))

    L._resample_to_i2d(paths, d, 'plain', crop_to_data=False)
    out = L._resample_event_masked(paths, {paths[0]: mask}, d, 'masked',
                                   crop_to_data=False)
    assert out == os.path.join(d, 'masked_i2d.fits')
    # the spike (50000 in one of two frames) enters the coadd at 25000 or at a quarter of
    # that if the output grid is half a pixel off, and is gone with the mask; what is
    # left is the +100 central source both frames carry
    assert peak('plain') > 2000.0
    assert peak('masked') < 200.0
    assert [n for n in os.listdir(d) if n.startswith('.eventmask_')] == []
    assert not os.path.exists(os.path.join(d, 'masked_asn.json'))


# --- build_mergedcat_residuals -> per-frame residual writer -------------------------

def _mask_of(frame_index):
    mask = np.zeros(E2E_SHAPE, dtype=bool)
    mask[10 + frame_index:14 + frame_index, 20:25] = True
    return mask


def _build(root, monkeypatch, masks_of):
    """Run build_mergedcat_residuals on the render-procs fixture with the detector
    replaced by ``masks_of(frames) -> {frame: mask}``; returns what the residual
    writer was called with: {output basename: bad_mask or 'no keyword'}."""
    for var in ('MERGEDCAT_SATSTAR_COVER', 'MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM',
                'MERGE_RENDER_PSF_SHAPE', 'MERGE_RENDER_FWHM_MULT',
                'MERGE_SATSTAR_RENDER_CAP', 'GC_INSTRUMENT_OVERRIDE'):
        monkeypatch.delenv(var, raising=False)
    root.mkdir()
    frames, merged = _setup(root)
    seen = {}

    def _save(input_filename, output_filename, data, clear_dq=False, **kw):
        seen[os.path.basename(output_filename)] = (kw.get('bad_mask', 'no keyword'),
                                                    clear_dq)

    monkeypatch.setattr(L.psf_preflight, 'preflight_psf_data', lambda *a, **k: None)
    monkeypatch.setattr(L, 'get_psf_model', lambda *a, module=None, **k: (_e2e_grid(), None))
    monkeypatch.setattr(filtering, 'get_fwhm', lambda hdr, *a, **k: (0.164, E2E_FWHM_PIX))
    monkeypatch.setattr(L, 'save_residual_datamodel', _save)
    monkeypatch.setattr(L, '_resample_to_i2d',
                        lambda files, pdir, name, **k: os.path.join(pdir, f'{name}_i2d.fits'))
    if masks_of is not None:
        monkeypatch.setattr(L, 'single_exposure_event_masks',
                            lambda files: masks_of(frames, files))
    L.build_mergedcat_residuals(
        str(root), str(root), merged, 'F480M', E2E_PID, E2E_FIELD, 'merged',
        _e2e_options(), frames, E2E_ITER, ['basic'], satstar_label=None,
        write_model_i2d=False)
    return frames, seen


def test_only_the_frame_with_an_event_gets_the_keyword_and_only_its_residual(
        tmp_path, monkeypatch):
    def masks_of(frames, files):
        assert sorted(files) == sorted(frames)               # full-frame run: the frames
        return {frames[1]: _mask_of(1)}

    frames, seen = _build(tmp_path / 'a', monkeypatch, masks_of)
    resid = {k: v for k, v in seen.items() if k.endswith('_mergedcat_residual.fits')}
    model = {k: v for k, v in seen.items() if k.endswith('_mergedcat_model.fits')}
    assert len(resid) == len(frames) == len(model)
    with_mask = [v[0] for v in resid.values() if not isinstance(v[0], str)]
    assert len(with_mask) == 1 and np.array_equal(with_mask[0], _mask_of(1))
    assert sum(isinstance(v[0], str) and v[0] == 'no keyword'
               for v in resid.values()) == len(frames) - 1
    assert all(v == ('no keyword', True) for v in model.values())   # models: never masked


def test_switch_off_calls_the_writer_as_before(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    frames, seen = _build(tmp_path / 'b', monkeypatch, None)       # the real detector entry
    assert len(seen) == 2 * len(frames)
    assert {v[0] for v in seen.values()} == {'no keyword'}


def test_cutout_run_masks_the_crops_and_maps_them_back_to_the_frames(monkeypatch, tmp_path):
    opts = _e2e_options()
    opts.cutout_region = 'x'
    monkeypatch.setattr(L, '_cutout_label_for', lambda o: 'lab')
    asked = []

    def masks(files):
        asked.extend(files)
        return {os.path.join('pdir', 'a_crf_cutout_lab.fits'): 'mask-a'}

    monkeypatch.setattr(L, 'single_exposure_event_masks', masks)
    got = L._event_masks_by_frame(['/x/a_crf.fits', '/x/b_crf.fits'], 'pdir', opts)
    assert asked == [os.path.join('pdir', 'a_crf_cutout_lab.fits'),
                     os.path.join('pdir', 'b_crf_cutout_lab.fits')]
    assert got == {'/x/a_crf.fits': 'mask-a'}
