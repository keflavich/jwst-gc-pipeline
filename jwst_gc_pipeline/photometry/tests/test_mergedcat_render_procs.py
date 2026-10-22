"""build_mergedcat_residuals with --mergedcat-render-procs: forked processes
render the frames, and every per-frame residual and model is the serial
render's, bit for bit (#1230).

Four synthetic NIRCam LW frames on two detectors, each seeing the merged
catalog's stars at different pixel positions, are rendered once serially and
once in three processes.  Only the PSF-grid builder, the OPD preflight, the
FWHM lookup, the JWST-datamodel writer and the resample are replaced (as in
test_mergedcat_satstar_cover's end-to-end tests); the writer saves each array
to disk with the pid that wrote it, so the test can tell which process
rendered which frame.
"""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as ccl
import jwst_gc_pipeline.reduction.filtering as filtering
from jwst_gc_pipeline.mast_names import jw_prefix
from jwst_gc_pipeline.photometry.naming import _inst_token, frame_identity
from jwst_gc_pipeline.photometry.tests.test_mergedcat_satstar_cover import (
    E2E_FIELD, E2E_FWHM_PIX, E2E_ITER, E2E_PID, E2E_SHAPE,
    _e2e_grid, _e2e_options, _e2e_wcs,
)

pytestmark = [
    pytest.mark.filterwarnings(
        'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning'),
    pytest.mark.filterwarnings(
        'ignore:This process .* is multi-threaded:DeprecationWarning'),
]

DETECTORS = ['nrcalong', 'nrcblong', 'nrcalong', 'nrcblong']


def _setup(root):
    """Write the frames, their raw basic residual/model pairs, and the merged
    catalog under ``root``.  Same seed every call, so two calls write the
    same inputs."""
    rng = np.random.default_rng(5)
    pipeline_dir = root / 'F480M' / 'pipeline'
    pipeline_dir.mkdir(parents=True)
    nstar = 25
    px, py = rng.uniform(4, 60, nstar), rng.uniform(4, 60, nstar)
    sky = _e2e_wcs().pixel_to_world(px, py)
    merged = str(root / 'merged_f480m_m7.fits')
    Table({'ra': sky.ra.deg, 'dec': sky.dec.deg,
           'flux_fit': rng.lognormal(9.0, 1.0, nstar),
           'replaced_saturated': np.zeros(nstar, bool)}).write(merged)

    options = _e2e_options()
    phdr = fits.Header({'INSTRUME': 'NIRCAM', 'TELESCOP': 'JWST',
                        'DATE-OBS': '2023-04-01', 'FILTER': 'F480M'})
    frames = []
    for k, det in enumerate(DETECTORS):
        ww = _e2e_wcs()
        # each frame sees the stars at other sub-pixel positions
        ww.wcs.crpix = [32.5 + 0.37 * k, 32.5 - 0.21 * k]
        frame = str(pipeline_dir / f'jw01234001001_02101_{k + 1:05d}_{det}_crf.fits')
        fits.HDUList([fits.PrimaryHDU(header=phdr),
                      fits.ImageHDU(np.zeros(E2E_SHAPE, 'float32'),
                                    header=ww.to_header(), name='SCI'),
                      fits.ImageHDU(np.ones(E2E_SHAPE, 'float32'), name='ERR'),
                      ]).writeto(frame)
        visit, vgroup, exposure, fdet = frame_identity(frame, field=E2E_FIELD)
        tokens = ccl._predict_output_tokens(options, visit, vgroup, exposure,
                                            E2E_ITER)
        stem = (f'{pipeline_dir}/{jw_prefix(E2E_PID)}-o{E2E_FIELD}_t001_'
                f'{_inst_token("F480M")}_clear-f480m-{fdet}{"".join(tokens)}'
                f'_daophot_basic')
        for suffix, loc in (('residual', 100.0), ('model', 0.0)):
            data = rng.normal(loc, 1.0, E2E_SHAPE).astype('float32')
            fits.HDUList([fits.PrimaryHDU(),
                          fits.ImageHDU(data, header=ww.to_header(), name='SCI'),
                          ]).writeto(f'{stem}_{suffix}.fits')
        frames.append(frame)
    return frames, merged


def _render(root, monkeypatch, n_procs):
    for var in ('MERGEDCAT_SATSTAR_COVER', 'MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM',
                'MERGE_RENDER_PSF_SHAPE', 'MERGE_RENDER_FWHM_MULT',
                'MERGE_SATSTAR_RENDER_CAP', 'GC_INSTRUMENT_OVERRIDE'):
        monkeypatch.delenv(var, raising=False)
    root.mkdir()
    frames, merged = _setup(root)
    grid = _e2e_grid()
    loads = root / 'loads'
    loads.mkdir()

    def _get_psf_model(*args, module=None, **kwargs):
        (loads / f'{os.getpid()}_{module}').write_text('')
        return grid, None

    def _save(input_filename, output_filename, data, clear_dq=False):
        np.save(f'{output_filename}.npy', np.asarray(data))
        with open(f'{output_filename}.pid', 'w') as fh:
            fh.write(str(os.getpid()))

    resampled = []

    def _resample(files, pipeline_dir, name, **kwargs):
        resampled.append([os.path.basename(f) for f in files])
        return os.path.join(pipeline_dir, f'{name}_i2d.fits')

    monkeypatch.setattr(ccl.psf_preflight, 'preflight_psf_data',
                        lambda *a, **k: None)
    monkeypatch.setattr(ccl, 'get_psf_model', _get_psf_model)
    monkeypatch.setattr(filtering, 'get_fwhm',
                        lambda hdr, *a, **k: (0.164, E2E_FWHM_PIX))
    monkeypatch.setattr(ccl, 'save_residual_datamodel', _save)
    monkeypatch.setattr(ccl, '_resample_to_i2d', _resample)

    options = _e2e_options()
    options.mergedcat_render_procs = n_procs
    record = []
    out = ccl.build_mergedcat_residuals(
        str(root), str(root), merged, 'F480M', E2E_PID, E2E_FIELD, 'merged',
        options, frames, E2E_ITER, ['basic'], satstar_label=None,
        write_model_i2d=False, perframe_record=record)
    assert list(out) == ['basic']
    arrays, pids = [], set()
    for rec in record:
        for path in rec['rendered']:
            arrays.append(np.load(f'{path}.npy'))
            with open(f'{path}.pid') as fh:
                pids.add(int(fh.read()))
    names = [[os.path.basename(p) for p in rec['rendered']] for rec in record]
    return dict(arrays=arrays, pids=pids, names=names, resampled=resampled,
                loads=sorted(p.name for p in loads.iterdir()))


def test_forked_render_equals_the_serial_render(tmp_path, monkeypatch, capfd):
    serial = _render(tmp_path / 'serial', monkeypatch, 1)
    forked = _render(tmp_path / 'forked', monkeypatch, 3)

    assert len(serial['arrays']) == len(forked['arrays']) == 2 * len(DETECTORS)
    for a, b in zip(serial['arrays'], forked['arrays']):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    # frames differ from each other, so a reordering could not pass
    assert not np.array_equal(serial['arrays'][0], serial['arrays'][2])
    # same per-frame products, recorded and resampled in frame order
    assert forked['names'] == serial['names']
    assert forked['resampled'] == serial['resampled']
    assert [n[0] for n in serial['names']] == [
        n for n in serial['resampled'][0]]

    # the serial render stays in this process; the forked one does not
    assert serial['pids'] == {os.getpid()}
    assert os.getpid() not in forked['pids']
    # both detectors' grids were loaded here, before the fork
    assert forked['loads'] == serial['loads'] == sorted(
        f'{os.getpid()}_{d}' for d in set(DETECTORS))
    assert 'rendering 4 frames across 3 processes (--mergedcat-render-procs)' in (
        capfd.readouterr().out)
