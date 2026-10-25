"""The merged-catalog render uses each frame's own detector PSF grid.

The per-frame fit loads its grid with ``module=<frame detector>`` (cataloging
sets ``file_module = file_detector``).  ``build_mergedcat_residuals`` used to
load ONE grid with the run's ``module``: ``'nrcb'`` maps to NRCB1, so nrcb2-4
were rendered with the NRCB1 PSF, and ``'merged'`` gave every frame the
channel's first grid.  A model rendered with a PSF other than the fitted one
leaves a ring or a hole at bright stars.
"""
import threading
import time

import numpy as np
import pytest
from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as ccl
import jwst_gc_pipeline.reduction.filtering as filtering
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import _RenderPSFGrids
from jwst_gc_pipeline.photometry.tests.test_mergedcat_satstar_cover import (
    E2E_FIELD, E2E_FWHM_PIX, E2E_ITER, E2E_PID, E2E_SATLABEL, E2E_SHAPE,
    _e2e_grid, _e2e_options, _e2e_setup)


def _recording_loader():
    calls = []

    def load(detector):
        calls.append(detector)
        return f'grid:{detector}'
    return load, calls


def test_sw_detectors_each_get_their_own_grid():
    load, calls = _recording_loader()
    grids = _RenderPSFGrids(load, 'F212N', 'NIRCam')
    got = [grids(d) for d in ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4')]
    assert got == ['grid:nrcb1', 'grid:nrcb2', 'grid:nrcb3', 'grid:nrcb4']
    assert calls == ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
    assert grids.loaded == ['NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']


def test_a_grid_is_loaded_once_per_detector():
    load, calls = _recording_loader()
    grids = _RenderPSFGrids(load, 'F212N', 'NIRCam')
    for _ in range(3):
        for d in ('nrca1', 'nrcb3'):
            grids(d)
    assert calls == ['nrca1', 'nrcb3']


def test_long_wave_tokens_naming_the_same_grid_share_one_load():
    load, calls = _recording_loader()
    grids = _RenderPSFGrids(load, 'F480M', 'NIRCam')
    assert grids('nrcblong') == 'grid:nrcblong'
    assert grids('nrcb5') == 'grid:nrcblong'
    assert grids('nrcalong') == 'grid:nrcalong'
    assert calls == ['nrcblong', 'nrcalong']
    assert grids.loaded == ['NRCA5', 'NRCB5']


def test_miri_frames_map_to_the_one_miri_grid():
    load, calls = _recording_loader()
    grids = _RenderPSFGrids(load, 'F770W', 'MIRI')
    grids('mirimage')
    grids('mirimage')
    assert calls == ['mirimage']
    assert grids.loaded == ['MIRIM']


def test_concurrent_frames_load_each_grid_once():
    calls = []

    def slow_load(detector):
        calls.append(detector)
        time.sleep(0.02)
        return f'grid:{detector}'
    grids = _RenderPSFGrids(slow_load, 'F212N', 'NIRCam')
    dets = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'] * 8
    out = {}

    def worker(i, d):
        out[i] = grids(d)
    threads = [threading.Thread(target=worker, args=(i, d))
               for i, d in enumerate(dets)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(calls) == ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
    assert all(out[i] == f'grid:{d}' for i, d in enumerate(dets))


def _wide_grid():
    """A grid unlike the frame's: twice the FWHM."""
    sigma = 2.0 * E2E_FWHM_PIX / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    yy, xx = np.mgrid[-12:13, -12:13]
    psf = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    psf /= psf.sum()
    ny, nx = E2E_SHAPE
    xy = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    return GriddedPSFModel(NDData(np.stack([psf] * len(xy)),
                                  meta={'grid_xypos': xy, 'oversampling': 1}))


@pytest.mark.parametrize('module', ['nrcb', 'merged'])
@pytest.mark.filterwarnings(
    'ignore::jwst_gc_pipeline.frame_wcs.MissingGwcsWarning')
def test_render_uses_the_frame_detector_grid_not_the_run_module(
        tmp_path, monkeypatch, module):
    """End to end: the frame is nrcblong, its residual base was made with the
    nrcblong grid, and the run module is 'nrcb' / 'merged'.  Only the frame
    detector's grid subtracts the stars cleanly; the module's grid (here a PSF
    twice as wide) leaves rings."""
    # 'accepted' cover: B is rendered from the catalog, so a clean residual
    # needs both B and C subtracted with the fitted PSF.
    monkeypatch.setenv('MERGEDCAT_SATSTAR_COVER', 'accepted')
    for var in ('MERGEDCAT_SATSTAR_COVER_RADIUS_FWHM',
                'MERGE_RENDER_PSF_SHAPE', 'MERGE_RENDER_FWHM_MULT',
                'MERGE_SATSTAR_RENDER_CAP', 'GC_INSTRUMENT_OVERRIDE'):
        monkeypatch.delenv(var, raising=False)
    right = _e2e_grid()
    frame, merged = _e2e_setup(tmp_path, right)
    requested = []

    def fake_get_psf_model(filtername, proposal_id, field, module=None, **kw):
        requested.append(module)
        return (right if module == 'nrcblong' else _wide_grid()), None

    monkeypatch.setattr(ccl, 'get_psf_model', fake_get_psf_model)
    monkeypatch.setattr(ccl.psf_preflight, 'preflight_psf_data',
                        lambda *a, **k: None)
    monkeypatch.setattr(filtering, 'get_fwhm',
                        lambda hdr, *a, **k: (0.164, E2E_FWHM_PIX))
    saved = {}

    def _save(input_filename, output_filename, data, clear_dq=False):
        saved[output_filename] = np.array(data, dtype=float)
    monkeypatch.setattr(ccl, 'save_residual_datamodel', _save)
    monkeypatch.setattr(ccl, '_resample_to_i2d',
                        lambda files, pdir, name, **k: f'{pdir}/{name}_i2d.fits')

    ccl.build_mergedcat_residuals(
        str(tmp_path), str(tmp_path), merged, 'F480M', E2E_PID, E2E_FIELD,
        module, _e2e_options(), [frame], E2E_ITER, ['basic'],
        satstar_label=E2E_SATLABEL, write_model_i2d=False)

    assert requested == ['nrcblong']
    resid = [v for k, v in saved.items() if k.endswith('_mergedcat_residual.fits')]
    assert len(resid) == 1
    # B (3e4) and C (2e4) rendered with the fitted PSF cancel to round-off;
    # with the wide grid the residual at B's peak is ~70% of B's peak.
    peak_b = 3.0e4 * float(right.data[0].max())
    assert np.abs(resid[0]).max() < 1e-4 * peak_b
