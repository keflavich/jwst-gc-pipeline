"""Regression: the merged / all-detectors PSF path must honor the on-disk grid
cache instead of rebuilding via MAST + nrc.psf_grid(all_detectors=True).

Before this fix, ``get_psf_model(module='merged', use_webbpsf=True)`` skipped the
per-detector cache lookup entirely (stpsf_detector_for_module('merged') is None),
fell through to MAST login + a full-channel ``psf_grid`` rebuild, and overwrote
the identical grids already sitting on disk -- ~7-8 h/run on brick, once per
(module, filter) merged-residual phase.

These tests build fake per-detector cache files and assert the merged path loads
them and NEVER reaches the MAST rebuild.  The rebuild is booby-trapped (expanduser
-> nonexistent token) so a regression raises FileNotFoundError instead of silently
re-downloading.
"""
import os

import pytest

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as ccl


class _FakeGrid:
    """Stand-in for a stpsf GriddedPSFModel returned from the cache."""

    def __init__(self, path):
        self.path = path


@pytest.fixture
def _trap_mast(monkeypatch):
    """Make the MAST-rebuild fallback fail loudly, and make the on-disk loader
    return a sentinel keyed on the file it was asked to load."""
    monkeypatch.setattr(ccl, 'to_griddedpsfmodel',
                        lambda fn: _FakeGrid(os.path.basename(fn)))
    # get_psf_model's rebuild branch opens os.path.expanduser('~/.mast_api_token')
    # first; point it at a path that cannot exist so any fall-through raises.
    monkeypatch.setattr(ccl.os.path, 'expanduser',
                        lambda p: '/nonexistent/should-not-reach-mast')


def _write_grid(psfdir, det, filt, samp=2):
    fn = os.path.join(
        psfdir, f'nircam_{det.lower()}_{filt.lower()}_fovp101_samp{samp}_npsf16.fits')
    with open(fn, 'w') as fh:
        fh.write('fake')
    return fn


def test_merged_lw_loads_from_cache_grid0_is_nrca5(tmp_path, _trap_mast):
    psfdir = str(tmp_path)
    _write_grid(psfdir, 'NRCA5', 'F356W')
    _write_grid(psfdir, 'NRCB5', 'F356W')
    grid, _model = ccl.get_psf_model(
        'F356W', '1182', '004', module='merged', use_webbpsf=True,
        use_grid=True, instrument='NIRCam', psf_cache_dir=psfdir)
    # downstream collapses the list to grid[0]; for LW that must be NRCA5
    # (webbpsf detector_list order), NOT a rebuilt grid.
    assert isinstance(grid, _FakeGrid)
    assert grid.path == 'nircam_nrca5_f356w_fovp101_samp2_npsf16.fits'


def test_merged_sw_loads_from_cache_grid0_is_nrca1(tmp_path, _trap_mast):
    psfdir = str(tmp_path)
    for det in ('NRCA1', 'NRCA2', 'NRCA3', 'NRCA4',
                'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4'):
        _write_grid(psfdir, det, 'F200W')
    grid, _model = ccl.get_psf_model(
        'F200W', '1182', '004', module='merged', use_webbpsf=True,
        use_grid=True, instrument='NIRCam', psf_cache_dir=psfdir)
    assert isinstance(grid, _FakeGrid)
    assert grid.path == 'nircam_nrca1_f200w_fovp101_samp2_npsf16.fits'


def test_merged_incomplete_cache_falls_through_to_rebuild(tmp_path, _trap_mast):
    # Only one of the two LW detectors cached -> the merged loader must NOT
    # partially load; it falls through to the (trapped) rebuild.
    psfdir = str(tmp_path)
    _write_grid(psfdir, 'NRCA5', 'F444W')  # NRCB5 missing
    with pytest.raises(FileNotFoundError):
        ccl.get_psf_model(
            'F444W', '1182', '004', module='merged', use_webbpsf=True,
            use_grid=True, instrument='NIRCam', psf_cache_dir=psfdir)
