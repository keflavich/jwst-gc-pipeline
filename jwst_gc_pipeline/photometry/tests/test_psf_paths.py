"""Tests for the centralized PSF-grid path resolver (psf_paths.py).

Pins the central-first / legacy-fallback read order and the central naming
(keyed only by instrument+module+filter+oversample+blur, not proposal/field)
so the shared store can't silently regress to per-field duplication.
"""
from jwst_gc_pipeline.photometry import psf_paths as PP

ROOT = '/jwst'


def test_central_name_keyed_by_physics_only():
    p = PP.central_merged_psf_grid_path(ROOT, 'NIRCam', 'nrcb5', 'f405n',
                                        oversample=1, blur=False)
    assert p == '/jwst/psfs_shared/nircam_nrcb5_F405N_PSFgrid_oversample1.fits'


def test_central_name_blur_and_oversample():
    p = PP.central_merged_psf_grid_path(ROOT, 'MIRI', 'mirimage', 'f770w',
                                        oversample=2, blur=True)
    assert p == '/jwst/psfs_shared/miri_mirimage_F770W_PSFgrid_oversample2_blur.fits'


def test_legacy_path_keeps_proposal_and_field():
    p = PP.legacy_merged_psf_grid_path(ROOT, 'brick', 'f405n', '1182', '001',
                                       oversample=1, blur=False)
    assert p == '/jwst/brick/psfs/F405N_1182_001_merged_PSFgrid_oversample1.fits'


def _resolve(present):
    return PP.resolve_merged_psf_grid_path(
        ROOT, 'brick', 'NIRCam', 'nrcb5', 'f405n', '1182', '001',
        exists=lambda path: path in present)


def test_resolve_prefers_central_when_present():
    central = PP.central_merged_psf_grid_path(ROOT, 'NIRCam', 'nrcb5', 'f405n')
    legacy = PP.legacy_merged_psf_grid_path(ROOT, 'brick', 'f405n', '1182', '001')
    assert _resolve({central, legacy}) == central


def test_resolve_falls_back_to_legacy():
    legacy = PP.legacy_merged_psf_grid_path(ROOT, 'brick', 'f405n', '1182', '001')
    assert _resolve({legacy}) == legacy


def test_resolve_returns_central_when_neither_exists():
    central = PP.central_merged_psf_grid_path(ROOT, 'NIRCam', 'nrcb5', 'f405n')
    assert _resolve(set()) == central
