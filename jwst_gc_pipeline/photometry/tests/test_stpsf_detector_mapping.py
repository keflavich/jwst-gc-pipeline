"""Unit tests for stpsf_detector_for_module: the single module->detector
mapping used by BOTH the PSF-grid cache lookup and the MAST/Poppy download
branch in get_psf_model.

Regression: the download branch used to fall through to ``module.upper()`` for
LW per-frame modules, producing 'NRCALONG'/'NRCBLONG' — not valid stpsf
detector names — so any cold-cache LW per-frame build raised instead of
downloading the grid.
"""
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    stpsf_detector_for_module,
)


def test_miri_always_mirim():
    assert stpsf_detector_for_module('mirimage', 'F770W', 'MIRI') == 'MIRIM'
    assert stpsf_detector_for_module('merged', 'F2550W', 'MIRI') == 'MIRIM'


def test_module_level_lw_filters_map_to_5():
    # module-level tokens with an LW filter (F3xx/F4xx) -> detector 5
    assert stpsf_detector_for_module('nrca', 'F405N', 'NIRCam') == 'NRCA5'
    assert stpsf_detector_for_module('nrcb', 'F466N', 'NIRCam') == 'NRCB5'
    assert stpsf_detector_for_module('nrca', 'F356W', 'NIRCam') == 'NRCA5'


def test_module_level_sw_filters_map_to_1():
    assert stpsf_detector_for_module('nrca', 'F182M', 'NIRCam') == 'NRCA1'
    assert stpsf_detector_for_module('nrcb', 'F212N', 'NIRCam') == 'NRCB1'


def test_long_detector_tokens_map_to_5_not_nrcalong():
    # THE bug: per-frame LW path passes the physical detector token; stpsf has
    # no 'NRCALONG'/'NRCBLONG' detector, so these must map to NRCA5/NRCB5.
    assert stpsf_detector_for_module('nrcalong', 'F405N', 'NIRCam') == 'NRCA5'
    assert stpsf_detector_for_module('nrcblong', 'F466N', 'NIRCam') == 'NRCB5'
    # case-insensitive on the module token
    assert stpsf_detector_for_module('NRCALONG', 'F405N', 'NIRCam') == 'NRCA5'


def test_sw_physical_detectors_map_directly():
    for det in ('nrca1', 'nrca2', 'nrca3', 'nrca4',
                'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'):
        assert stpsf_detector_for_module(det, 'F182M', 'NIRCam') == det.upper()


def test_merged_returns_none_for_all_detectors_path():
    assert stpsf_detector_for_module('merged', 'F405N', 'NIRCam') is None
