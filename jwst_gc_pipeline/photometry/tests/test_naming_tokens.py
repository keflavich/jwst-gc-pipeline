"""Unit tests for the filename-token single-source-of-truth (naming.py).

These tokens define the on-disk product naming convention; a silent change
breaks skip-if-done prediction and every downstream glob.
"""
import pytest

from jwst_gc_pipeline.photometry import naming as N


def test_instrument_from_filter():
    assert N._instrument_from_filter('F770W') == 'MIRI'
    assert N._instrument_from_filter('f2550w') == 'MIRI'
    assert N._instrument_from_filter('F182M') == 'NIRCam'
    assert N._inst_token('F182M') == 'nircam'
    assert N._svo_filter_id('f480m') == 'JWST/NIRCam.F480M'
    assert N._svo_filter_id('F770W') == 'JWST/MIRI.F770W'


def test_chunk_token_roundtrip():
    assert N._chunk_token(0, 1) == ''
    assert N._chunk_token(0, None) == ''
    tok = N._chunk_token(3, 8)
    assert tok == '_chunk03of08'
    assert N._strip_chunk(f'iter3{tok}') == 'iter3'
    assert N._strip_chunk('iter3') == 'iter3'
    assert N._strip_chunk(None) is None
    with pytest.raises(ValueError):
        N._chunk_token(8, 8)


def test_iteration_and_bgsub_tokens():
    assert N._iteration_token('') == ''
    assert N._iteration_token(None) == ''
    assert N._iteration_token('m5') == '_m5'
    assert N._iteration_token('_m5') == '_m5'
    assert N._bgsub_token_from_flags(False) == ''
    assert N._bgsub_token_from_flags(True) == '_bgsub'
    assert N._bgsub_token_from_flags(True, resbgsub=True) == '_bgsub_resbgsub'
    assert N._bgsub_token_from_flags(False, resbgsub=True) == '_resbgsub'


def test_residual_i2d_name_family():
    base = '/x/jw02221-o001_clear-f182m-merged_m6_mergedcat_residual_i2d.fits'
    bg = N.residual_to_smoothed_bg_i2d(base)
    assert bg.endswith('_residual_smoothed_bg_i2d.fits')
    # bg -> detection returns to the residual name (round trip)
    assert N.smoothed_bg_to_detection_i2d(bg) == base
    assert N.residual_to_model_i2d(base).endswith('_model_i2d.fits')
    assert N.residual_to_infilled_i2d(base).endswith('_residual_infilled_i2d.fits')
