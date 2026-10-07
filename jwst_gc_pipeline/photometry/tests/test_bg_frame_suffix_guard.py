"""A smoothed residual background must be subtracted from the frames it was
built from (#1130).

The per-frame residuals keep the pristine frame level and the smoother adds
and removes none, so the background carries the sky level of the frames the
phase fit.  wd2 SW ran m1-m6 on ``destreak_o005_crf`` (sky removed) and m7 on
``align_o005_crf`` (sky kept): every m7 frame kept 0.6-7.5 MJy/sr after the
m6 background came off, and the local-S/N gate passed 96-99% of the daofind
peaks.  The background now carries the frame suffix it was built from
(``BGFRMSUF``) and the subtraction refuses a different one.
"""
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging as C

FILT = 'F200W'          # present in reduction/fwhm_table.ecsv
PIX_AS = 0.031


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [50, 50]
    w.wcs.crval = [161.03, -59.75]
    w.wcs.cdelt = [-PIX_AS / 3600, PIX_AS / 3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return w


def _consumer_header(path):
    """The header ``_prepare_frame_for_photometry`` reads the stamp from."""
    with fits.open(path) as h:
        hdu = h['SCI'] if 'SCI' in [x.name for x in h] else h[0]
        return hdu.header.copy()


def _source_masked_bg(tmp_path):
    w = _wcs()
    mc = str(tmp_path / 'sim_clear-f200w-nrca_resbgsub_m6_daophot_basic_mergedcat_residual_i2d.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=np.full((100, 100), 0.4, 'float32'),
                                header=w.to_header(), name='SCI')]).writeto(mc)
    vet = str(tmp_path / 'vetted.fits')
    Table({'skycoord': w.pixel_to_world(np.array([10.0]), np.array([10.0]))}
          ).write(vet, overwrite=True)
    return C._build_source_masked_bg(mc, vet, FILT)


# ---------------------------------------------------------------------------
# stamp
# ---------------------------------------------------------------------------
def test_stamp_on_source_masked_bg_reaches_the_consumer(tmp_path):
    bg = _source_masked_bg(tmp_path)
    assert C.BG_FRAME_SUFFIX_KEY not in _consumer_header(bg)
    before = fits.getdata(bg).copy()
    C._stamp_bg_frame_suffix(bg, 'destreak_o005_crf')
    assert _consumer_header(bg)[C.BG_FRAME_SUFFIX_KEY] == 'destreak_o005_crf'
    # the map itself is untouched
    np.testing.assert_array_equal(fits.getdata(bg), before)


def test_stamp_on_sci_extension_bg(tmp_path):
    p = str(tmp_path / 'bg_sci.fits')
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=np.zeros((4, 4), 'float32'), name='SCI')]).writeto(p)
    C._stamp_bg_frame_suffix(p, 'align_o005_crf')
    assert _consumer_header(p)[C.BG_FRAME_SUFFIX_KEY] == 'align_o005_crf'
    assert C.BG_FRAME_SUFFIX_KEY not in fits.getheader(p, 0)


def test_restamp_overwrites(tmp_path):
    bg = _source_masked_bg(tmp_path)
    C._stamp_bg_frame_suffix(bg, 'destreak_o005_crf')
    C._stamp_bg_frame_suffix(bg, 'align_o005_crf')
    assert _consumer_header(bg)[C.BG_FRAME_SUFFIX_KEY] == 'align_o005_crf'


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('stamp', [None, '', '   '])
def test_unstamped_passes(monkeypatch, stamp):
    monkeypatch.delenv(C.BG_FRAME_SUFFIX_OVERRIDE_ENV, raising=False)
    h = fits.Header()
    if stamp is not None:
        h[C.BG_FRAME_SUFFIX_KEY] = stamp
    assert C._check_bg_frame_suffix(h, 'align_o005_crf') == 'unstamped'


def test_match_passes(monkeypatch):
    monkeypatch.delenv(C.BG_FRAME_SUFFIX_OVERRIDE_ENV, raising=False)
    h = fits.Header({C.BG_FRAME_SUFFIX_KEY: 'align_o005_crf'})
    assert C._check_bg_frame_suffix(h, ' align_o005_crf ') == 'match'


def test_wd2_mismatch_raises_and_names_both(monkeypatch):
    monkeypatch.delenv(C.BG_FRAME_SUFFIX_OVERRIDE_ENV, raising=False)
    h = fits.Header({C.BG_FRAME_SUFFIX_KEY: 'destreak_o005_crf'})
    with pytest.raises(RuntimeError) as ei:
        C._check_bg_frame_suffix(h, 'align_o005_crf', label='m6_bg.fits')
    msg = str(ei.value)
    assert "'destreak_o005_crf'" in msg and "'align_o005_crf'" in msg
    assert 'm6_bg.fits' in msg
    assert C.BG_FRAME_SUFFIX_OVERRIDE_ENV in msg


def test_override_warns_instead(monkeypatch, capsys):
    monkeypatch.setenv(C.BG_FRAME_SUFFIX_OVERRIDE_ENV, '1')
    h = fits.Header({C.BG_FRAME_SUFFIX_KEY: 'destreak_o005_crf'})
    assert C._check_bg_frame_suffix(h, 'align_o005_crf') == 'mismatch'
    assert 'WARNING' in capsys.readouterr().out


def test_override_needs_exactly_one(monkeypatch):
    monkeypatch.setenv(C.BG_FRAME_SUFFIX_OVERRIDE_ENV, 'yes')
    h = fits.Header({C.BG_FRAME_SUFFIX_KEY: 'destreak_o005_crf'})
    with pytest.raises(RuntimeError):
        C._check_bg_frame_suffix(h, 'align_o005_crf')


def test_per_filter_override_suffix_is_what_gets_compared(monkeypatch):
    """The consumer compares against ``_resolve_each_suffix``, so a filter
    listed in ``--each-suffix-overrides`` is checked against its own crf."""
    monkeypatch.delenv(C.BG_FRAME_SUFFIX_OVERRIDE_ENV, raising=False)
    opts = types.SimpleNamespace(each_suffix='align_o005_crf',
                                 each_suffix_overrides='F187N:destreak_o005_crf')
    h = fits.Header({C.BG_FRAME_SUFFIX_KEY: 'destreak_o005_crf'})
    assert C._check_bg_frame_suffix(h, C._resolve_each_suffix(opts, 'F187N')) == 'match'
    with pytest.raises(RuntimeError):
        C._check_bg_frame_suffix(h, C._resolve_each_suffix(opts, 'F200W'))
