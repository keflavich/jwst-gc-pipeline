"""A cached per-exposure satstar catalog stores the sky position the star had
when it was FIT.  The cache is deliberately not invalidated when the frame's
WCS changes (``_satstar_recovery_signature`` keys it on the recovery/deblend
config only), so after an offsets-table correction + frame regeneration the
stored ``skycoord_fit`` no longer matches its own pixel centroid, and the
consolidated satstar catalog publishes the OLD frame's astrometry.

Measured on brick F200W (issue #193): stored minus current-GWCS = +56.8/+88.7
mas over 3279 saturated stars in 40 exposures, against a +58.7/+88.2 mas
saturated-versus-unsaturated position excess in the m6 catalog built from them.

The pixel centroids are fine; only the projection is stale.  So every read
re-projects them through the frame's current GWCS.
"""
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, Sip
from astropy import units as u

from jwst_gc_pipeline.photometry.satstar_wcs_refresh import (
    MissingSatstarFrameWarning, frame_path_for_satstar_catalog,
    refresh_satstar_skycoords)

PIXSCALE = 31.2 / 3600.0 / 1000.0  # NIRCam SW, deg/px


def _wcs(crval1=266.5, crval2=-28.7):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [64.5, 64.5]
    w.wcs.crval = [crval1, crval2]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


def _satstar_table(wcs_at_fit_time, x, y):
    """A per-exposure satstar catalog as the fitter writes it: pixel centroids
    plus the sky position they projected to under ``wcs_at_fit_time``."""
    sky = wcs_at_fit_time.pixel_to_world(x, y)
    tbl = Table({'xcentroid': np.asarray(x, dtype=float),
                 'ycentroid': np.asarray(y, dtype=float),
                 'flux_fit': np.ones(len(x))})
    tbl['skycoord_fit'] = sky
    tbl.meta.update(wcs_at_fit_time.to_header(relax=True))
    return tbl


def _write_frame(path, wcs_now, shape=(128, 128)):
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(np.zeros(shape, dtype='float32'), name='SCI')
    hdu1.header.update(wcs_now.to_header(relax=True))
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


def _seps(tbl, reference_sky):
    got = SkyCoord(tbl['skycoord_fit'])
    return got.separation(reference_sky).to(u.mas).value


def test_stale_sky_is_reprojected_onto_the_moved_frame(tmp_path):
    """The frame moved 100 mas after the fit; the refresh must move the stored
    sky by exactly that, NOT leave it at the fit-time value."""
    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.5 + 100e-3 / 3600.0 / np.cos(np.radians(-28.7)),
                   crval2=-28.7 + 30e-3 / 3600.0)
    x = np.array([10.0, 64.0, 120.0])
    y = np.array([20.0, 64.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    stale = SkyCoord(tbl['skycoord_fit'])

    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)
    cat = str(tmp_path / 'exp_crf_m3_satstar_catalog.fits')

    out, shift = refresh_satstar_skycoords(tbl, frame_path=frame,
                                           catalog_path=cat)

    # positions now agree with the CURRENT frame, to sub-mas
    assert np.all(_seps(out, now_wcs.pixel_to_world(x, y)) < 0.01)
    # and they are no longer what the cache stored
    assert np.all(_seps(out, stale) > 50)
    assert np.hypot(*shift) == pytest.approx(np.hypot(100.0, 30.0), rel=0.02)


def test_pixel_centroids_are_untouched(tmp_path):
    """Re-projection is the whole fix: the FIT is good, so the stored pixel
    centroid must survive the refresh byte for byte."""
    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.6)
    x = np.array([10.0, 64.0, 120.0])
    y = np.array([20.0, 64.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)

    out, _ = refresh_satstar_skycoords(tbl, frame_path=frame)
    np.testing.assert_array_equal(np.asarray(out['xcentroid']), x)
    np.testing.assert_array_equal(np.asarray(out['ycentroid']), y)


def test_unmoved_frame_is_a_no_op(tmp_path):
    """A field whose frames have not moved since the fit must not be perturbed
    -- the refresh has to be safe to run on every read."""
    fit_wcs = _wcs()
    x = np.array([10.0, 64.0, 120.0])
    y = np.array([20.0, 64.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    before = SkyCoord(tbl['skycoord_fit'])
    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, fit_wcs)

    out, shift = refresh_satstar_skycoords(tbl, frame_path=frame)
    assert np.all(_seps(out, before) < 0.01)
    assert np.hypot(*shift) < 0.01


def test_component_anchor_follows_the_frame(tmp_path):
    """``sat_com_ra``/``sat_com_dec`` is the dedup's stable per-component sky
    anchor and is stored as sky only.  If it is left at the fit-time frame while
    ``skycoord_fit`` moves, one physical star splits into two rows in the
    consolidated catalog."""
    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.5 + 200e-3 / 3600.0 / np.cos(np.radians(-28.7)))
    x = np.array([30.0, 90.0])
    y = np.array([40.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    anchor_sky = fit_wcs.pixel_to_world(x + 1.0, y - 1.0)
    tbl['sat_com_ra'] = anchor_sky.ra.deg
    tbl['sat_com_dec'] = anchor_sky.dec.deg

    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)
    out, _ = refresh_satstar_skycoords(tbl, frame_path=frame)

    expected = now_wcs.pixel_to_world(x + 1.0, y - 1.0)
    got = SkyCoord(np.asarray(out['sat_com_ra']) * u.deg,
                   np.asarray(out['sat_com_dec']) * u.deg)
    assert np.all(got.separation(expected).to(u.mas).value < 0.5)


def test_missing_frame_warns_rather_than_silently_serving_stale_sky(tmp_path):
    fit_wcs = _wcs()
    tbl = _satstar_table(fit_wcs, np.array([10.0]), np.array([20.0]))
    cat = str(tmp_path / 'gone_crf_m3_satstar_catalog.fits')
    with pytest.warns(MissingSatstarFrameWarning):
        out, shift = refresh_satstar_skycoords(tbl, catalog_path=cat)
    assert not np.isfinite(shift[0])
    assert len(out) == 1


@pytest.mark.parametrize('suffix', ['_m3', '_m12', '_iter2', '_resbgsub_m4'])
def test_frame_path_recovers_every_file_suffix(tmp_path, suffix):
    """The writer names the cache ``frame.replace('.fits', suffix +
    '_satstar_catalog.fits')`` for whatever the run's file suffix is, so the
    frame lookup has to strip all of them."""
    frame = tmp_path / 'jw01182004001_04101_00001_nrca1_destreak_o004_crf.fits'
    frame.write_text('')
    cat = str(frame).replace('.fits', f'{suffix}_satstar_catalog.fits')
    assert frame_path_for_satstar_catalog(cat) == str(frame)


def test_frame_path_returns_none_for_a_non_satstar_name(tmp_path):
    assert frame_path_for_satstar_catalog(str(tmp_path / 'x_crf.fits')) is None


def test_consolidated_build_reads_caches_on_the_current_frame(tmp_path):
    """``merge_catalogs.load_satstar_catalog`` builds the consolidated
    per-filter satstar catalog whose positions reach the merged photometry.  It
    must read each per-exposure cache THROUGH the frame's current WCS; reading
    it raw is how a June fit published June's astrometry from an August frame.
    """
    from jwst_gc_pipeline.photometry import merge_catalogs as MC

    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.5 + 90e-3 / 3600.0 / np.cos(np.radians(-28.7)))
    x = np.array([25.0, 75.0])
    y = np.array([35.0, 85.0])
    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)
    cat = str(tmp_path / 'exp_crf_m3_satstar_catalog.fits')
    _satstar_table(fit_wcs, x, y).write(cat, overwrite=True)

    got = MC._read_satstar_catalog_on_current_frame(cat, {})
    assert np.all(_seps(got, now_wcs.pixel_to_world(x, y)) < 0.01)
    assert np.all(_seps(got, fit_wcs.pixel_to_world(x, y)) > 50)


def test_consolidated_cache_generation_forces_a_rebuild():
    """The consolidated satstar cache on disk was built from raw (stale-sky)
    reads, and it is keyed by file count + dedup radius + algorithm token --
    none of which changed here.  The algorithm token must move so those caches
    rebuild instead of serving the old positions."""
    from jwst_gc_pipeline.photometry import merge_catalogs as MC
    assert MC._SATSTAR_DEDUP_ALG != 'fp3'


def test_off_detector_seeds_keep_their_stored_sky(tmp_path):
    """An outside-FOV satstar seed is fit at a pixel OFF the detector (its star
    is off the frame; only the spikes are on it).  A GWCS returns NaN there --
    the right sentinel, but re-projecting those rows to NaN would delete
    positions the off-FOV flux reconciliation reads.  Keep the stored value and
    say so, rather than silently NaN-ing them."""
    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.5 + 100e-3 / 3600.0 / np.cos(np.radians(-28.7)))
    x = np.array([30.0, 90.0, -400.0])          # third is off the 128x128 frame
    y = np.array([40.0, 100.0, -400.0])
    tbl = _satstar_table(fit_wcs, x, y)
    stored = SkyCoord(tbl['skycoord_fit'])

    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)

    class _Bounded:
        """Stands in for a GWCS: NaN outside the detector."""
        def pixel_to_world(self, px, py):
            sky = now_wcs.pixel_to_world(np.asarray(px), np.asarray(py))
            bad = (px < 0) | (px > 127) | (py < 0) | (py > 127)
            ra, dec = sky.ra.deg.copy(), sky.dec.deg.copy()
            ra[bad] = np.nan
            dec[bad] = np.nan
            return SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')

    with pytest.warns(MissingSatstarFrameWarning):
        out, shift = refresh_satstar_skycoords(tbl, wcs=_Bounded())

    got = SkyCoord(out['skycoord_fit'])
    assert np.all(np.isfinite(got.ra.deg))                      # nothing NaN-ed
    assert got[2].separation(stored[2]).to(u.mas).value < 0.01   # kept as stored
    assert np.all(_seps(out, now_wcs.pixel_to_world(x, y))[:2] < 0.01)
    assert np.hypot(*shift) == pytest.approx(100.0, rel=0.05)


def _sip_wcs(crval1=266.5, crval2=-28.7):
    """A JWST-shaped ``RA---TAN-SIP``: the projection every real detector-frame
    satstar catalog stamps into its meta."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']
    w.wcs.crpix = [64.5, 64.5]
    w.wcs.crval = [crval1, crval2]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    a[2, 0], a[1, 1], a[0, 2] = 4e-5, -2e-5, 3e-5
    b[2, 0], b[1, 1], b[0, 2] = -3e-5, 5e-5, 2e-5
    w.sip = Sip(a, b, None, None, w.wcs.crpix)
    return w


def test_anchor_transport_survives_a_sip_projection(tmp_path):
    """The component anchor is stored as sky only.  Recovering its pixel by
    inverting a WCS rebuilt from the meta's LINEAR cards drops the SIP terms and
    lands on the wrong pixel -- 54.87 mas median / 224.30 mas max (1.79 / 7.26 px)
    on brick F182M nrcb1, an UNMOVED frame that should have refreshed by 0.00.
    Transport the anchor by the offset ``skycoord_fit`` itself moved by instead,
    which needs no fit-time WCS and so cannot lose the distortion."""
    fit_wcs = _sip_wcs()
    x = np.array([12.0, 40.0, 64.0, 100.0, 124.0])
    y = np.array([18.0, 110.0, 64.0, 30.0, 121.0])
    tbl = _satstar_table(fit_wcs, x, y)
    # anchor = the saturated component's bbox centre, a few px off the centroid
    anchor_sky = fit_wcs.pixel_to_world(x + 3.0, y - 2.0)
    tbl['sat_com_ra'] = anchor_sky.ra.deg
    tbl['sat_com_dec'] = anchor_sky.dec.deg

    # the frame has NOT moved: the fit-time WCS is the current one
    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, fit_wcs)

    out, shift = refresh_satstar_skycoords(tbl, frame_path=frame)

    got = SkyCoord(np.asarray(out['sat_com_ra']) * u.deg,
                   np.asarray(out['sat_com_dec']) * u.deg)
    assert np.hypot(*shift) < 0.01
    # an unmoved frame must not move the anchor either
    assert np.all(got.separation(anchor_sky).to(u.mas).value < 0.01)


def test_anchor_transport_matches_the_exact_pixel_transport(tmp_path):
    """With a SIP projection AND a moved frame, the transported anchor has to
    agree with the exact answer -- invert the fit-time WCS to the anchor pixel,
    project that pixel through the current one."""
    fit_wcs = _sip_wcs()
    now_wcs = _sip_wcs(crval1=266.5 + 2.0 / 3600.0 / np.cos(np.radians(-28.7)),
                       crval2=-28.7 + 1.0 / 3600.0)
    x = np.array([12.0, 40.0, 64.0, 100.0, 124.0])
    y = np.array([18.0, 110.0, 64.0, 30.0, 121.0])
    ax, ay = x + 3.0, y - 2.0
    tbl = _satstar_table(fit_wcs, x, y)
    anchor_sky = fit_wcs.pixel_to_world(ax, ay)
    tbl['sat_com_ra'] = anchor_sky.ra.deg
    tbl['sat_com_dec'] = anchor_sky.dec.deg

    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)
    out, _ = refresh_satstar_skycoords(tbl, frame_path=frame)

    exact = now_wcs.pixel_to_world(ax, ay)
    got = SkyCoord(np.asarray(out['sat_com_ra']) * u.deg,
                   np.asarray(out['sat_com_dec']) * u.deg)
    # it MOVED (the frame moved 2.2")
    assert np.all(got.separation(anchor_sky).arcsec > 2.0)
    # and it landed where the exact pixel transport puts it
    assert np.all(got.separation(exact).to(u.mas).value < 0.5)


def test_module_builds_no_wcs_from_a_stamped_header():
    """ASTROMETRY RULE #2: the only WCS this module reads is the frame's GWCS
    through ``frame_wcs``.  Rebuilding the fit-time WCS from the catalog meta's
    header cards is what lost the SIP terms."""
    import inspect
    from jwst_gc_pipeline.photometry import satstar_wcs_refresh as mod
    src = inspect.getsource(mod)
    for token in ('astropy_wcs.WCS(', 'wcs.WCS(', 'all_world2pix', 'wcs_world2pix'):
        assert token not in src, f"{token} rebuilds a header WCS in {mod.__name__}"


# ---------------------------------------------------------------------------
# The consolidated cache must go stale when the FRAMES move, not only when the
# per-exposure satstar catalogs do.
# ---------------------------------------------------------------------------

def test_frame_state_signature_moves_when_a_frame_is_rewritten(tmp_path):
    """Re-aligning or regenerating an exposure rewrites the FRAME and touches no
    satstar catalog, so a signature over the satstar catalogs alone cannot see
    it.  This one is taken over the frames they resolve to."""
    from jwst_gc_pipeline.photometry.satstar_wcs_refresh import (
        satstar_frame_state_signature)

    frame = tmp_path / 'exp_crf.fits'
    _write_frame(str(frame), _wcs())
    cat = str(tmp_path / 'exp_crf_m3_satstar_catalog.fits')
    _satstar_table(_wcs(), np.array([10.0]), np.array([20.0])).write(cat)

    before = satstar_frame_state_signature([cat])
    assert before and satstar_frame_state_signature([cat]) == before  # stable

    _write_frame(str(frame), _wcs(crval1=266.6))   # the frame moved
    os.utime(frame, (1e9, 1e9))                    # ...to a distinct mtime
    assert satstar_frame_state_signature([cat]) != before


def test_frame_state_signature_marks_a_missing_frame(tmp_path):
    """A catalog whose frame is gone must not silently contribute nothing --
    that would make an appearing/disappearing frame invisible to the cache."""
    from jwst_gc_pipeline.photometry.satstar_wcs_refresh import (
        satstar_frame_state_signature)

    cat = str(tmp_path / 'exp_crf_m3_satstar_catalog.fits')
    _satstar_table(_wcs(), np.array([10.0]), np.array([20.0])).write(cat)
    without = satstar_frame_state_signature([cat])

    _write_frame(str(tmp_path / 'exp_crf.fits'), _wcs())
    assert satstar_frame_state_signature([cat]) != without
    assert satstar_frame_state_signature([]) == ''


def test_consolidated_cache_rebuilds_after_the_frames_move(tmp_path, capsys):
    """The durable half of the fix.  ``load_satstar_catalog``'s freshness test
    is cache mtime + source count + dedup radius + algorithm -- all properties of
    the satstar catalogs.  Correcting the offsets table and regenerating from
    ``_cal`` changes none of them, so without a frame term the consolidated
    catalog (the file that feeds the merged photometry) keeps serving the
    pre-correction sky."""
    from jwst_gc_pipeline.photometry import merge_catalogs as MC

    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    x = np.array([25.0, 75.0])
    y = np.array([35.0, 85.0])
    fit_wcs = _wcs()
    frames = []
    for i in (1, 2):
        frame = pdir / f'jw02221001001_02101_0000{i}_nrca1_o001_crf.fits'
        _write_frame(str(frame), fit_wcs)
        frames.append(frame)
        _satstar_table(fit_wcs, x, y).write(
            str(frame).replace('.fits', '_m6_satstar_catalog.fits'))

    first = MC.load_satstar_catalog('f182m', target='brick',
                                    basepath=str(tmp_path) + '/')
    assert first is not None and len(first) == 2
    assert np.all(_seps(first, fit_wcs.pixel_to_world(x, y)) < 0.01)

    # The frames move 90 mas (offsets-table correction + regeneration from _cal).
    # No satstar catalog is touched, and the consolidated cache stays newer than
    # all of them -- every pre-existing freshness term still says "fresh".
    now_wcs = _wcs(crval1=266.5 + 90e-3 / 3600.0 / np.cos(np.radians(-28.7)))
    for frame in frames:
        _write_frame(str(frame), now_wcs)
    cache = tmp_path / 'catalogs' / 'f182m_consolidated_satstar_catalog.fits'
    assert cache.exists()
    newest_cat = max(os.path.getmtime(str(f).replace('.fits',
                                                     '_m6_satstar_catalog.fits'))
                     for f in frames)
    assert os.path.getmtime(cache) >= newest_cat

    capsys.readouterr()
    second = MC.load_satstar_catalog('f182m', target='brick',
                                     basepath=str(tmp_path) + '/')
    out = capsys.readouterr().out
    assert 'the exposures moved under it' in out, out
    assert np.all(_seps(second, now_wcs.pixel_to_world(x, y)) < 0.01)
    assert np.all(_seps(second, fit_wcs.pixel_to_world(x, y)) > 50)


def test_consolidated_cache_is_still_reused_when_nothing_moved(tmp_path, capsys):
    """The frame term must not defeat the cache: the consolidation is the
    dominant cost of every merge (~1400 files, dozens of times per run), so an
    unchanged field has to hit."""
    from jwst_gc_pipeline.photometry import merge_catalogs as MC

    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    fit_wcs = _wcs()
    frame = pdir / 'jw02221001001_02101_00001_nrca1_o001_crf.fits'
    _write_frame(str(frame), fit_wcs)
    _satstar_table(fit_wcs, np.array([25.0]), np.array([35.0])).write(
        str(frame).replace('.fits', '_m6_satstar_catalog.fits'))

    MC.load_satstar_catalog('f182m', target='brick', basepath=str(tmp_path) + '/')
    capsys.readouterr()
    MC.load_satstar_catalog('f182m', target='brick', basepath=str(tmp_path) + '/')
    out = capsys.readouterr().out
    assert 'Using consolidated satstar catalog' in out, out
    assert 'Rebuilding' not in out, out
