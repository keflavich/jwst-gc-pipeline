"""Satstar position QA: skycoord_fit against the saturated-component anchor (#951)."""
import numpy as np
import pytest
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry.satstar_position_qa import (
    satstar_position_qa, format_satstar_position_qa, frame_pixel_scale_mas,
    log_satstar_position_qa)

RA0 = 266.5
DEC0 = -28.7
MAS = 1.0 / 3.6e6


def _tbl(dec_off_mas=None, ra_off_mas=None, n=None, kinds=None):
    """Fit positions displaced from a fixed anchor by the given offsets."""
    if dec_off_mas is None:
        dec_off_mas = np.zeros(n)
    dec_off_mas = np.asarray(dec_off_mas, dtype=float)
    n = len(dec_off_mas)
    ra_off = np.zeros(n) if ra_off_mas is None else np.asarray(ra_off_mas, float)
    t = Table()
    t['skycoord_fit'] = SkyCoord((RA0 + ra_off * MAS / np.cos(np.deg2rad(DEC0))) * u.deg,
                                 (DEC0 + dec_off_mas * MAS) * u.deg)
    t['sat_com_ra'] = np.full(n, RA0)
    t['sat_com_dec'] = np.full(n, DEC0)
    if kinds is not None:
        t['seed_kind'] = np.asarray(kinds)
    return t


def test_known_offsets_give_expected_statistics():
    t = _tbl([0.0 + 1e-3, 10.0, 20.0, 10.0, 10.0])
    qa = satstar_position_qa(t, pixscale_mas=31.0)
    assert qa['available'] and qa['n'] == 5 and qa['n_finite_anchor'] == 5
    assert qa['median_mas'] == pytest.approx(10.0, abs=1e-3)
    assert qa['p84_mas'] == pytest.approx(np.percentile([0.001, 10, 20, 10, 10], 84),
                                          abs=1e-3)
    assert qa['frac_gt_200mas'] == 0.0
    # half a pixel is 15.5 mas: only the 20 mas row exceeds it
    assert qa['frac_gt_halfpix'] == pytest.approx(0.2)


def test_gross_outlier_fraction():
    qa = satstar_position_qa(_tbl([5.0, 5.0, 5.0, 300.0]))
    assert qa['frac_gt_200mas'] == pytest.approx(0.25)


def test_ra_offset_is_scaled_by_cos_dec():
    # 30 mas on the sky along RA at dec -28.7 is 30/cos(dec) mas in RA
    t = _tbl(np.zeros(3), ra_off_mas=[30.0, 30.0, 30.0])
    qa = satstar_position_qa(t)
    assert qa['median_mas'] == pytest.approx(30.0, abs=1e-3)
    ra_diff_mas = (t['skycoord_fit'].ra.deg[0] - RA0) * 3.6e6
    assert ra_diff_mas == pytest.approx(30.0 / np.cos(np.deg2rad(DEC0)), rel=1e-6)


def test_ra_wrap_across_zero():
    t = Table()
    t['skycoord_fit'] = SkyCoord([359.99999] * u.deg, [0.0] * u.deg)
    t['sat_com_ra'] = [0.00001]
    t['sat_com_dec'] = [0.0]
    qa = satstar_position_qa(t)
    assert qa['median_mas'] == pytest.approx(0.00002 * 3.6e6, rel=1e-3)


def test_locked_catalog_is_flagged():
    qa = satstar_position_qa(_tbl(np.zeros(20)))
    assert qa['frac_locked'] == 1.0
    assert qa['lock_suspect']


def test_sickle_f480m_pattern_is_flagged_despite_zero_median():
    # 60% of rows exactly on the anchor, the rest in a ~31 mas tail
    off = np.r_[np.zeros(12), np.full(8, 31.6)]
    qa = satstar_position_qa(_tbl(off))
    assert qa['median_mas'] < 0.05
    assert qa['frac_locked'] == pytest.approx(0.6)
    assert qa['lock_suspect']


def test_unlocked_catalog_is_not_flagged():
    qa = satstar_position_qa(_tbl(np.linspace(3, 15, 40)))
    assert qa['frac_locked'] == 0.0
    assert not qa['lock_suspect']


def test_few_rows_are_not_flagged_as_locked():
    assert not satstar_position_qa(_tbl(np.zeros(5)))['lock_suspect']


def test_pre_anchor_catalog_reports_unavailable(capsys):
    t = _tbl(np.zeros(3))
    t.remove_columns(['sat_com_ra', 'sat_com_dec'])
    assert satstar_position_qa(t) == {'available': False,
                                      'reason': 'no_anchor_columns'}
    qa = log_satstar_position_qa(t)
    assert qa['available'] is False
    assert 'pre-anchor catalog vintage' in capsys.readouterr().out


def test_nan_anchors_are_excluded_and_counted():
    t = _tbl([10.0, 10.0, 10.0, 10.0])
    anchor = np.array(t['sat_com_ra'])
    anchor[1] = np.nan
    t['sat_com_ra'] = anchor
    qa = satstar_position_qa(t)
    assert qa['n'] == 4 and qa['n_finite_anchor'] == 3
    assert qa['median_mas'] == pytest.approx(10.0, abs=1e-3)


def test_masked_anchor_is_excluded():
    t = _tbl([10.0, 10.0, 10.0])
    t['sat_com_dec'] = np.ma.masked_array(np.full(3, DEC0), mask=[False, True, False])
    assert satstar_position_qa(t)['n_finite_anchor'] == 2


def test_all_anchors_nan_gives_nan_statistics_without_error():
    t = _tbl([10.0, 10.0])
    t['sat_com_ra'] = np.full(2, np.nan)
    qa = satstar_position_qa(t, pixscale_mas=31.0)
    assert qa['n_finite_anchor'] == 0
    assert np.isnan(qa['median_mas']) and not qa['lock_suspect']
    assert '0/2 rows carry both a finite fit position and a finite component anchor' \
        in format_satstar_position_qa(qa)


def test_without_pixscale_the_halfpixel_fraction_is_omitted():
    qa = satstar_position_qa(_tbl([10.0, 20.0]))
    assert 'frac_gt_halfpix' not in qa
    assert 'half a pixel' not in format_satstar_position_qa(qa)


def test_median_per_seed_kind():
    t = _tbl([5.0, 7.0, 15.0, 17.0], kinds=['peak', 'peak', 'dqsat', 'dqsat'])
    qa = satstar_position_qa(t)
    assert qa['median_by_seed_kind']['peak'] == pytest.approx(6.0, abs=1e-3)
    assert qa['median_by_seed_kind']['dqsat'] == pytest.approx(16.0, abs=1e-3)


def test_bytes_seed_kind_is_decoded():
    t = _tbl([5.0, 15.0], kinds=[b'peak', b'dqsat'])
    assert set(satstar_position_qa(t)['median_by_seed_kind']) == {'peak', 'dqsat'}


class _FakeWCS:
    """Plane at 31 mas per pixel, with RA increasing toward smaller x."""

    def pixel_to_world(self, x, y):
        return SkyCoord((RA0 - x * 31 * MAS / np.cos(np.deg2rad(DEC0))) * u.deg,
                        (DEC0 + y * 31 * MAS) * u.deg)


def test_frame_pixel_scale_from_world_differences():
    assert frame_pixel_scale_mas(_FakeWCS(), 100.0, 200.0) == pytest.approx(31.0, rel=1e-3)


def test_frame_pixel_scale_none_for_nan_world():
    class _Off:
        def pixel_to_world(self, x, y):
            return SkyCoord(np.nan * u.deg, np.nan * u.deg)
    assert frame_pixel_scale_mas(_Off(), 0.0, 0.0) is None


def test_log_line_is_informational_and_states_reference(capsys):
    t = _tbl(np.linspace(3, 15, 40), kinds=['peak'] * 40)
    log_satstar_position_qa(t, frame_path=None, label='x_satstar_catalog.fits')
    line = capsys.readouterr().out.strip()
    assert line.startswith('Satstar position QA x_satstar_catalog.fits:')
    assert 'median' in line and 'beyond 200 mas' in line
    assert "seed component's box centre" in line


# --------------------------------------------------------------------------
# Missing-column wording (the cause differs, so does the message)
# --------------------------------------------------------------------------

def test_missing_skycoord_fit_is_reported_as_such_not_as_pre_anchor(capsys):
    t = _tbl(np.zeros(3))
    t.remove_column('skycoord_fit')
    assert satstar_position_qa(t) == {'available': False,
                                      'reason': 'no_skycoord_fit'}
    log_satstar_position_qa(t)
    line = capsys.readouterr().out
    assert 'no skycoord_fit column' in line
    assert 'pre-anchor' not in line


def test_missing_anchor_columns_keep_the_pre_anchor_wording(capsys):
    t = _tbl(np.zeros(3))
    t.remove_columns(['sat_com_ra', 'sat_com_dec'])
    assert satstar_position_qa(t)['reason'] == 'no_anchor_columns'
    log_satstar_position_qa(t)
    assert 'pre-anchor catalog vintage' in capsys.readouterr().out


def test_nan_fit_with_finite_anchor_is_not_called_an_anchor_problem():
    t = _tbl([10.0, 10.0])
    t['skycoord_fit'] = SkyCoord([np.nan, np.nan] * u.deg, [np.nan, np.nan] * u.deg)
    qa = satstar_position_qa(t)
    line = format_satstar_position_qa(qa)
    assert qa['n_finite_anchor'] == 0
    assert 'both a finite fit position and a finite component anchor' in line


def test_fit_in_another_frame_is_compared_in_icrs():
    t = _tbl([10.0, 10.0, 10.0])
    t['skycoord_fit'] = t['skycoord_fit'].galactic
    qa = satstar_position_qa(t)
    assert qa['median_mas'] == pytest.approx(10.0, abs=1e-2)


# --------------------------------------------------------------------------
# Thresholds, each pinned on both sides of its boundary
# --------------------------------------------------------------------------

def test_gross_fraction_counts_only_rows_beyond_the_limit():
    qa = satstar_position_qa(_tbl([5.0, 5.0, 199.0, 201.0]))
    assert qa['frac_gt_200mas'] == pytest.approx(0.25)


def test_gross_boundary_equal_is_not_counted(monkeypatch):
    from jwst_gc_pipeline.photometry import satstar_position_qa as Q
    # Put the limit exactly on the separation the table produces.
    t = _tbl([40.0, 40.0, 40.0, 40.0])
    sep = satstar_position_qa(t)['median_mas']
    monkeypatch.setattr(Q, 'GROSS_MAS', sep)
    assert Q.satstar_position_qa(t)['frac_gt_200mas'] == 0.0
    monkeypatch.setattr(Q, 'GROSS_MAS', sep - 1e-6)
    assert Q.satstar_position_qa(t)['frac_gt_200mas'] == 1.0


def test_lock_tolerance_is_strict(monkeypatch):
    from jwst_gc_pipeline.photometry import satstar_position_qa as Q
    t = _tbl(np.full(12, 0.04))
    assert Q.satstar_position_qa(t)['frac_locked'] == 1.0       # below 0.05 mas
    t = _tbl(np.full(12, 0.06))
    assert Q.satstar_position_qa(t)['frac_locked'] == 0.0       # above 0.05 mas
    # separation equal to the tolerance is not locked (strict <)
    sep = Q.satstar_position_qa(t)['median_mas']
    monkeypatch.setattr(Q, 'LOCK_TOL_MAS', sep)
    assert Q.satstar_position_qa(t)['frac_locked'] == 0.0


def test_lock_fraction_is_strict_at_a_quarter():
    # 12 rows, 3 locked = exactly 25%: not suspect; 4 locked: suspect
    off = np.r_[np.zeros(3), np.full(9, 20.0)]
    qa = satstar_position_qa(_tbl(off))
    assert qa['frac_locked'] == pytest.approx(0.25) and not qa['lock_suspect']
    off = np.r_[np.zeros(4), np.full(8, 20.0)]
    assert satstar_position_qa(_tbl(off))['lock_suspect']


def test_lock_minimum_rows_is_inclusive_at_ten():
    assert satstar_position_qa(_tbl(np.zeros(10)))['lock_suspect']
    assert not satstar_position_qa(_tbl(np.zeros(9)))['lock_suspect']


def test_seed_kind_of_excluded_rows_is_not_reported():
    t = _tbl([10.0, 10.0, 10.0], kinds=['peak', 'forced', 'peak'])
    anchor = np.array(t['sat_com_ra'])
    anchor[1] = np.nan
    t['sat_com_ra'] = anchor
    assert set(satstar_position_qa(t)['median_by_seed_kind']) == {'peak'}


def test_log_line_carries_the_lock_suspect_message_only_when_raised():
    locked = format_satstar_position_qa(satstar_position_qa(_tbl(np.zeros(20))))
    assert 'locked to the anchor in a large share of rows' in locked
    free = format_satstar_position_qa(satstar_position_qa(_tbl(np.linspace(3, 15, 20))))
    assert 'locked to the anchor' not in free


# --------------------------------------------------------------------------
# Pixel-scale helper (_catalog_pixel_scale_mas)
# --------------------------------------------------------------------------

class _AnisoWCS:
    """Plane with a different scale along x and y, so a geometric mean differs
    from either axis.  Scale depends on position to expose which pixel is used."""

    def __init__(self, sx_mas=20.0, sy_mas=45.0):
        self.sx, self.sy = sx_mas, sy_mas

    def pixel_to_world(self, x, y):
        return SkyCoord((RA0 - x * self.sx * MAS / np.cos(np.deg2rad(DEC0))) * u.deg,
                        (DEC0 + y * self.sy * MAS) * u.deg)


def test_pixel_scale_is_the_geometric_mean_of_the_axes():
    assert frame_pixel_scale_mas(_AnisoWCS(20.0, 45.0), 5.0, 5.0) == pytest.approx(
        np.sqrt(20.0 * 45.0), rel=1e-3)


def _centroid_tbl(xc, yc, x_fit, y_fit):
    t = Table()
    t['xcentroid'] = xc
    t['ycentroid'] = yc
    t['x_fit'] = x_fit
    t['y_fit'] = y_fit
    return t


class _PositionWCS:
    """Records where it was evaluated; scale grows with x."""

    def __init__(self):
        self.calls = []

    def pixel_to_world(self, x, y):
        self.calls.append((x, y))
        return SkyCoord(
            (RA0 - 0.5 * 1e-3 * x ** 2 * MAS / np.cos(np.deg2rad(DEC0))) * u.deg,
            (DEC0 + y * 30 * MAS) * u.deg)


def test_catalog_pixel_scale_reads_xcentroid_not_x_fit(monkeypatch):
    from jwst_gc_pipeline.photometry import satstar_position_qa as Q
    import jwst_gc_pipeline.frame_wcs as fw
    w = _PositionWCS()
    monkeypatch.setattr(fw, 'frame_wcs', lambda path, **kw: w)
    # centroids around (100, 200); stamp-local x_fit/y_fit near (3, 4)
    t = _centroid_tbl([90.0, 100.0, 110.0], [190.0, 200.0, 210.0],
                      [2.0, 3.0, 4.0], [3.0, 4.0, 5.0])
    scale = Q._catalog_pixel_scale_mas(t, 'frame.fits')
    assert scale is not None
    assert w.calls[0] == (100.0, 200.0)       # median centroid, not median x_fit
    # the on-sky x step from x=100 to 101 is 0.5e-3 * (101**2 - 100**2) mas;
    # the y step is 30 mas.  At the x_fit position (3) the x step would be
    # ~0.003 mas, giving a scale ~30x smaller.
    sx = 0.5e-3 * (101.0 ** 2 - 100.0 ** 2)
    assert scale == pytest.approx(np.sqrt(sx * 30.0), rel=1e-2)


def test_catalog_pixel_scale_none_without_a_frame_or_centroids():
    from jwst_gc_pipeline.photometry import satstar_position_qa as Q
    t = _centroid_tbl([1.0], [2.0], [1.0], [2.0])
    assert Q._catalog_pixel_scale_mas(t, None) is None
    t.remove_column('xcentroid')
    assert Q._catalog_pixel_scale_mas(t, 'frame.fits') is None
    t = _centroid_tbl([np.nan], [np.nan], [1.0], [2.0])
    assert Q._catalog_pixel_scale_mas(t, 'frame.fits') is None


def test_catalog_pixel_scale_none_when_frame_unreadable(monkeypatch):
    from jwst_gc_pipeline.photometry import satstar_position_qa as Q
    import jwst_gc_pipeline.frame_wcs as fw

    def _boom(path, **kw):
        raise OSError('no such frame')
    monkeypatch.setattr(fw, 'frame_wcs', _boom)
    t = _centroid_tbl([1.0], [2.0], [1.0], [2.0])
    assert Q._catalog_pixel_scale_mas(t, 'missing.fits') is None


def test_halfpixel_fraction_is_logged_when_the_frame_resolves(monkeypatch, capsys):
    import jwst_gc_pipeline.frame_wcs as fw
    monkeypatch.setattr(fw, 'frame_wcs', lambda path, **kw: _FakeWCS())
    t = _tbl([5.0, 5.0, 5.0, 20.0])          # half a 31 mas pixel is 15.5 mas
    t['xcentroid'] = [10.0] * 4
    t['ycentroid'] = [20.0] * 4
    qa = log_satstar_position_qa(t, frame_path='frame.fits')
    assert qa['frac_gt_halfpix'] == pytest.approx(0.25)
    assert '25.0% beyond half a pixel' in capsys.readouterr().out


def test_pixel_scale_lookup_leaves_the_wcs_resolution_tally_alone(tmp_path):
    """The QA is log-only: reading the frame must not change the WCSSRC /
    WCSNGW / WCSNSIP cards stamped into catalogs written later (#951)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    import jwst_gc_pipeline.frame_wcs as fw
    from jwst_gc_pipeline.photometry import satstar_position_qa as Q

    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [50, 50]
    w.wcs.cdelt = [-31 * MAS, 31 * MAS]
    path = tmp_path / 'frame.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros((100, 100)), header=w.to_header(),
                                name='SCI')]).writeto(path)
    t = _centroid_tbl([50.0], [50.0], [1.0], [1.0])

    before = dict(fw._RESOLUTION_TALLY)
    # the frame has no GWCS, so a bare frame_wcs() call is a SIP resolution
    fw.frame_wcs(str(path), warn_missing=False)
    assert fw._RESOLUTION_TALLY != before, "probe: frame_wcs should tally"
    fw._RESOLUTION_TALLY.clear()
    fw._RESOLUTION_TALLY.update(before)

    scale = Q._catalog_pixel_scale_mas(t, str(path))
    assert scale == pytest.approx(31.0, rel=1e-2)
    assert fw._RESOLUTION_TALLY == before


def test_tally_is_restored_when_the_frame_read_fails(monkeypatch):
    import jwst_gc_pipeline.frame_wcs as fw
    from jwst_gc_pipeline.photometry import satstar_position_qa as Q
    before = dict(fw._RESOLUTION_TALLY)

    def _count_then_fail(path, **kw):
        fw._RESOLUTION_TALLY['sip'] += 1
        raise RuntimeError('unreadable GWCS')
    monkeypatch.setattr(fw, 'frame_wcs', _count_then_fail)
    t = _centroid_tbl([1.0], [2.0], [1.0], [2.0])
    assert Q._catalog_pixel_scale_mas(t, 'frame.fits') is None
    assert fw._RESOLUTION_TALLY == before
