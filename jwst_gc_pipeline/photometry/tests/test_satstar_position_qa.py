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
    assert satstar_position_qa(t) == {'available': False}
    qa = log_satstar_position_qa(t)
    assert qa == {'available': False}
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
    assert '0/2 rows carry a finite component anchor' in format_satstar_position_qa(qa)


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
