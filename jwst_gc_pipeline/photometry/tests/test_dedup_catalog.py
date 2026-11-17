"""Unit tests for post-merge de-duplication (dedup_catalog)."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import dedup_catalog as dc


def _row(ra, dec, **bandvals):
    """bandvals like f405n=(mag, detected) -> fills mag/mask/emag/flux_jy."""
    return ra, dec, bandvals


def _build(rows, bands=('f405n', 'f187n')):
    n = len(rows)
    t = Table()
    ras = np.array([r[0] for r in rows])
    decs = np.array([r[1] for r in rows])
    t['skycoord_ref'] = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
    t['skycoord_ref_filtername'] = ['f405n'] * n
    for b in bands:
        mag = np.full(n, np.nan)
        mask = np.ones(n, bool)
        emag = np.full(n, np.nan)
        flux = np.full(n, np.nan)
        for i, r in enumerate(rows):
            if b in r[2]:
                m, det = r[2][b]
                mag[i] = m
                if det:
                    mask[i] = False
                    emag[i] = 0.02
                    flux[i] = 10 ** (-0.4 * m) * 1e-7
        t[f'mag_vega_{b}'] = mag
        t[f'mag_ab_{b}'] = mag + 0.5
        t[f'mask_{b}'] = mask
        t[f'emag_ab_{b}'] = emag
        t[f'flux_jy_{b}'] = flux
    return t


def test_complementary_pair_merges(tmp_path):
    # one star split: row A has f187n only, row B (0.04") has f405n only
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.04 / 3600., f405n=(16.0, True)),
        _row(1.0, 1.0, f187n=(14.0, True), f405n=(14.5, True)),  # isolated
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False)
    out = Table.read(o)
    assert len(out) == 2  # the split pair collapsed to one
    # the merged primary carries BOTH bands
    merged = out[out['n_merged'] > 1][0]
    assert np.isfinite(merged['mag_vega_f187n'])
    assert np.isfinite(merged['mag_vega_f405n'])
    assert not bool(merged['mask_f405n'])
    assert not bool(merged['mask_f187n'])


def test_resolved_binary_preserved(tmp_path):
    # two real stars 0.05" apart, both detected in f187n with 2-mag difference
    rows = [
        _row(0.0, 0.0, f187n=(14.0, True), f405n=(15.0, True)),
        _row(0.0, 0.05 / 3600., f187n=(16.0, True)),
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False)
    out = Table.read(o)
    assert len(out) == 2  # NOT merged -- binary protected


def test_near_identical_collision_merges(tmp_path):
    # same star detected in f187n in both rows with ~identical mag -> merge
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True), f405n=(16.0, True)),
        _row(0.0, 0.03 / 3600., f187n=(15.03, True)),
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), dmag_collision=0.1, verbose=False)
    out = Table.read(o)
    assert len(out) == 1


def test_far_pair_untouched(tmp_path):
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.5 / 3600., f405n=(16.0, True)),  # 0.5" away
    ]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False)
    out = Table.read(o)
    assert len(out) == 2


def test_complementary_split_flagged():
    # two rows 60 mas apart with disjoint bands, plus an isolated row
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.06 / 3600., f405n=(16.0, True)),
        _row(1.0, 1.0, f187n=(14.0, True), f405n=(14.5, True)),
    ]
    t = _build(rows)
    before = {c: np.array(t[c]) for c in t.colnames if c != 'skycoord_ref'}
    out = dc.flag_complementary_splits(t)
    assert len(out) == 3  # nothing dropped
    assert list(out['complementary_split']) == [True, True, False]
    # one group id, the lowest member row index; -1 where unflagged
    assert list(out['split_group']) == [0, 0, -1]
    for c, v in before.items():
        np.testing.assert_array_equal(np.array(out[c]), v)


def test_split_group_chain_uses_lowest_index():
    # three rows in a line, 60 mas apart: A-B and B-C are complementary
    rows = [
        _row(1.0, 1.0, f187n=(15.0, True)),
        _row(1.0, 1.0 + 0.06 / 3600., f405n=(16.0, True)),
        _row(1.0, 1.0 + 0.12 / 3600., f187n=(15.0, True)),
    ]
    # reorder so the middle row has the lowest index
    t = _build([rows[1], rows[0], rows[2]])
    dc.flag_complementary_splits(t)
    assert all(t['complementary_split'])
    assert set(t['split_group']) == {0}


def test_resolved_pair_not_flagged():
    # both rows detected in f187n with a 2 mag difference: resolved binary
    rows = [
        _row(0.0, 0.0, f187n=(14.0, True), f405n=(15.0, True)),
        _row(0.0, 0.05 / 3600., f187n=(16.0, True)),
    ]
    t = dc.flag_complementary_splits(_build(rows))
    assert not t['complementary_split'].any()
    assert (t['split_group'] == -1).all()


def test_redundant_pair_and_far_pair_not_flagged():
    rows = [
        # same band, near-identical magnitude: a duplicate, not a split
        _row(0.0, 0.0, f187n=(15.00, True), f405n=(16.0, True)),
        _row(0.0, 0.03 / 3600., f187n=(15.03, True)),
        # complementary but 0.5" apart
        _row(2.0, 2.0, f187n=(15.0, True)),
        _row(2.0, 2.0 + 0.5 / 3600., f405n=(16.0, True)),
    ]
    t = dc.flag_complementary_splits(_build(rows))
    assert not t['complementary_split'].any()


def test_forced_fill_does_not_count_as_a_detection():
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.04 / 3600., f405n=(16.0, True)),
    ]
    t = _build(rows)
    # row 0 carries a forced fill in f405n: no independent detection there
    t['mag_vega_f405n'][0] = 20.0
    t['forced_filled_f405n'] = [True, False]
    dc.flag_complementary_splits(t)
    assert t['complementary_split'].all()


def test_flag_is_idempotent_and_handles_empty_tables():
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.06 / 3600., f405n=(16.0, True)),
    ]
    t = _build(rows)
    dc.flag_complementary_splits(t)
    first = (np.array(t['complementary_split']), np.array(t['split_group']))
    ncols = len(t.colnames)
    dc.flag_complementary_splits(t)
    assert len(t.colnames) == ncols
    np.testing.assert_array_equal(t['complementary_split'], first[0])
    np.testing.assert_array_equal(t['split_group'], first[1])

    empty = dc.flag_complementary_splits(Table({'x': [1.0]}))
    assert not empty['complementary_split'].any()
    assert empty['split_group'][0] == -1


def test_dedup_output_flags_pairs_it_leaves_split(tmp_path):
    # four complementary rows chained at 60 mas exceed max_component=3, so
    # dedup leaves them split and the output rows must carry the flag
    d = 0.06 / 3600.
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, d, f405n=(16.0, True)),
        _row(0.0, 2 * d, f187n=(15.0, True)),
        _row(0.0, 3 * d, f405n=(16.0, True)),
        _row(1.0, 1.0, f187n=(14.0, True), f405n=(14.5, True)),  # isolated
    ]
    t = _build(rows)
    # stale m7 values must be overwritten, not inherited
    t['complementary_split'] = np.ones(len(t), bool)
    t['split_group'] = np.full(len(t), 99)
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), max_component=3, verbose=False)
    out = Table.read(o)
    assert len(out) == 5
    assert list(out['complementary_split']) == [True] * 4 + [False]
    assert list(out['split_group']) == [0, 0, 0, 0, -1]
    assert not (out['split_group'] == 99).any()


def test_dedup_collapse_unchanged_and_flag_cleared(tmp_path):
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.04 / 3600., f405n=(16.0, True)),
        _row(1.0, 1.0, f187n=(14.0, True), f405n=(14.5, True)),
    ]
    t = _build(rows)
    dc.flag_complementary_splits(t)
    assert t['complementary_split'].sum() == 2
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False)
    out = Table.read(o)
    assert len(out) == 2
    # the collapsed pair is one row now, so nothing is left to flag
    assert not out['complementary_split'].any()
    assert (out['split_group'] == -1).all()


def _pair(sep_arcsec, **kw):
    """Two complementary rows (f187n only / f405n only) ``sep_arcsec`` apart."""
    return _build([
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, sep_arcsec / 3600., f405n=(16.0, True)),
    ], **kw)


@pytest.mark.parametrize('sep, flagged', [
    (0.0999, True), (0.1001, False), (0.15, False), (0.19, False), (0.21, False)])
def test_flag_radius_boundary(sep, flagged):
    # default radius 0.10": a pair at 0.15-0.2" must stay unflagged
    t = dc.flag_complementary_splits(_pair(sep))
    assert bool(t['complementary_split'].all()) is flagged
    assert bool(t['complementary_split'].any()) is flagged


def test_flag_custom_radius():
    t = dc.flag_complementary_splits(_pair(0.15), radius=0.2 * u.arcsec)
    assert t['complementary_split'].all()
    t = dc.flag_complementary_splits(_pair(0.07), radius=0.05 * u.arcsec)
    assert not t['complementary_split'].any()


def test_dedup_output_flag_uses_link_radius(tmp_path):
    # a 4-row complementary chain at 70 mas, too large to collapse
    # (max_component=3); with link_radius=0.05" none of the pairs is close
    # enough to flag, with the default 0.10" all of them are
    d = 0.07 / 3600.
    rows = [_row(0.0, k * d, **{('f187n' if k % 2 == 0 else 'f405n'):
                                (15.0 + k, True)}) for k in range(4)]
    t = _build(rows)
    p = tmp_path / 'in.fits'
    t.write(p)
    for link, expect in ((0.05 * u.arcsec, False), (0.10 * u.arcsec, True)):
        o = tmp_path / f'out_{link.value}.fits'
        dc.dedup_merged_catalog(str(p), str(o), link_radius=link,
                                max_component=3, verbose=False)
        out = Table.read(o)
        assert len(out) == 4
        assert bool(out['complementary_split'].all()) is expect
        assert bool(out['complementary_split'].any()) is expect


def test_row_without_independent_detection_not_flagged():
    # row 1 has no detected band (only a masked/non-detected magnitude): there
    # is no shared band, but it is not evidence of a split either
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(0.0, 0.04 / 3600., f405n=(20.0, False)),
    ]
    t = dc.flag_complementary_splits(_build(rows))
    assert not t['complementary_split'].any()
    assert (t['split_group'] == -1).all()
    # the same rows flag once the second one has a real detection
    rows[1] = _row(0.0, 0.04 / 3600., f405n=(20.0, True))
    assert dc.flag_complementary_splits(_build(rows))['complementary_split'].all()


def test_both_rows_without_detection_not_flagged():
    rows = [
        _row(0.0, 0.0, f187n=(20.0, False)),
        _row(0.0, 0.04 / 3600., f405n=(20.0, False)),
    ]
    t = dc.flag_complementary_splits(_build(rows))
    assert not t['complementary_split'].any()


def test_satstar_wide_pair_is_not_flagged():
    # a complementary pair 0.3" apart with one saturated row: dedup links it
    # through the satstar radius, the flag only looks within ``radius``
    t = _pair(0.30)
    t['is_saturated_f187n'] = [True, False]
    dc.flag_complementary_splits(t)
    assert not t['complementary_split'].any()
    dc.flag_complementary_splits(t, radius=0.5 * u.arcsec)
    assert t['complementary_split'].all()


def test_flag_has_no_magnitude_threshold():
    # complementary pairs share no band, so no magnitude can matter
    assert 'dmag_collision' not in dc.flag_complementary_splits.__code__.co_varnames[
        :dc.flag_complementary_splits.__code__.co_argcount]
    for dm in (0.0, 5.0):
        rows = [_row(0.0, 0.0, f187n=(15.0, True)),
                _row(0.0, 0.04 / 3600., f405n=(15.0 + dm, True))]
        assert dc.flag_complementary_splits(_build(rows))['complementary_split'].all()


def test_nan_skycoord_ref_raises_valueerror():
    t = _pair(0.04)
    sc = t['skycoord_ref']
    ra = np.array(sc.ra.deg)
    ra[1] = np.nan
    t['skycoord_ref'] = SkyCoord(ra=ra * u.deg, dec=sc.dec)
    with pytest.raises(ValueError, match='non-finite'):
        dc.flag_complementary_splits(t)
