"""Unit tests for post-merge de-duplication (dedup_catalog)."""
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import dedup_catalog as dc


def _row(ra, dec, **bandvals):
    """bandvals like f405n=(mag, detected) -> fills mag/mask/emag/flux_jy."""
    return ra, dec, bandvals


def _build(rows, bands=('f405n', 'f187n'), emag=None, filled=None, pos=None):
    """``emag`` maps (row, band) -> emag (default 0.02); ``filled`` is a set of
    (row, band) forced fills; ``pos`` maps (row, band) -> (ra, dec) for the
    ``skycoord_<band>`` columns (default: the row's own position)."""
    n = len(rows)
    t = Table()
    ras = np.array([r[0] for r in rows])
    decs = np.array([r[1] for r in rows])
    t['skycoord_ref'] = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
    t['skycoord_ref_filtername'] = ['f405n'] * n
    emag = emag or {}
    filled = filled or set()
    pos = pos or {}
    for b in bands:
        if pos:
            ra_b = np.array([pos.get((i, b), (ras[i], decs[i]))[0] for i in range(n)])
            dec_b = np.array([pos.get((i, b), (ras[i], decs[i]))[1] for i in range(n)])
            t[f'skycoord_{b}'] = SkyCoord(ra=ra_b * u.deg, dec=dec_b * u.deg)
        if filled:
            t[f'forced_filled_{b}'] = np.array([(i, b) in filled for i in range(n)])
        mag = np.full(n, np.nan)
        mask = np.ones(n, bool)
        em = np.full(n, np.nan)
        flux = np.full(n, np.nan)
        for i, r in enumerate(rows):
            if b in r[2]:
                m, det = r[2][b]
                mag[i] = m
                if det:
                    mask[i] = False
                    em[i] = emag.get((i, b), 0.02)
                    flux[i] = 10 ** (-0.4 * m) * 1e-7
        t[f'mag_vega_{b}'] = mag
        t[f'mag_ab_{b}'] = mag + 0.5
        t[f'mask_{b}'] = mask
        t[f'emag_ab_{b}'] = em
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


ARCSEC_MAS = 1000.0


def _mas(x):
    """mas -> degrees on the sky for a dec offset."""
    return x / ARCSEC_MAS / 3600.


def _run(tmp_path, t, **kw):
    p = tmp_path / 'in.fits'
    o = tmp_path / 'out.fits'
    t.write(p, overwrite=True)
    dc.dedup_merged_catalog(str(p), str(o), verbose=False, **kw)
    return Table.read(o)


def _near_tie_table(emag_b=0.0422):
    # A: f187n+f405n, B: f187n only (a near-tie f187n collision); both ~equal
    # brightness so the pair merges as redundant.  B has the (slightly) lower
    # emag in f187n; A is the primary (two bands).
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True), f405n=(16.0, True)),
        _row(0.0, _mas(40), f187n=(15.03, True)),
    ]
    pos = {(1, 'f187n'): (0.0, _mas(100)), (0, 'f187n'): (0.0, 0.0),
           (0, 'f405n'): (0.0, _mas(30)), (1, 'f405n'): (0.0, _mas(40))}
    return _build(rows, emag={(0, 'f187n'): 0.0422, (1, 'f187n'): emag_b},
                  pos=pos)


def test_provenance_columns_unmerged_rows(tmp_path):
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True)),
        _row(1.0, 1.0, f187n=(14.0, True), f405n=(14.5, True)),
    ]
    out = _run(tmp_path, _build(rows))
    assert list(out['dedup_primary']) == [0, 1]
    assert list(out['dedup_src_f187n']) == [0, 1]
    assert not np.any(out['dedup_band_swapped_f187n'])
    assert np.all(out['dedup_pos_spread_mas'] == 0)
    assert 'DEDUPSRC' in out.meta and 'in.fits' in out.meta['DEDUPSRC']


def test_provenance_columns_swapped_band(tmp_path):
    # default rule: the lower-emag member (row 1) supplies f187n
    out = _run(tmp_path, _near_tie_table(emag_b=0.0416))
    assert len(out) == 1
    r = out[0]
    assert r['n_merged'] == 2
    assert r['dedup_primary'] == 0
    assert r['dedup_src_f187n'] == 1 and bool(r['dedup_band_swapped_f187n'])
    assert r['dedup_src_f405n'] == 0 and not bool(r['dedup_band_swapped_f405n'])
    # output f187n at 100 mas, f405n at 30 mas
    assert r['dedup_pos_spread_mas'] == pytest.approx(70.0, abs=0.5)


def test_provenance_can_be_disabled(tmp_path):
    out = _run(tmp_path, _near_tie_table(), write_provenance=False)
    assert not [c for c in out.colnames if c.startswith('dedup_')]


def test_default_ratio_none_keeps_lowest_emag_rule(tmp_path):
    out = _run(tmp_path, _near_tie_table(emag_b=0.0416))
    assert out[0]['dedup_src_f187n'] == 1


def test_winner_hysteresis_keeps_primary_on_near_tie(tmp_path):
    out = _run(tmp_path, _near_tie_table(emag_b=0.0416), winner_emag_ratio=0.8)
    r = out[0]
    assert r['dedup_src_f187n'] == 0 and not bool(r['dedup_band_swapped_f187n'])
    assert np.all(out['dedup_pos_spread_mas'] == 0)
    assert r['mag_vega_f187n'] == pytest.approx(15.00)


def test_winner_hysteresis_clear_win_still_swaps(tmp_path):
    out = _run(tmp_path, _near_tie_table(emag_b=0.01), winner_emag_ratio=0.8)
    r = out[0]
    assert r['dedup_src_f187n'] == 1
    assert r['mag_vega_f187n'] == pytest.approx(15.03)


def test_primary_without_independent_detection_takes_challenger(tmp_path):
    # primary (row 0, two bands) has f187n only as a forced fill; row 1
    # detects f187n independently, so the primary must take it even with
    # a hysteresis ratio.
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True), f405n=(16.0, True)),
        _row(0.0, _mas(40), f187n=(15.03, True)),
    ]
    t = _build(rows, filled={(0, 'f187n')})
    out = _run(tmp_path, t, winner_emag_ratio=0.8)
    assert len(out) == 1
    assert out[0]['dedup_src_f187n'] == 1
    assert out[0]['dedup_src_f405n'] == 0


def test_forced_fill_does_not_count_as_detection(tmp_path):
    # row 0 has a forced f405n fill, row 1 an independent f405n detection with
    # a very different magnitude: no collision, so the pair merges and the
    # f405n comes from row 1
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True), f405n=(19.0, True)),
        _row(0.0, _mas(40), f187n=(15.03, True), f405n=(16.0, True)),
    ]
    t = _build(rows, filled={(0, 'f405n')})
    out = _run(tmp_path, t, winner_emag_ratio=0.8)
    assert len(out) == 1
    assert out[0]['dedup_src_f405n'] == 1


def _resolved_table():
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True), f405n=(16.0, True)),
        _row(0.0, _mas(40), f187n=(15.05, True), f405n=(16.02, True)),
    ]
    pos = {(0, 'f187n'): (0.0, 0.0), (1, 'f187n'): (0.0, _mas(100)),
           (0, 'f405n'): (0.0, 0.0), (1, 'f405n'): (0.0, _mas(5))}
    return _build(rows, pos=pos)


def test_resolved_in_band_guard_default_off(tmp_path):
    assert len(_run(tmp_path, _resolved_table())) == 1


def test_resolved_in_band_guard_blocks_link(tmp_path):
    out = _run(tmp_path, _resolved_table(), band_resolve_mas=60)
    assert len(out) == 2
    # a threshold above the separation still merges
    assert len(_run(tmp_path, _resolved_table(), band_resolve_mas=150)) == 1


def test_resolved_in_band_guard_per_band_mapping(tmp_path):
    # guard applied to f405n only: its 5 mas separation does not block
    out = _run(tmp_path, _resolved_table(), band_resolve_mas={'f405n': 60})
    assert len(out) == 1
    out = _run(tmp_path, _resolved_table(), band_resolve_mas={'f187n': 60})
    assert len(out) == 2


def test_resolved_in_band_guard_component_check(tmp_path):
    # A-B and B-C are each within the guard, A-C is not: the component is
    # left intact although every pairwise link passes
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True)),
        _row(0.0, _mas(40), f187n=(15.03, True)),
        _row(0.0, _mas(80), f187n=(15.06, True)),
    ]
    pos = {(0, 'f187n'): (0.0, 0.0), (1, 'f187n'): (0.0, _mas(50)),
           (2, 'f187n'): (0.0, _mas(100))}
    out = _run(tmp_path, _build(rows, pos=pos), band_resolve_mas=60)
    assert len(out) == 3


def test_resolved_in_band_guard_pairwise_block(tmp_path):
    # rows at 0, 40, 80 mas on the reference sky; f187n positions 0, 100, 110
    # mas.  A-B and A-C are resolved in f187n (threshold 60), B-C is not.  The
    # pairwise test drops the two resolved links, so B and C merge and A stays;
    # the component check alone would see {A, B, C} with one resolved pair and
    # leave all three intact, so this pins the pairwise block separately.
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True)),
        _row(0.0, _mas(40), f187n=(15.03, True)),
        _row(0.0, _mas(80), f187n=(15.06, True)),
    ]
    pos = {(0, 'f187n'): (0.0, 0.0), (1, 'f187n'): (0.0, _mas(100)),
           (2, 'f187n'): (0.0, _mas(110))}
    out = _run(tmp_path, _build(rows, pos=pos), band_resolve_mas=60)
    assert len(out) == 2
    assert sorted(out['n_merged']) == [1, 2]
    # the unguarded run merges all three
    assert len(_run(tmp_path, _build(rows, pos=pos))) == 1


def test_resolved_in_band_guard_threshold_is_strict(tmp_path):
    # a separation just under / over the threshold merges / splits
    for sep, nrows in ((59.0, 1), (61.0, 2)):
        rows = [
            _row(0.0, 0.0, f187n=(15.00, True)),
            _row(0.0, _mas(40), f187n=(15.03, True)),
        ]
        pos = {(0, 'f187n'): (0.0, 0.0), (1, 'f187n'): (0.0, _mas(sep))}
        out = _run(tmp_path, _build(rows, pos=pos), band_resolve_mas=60)
        assert len(out) == nrows


def _resolved_pair(bands_order, mag_collision_band):
    # two rows 40 mas apart on the sky, resolved in f187n (100 mas apart in
    # that band) and, in ``mag_collision_band``, 1 mag apart
    rows = [
        _row(0.0, 0.0, f187n=(15.0, True), f405n=(16.0, True)),
        _row(0.0, _mas(40), f187n=(15.02, True), f405n=(16.0, True)),
    ]
    if mag_collision_band == 'f405n':
        rows[1][2]['f405n'] = (17.0, True)
    pos = {(0, 'f187n'): (0.0, 0.0), (1, 'f187n'): (0.0, _mas(100))}
    return _build(rows, bands=bands_order, pos=pos)


@pytest.mark.parametrize('order', [('f187n', 'f405n'), ('f405n', 'f187n')])
def test_resolved_only_counter_does_not_depend_on_band_order(tmp_path, capsys,
                                                             order):
    # blocked by the magnitude test as well as resolved -> not counted as
    # blocked only by the guard, whichever band comes first
    t = _resolved_pair(order, 'f405n')
    p, o = tmp_path / 'in.fits', tmp_path / 'out.fits'
    t.write(p)
    dc.dedup_merged_catalog(str(p), str(o), band_resolve_mas=60)
    assert '(0 blocked only by the resolved-in-band guard)' in capsys.readouterr().out
    # resolved only: counted once
    t = _resolved_pair(order, None)
    t.write(p, overwrite=True)
    dc.dedup_merged_catalog(str(p), str(o), band_resolve_mas=60)
    assert '(1 blocked only by the resolved-in-band guard)' in capsys.readouterr().out


def test_parse_band_resolve_mas():
    assert dc.parse_band_resolve_mas(None) is None
    assert dc.parse_band_resolve_mas('') is None
    assert dc.parse_band_resolve_mas('60') == 60.0
    assert dc.parse_band_resolve_mas(60) == 60.0
    assert dc.parse_band_resolve_mas('F212N=60, f200w=50') == {
        'f212n': 60.0, 'f200w': 50.0}
    # an already-parsed mapping is validated and passed through
    assert dc.parse_band_resolve_mas({'F212N': 60}) == {'f212n': 60.0}


@pytest.mark.parametrize('bad', [
    'abc', 'f212n=', '60,50', 'f212n=60,,f200w=50', '=60', 'f212n=60=50',
    '-5', '0', 'nan', 'inf', 'f187n=-1', 'f187n=nan', 'f187n=0', 'f187n=inf',
    {}, {'f187n': -1}, {'': 60}])
def test_parse_band_resolve_mas_rejects_malformed_or_inert(bad):
    # malformed text, or a value that would leave the guard silently off
    with pytest.raises(ValueError):
        dc.parse_band_resolve_mas(bad)


@pytest.mark.parametrize('bad', [float('nan'), 0, 0.0, -0.8, float('inf'), 'x'])
def test_winner_emag_ratio_rejects_inert_values(bad):
    with pytest.raises(ValueError):
        dc.check_winner_emag_ratio(bad)


def test_winner_emag_ratio_accepts_and_warns_above_one():
    assert dc.check_winner_emag_ratio(None) is None
    assert dc.check_winner_emag_ratio(0.8) == 0.8
    assert dc.check_winner_emag_ratio(1.0) == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        dc.check_winner_emag_ratio(0.8)
    with pytest.warns(UserWarning, match='default lowest-emag rule'):
        assert dc.check_winner_emag_ratio(1.5) == 1.5


def test_bad_options_fail_before_the_input_is_read(tmp_path):
    # the input does not exist: the option error must come first
    with pytest.raises(ValueError, match='band_resolve_mas'):
        dc.dedup_merged_catalog(str(tmp_path / 'nope.fits'),
                                str(tmp_path / 'out.fits'),
                                band_resolve_mas='nan')
    with pytest.raises(ValueError, match='winner_emag_ratio'):
        dc.dedup_merged_catalog(str(tmp_path / 'nope.fits'),
                                str(tmp_path / 'out.fits'),
                                winner_emag_ratio=float('nan'))


def test_guard_band_not_in_table_warns_and_is_off(tmp_path):
    with pytest.warns(UserWarning, match=r'f999w \(not a band of this table'):
        out = _run(tmp_path, _resolved_table(), band_resolve_mas={'f999w': 60})
    # nothing was guarded: the resolved pair merged, no DEDUPRES, "guard is OFF"
    assert len(out) == 1
    assert 'DEDUPRES' not in out.meta


def test_guard_band_without_skycoord_warns_and_is_off(tmp_path):
    # _build without ``pos`` writes no skycoord_<band> columns
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True)),
        _row(0.0, _mas(40), f187n=(15.03, True)),
    ]
    with pytest.warns(UserWarning, match='no skycoord_f187n SkyCoord column.*guard is OFF'):
        out = _run(tmp_path, _build(rows), band_resolve_mas=60)
    assert len(out) == 1 and 'DEDUPRES' not in out.meta


def test_guard_partially_applied_warns_but_stays_on(tmp_path):
    with pytest.warns(UserWarning) as rec:
        out = _run(tmp_path, _resolved_table(),
                   band_resolve_mas={'f187n': 60, 'f999w': 60})
    msgs = [str(w.message) for w in rec]
    assert any('f999w' in m and 'guard is OFF' not in m for m in msgs)
    assert len(out) == 2 and out.meta['DEDUPRES'] == 'f187n=60'


def test_header_records_input_name_row_count_and_options(tmp_path):
    out = _run(tmp_path, _resolved_table(), winner_emag_ratio=0.8,
               band_resolve_mas={'f187n': 60})
    assert out.meta['DEDUPSRC'] == 'pre-dedup row indices refer to in.fits'
    assert out.meta['DEDUPN0'] == 2
    assert out.meta['DEDUPRAT'] == 0.8
    assert out.meta['DEDUPRES'] == 'f187n=60'
    plain = _run(tmp_path, _resolved_table())
    assert plain.meta['DEDUPN0'] == 2
    assert 'DEDUPRAT' not in plain.meta and 'DEDUPRES' not in plain.meta
    off = _run(tmp_path, _resolved_table(), write_provenance=False)
    assert not [k for k in off.meta if k.startswith('DEDUP')]


def test_position_spread_counts_only_independent_detections(tmp_path):
    # f405n is a forced fill in both rows (at a distant position) and so
    # takes no part in the spread; f187n alone gives no pair
    rows = [
        _row(0.0, 0.0, f187n=(15.00, True), f405n=(16.0, True)),
        _row(0.0, _mas(40), f187n=(15.03, True), f405n=(16.0, True)),
    ]
    pos = {(0, 'f187n'): (0.0, 0.0), (1, 'f187n'): (0.0, _mas(100)),
           (0, 'f405n'): (0.0, _mas(300)), (1, 'f405n'): (0.0, _mas(300))}
    t = _build(rows, emag={(0, 'f187n'): 0.0422, (1, 'f187n'): 0.0416},
               filled={(0, 'f405n'), (1, 'f405n')}, pos=pos)
    out = _run(tmp_path, t)
    assert len(out) == 1 and out[0]['dedup_src_f187n'] == 1
    assert out[0]['dedup_pos_spread_mas'] == 0


def test_position_spread_reads_the_supplying_row(tmp_path):
    # f187n comes from row 1 (at 100 mas), f405n stays row 0's (at 30 mas);
    # row 0's own f187n position (0 mas) must not be used
    out = _run(tmp_path, _near_tie_table(emag_b=0.0416))
    assert out[0]['dedup_pos_spread_mas'] == pytest.approx(70.0, abs=0.5)


def test_rerun_on_a_dedup_output_does_not_swap_provenance_columns(tmp_path):
    t = _near_tie_table(emag_b=0.0416)
    t['dedup_src_f187n'] = [10, 20]
    t['dedup_primary'] = [10, 20]
    out = _run(tmp_path, t, write_provenance=False)
    assert len(out) == 1
    assert out['dedup_src_f187n'][0] == 10


def _indep_table():
    t = _near_tie_table(emag_b=0.0416)
    # row 0 (primary): independent in f405n only; row 1: in f187n only
    t['independently_detected_f187n'] = [False, True]
    t['independently_detected_f405n'] = [True, False]
    t['n_filt_independent'] = np.array([1, 1], dtype='i4')
    return t


def test_n_filt_independent_follows_the_swapped_bands(tmp_path):
    # f187n (and its independently_detected flag) moves from row 1 to the
    # primary, so the primary is independent in both bands now (#1329)
    out = _run(tmp_path, _indep_table())
    assert len(out) == 1
    assert out[0]['independently_detected_f187n']
    assert out[0]['independently_detected_f405n']
    assert out[0]['n_filt_independent'] == 2
    assert out['n_filt_independent'].dtype.kind == 'i'
    assert out['n_filt_independent'].dtype.itemsize == 4


def test_n_filt_independent_kept_where_no_band_swaps(tmp_path):
    # hysteresis keeps the primary's f187n: nothing swapped, count untouched
    # (even when the stored value disagrees with the flags)
    t = _indep_table()
    t['n_filt_independent'] = np.array([7, 1], dtype='i4')
    out = _run(tmp_path, t, winner_emag_ratio=0.8)
    assert len(out) == 1 and out[0]['n_filt_independent'] == 7


def test_n_filt_independent_of_unmerged_rows_untouched(tmp_path):
    rows = [_row(0.0, 0.0, f187n=(15.0, True), f405n=(16.0, True)),
            _row(0.0, _mas(40), f187n=(15.03, True)),
            _row(1.0, 1.0, f187n=(14.0, True), f405n=(14.5, True))]
    t = _build(rows, emag={(0, 'f187n'): 0.0422, (1, 'f187n'): 0.0416})
    t['independently_detected_f187n'] = [False, True, True]
    t['independently_detected_f405n'] = [True, False, True]
    t['n_filt_independent'] = np.array([1, 1, 5], dtype='i4')
    out = _run(tmp_path, t)
    assert sorted(out['n_filt_independent']) == [2, 5]


def test_tables_without_n_filt_independent_are_unchanged(tmp_path):
    out = _run(tmp_path, _near_tie_table(emag_b=0.0416))
    assert 'n_filt_independent' not in out.colnames
