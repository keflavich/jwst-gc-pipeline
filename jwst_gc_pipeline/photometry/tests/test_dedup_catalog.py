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


def test_parse_band_resolve_mas():
    assert dc.parse_band_resolve_mas(None) is None
    assert dc.parse_band_resolve_mas('') is None
    assert dc.parse_band_resolve_mas('60') == 60.0
    assert dc.parse_band_resolve_mas('F212N=60, f200w=50') == {
        'f212n': 60.0, 'f200w': 50.0}
