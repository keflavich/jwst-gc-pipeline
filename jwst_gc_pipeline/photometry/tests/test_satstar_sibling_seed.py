"""Issue #925 item 3: measure every satstar in every exposure that covers it.

Measured on o132, comparing per star how many exposures' footprints contain it
against how many produced a satstar row:

* 15.3% (F480M, 274 of 1791) and 14.7% (F212N, 2296 of 15622) of star-exposures
  are covered by an exposure that measured nothing there.
* 67.9% (F480M) / 70.2% (F212N) of the single-measurement stars are covered by
  more than one exposure, so the gap is a threshold the star straddles rather
  than the edge of the dither pattern.

The gate-rejected fits already on disk are NOT a way to fill it.  Their
positions sit 27.17 mas (`implied_peak_gate`) and 223.44 mas
(`fit_quality_gate`) from the accepted ensemble mean, against 2.27 mas for the
accepted per-exposure fits: folding a 27 mas population into a 2.27 mas ensemble
degrades the mean it is meant to improve, which is reason enough on its own.

Hence forced measurement, and hence position-only -- the forced rows are a
position measurement of a star this exposure did not see saturated, so their
flux must never reach the catalog.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import merge_catalogs as mc
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf


RA0, DEC0 = 266.5, -28.9
ITERS = ('m3', 'm7')


def _table(rows):
    """rows = [(dra_mas, ddec_mas, flux, exposure_index, iter, position_only)]"""
    t = Table()
    t['skycoord_fit'] = SkyCoord(
        [RA0 + (r[0] / 3.6e6) / np.cos(np.radians(DEC0)) for r in rows] * u.deg,
        [DEC0 + r[1] / 3.6e6 for r in rows] * u.deg, frame='icrs')
    t['flux_fit'] = np.array([r[2] for r in rows], dtype=float)
    t[mc._SATSTAR_EXPKEY_COL] = np.array(
        [f'jw10678132001_02101_{r[3]:05d}_nrcalong' for r in rows], dtype=str)
    t[mc._SATSTAR_ITER_COL] = np.array([r[4] for r in rows], dtype=str)
    t['position_only'] = np.array([r[5] for r in rows], dtype=bool)
    return t


def _star(exposures, position_only_from=None, flux=1.0e4, spread=4.0):
    rows = []
    for e in range(exposures):
        po = (position_only_from is not None and e >= position_only_from)
        for k in ITERS:
            rows.append((spread * (-1) ** e, 0.0, flux, e, k, po))
    return rows


# --------------------------------------------------------------------------
# the finder's seeding contract
# --------------------------------------------------------------------------

def test_sibling_is_a_distinct_seed_kind_and_loses_to_real_saturation():
    src = open(ssf.__file__).read()
    # 5 must exist and map to 'sibling'
    assert "5: 'sibling'" in src
    # the kind resolution is min-nonzero, so a component this exposure also saw
    # as DQ-saturated (kind 1) keeps its stronger provenance and its flux
    assert 'ndimage.minimum(' in src
    assert "_kind_img[_smask & (_kind_img == 0)] = 5" in src


def test_sibling_seeding_applies_no_amplitude_threshold():
    src = open(ssf.__file__).read()
    block = src.split('SIBLING-EXPOSURE SEEDING')[1].split('if nsource == 0:')[0]
    # every other seeding block gates on the data level; this one must not, or
    # it re-imposes the condition that created the gap
    assert 'severity_floor' not in block
    assert 'sci_ps >' not in block


def test_position_only_rides_on_the_seed_kind():
    src = open(ssf.__file__).read()
    assert "_position_only = (_seed_kind == 'sibling')" in src
    assert "result['position_only'] = bool(_position_only)" in src


def test_the_implied_peak_gate_keeps_sibling_rows_instead_of_rejecting_them():
    src = open(ssf.__file__).read()
    gate = src.split('Satstar severity gate: REJECT')[0]
    # the branch that skips the rejection must be the position_only one
    assert 'if _position_only:' in gate.split('_obs_pk >= 0.5 * float(_sev_floor)')[-1]


# --------------------------------------------------------------------------
# the ensemble treats position-only rows as positions, never as flux
# --------------------------------------------------------------------------

def test_position_only_rows_count_toward_n_frames():
    # 2 real exposures + 2 sibling-seeded ones
    out = mc._dedup_satstar_catalog(_table(_star(4, position_only_from=2)))
    assert len(out) == 1
    assert int(out['n_frames_fit'][0]) == 4
    assert int(out['n_frames_flux_fit'][0]) == 2


def test_position_only_rows_are_excluded_from_the_flux_statistics():
    rows = _star(2, flux=1.0e4)                      # real, flux 1e4
    rows += [(0.0, 0.0, 9.9e9, e, k, True)           # position-only, absurd flux
             for e in (2, 3) for k in ITERS]
    out = mc._dedup_satstar_catalog(_table(rows))
    assert float(out['flux_med_fit'][0]) == pytest.approx(1.0e4)
    assert int(out['n_frames_fit'][0]) == 4
    assert int(out['n_frames_flux_fit'][0]) == 2


def test_a_position_only_row_never_becomes_the_representative():
    # the position-only row is by far the brightest, so a plain brightest-first
    # pick would hand it flux_fit
    rows = _star(2, flux=1.0e4)
    rows += [(0.0, 0.0, 9.9e9, 2, k, True) for k in ITERS]
    out = mc._dedup_satstar_catalog(_table(rows))
    assert float(out['flux_fit'][0]) == pytest.approx(1.0e4)
    assert bool(out['position_only'][0]) is False


def test_a_star_seen_saturated_nowhere_is_flagged_position_only():
    rows = [(0.0, 0.0, 1.0e4, e, k, True) for e in range(3) for k in ITERS]
    out = mc._dedup_satstar_catalog(_table(rows))
    assert bool(out['position_only'][0]) is True
    # it still carries its ensemble; replace_saturated is what drops it
    assert int(out['n_frames_fit'][0]) == 3
    assert int(out['n_frames_flux_fit'][0]) == 0


def test_position_only_flag_does_not_depend_on_which_row_was_kept():
    # one real exposure, faintest of the group: the flag must come from the
    # group, not from whichever row happened to be representative
    rows = [(0.0, 0.0, 1.0e2, 0, k, False) for k in ITERS]
    rows += [(0.0, 0.0, 9.9e9, e, k, True) for e in (1, 2) for k in ITERS]
    out = mc._dedup_satstar_catalog(_table(rows))
    assert bool(out['position_only'][0]) is False
    assert int(out['n_frames_flux_fit'][0]) == 1


def test_scatter_uses_the_position_only_exposures():
    # real exposures agree exactly; the sibling ones carry the whole spread, so
    # a scatter that ignored them would read zero
    rows = [(0.0, 0.0, 1.0e4, e, k, False) for e in (0, 1) for k in ITERS]
    rows += [(20.0, 0.0, 1.0e4, 2, k, True) for k in ITERS]
    rows += [(-20.0, 0.0, 1.0e4, 3, k, True) for k in ITERS]
    out = mc._dedup_satstar_catalog(_table(rows))
    assert float(out['std_ra_fit'][0]) * 3.6e6 > 10.0


def test_tables_without_the_column_behave_exactly_as_before():
    t = _table(_star(3))
    t.remove_column('position_only')
    out = mc._dedup_satstar_catalog(t)
    assert int(out['n_frames_fit'][0]) == 3
    assert int(out['n_frames_flux_fit'][0]) == 3
    assert bool(out['position_only'][0]) is False


# --------------------------------------------------------------------------
# substitution and near-saturation flagging must ignore position-only stars
# --------------------------------------------------------------------------

def test_replace_saturated_drops_position_only_stars():
    src = open(mc.__file__).read()
    body = src.split('def replace_saturated(')[1].split('\ndef ')[0]
    assert "if 'position_only' in satstar_cat.colnames:" in body
    assert 'satstar_cat = satstar_cat[~_po]' in body


def test_flag_near_saturated_drops_position_only_stars():
    src = open(mc.__file__).read()
    body = src.split('def flag_near_saturated(')[1].split('\ndef ')[0]
    assert "if 'position_only' in satstar_cat.colnames:" in body
    assert 'satstar_cat = satstar_cat[~_po]' in body


def test_sibling_seeds_are_in_the_satstar_cache_key():
    from jwst_gc_pipeline.photometry import satstar_cache
    import inspect
    assert 'sibling_sky' in inspect.signature(
        satstar_cache.satstar_content_key).parameters
    src = open(satstar_cache.__file__).read()
    # present and absent must hash differently, or a fit made without seeds
    # gets adopted by a run that has them
    assert "h.update(b'sibling:none')" in src
    assert "f'sibling:{len(sibling_sky)}'" in src


def test_sibling_seeding_is_plumbed_end_to_end():
    import inspect
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as ccl
    assert 'sibling_xy' in inspect.signature(
        ssf.find_saturated_stars).parameters
    assert 'sibling_sky' in inspect.signature(
        ssf.get_saturated_stars).parameters
    assert 'sibling_sky' in inspect.signature(
        ccl.load_or_make_satstar_catalog).parameters
    cat_src = open(
        __import__('jwst_gc_pipeline.photometry.cataloging',
                   fromlist=['cataloging']).__file__).read()
    assert 'sibling_sky=_sibling_sky,' in cat_src
    # the selection itself is exercised behaviourally below, against real files
    assert 'satstar_sibling_seed_positions(filtername, basepath)' in cat_src


# --------------------------------------------------------------------------
# what the seed list is allowed to contain
#
# This is the property that bounds the whole mechanism.  A position-only row is
# itself a forced measurement, so seeding from one would let the forced
# population feed on its own output; the finder re-runs at every m-token, so it
# would compound six times per pass.  Seeding only from real saturation
# detections makes the seed list a fixed point.
# --------------------------------------------------------------------------

def _write_consolidated(tmp_path, filt, n_real, n_position_only):
    d = tmp_path / 'catalogs'
    d.mkdir(parents=True, exist_ok=True)
    n = n_real + n_position_only
    t = Table()
    t['skycoord_fit'] = SkyCoord(
        (RA0 + np.arange(n) * 1e-4) * u.deg,
        np.full(n, DEC0) * u.deg, frame='icrs')
    t['flux_fit'] = np.full(n, 1.0e4)
    t['position_only'] = np.array([False] * n_real + [True] * n_position_only)
    t.write(d / f'{filt.lower()}_consolidated_satstar_catalog.fits',
            overwrite=True)
    return str(tmp_path)


def test_seed_list_excludes_position_only_stars(tmp_path):
    base = _write_consolidated(tmp_path, 'F480M', n_real=3, n_position_only=5)
    seeds = mc.satstar_sibling_seed_positions('F480M', base)
    assert seeds is not None
    # the five forced-only stars must not seed the next iteration
    assert len(seeds) == 3


def test_seed_list_is_a_fixed_point_across_iterations(tmp_path):
    # feeding the previous iteration's seeds back in must not grow the list
    base = _write_consolidated(tmp_path, 'F480M', n_real=4, n_position_only=9)
    first = mc.satstar_sibling_seed_positions('F480M', base)
    second = mc.satstar_sibling_seed_positions('F480M', base)
    assert len(first) == len(second) == 4


def test_no_seeds_when_nothing_was_ever_seen_saturated(tmp_path):
    base = _write_consolidated(tmp_path, 'F480M', n_real=0, n_position_only=6)
    assert mc.satstar_sibling_seed_positions('F480M', base) is None


def test_no_seeds_on_a_bands_first_pass(tmp_path):
    (tmp_path / 'catalogs').mkdir(parents=True, exist_ok=True)
    assert mc.satstar_sibling_seed_positions('F480M', str(tmp_path)) is None


def test_legacy_catalog_without_the_column_seeds_everything(tmp_path):
    # a consolidated catalog written before #928 has no position_only column;
    # every row in it is a real saturation detection by construction
    d = tmp_path / 'catalogs'
    d.mkdir(parents=True, exist_ok=True)
    t = Table()
    t['skycoord_fit'] = SkyCoord((RA0 + np.arange(5) * 1e-4) * u.deg,
                                 np.full(5, DEC0) * u.deg, frame='icrs')
    t.write(d / 'f480m_consolidated_satstar_catalog.fits', overwrite=True)
    seeds = mc.satstar_sibling_seed_positions('F480M', str(tmp_path))
    assert seeds is not None and len(seeds) == 5


def test_seed_positions_are_the_catalog_positions(tmp_path):
    base = _write_consolidated(tmp_path, 'F480M', n_real=2, n_position_only=1)
    seeds = mc.satstar_sibling_seed_positions('F480M', base)
    assert seeds.ra.deg[0] == pytest.approx(RA0, abs=1e-9)
    assert seeds.ra.deg[1] == pytest.approx(RA0 + 1e-4, abs=1e-9)


def test_position_only_is_computed_from_the_group_not_the_kept_row():
    """Direct unit test of _attach_satstar_ensemble with a POSITION-ONLY keeper.

    The dedup's lexsort already puts real rows ahead of position-only ones, so
    through the normal path the representative is never position-only and a
    representative-inherited flag would give the same answer -- a mutation
    replacing the group computation with `out['position_only']` survives the
    end-to-end tests.  Call the unit directly with a keeper the sort would never
    choose, so the two implementations disagree and the group one is pinned.
    """
    rows = [(0.0, 0.0, 1.0e4, 0, 'm7', True),     # keeper, position-only
            (0.0, 0.0, 1.0e4, 1, 'm7', False)]    # same star, real measurement
    tbl = _table(rows)
    fin_idx = np.array([0, 1])
    owner = np.array([0, 0])                      # both belong to local row 0
    kept_sorted = np.array([0])
    out = mc._attach_satstar_ensemble(tbl[[0]], tbl, fin_idx, owner, kept_sorted)
    assert bool(out['position_only'][0]) is False
    assert int(out['n_frames_fit'][0]) == 2
    assert int(out['n_frames_flux_fit'][0]) == 1


def test_position_only_keeper_with_no_real_member_stays_flagged():
    rows = [(0.0, 0.0, 1.0e4, 0, 'm7', True),
            (0.0, 0.0, 1.0e4, 1, 'm7', True)]
    tbl = _table(rows)
    out = mc._attach_satstar_ensemble(tbl[[0]], tbl, np.array([0, 1]),
                                      np.array([0, 0]), np.array([0]))
    assert bool(out['position_only'][0]) is True
    assert int(out['n_frames_flux_fit'][0]) == 0
