"""The dashboard's per-tile numbers must come from the map that GATED the tie.

Issue #668.  Since #667 a small verified tie on a dense reference is gated by
the same-star region map (``per_tile_source == 'same-star-region'``) and the
histogram grid in ``per_tile`` is kept only as a diagnostic.  The monitor used
to publish the grid's worst cell regardless -- on cloudef that is a 5953.8 mas
noise bin beside a checkpoint that passes -- with nothing on the page saying
which map decided.  These tests pin: the scanner follows ``per_tile_source``,
keeps the grid under a labelled diagnostic key, the check follows the region
map's own verdict, and a histogram-gated record is unchanged.
"""
import json
import os

from jwst_gc_pipeline.monitoring import checks, render, scan


def _grid():
    return {'n_ok': 36, 'n_total': 36, 'clean': False,
            'worst_off_mas': 5953.8, 'worst_off_cell': {'ix': 5, 'iy': 0},
            'min_contrast_seen': 1.2,
            'cells': [{'ix': 5, 'iy': 0, 'off_mas': 5953.8, 'dra': 5000.0,
                       'ddec': 3000.0, 'contrast': 1.2, 'npairs': 40},
                      {'ix': 2, 'iy': 2, 'off_mas': 3.0, 'dra': 2.0,
                       'ddec': 2.0, 'contrast': 30.0, 'npairs': 400}]}


def _region(flag_worst=False):
    return {'n_cells': 2, 'n_measured': 2, 'n_flagged': int(flag_worst),
            'n_uncovered': 0, 'measurable': True, 'clean': not flag_worst,
            'worst_off_mas': 21.5, 'tol_mas': 26.0, 'tol_source': 'adaptive',
            'cells': [{'ix': 1, 'iy': 0, 'n': 120, 'resid_off_mas': 21.5,
                       'resid_dra_mas': 20.0, 'resid_ddec_mas': 8.0,
                       'flagged': flag_worst},
                      {'ix': 0, 'iy': 0, 'n': 300, 'resid_off_mas': 2.0,
                       'resid_dra_mas': 1.0, 'resid_ddec_mas': 1.7,
                       'flagged': False}]}


def _scan(tmp_path, tie):
    ckdir = tmp_path / 'astrometry_checkpoints'
    os.makedirs(ckdir, exist_ok=True)
    rec = {'stage': 'm2', 'date': '2026-09-10T00:00:00Z', 'passed': True,
           'failures': [],
           'visits': [{'visit': '1', 'consensus': {'consensus_ok': True},
                       'reference_tie': tie,
                       'exposures': [{'off': 1.0, 'contrast': 40.0, 'ok': True}]}]}
    (ckdir / 'checkpoint_m2_F200W_latest.json').write_text(json.dumps(rec))
    return scan.astrometry_checkpoints(str(tmp_path))


def _run(astrom):
    return {'target': 'cloudef', 'proposal': '1182', 'obsid': '001',
            'astrometry': astrom, 'per_filter': {}}


def _tile_verdicts(astrom):
    return [v for v in checks.check_astrometry(_run(astrom))
            if v['name'].startswith(('astrometry-worst-tile',
                                     'astrometry-tile-contrast'))]


def test_region_gated_record_publishes_the_region_map(tmp_path):
    astrom = _scan(tmp_path, {'off_mas': 0.334, 'per_tile': _grid(),
                              'per_tile_source': 'same-star-region',
                              'per_tile_same_star': _region()})
    visit = astrom['F200W']['visits'][0]
    assert visit['tile_source'] == 'same-star-region'
    assert visit['worst_tile_mas'] == 21.5
    assert visit['worst_tile_cell'] == '(1,0)'
    assert visit['tiles_ok'] == 2 and visit['tiles_total'] == 2
    assert visit['tile_tol_mas'] == 26.0
    assert visit['min_tile_contrast'] is None
    assert [c['off_mas'] for c in visit['cells']] == [21.5, 2.0]
    # The grid is kept, labelled as the diagnostic it is.
    assert visit['grid_diagnostic']['worst_tile_mas'] == 5953.8


def test_region_gated_clean_map_raises_no_tile_finding(tmp_path):
    """21.5 mas is over LOCAL_CELL_TOL_MAS but inside the map's own 26 mas
    tolerance and unflagged, so the checkpoint passed and the page agrees."""
    astrom = _scan(tmp_path, {'off_mas': 0.334, 'per_tile': _grid(),
                              'per_tile_source': 'same-star-region',
                              'per_tile_same_star': _region()})
    assert _tile_verdicts(astrom) == []


def test_region_gated_flagged_map_still_warns(tmp_path):
    astrom = _scan(tmp_path, {'off_mas': 0.334, 'per_tile': _grid(),
                              'per_tile_source': 'same-star-region',
                              'per_tile_same_star': _region(flag_worst=True)})
    names = [v['name'] for v in _tile_verdicts(astrom)]
    assert names == ['astrometry-worst-tile-F200W-v1']
    assert 'Same-star region map' in _tile_verdicts(astrom)[0]['detail']


def test_histogram_gated_record_is_unchanged(tmp_path):
    astrom = _scan(tmp_path, {'off_mas': 0.334, 'per_tile': _grid()})
    visit = astrom['F200W']['visits'][0]
    assert visit['tile_source'] == 'histogram-grid'
    assert visit['worst_tile_mas'] == 5953.8
    assert visit['grid_diagnostic'] is None
    names = sorted(v['name'] for v in _tile_verdicts(astrom))
    assert names == ['astrometry-tile-contrast-F200W-v1',
                     'astrometry-worst-tile-F200W-v1']


def test_table_names_the_gating_map(tmp_path):
    astrom = _scan(tmp_path, {'off_mas': 0.334, 'per_tile': _grid(),
                              'per_tile_source': 'same-star-region',
                              'per_tile_same_star': _region()})
    html = render._astrometry_table(_run(astrom))
    assert 'same-star-region' in html
    assert 'histogram grid (diagnostic, not gating)' in html
    assert '5953.8' not in html.split('title=')[0]
