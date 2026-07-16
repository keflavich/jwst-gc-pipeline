"""Unit tests for the operational scripts in scripts/reduction/ (not part of
the package; imported by path)."""
import importlib.util
import json
import os
import time

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reduction')


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPTS, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rename_stale_band_token():
    m = _load('rename_stale_mosaics')
    assert m.band_of('jw02221-o001_t001_nircam_clear-f182m-merged-reproject-vvv_i2d.fits') == 'f182m'
    assert m.band_of('jw02221-o002_t001_miri_f2550w_realigned-to-vvv.fits') == 'f2550w'
    assert m.band_of('jw02221-o001_t001_nircam_clear-F405N-merged_realigned-to-vvv.fits') == 'f405n'
    assert m.band_of('no_band_here.fits') is None


def test_rename_stale_staleness_logic(tmp_path):
    """A pre-campaign realigned mosaic is renamed; a same-campaign one is kept."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = tmp_path / 'myfield' / 'F182M' / 'pipeline'
    pipe.mkdir(parents=True)
    ref = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged_data_i2d.fits'
    stale = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged-reproject-vvv_i2d.fits'
    fresh = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged_realigned-to-refcat.fits'
    now = time.time()
    for p, age_days in ((ref, 0), (stale, 400), (fresh, 0.5)):
        p.write_bytes(b'x')
        os.utime(p, (now - age_days * 86400,) * 2)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert len(plan) == 1
    assert not stale.exists()
    assert (str(stale) + m.SUFFIX) == str(stale) + '_badastrometry_stale'
    assert os.path.exists(str(stale) + m.SUFFIX)
    assert fresh.exists()


def test_purge_satstar_caches(tmp_path):
    m = _load('purge_satstar_caches')
    pipe = tmp_path / 'brick' / 'F182M' / 'pipeline'
    cats = tmp_path / 'brick' / 'catalogs'
    pipe.mkdir(parents=True)
    cats.mkdir(parents=True)
    a = pipe / 'exp1_m12_satstar_catalog.fits'
    b = cats / 'f182m_consolidated_satstar_catalog.fits'
    other = pipe / 'exp1_m12_daophot_basic.fits'
    for p in (a, b, other):
        p.write_bytes(b'x')
    # dry run: nothing moves
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=False)
    assert n == 2 and a.exists() and b.exists()
    # execute: both cache levels sidelined, unrelated file untouched
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=True)
    assert n == 2
    assert not a.exists() and not b.exists()
    assert os.path.exists(str(a) + m.SUFFIX) and os.path.exists(str(b) + m.SUFFIX)
    assert other.exists()
    # idempotent: second execute finds nothing
    assert m.purge(str(tmp_path), 'brick', ['F182M'], execute=True) == 0


# --- apply_m2_checkpoint_corrections: per-exposure extension ---------------

def _write_m2_record(records_dir, filt, visit_exposures):
    """visit_exposures: {visit_int: [exposure ints]} -> a minimal m2 record with
    the full exposure enumeration (one detector), no corrections needed."""
    visits = []
    for vnum, exps in visit_exposures.items():
        visits.append(dict(visit=str(vnum), filtername=filt, exposures=[
            dict(key=[str(vnum), e, 'nrca1', filt]) for e in exps]))
    rec = dict(stage='m2', filtername=filt, visits=visits, corrections=[])
    with open(os.path.join(records_dir, f'checkpoint_m2_{filt}_latest.json'), 'w') as fh:
        json.dump(rec, fh)


def test_exposure_universe_keyed_by_visit_and_filter(tmp_path):
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    # 1182-like: two visits sharing exposure numbers 1..12; plus a 2221-like
    # single-visit filter with 1..3
    _write_m2_record(str(rd), 'F200W', {1: list(range(1, 13)), 2: list(range(1, 13))})
    _write_m2_record(str(rd), 'F182M', {1: [1, 2, 3]})
    u = m.load_exposure_universe(str(rd))
    assert u[(1, 'F200W')] == list(range(1, 13))
    assert u[(2, 'F200W')] == list(range(1, 13))
    assert u[(1, 'F182M')] == [1, 2, 3]
    assert (2, 'F182M') not in u


def test_extend_covers_subfloor_exposure_no_phantom_rows(tmp_path):
    """The reviewer's gap: an exposure sub-floor in EVERY filter carries no
    correction but is a real frame -- it must still get a row (from the record
    universe), and a visit must never receive another visit's exposure numbers."""
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    # F200W tiles two visits, exposures 1..3 each; NONE carry a correction here
    _write_m2_record(str(rd), 'F200W', {1: [1, 2, 3], 2: [1, 2, 3]})
    universe = m.load_exposure_universe(str(rd))

    # pristine per-visit table (no Exposure column): one row per (visit, filter)
    tbl = Table(dict(
        Visit=['jw01182004001', 'jw01182004002'],
        Filter=['F200W', 'F200W'],
        **{'dra (arcsec)': [0.0, 0.0], 'ddec (arcsec)': [0.0, 0.0]}))
    tp = tmp_path / 'offsets.csv'
    tbl.write(str(tp), overwrite=True)

    out, extended = m.extend_table_to_per_exposure(
        str(tp), universe, extend_filters={'F200W'})
    assert extended
    assert 'Exposure' in out.colnames
    # visit 1 gets exposures 1,2,3; visit 2 gets exposures 1,2,3 -- 6 rows total
    v1 = sorted(int(r['Exposure']) for r in out if m._table_visit_number(r['Visit']) == 1)
    v2 = sorted(int(r['Exposure']) for r in out if m._table_visit_number(r['Visit']) == 2)
    assert v1 == [1, 2, 3]      # incl. exp 3, which carried NO correction
    assert v2 == [1, 2, 3]      # no phantom exposure numbers from the other visit
    assert len(out) == 6


def test_extend_leaves_unextended_filter_as_single_visit_row(tmp_path):
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    _write_m2_record(str(rd), 'F410M', {1: [1, 2, 3, 4]})
    universe = m.load_exposure_universe(str(rd))
    tbl = Table(dict(Visit=['jw02221001001'], Filter=['F410M'],
                     **{'dra (arcsec)': [0.0], 'ddec (arcsec)': [0.0]}))
    tp = tmp_path / 'o.csv'
    tbl.write(str(tp), overwrite=True)
    # filter NOT in extend_filters -> keeps one per-visit row, Exposure = -1
    out, extended = m.extend_table_to_per_exposure(
        str(tp), universe, extend_filters=set())
    assert extended is False


def test_extend_idempotent_when_exposure_column_present(tmp_path):
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    tbl = Table(dict(Visit=['jw02221001001'], Filter=['F410M'], Exposure=[1],
                     **{'dra (arcsec)': [0.0], 'ddec (arcsec)': [0.0]}))
    tp = tmp_path / 'o.csv'
    tbl.write(str(tp), overwrite=True)
    out, extended = m.extend_table_to_per_exposure(str(tp), {}, {'F410M'})
    assert extended is False
    assert len(out) == 1
