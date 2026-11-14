"""``current_records``: a superseded untokened checkpoint record is not current (#827).

``checkpoint_m2_F444W_latest.json`` (pre-migration) and
``checkpoint_m2_F444W_o001_latest.json`` both end in ``_latest.json``, so a glob
reads both.  Two investigations were misled by the stale one (a w51 F444W 29"
"misalignment"; wd1 4-tuple vs 5-tuple keys reading as "no m2 baseline").
"""
import json
import os
import sys

import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    current_records, parse_record_name, prefer_tokenised)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
for _sub in ('release', 'reduction'):
    _p = os.path.join(_REPO, 'scripts', _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _touch(d, *names, mtime=None):
    for n in names:
        p = d / n
        p.write_text('{}')
        if mtime is not None:
            os.utime(p, (mtime, mtime))


def _names(rec):
    return sorted(os.path.basename(p) for p in rec.values())


def test_untokened_dropped_when_a_tokened_sibling_exists(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F444W_latest.json',
           'checkpoint_m2_F444W_o001_latest.json')
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m2_F444W_o001_latest.json']


def test_untokened_kept_when_no_tokened_sibling(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F444W_latest.json',
           'checkpoint_m2_F200W_o001_latest.json')
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m2_F200W_o001_latest.json',
        'checkpoint_m2_F444W_latest.json']


def test_proposal_token_counts_as_tokened(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F090W_latest.json',
           'checkpoint_m2_F090W_j7213_latest.json')
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m2_F090W_j7213_latest.json']


def test_stage_and_filter_scope_the_supersession(tmp_path):
    # A tokened m2 record does not supersede an untokened m3 one.
    _touch(tmp_path, 'checkpoint_m3_F444W_latest.json',
           'checkpoint_m2_F444W_o001_latest.json')
    assert len(current_records(tmp_path)) == 2


def test_m7_crossfilter_is_keyed_per_module(tmp_path):
    _touch(tmp_path,
           'checkpoint_m7_crossfilter_nrca_o001_latest.json',
           'checkpoint_m7_crossfilter_nrcb_o001_latest.json',
           'checkpoint_m7_crossfilter_nrcb_latest.json',
           'checkpoint_m7_crossfilter_merged_latest.json')
    # The untokened nrcb record is superseded by its own module's tokened one;
    # the untokened `merged` record has no tokened `merged` sibling and stays.
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m7_crossfilter_merged_latest.json',
        'checkpoint_m7_crossfilter_nrca_o001_latest.json',
        'checkpoint_m7_crossfilter_nrcb_o001_latest.json']


def test_module_less_m7_record_with_only_module_siblings_stays(tmp_path):
    _touch(tmp_path, 'checkpoint_m7_crossfilter_latest.json',
           'checkpoint_m7_crossfilter_nrcb_o001_latest.json')
    assert len(current_records(tmp_path)) == 2


def test_module_less_m7_record_is_superseded_by_a_module_less_tokened_one(tmp_path):
    _touch(tmp_path, 'checkpoint_m7_crossfilter_latest.json',
           'checkpoint_m7_crossfilter_o001_latest.json',
           'checkpoint_m7_crossfilter_o004_latest.json')
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m7_crossfilter_o001_latest.json',
        'checkpoint_m7_crossfilter_o004_latest.json']


def test_joint_obs_token(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F150W_latest.json',
           'checkpoint_m2_F150W_o002-998_latest.json')
    rec = current_records(tmp_path)
    assert list(rec) == [('m2', 'F150W', None, '002-998', None)]


def test_key_shape(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F444W_o001_latest.json')
    assert list(current_records(tmp_path)) == [('m2', 'F444W', None, '001', None)]


def test_timestamped_files_excluded_by_default(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F444W_o001_latest.json',
           'checkpoint_m2_F444W_o001_20260816T143240Z.json')
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m2_F444W_o001_latest.json']


def test_history_newest_mtime_wins_inside_one_key(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F444W_o001_20260816T143240Z.json', mtime=1000)
    _touch(tmp_path, 'checkpoint_m2_F444W_o001_20260817T143240Z.json', mtime=3000)
    _touch(tmp_path, 'checkpoint_m2_F444W_o001_latest.json', mtime=2000)
    got = current_records(tmp_path, include_history=True)
    assert _names(got) == ['checkpoint_m2_F444W_o001_20260817T143240Z.json']


def test_unparseable_names_are_skipped(tmp_path):
    _touch(tmp_path, 'checkpoint_notastage_latest.json',
           'checkpoint_m2_F444W_o001_latest.json', 'README_latest.json')
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m2_F444W_o001_latest.json']


def test_tokened_wins_even_when_the_untokened_file_is_newer(tmp_path):
    _touch(tmp_path, 'checkpoint_m2_F444W_o001_latest.json', mtime=1000)
    _touch(tmp_path, 'checkpoint_m2_F444W_latest.json', mtime=9000)
    assert _names(current_records(tmp_path)) == [
        'checkpoint_m2_F444W_o001_latest.json']


def test_missing_directory_is_empty(tmp_path):
    assert current_records(tmp_path / 'absent') == {}


def test_prefer_tokenised_accepts_a_custom_accessor():
    items = [parse_record_name('checkpoint_m3_F187N_latest.json'),
             parse_record_name('checkpoint_m3_F187N_o007_latest.json')]
    kept = prefer_tokenised(items, info_of=lambda i: i)
    assert [i['obs'] for i in kept] == ['007']


# -- the readers route through the helper --------------------------------------

def _m2_record(date, off):
    return {'filtername': 'F444W', 'date': date, 'correcting': False,
            'corrections': [],
            'visits': [{'visit': '1', 'exposures': [],
                        'reference_tie': {'vs_sparse': {'off': off,
                                                         'contrast': 20.0}}}]}


def _write(d, name, rec):
    (d / name).write_text(json.dumps(rec))


def test_gate_records_drop_the_stale_untokened_record(tmp_path, monkeypatch):
    import check_astrometry_checkpoints as gate
    ck = tmp_path / 'w51' / 'astrometry_checkpoints'
    ck.mkdir(parents=True)
    _write(ck, 'checkpoint_m2_F444W_latest.json', {'date': '2026-08-01'})
    _write(ck, 'checkpoint_m2_F444W_o001_latest.json', {'date': '2026-09-01'})
    monkeypatch.setattr(gate, 'BASE', str(tmp_path))
    found, unreadable = gate.records('w51')
    assert [os.path.basename(p) for p, _i, _r in found] == [
        'checkpoint_m2_F444W_o001_latest.json']
    assert not unreadable


def test_provenance_ties_use_the_tokened_record_not_the_29_arcsec_stale_one(tmp_path):
    import astrometry_provenance as prov
    ck = tmp_path / 'astrometry_checkpoints'
    ck.mkdir()
    # The stale record is the NEWER one by date, so only the supersession rule
    # (not the date comparison) can pick the right one.
    _write(ck, 'checkpoint_m2_F444W_latest.json', _m2_record('2026-10-01', 29000.0))
    _write(ck, 'checkpoint_m2_F444W_o001_latest.json', _m2_record('2026-09-01', 1.5))
    ties = prov.reference_ties({'data_dir': str(tmp_path)})
    assert ties['F444W']['sparse_mas'] == 1.5


def test_stability_records_drop_the_stale_untokened_record(tmp_path):
    import astrometry_stability_report as stab
    ck = tmp_path / 'astrometry_checkpoints'
    ck.mkdir()
    _write(ck, 'checkpoint_m2_F444W_latest.json', {})
    _write(ck, 'checkpoint_m2_F444W_o001_latest.json', {})
    recs, _bad = stab._records(str(tmp_path))
    assert list(recs) == [('m2', 'F444W', 'o001')]


def test_stability_record_name_reads_module_and_joint_tokens():
    import astrometry_stability_report as stab
    assert stab._record_name('checkpoint_m7_crossfilter_nrcb_o004_latest.json') == (
        'm7_crossfilter', 'nrcb', 'o004')
    assert stab._record_name('checkpoint_m2_F150W_o002-998_latest.json') == (
        'm2', 'F150W', 'o002-998')
    assert stab._record_name('checkpoint_m2_F090W_j7213_latest.json') == (
        'm2', 'F090W', 'j7213')
    assert stab._record_name('checkpoint_m2_F444W_latest.json') == (
        'm2', 'F444W', '')


def test_monitor_scan_shows_only_the_tokened_record(tmp_path):
    from jwst_gc_pipeline.monitoring import scan
    ck = tmp_path / 'astrometry_checkpoints'
    ck.mkdir()
    _write(ck, 'checkpoint_m2_F444W_latest.json', _m2_record('2026-10-01', 29000.0))
    _write(ck, 'checkpoint_m2_F444W_o001_latest.json', _m2_record('2026-09-01', 1.5))
    got = scan.astrometry_checkpoints(str(tmp_path))
    assert sorted(got) == ['F444W_o001']


def test_apply_m2_blind_glob_ignores_a_superseded_untokened_record(tmp_path):
    import apply_m2_checkpoint_corrections as apply_m2
    corr = {'visit': '1', 'exposure': 1}
    _write(tmp_path, 'checkpoint_m2_F444W_latest.json', {'corrections': [corr]})
    _write(tmp_path, 'checkpoint_m2_F444W_o001_latest.json',
           {'corrections': [corr]})
    records, corrections = apply_m2.load_corrections(str(tmp_path))
    assert [os.path.basename(p) for p in records] == [
        'checkpoint_m2_F444W_o001_latest.json']
    assert len(corrections) == 1


def test_apply_m2_still_refuses_two_live_observations(tmp_path):
    import apply_m2_checkpoint_corrections as apply_m2
    _write(tmp_path, 'checkpoint_m2_F444W_o001_latest.json', {'corrections': []})
    _write(tmp_path, 'checkpoint_m2_F444W_o002_latest.json', {'corrections': []})
    with pytest.raises(SystemExit):
        apply_m2.load_corrections(str(tmp_path))
