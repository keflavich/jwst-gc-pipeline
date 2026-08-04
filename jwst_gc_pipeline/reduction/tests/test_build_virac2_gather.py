"""``build_virac2_offsets._gather`` / ``_solve`` identity rules.

The offsets-table row key is ``(Visit, Filter, Exposure[, Module][, Vgroup])``
and ``Visit`` is the full ``jw<prop><obs><visit>`` token.  A per-frame catalog
basename carries only ``visit001``, which is the FIRST visit of whatever
observation produced it -- so several observations reduced into one directory
are indistinguishable by name, and gc2211 additionally REUSES its visit-group
ids across observations.  The observation therefore has to come from the crf the
catalog was fit on, and a catalog from another observation must be refused rather
than relabelled (that is what a cloudef5 build was silently doing).
"""
import os

import numpy as np
import pytest

from jwst_gc_pipeline.reduction import build_virac2_offsets as bvo


def _fake_catalogs(tmp_path, spec, filt='f200w', sub='F200W'):
    """Create empty catalog files named as the pipeline names them.

    ``spec``: {catalog_basename: crf_basename}.  The content is irrelevant --
    ``load_siaf`` is monkeypatched -- only the names and the crf mapping matter.
    """
    d = tmp_path / sub
    d.mkdir(parents=True, exist_ok=True)
    for name in spec:
        (d / name).write_text('')
    return str(tmp_path)


def _patch_load_siaf(monkeypatch, spec):
    def fake(f):
        crf = spec[os.path.basename(f)]
        ra = np.array([266.0, 266.001]); dec = np.array([-28.9, -28.901])
        return ra, dec, 0.0, 0.0, crf
    monkeypatch.setattr(bvo, 'load_siaf', fake)


def test_gather_keys_on_the_crf_observation_not_the_basename(tmp_path, monkeypatch):
    """gc2211's shape: one directory, two observations, SAME visit + vgroup +
    exposure in the catalog name.  They must land on separate keys."""
    spec = {
        'f200w_nrca1_o023_visit001_vgroup02201_exp00001_m3_daophot_basic.fits':
            '/x/jw02211023001_02201_00001_nrca1_destreak_o023_crf.fits',
    }
    base = _fake_catalogs(tmp_path, spec)
    _patch_load_siaf(monkeypatch, spec)
    byve, byv, coarse = bvo._gather('f200w', base, 'F200W', '_m3', ['nrca1'],
                                    prop='2211', field='023', otag='_o023')
    assert list(byv) == ['jw02211023001']
    assert list(byve) == [('jw02211023001', '2201', 1)]


def test_gather_refuses_a_catalog_from_another_observation(tmp_path, monkeypatch):
    """THE cloudef5 case: every globbed file belongs to obs 002 while the region
    asks for obs 005.  Relabelling them would apply obs 002's tie to obs 005."""
    spec = {
        'f162m_nrca1_visit001_vgroup02101_exp00001_m3_daophot_basic.fits':
            '/x/jw02092002001_02101_00001_nrca1_destreak_o002_crf.fits',
    }
    base = _fake_catalogs(tmp_path, spec, sub='F162M')
    _patch_load_siaf(monkeypatch, spec)
    with pytest.raises(bvo.WrongObservationError, match='jw02092002001'):
        bvo._gather('f162m', base, 'F162M', '_m3', ['nrca1'],
                    prop='2092', field='005')


def test_gather_refuses_two_catalogs_claiming_one_frame(tmp_path, monkeypatch):
    """Both the zero-padded and the bare spelling of one visit group exist on disk
    (brick/F182M holds vgroup07101 AND vgroup7101, 192 files each).  Canonicalising
    the token -- which is what makes the producer and consumer agree -- makes those
    two files collide on ONE key, so they must be refused rather than pooled and
    double-weighted."""
    spec = {
        'f200w_nrca1_visit001_vgroup07101_exp00001_m3_daophot_basic.fits':
            '/x/jw02211046001_07101_00001_nrca1_destreak_o046_crf.fits',
        'f200w_nrca1_visit001_vgroup7101_exp00001_m3_daophot_basic.fits':
            '/x/jw02211046001_07101_00001_nrca1_destreak_o046_crf.fits',
    }
    base = _fake_catalogs(tmp_path, spec)
    _patch_load_siaf(monkeypatch, spec)
    with pytest.raises(bvo.WrongObservationError, match='(?i)same frame'):
        bvo._gather('f200w', base, 'F200W', '_m3', ['nrca1'],
                    prop='2211', field='046')


def test_gather_refuses_when_nothing_matched(tmp_path):
    """An empty glob previously produced an empty table and a silent no-op.

    The type is NoPerFrameCatalogsError, not WrongObservationError: "this module
    has no catalogs" and "these catalogs belong to another observation" need
    telling apart WITHOUT sniffing the message, because lock_filter may skip the
    first (an unobserved module) and must never skip the second.
    """
    (tmp_path / 'F200W').mkdir()
    with pytest.raises(bvo.NoPerFrameCatalogsError, match='(?i)no per-frame catalogs'):
        bvo._gather('f200w', str(tmp_path), 'F200W', '_m3', ['nrca1'],
                    prop='2211', field='050')


def test_solve_writes_the_crf_visit_token(monkeypatch):
    """_solve must emit the key _gather built, not a token synthesised from the
    region's own field."""
    monkeypatch.setattr(bvo, 'build_consensus',
                        lambda pairs: bvo.SkyCoord([266.0] * 3 * bvo.u.deg,
                                                   [-28.9] * 3 * bvo.u.deg))
    monkeypatch.setattr(bvo, 'coarse_xcorr', lambda *a, **k: (None,) * 6)
    monkeypatch.setattr(bvo, 'coord_shift',
                        lambda ra, dec, ref: (0.001, -0.002, 0.5, 0.5, 42))
    byve = {('jw02211046001', '2201', 1): [[np.array([266.0])], [np.array([-28.9])]]}
    byv = {'jw02211046001': [(np.array([266.0]), np.array([-28.9]))]}
    coarse = {'jw02211046001': [[0.0], [0.0]]}
    rows = bvo._solve(byve, byv, coarse, 0.0, 0.0, None, 'f200w')
    assert len(rows) == 1
    assert rows[0]['Visit'] == 'jw02211046001'
    assert rows[0]['Vgroup'] == '2201' and rows[0]['Exposure'] == 1


def test_solve_refuses_a_visit_whose_every_exposure_failed(monkeypatch):
    """A visit with no rows leaves every one of its frames without an offset."""
    monkeypatch.setattr(bvo, 'build_consensus',
                        lambda pairs: bvo.SkyCoord([266.0] * 3 * bvo.u.deg,
                                                   [-28.9] * 3 * bvo.u.deg))
    monkeypatch.setattr(bvo, 'coarse_xcorr', lambda *a, **k: (None,) * 6)
    monkeypatch.setattr(bvo, 'coord_shift', lambda ra, dec, ref: None)
    byve = {('jw02211046001', '2201', 1): [[np.array([266.0])], [np.array([-28.9])]]}
    byv = {'jw02211046001': [(np.array([266.0]), np.array([-28.9]))]}
    coarse = {'jw02211046001': [[0.0], [0.0]]}
    with pytest.raises(SystemExit, match='(?i)all 1 exposures failed'):
        bvo._solve(byve, byv, coarse, 0.0, 0.0, None, 'f200w')


def test_regions_sharing_a_directory_are_separable():
    """Two regions that glob the SAME (basepath, subdir, filter) must differ in
    something the glob or the crf check can act on -- an ``otag``, or an ``mtag``
    naming the stage that region's OWN observation reached.

    cloudef2 (2092/002) and cloudef5 (2092/005) catalog into one directory with no
    observation token in the filenames, so the stage is the only separator: obs 005
    reached m2, obs 002 reached m7, and cloudef5 globbing ``_m3`` matched 100%
    obs-002 files.  The five gc2211 regions share a directory too and separate by
    ``otag``.
    """
    from collections import defaultdict
    by_dir = defaultdict(list)
    for key, rc in bvo.REGION.items():
        for filt, (sub, _ep, mtag) in rc['filts'].items():
            by_dir[(rc['basepath'], sub, filt)].append(
                (key, mtag, bool(rc.get('otag'))))
    for where, entries in sorted(by_dir.items()):
        if len(entries) < 2:
            continue
        assert all(e[2] for e in entries) or len({e[1] for e in entries}) == len(entries), (
            f"regions {[e[0] for e in entries]} all glob {where} and cannot be "
            f"told apart: none carries an otag and they share an mtag")


def test_cloudef_ties_each_observation_at_its_own_deepest_stage():
    """Regression on the specific mis-scope: cloudef5 at ``_m3`` could only ever
    match obs-002 catalogs (obs 005 has no m3+ products)."""
    assert {v[2] for v in bvo.REGION['cloudef5']['filts'].values()} == {'_m2'}
    assert {v[2] for v in bvo.REGION['cloudef2']['filts'].values()} == {'_m3'}


def test_gc2211_regions_are_registered_per_observation():
    keys = [k for k in bvo.REGION if k.startswith('gc2211')]
    assert keys == ['gc2211_023', 'gc2211_028', 'gc2211_046',
                    'gc2211_049', 'gc2211_050']
    for k in keys:
        rc = bvo.REGION[k]
        assert rc['proposal'] == '2211' and rc['otag'] is True
        assert set(rc['filts']) == {'f150w', 'f200w', 'f277w'}
        # each observation gets its OWN Visit prefix, which is the whole point
        assert rc['field'] == k.split('_')[1]
