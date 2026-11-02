"""The m7 cross-band seed refuses m6 catalogs that mix DVA states (#1128).

wd2 F150W was rebuilt after the inter-detector DVA correction went on by
default (``DVACORR = True``); the other 15 bands predate it.  The correction
moves each detector by 10-25 mas about V1, and wd2 has no reference tie to
absorb the common part, so F150W sits ~23 mas from the other bands.  Seeds
from the other bands landed ~0.75 px off-star in F150W and the overshoot
refit, which keeps the seed position, read 0.5 mag faint.  The catalogs carry
the frame's ``DVACORR`` as the ``WCSGDVA`` stamp; the seed builder now
compares it across its inputs.
"""
import os
import types

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging as C


def _opts():
    return types.SimpleNamespace(
        desaturated=False, bgsub=False, blur=False,
        proposal_id='4147', field='012', modules='merged')


def _write_m6(cut_bp, filt, dva=None):
    os.makedirs(f'{cut_bp}/catalogs', exist_ok=True)
    t = Table()
    t['skycoord'] = SkyCoord([10.0] * u.deg, [20.0] * u.deg)
    t['flux'] = [1000.0]
    t['flux_err'] = [50.0]
    t['qfit'] = [0.05]
    if dva is not None:
        t.meta['WCSGDVA'] = dva
    p = f'{cut_bp}/catalogs/{filt}_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'
    t.write(p, overwrite=True)
    return p


@pytest.mark.parametrize('val, want', [
    (None, False), (True, True), (False, False), (np.bool_(True), True),
    ('T', True), ('true', True), ('F', False), ('', False)])
def test_catalog_dva_state(val, want):
    meta = {} if val is None else {'WCSGDVA': val}
    assert C._catalog_dva_state(meta) is want


def test_stamp_survives_a_fits_round_trip(tmp_path):
    p = _write_m6(str(tmp_path), 'f150w', dva=True)
    assert C._catalog_dva_state(Table.read(p).meta) is True


@pytest.mark.parametrize('states', [
    {'f150w/merged': False, 'f200w/merged': False},
    {'f150w/merged': True, 'f200w/merged': True},
    {}])
def test_consistent_states_pass(monkeypatch, states):
    monkeypatch.delenv('CROSSBAND_ALLOW_DVA_MISMATCH', raising=False)
    C._check_crossband_dva_consistency(states)


def test_mixed_states_raise_and_name_both_groups(monkeypatch):
    monkeypatch.delenv('CROSSBAND_ALLOW_DVA_MISMATCH', raising=False)
    with pytest.raises(RuntimeError, match='mix DVA states') as ei:
        C._check_crossband_dva_consistency(
            {'f150w/merged': True, 'f200w/merged': False, 'f115w/merged': False})
    msg = str(ei.value)
    assert "Corrected: ['f150w/merged']" in msg
    assert "unstamped: ['f115w/merged', 'f200w/merged']" in msg
    assert 'CROSSBAND_ALLOW_DVA_MISMATCH' in msg


def test_override_warns_instead(monkeypatch, capsys):
    monkeypatch.setenv('CROSSBAND_ALLOW_DVA_MISMATCH', '1')
    C._check_crossband_dva_consistency({'f150w/merged': True, 'f200w/merged': False})
    assert 'WARNING (override)' in capsys.readouterr().out


def test_malformed_override_raises(monkeypatch):
    monkeypatch.setenv('CROSSBAND_ALLOW_DVA_MISMATCH', 'maybe')
    with pytest.raises(ValueError, match='CROSSBAND_ALLOW_DVA_MISMATCH'):
        C._check_crossband_dva_consistency({'f150w/merged': True, 'f200w/merged': False})


# ---------------------------------------------------------------------------
# _build_crossband_seed
# ---------------------------------------------------------------------------
def test_seed_refuses_wd2_like_mix(tmp_path, monkeypatch):
    monkeypatch.delenv('CROSSBAND_ALLOW_DVA_MISMATCH', raising=False)
    cut_bp = str(tmp_path)
    _write_m6(cut_bp, 'f150w', dva=True)
    _write_m6(cut_bp, 'f200w')
    _write_m6(cut_bp, 'f115w')
    with pytest.raises(RuntimeError, match='f150w/merged'):
        C._build_crossband_seed(cut_bp, ['merged'], ['f115w', 'f150w', 'f200w'], _opts())
    assert not os.path.exists(C.crossband_seed_file(cut_bp, _opts()))


@pytest.mark.parametrize('dva', [None, True])
def test_seed_builds_when_states_agree(tmp_path, monkeypatch, dva):
    monkeypatch.delenv('CROSSBAND_ALLOW_DVA_MISMATCH', raising=False)
    cut_bp = str(tmp_path)
    for f in ('f115w', 'f150w', 'f200w'):
        _write_m6(cut_bp, f, dva=dva)
    out = C._build_crossband_seed(cut_bp, ['merged'], ['f115w', 'f150w', 'f200w'], _opts())
    assert len(Table.read(out)) == 1


def test_seed_builds_under_override(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('CROSSBAND_ALLOW_DVA_MISMATCH', 'on')
    cut_bp = str(tmp_path)
    _write_m6(cut_bp, 'f150w', dva=True)
    _write_m6(cut_bp, 'f200w')
    out = C._build_crossband_seed(cut_bp, ['merged'], ['f150w', 'f200w'], _opts())
    assert len(Table.read(out)) == 1
    assert 'WARNING (override)' in capsys.readouterr().out
