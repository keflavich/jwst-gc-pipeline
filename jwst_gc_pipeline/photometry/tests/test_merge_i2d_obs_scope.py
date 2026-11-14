"""The merge's i2d glob must follow the observation being merged (#817).

``Obs.glob_obsid`` answers ``'*'`` for a proposal that lists several
observations.  For the per-observation merge proposals (gc1266, gc3571,
gc-treasury, ...) the merge knows its observation and used to ignore it, so the
glob pooled every pointing's mosaic and ``merge_daophot`` took the first match
as the WCS and pixel-area source for all of them.
"""
import glob
import os
import types

import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline import fields as _fields
from jwst_gc_pipeline.mast_names import jw_prefix
from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import naming


class _Stop(Exception):
    """Raised by the stubbed ``frame_wcs`` once it has recorded its argument."""


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')


def _write_m2(basepath, filt, module, obs):
    t = Table({'skycoord': SkyCoord([266.0] * u.deg, [-28.9] * u.deg),
               'flux_fit': [1000.0], 'flux_err': [1.0]})
    t.meta['PIXSCALE'] = 0.11
    (basepath / 'catalogs').mkdir(exist_ok=True)
    t.write(basepath / 'catalogs' /
            f'{filt}_{module}_o{obs}_indivexp_merged_dao_basic.fits')


def _run_daophot(tmp_path, monkeypatch, field, obs_files, target='gc3571',
                 progid='3571', filt='f770w', module='mirimage'):
    """Run ``merge_daophot`` up to its ``frame_wcs`` call; return what it was given."""
    seen = []

    def fake_frame_wcs(fn):
        seen.append(fn)
        raise _Stop

    monkeypatch.setattr(MC, 'frame_wcs', fake_frame_wcs)
    for obs in obs_files:
        _touch(tmp_path / filt.upper() / 'pipeline' /
               f'{jw_prefix(progid)}-o{obs}_t001_miri_{filt}_{module}_i2d.fits')
    _write_m2(tmp_path, filt, module, field if field else obs_files[0])
    with pytest.raises(_Stop):
        MC.merge_daophot(module=module, daophot_type='basic', indivexp=True,
                         target=target, basepath=str(tmp_path), progid=progid,
                         filternames_override=[filt], field=field)
    return seen


# ---------------------------------------------------------------------------
# _obsid_for_glob
# ---------------------------------------------------------------------------

def test_registry_answer_for_gc1266_and_gc3571_is_the_wildcard():
    """The premise: both fields list several MIRI observations and declare no
    ``glob_obsid``, so the registry alone answers ``'*'``."""
    assert _fields.glob_obsid('gc1266', '1266', 'miri') == '*'
    assert _fields.glob_obsid('gc3571', '3571', 'miri') == '*'


@pytest.mark.parametrize('field,expected', [('9', '009'), ('009', '009'),
                                            ('08', '008'), ('10', '010')])
def test_per_obs_proposal_globs_its_own_observation(field, expected):
    assert MC._obsid_for_glob('gc1266', '1266', 'f770w', field=field) == expected


def test_field_the_registry_does_not_list_raises():
    """A typo must not build a glob that matches nothing without saying so."""
    with pytest.raises(naming.ObservationFieldError, match='not one of'):
        MC._obsid_for_glob('gc1266', '1266', 'f770w', field='011')
    # 004 exists for gc1266 MIRI but 011 does not; gc3571 has no 008 either
    with pytest.raises(naming.ObservationFieldError, match='not one of'):
        MC._obsid_for_glob('gc3571', '3571', 'f770w', field='008')


def test_per_obs_proposal_without_a_field_raises_rather_than_pooling():
    with pytest.raises(naming.ObservationFieldError, match='--field'):
        MC._obsid_for_glob('gc3571', '3571', 'f770w')
    with pytest.raises(naming.ObservationFieldError, match='--field'):
        MC._obsid_for_glob('gc3571', '3571', 'f770w', field='')


@pytest.mark.parametrize('bad', ['*', 'abc', '002-998'])
def test_a_field_that_names_no_single_observation_raises(bad):
    with pytest.raises(naming.ObservationFieldError):
        MC._obsid_for_glob('gc3571', '3571', 'f770w', field=bad)


def test_wildcard_registration_accepts_a_field_on_shape():
    """gc-treasury claims every observation of 10678 (``obsids: '*'``), so
    there is no list to test membership against; testing it anyway would
    reject every real obsid (``'042' not in {'*'}``)."""
    assert MC._obsid_for_glob('gc-treasury', '10678', 'f212n', field='42') == '042'
    assert MC._obsid_for_glob('gc-treasury', '10678', 'f212n', field='139') == '139'
    with pytest.raises(naming.ObservationFieldError):
        MC._obsid_for_glob('gc-treasury', '10678', 'f212n', field='1234')


def test_single_observation_registrations_ignore_field():
    """The registry answer is already one observation; ``field`` changes nothing."""
    before = _fields.glob_obsid('cloudc', '2221', 'nircam')
    assert MC._obsid_for_glob('cloudc', '2221', 'f405n', field=None) == \
        MC._obsid_for_glob('cloudc', '2221', 'f405n', field='999')
    assert before == MC._obsid_for_glob('cloudc', '2221', 'f405n')


def test_non_per_obs_proposal_keeps_the_registry_wildcard():
    """cloudef MIRI lists three observations under ``'*'`` and is not a
    per-observation merge, so it keeps its pooled glob (the guard warns)."""
    assert MC._obsid_for_glob('cloudef', '2092', 'f770w') == \
        _fields.glob_obsid('cloudef', '2092', 'miri')


def test_unobserved_instrument_is_still_none():
    assert MC._obsid_for_glob('gc3571', '3571', 'f212n', field='003') is None


# ---------------------------------------------------------------------------
# merge_daophot reads the observation's own mosaic
# ---------------------------------------------------------------------------

def test_merge_daophot_takes_the_wcs_from_its_own_observation(tmp_path, monkeypatch):
    seen = _run_daophot(tmp_path, monkeypatch, '003', ['001', '003'])
    assert [os.path.basename(f) for f in seen] == \
        ['jw03571-o003_t001_miri_f770w_mirimage_i2d.fits']


def test_unscoped_glob_matches_every_tile_and_leads_with_o001(tmp_path):
    """The failure this fixes: the unscoped glob matches both tiles and the
    first (o001) is the WCS source for the o003 merge."""
    for obs in ('001', '003'):
        _touch(tmp_path / 'F770W' / 'pipeline' /
               f'jw03571-o{obs}_t001_miri_f770w_mirimage_i2d.fits')
    pooled = sorted(os.path.basename(x) for x in glob.glob(
        f"{tmp_path}/F770W/pipeline/jw03571-o*_t001_miri*f770w*mirimage_i2d.fits"))
    assert pooled[0].startswith('jw03571-o001')
    assert len(pooled) == 2


def _fake_svo(monkeypatch):
    jfilts = Table({'filterID': ['JWST/MIRI.F770W'], 'ZeroPoint': [1.0]})
    monkeypatch.setattr(
        MC, 'SvoFps', types.SimpleNamespace(get_filter_list=lambda fac: jfilts))


def test_merge_daophot_with_no_matching_mosaic_keeps_the_fallback(
        tmp_path, monkeypatch, capsys):
    """The observation's mosaic not being on disk is the existing no-match
    path: a warning, no ``frame_wcs`` call, and the catalogue's own
    ``skycoord`` and ``PIXSCALE``.  o001's mosaic must not stand in for o003's."""
    monkeypatch.setattr(MC, 'frame_wcs',
                        lambda fn: pytest.fail(f'frame_wcs called on {fn}'))
    _fake_svo(monkeypatch)
    monkeypatch.setattr(MC, 'sanity_check_individual_table', lambda t: None)
    calls = []
    monkeypatch.setattr(MC, 'merge_catalogs', lambda tbls, **kw: calls.append(tbls))
    _write_m2(tmp_path, 'f770w', 'mirimage', '003')
    (tmp_path / 'reduction').mkdir()
    Table({'Filter': ['F770W'], 'PSF FWHM (arcsec)': [0.27],
           'PSF FWHM (pixel)': [2.5]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv')
    _touch(tmp_path / 'F770W' / 'pipeline' /
           'jw03571-o001_t001_miri_f770w_mirimage_i2d.fits')
    MC.merge_daophot(module='mirimage', daophot_type='basic', indivexp=True,
                     target='gc3571', basepath=str(tmp_path), progid='3571',
                     filternames_override=['f770w'], field='003')
    assert len(calls) == 1
    assert 'no mirimage i2d for filter f770w' in capsys.readouterr().out


def test_merge_daophot_without_a_field_raises_for_a_per_obs_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, 'frame_wcs', lambda fn: pytest.fail('unreachable'))
    _write_m2(tmp_path, 'f770w', 'mirimage', '003')
    with pytest.raises(naming.ObservationFieldError):
        MC.merge_daophot(module='mirimage', daophot_type='basic', indivexp=True,
                         target='gc3571', basepath=str(tmp_path), progid='3571',
                         filternames_override=['f770w'])


# ---------------------------------------------------------------------------
# the residual guard
# ---------------------------------------------------------------------------

def test_guard_raises_for_pooled_observations_of_a_per_obs_proposal():
    matches = ['x/jw03571-o001_t001_miri_f770w_mirimage_i2d.fits',
               'x/jw03571-o003_t001_miri_f770w_mirimage_i2d.fits']
    with pytest.raises(ValueError, match=r"observations \['001', '003'\]"):
        MC._check_single_observation(matches, 'gc3571', '3571', 'f770w')


def test_guard_warns_for_other_proposals(capsys):
    matches = ['x/jw02092-o004_t001_miri_f770w_mirimage_i2d.fits',
               'x/jw02092-o006_t001_miri_f770w_mirimage_i2d.fits']
    MC._check_single_observation(matches, 'cloudef', '2092', 'f770w')
    assert 'WARNING' in capsys.readouterr().out


def test_guard_passes_a_registered_joint_set(capsys):
    """sgrb2 MIRI 002 and 998 tile one field between them."""
    matches = ['x/jw05365-o002_t001_miri_f770w_mirimage_i2d.fits',
               'x/jw05365-o998_t001_miri_f770w_mirimage_i2d.fits']
    MC._check_single_observation(matches, 'sgrb2', '5365', 'f770w')
    assert 'WARNING' not in capsys.readouterr().out


def test_guard_is_quiet_for_one_observation_or_none(capsys):
    MC._check_single_observation([], 'gc3571', '3571', 'f770w')
    MC._check_single_observation(
        ['x/jw03571-o003_t001_miri_f770w_mirimage_i2d.fits'], 'gc3571', '3571', 'f770w')
    assert capsys.readouterr().out == ''


# ---------------------------------------------------------------------------
# merge_crowdsource and the CLI
# ---------------------------------------------------------------------------

def test_merge_crowdsource_accepts_field():
    import inspect
    assert 'field' in inspect.signature(MC.merge_crowdsource).parameters
    assert inspect.signature(MC.merge_crowdsource).parameters['field'].default is None


def test_merge_crowdsource_scopes_its_i2d_list_to_the_field(tmp_path, monkeypatch):
    """One mosaic per filter for the named observation, so the list lines up
    with the per-filter catalogues instead of holding every tile."""
    for obs in ('001', '003'):
        _touch(tmp_path / 'F770W' / 'pipeline' /
               f'jw03571-o{obs}_t001_miri_f770w_mirimage_i2d.fits')
    seen = []

    def fake_frame_wcs(fn):
        seen.append(fn)
        raise _Stop

    monkeypatch.setattr(MC, 'frame_wcs', fake_frame_wcs)
    monkeypatch.setattr(MC, '_obs_filters_for', lambda target: {'3571': ['f770w']})
    t = Table({'x': [1.0], 'y': [1.0], 'qf': [1.0], 'flux': [1.0], 'dflux': [1.0],
               'fwhm': [2.0]})
    (tmp_path / 'catalogs').mkdir()
    t.write(tmp_path / 'catalogs' / 'f770w_mirimage_o003_indivexp_merged_crowdsource.fits')
    _fake_svo(monkeypatch)
    with pytest.raises(_Stop):
        MC.merge_crowdsource(module='mirimage', target='gc3571', indivexp=True,
                             basepath=str(tmp_path), field='003')
    assert [os.path.basename(f) for f in seen] == \
        ['jw03571-o003_t001_miri_f770w_mirimage_i2d.fits']


def test_merge_crowdsource_without_a_field_raises_for_a_per_obs_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, '_obs_filters_for', lambda target: {'3571': ['f770w']})
    with pytest.raises(naming.ObservationFieldError):
        MC.merge_crowdsource(module='mirimage', target='gc3571', indivexp=True,
                             basepath=str(tmp_path))


# ---------------------------------------------------------------------------
# the MIRI mosaic spelling: recorded so a later change is deliberate
# ---------------------------------------------------------------------------

def test_miri_mosaic_names_on_disk_do_not_match_the_i2d_glob():
    """Recorded, not endorsed.  The MIRI mosaics are named
    ``jw03571-o001_t001_miri_clear-f770w-mirimage_data_i2d.fits`` and
    ``..._miri_f770w_i2d.fits``, and the glob ends in ``{module}_i2d.fits``,
    which neither spelling satisfies for module ``mirimage``.  Every MIRI merge
    therefore takes the catalogue-``PIXSCALE`` fallback.  Widening the glob would
    switch it to ``proj_plane_pixel_area()`` of the i2d and change ``flux_jy``,
    so it needs its own comparison; this test fails when that happens."""
    import fnmatch
    pattern = 'jw03571-o001_t001_miri*f770w*mirimage_i2d.fits'
    for name in ('jw03571-o001_t001_miri_clear-f770w-mirimage_data_i2d.fits',
                 'jw03571-o001_t001_miri_f770w_i2d.fits'):
        assert not fnmatch.fnmatch(name, pattern)


# ---------------------------------------------------------------------------
# main() hands --field to the all-filter merges
# ---------------------------------------------------------------------------

def test_main_passes_field_to_the_all_filter_merges(tmp_path, monkeypatch):
    """The all-filter step used to pass no ``field``, so the i2d glob (and the
    merged catalogue name) could not follow the observation."""
    import sys
    calls = {}
    monkeypatch.setenv('GC_BASEPATH_OVERRIDE', str(tmp_path))
    monkeypatch.delenv('SLURM_ARRAY_TASK_ID', raising=False)
    monkeypatch.delenv('RUN_REGISTRATION_GATE', raising=False)
    monkeypatch.setattr(sys, 'argv', ['merge_catalogs', '--target', 'gc3571',
                                      '--modules', 'mirimage', '--field', '3'])
    monkeypatch.setattr(MC, 'merge_crowdsource',
                        lambda **kw: calls.setdefault('crowdsource', kw))
    monkeypatch.setattr(MC, 'merge_daophot',
                        lambda **kw: calls.setdefault('daophot', kw))
    MC.main()
    assert calls['crowdsource']['field'] == '3'
    assert calls['daophot']['field'] == '3'
