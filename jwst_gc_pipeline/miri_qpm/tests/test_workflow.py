"""Local discovery, safe staging, resume validity, and command orchestration."""
from pathlib import Path
import shutil

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.miri_qpm import workflow as w

STEM = 'jw10678001001_02201_00001_mirimage'


def exposure(path, filt='F770W', **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header(dict(FILTER=filt, SUBARRAY='FULL', INSTRUME='MIRI', EXP_TYPE='MIR_IMAGE', **extra))
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), hdr).writeto(path)
    return str(path)


def config(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    exposure(a/f'{STEM}_cal.fits')
    exposure(b/f'{STEM}_rate.fits')
    return dict(output_dir=str(tmp_path/'out'), fields=[dict(name='test', proposal='10678',
                observations=['001'], roots=[str(a), str(b)])])


def test_discovery_pairs_across_nested_roots_and_excludes_other_bands(tmp_path):
    cfg = config(tmp_path)
    exposure(tmp_path/'a/nested'/f'{STEM.replace("00001_mir", "00002_mir")}_cal.fits', filt='F1000W')
    found = w.plan(cfg)['fields'][0]
    assert len(found['exposures']) == 1
    assert found['exposures'][0]['rate'].startswith(str(tmp_path/'b'))
    assert len(found['ignored']) == 1


def test_conflicting_duplicates_refused_identical_copies_allowed(tmp_path):
    cfg = config(tmp_path)
    source = tmp_path/'a'/f'{STEM}_cal.fits'
    duplicate = tmp_path/'b'/source.name
    shutil.copy2(source, duplicate)
    assert len(w.plan(cfg)['fields'][0]['exposures']) == 1
    fits.setval(duplicate, 'TEST', value=1)
    with pytest.raises(ValueError, match='Conflicting cal'):
        w.plan(cfg)


def test_discovery_requires_rate_or_ramp_and_complete_observation_selection(tmp_path):
    cfg = config(tmp_path)
    (tmp_path/'b'/f'{STEM}_rate.fits').unlink()
    with pytest.raises(ValueError, match='rate or uncal'):
        w.plan(cfg)
    exposure(tmp_path/'b'/f'{STEM}_uncal.fits')
    assert w.plan(cfg)['fields'][0]['exposures'][0]['uncal']
    cfg['fields'][0]['observations'].append('002')
    with pytest.raises(ValueError, match='observations'):
        w.plan(cfg)


def test_background_subtracted_cal_is_regenerated_from_rate(tmp_path):
    cfg = config(tmp_path)
    fits.setval(tmp_path/'a'/f'{STEM}_cal.fits', 'S_BKDSUB', value='COMPLETE')
    field = w.plan(cfg)['fields'][0]
    assert field['exposures'][0]['cal'] is None
    assert 'background subtraction' in field['ignored'][0]['reason']


def test_gather_is_repeatable_but_new_exposures_require_new_run(tmp_path):
    cfg = config(tmp_path)
    first = w.gather(cfg)
    assert w.gather(cfg) == first
    exposure(tmp_path/'a'/f'{STEM.replace("00001_mir", "00002_mir")}_rate.fits')
    with pytest.raises(ValueError, match='new output_dir'):
        w.gather(cfg)


def test_prepare_creates_missing_stages_and_separate_image3_inputs(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    (tmp_path/'a'/f'{STEM}_cal.fits').unlink()
    (tmp_path/'b'/f'{STEM}_rate.fits').unlink()
    exposure(tmp_path/'a'/f'{STEM}_uncal.fits')
    calls = []
    def stage(kind, source, directory, settings):
        calls.append(kind)
        return Path(exposure(directory/f'{STEM}_{kind}.fits'))
    def reference(field, cals, directory, settings):
        assert Path(cals[0]).parent.name == 'imager_inputs'
        fits.setval(cals[0], 'S_BKDSUB', value='COMPLETE')
        return Path(exposure(directory/'test_i2d.fits'))
    monkeypatch.setattr(w, '_stage', stage)
    monkeypatch.setattr(w, '_reference', reference)
    prepared = w.prepare(w.gather(cfg))
    assert calls == ['rate', 'cal']
    assert fits.getheader(prepared['fields'][0]['exposures'][0]['cal']).get('S_BKDSUB') is None
    assert w.prepare(w.gather(cfg)) == prepared
    assert calls == ['rate', 'cal']


def test_config_resolves_relative_paths_and_rejects_unsafe_names(tmp_path):
    path = tmp_path/'run.toml'
    path.write_text('output_dir="out"\nroots=["inputs"]\n[[fields]]\nname="field"\nproposal=10678\nobservations=[1,2]\n')
    cfg = w.load_config(path)
    assert cfg['fields'][0]['observations'] == ['001','002']
    assert cfg['fields'][0]['roots'] == [str(tmp_path/'inputs')]
    path.write_text(path.read_text().replace('name="field"', 'name="../outside"'))
    with pytest.raises(ValueError, match='field name'):
        w.load_config(path)


def test_image_failure_never_writes_completion(tmp_path, monkeypatch):
    from jwst_gc_pipeline.miri_qpm import pipeline
    cfg = config(tmp_path)
    field = {**cfg['fields'][0], **w.plan(cfg)['fields'][0], 'reference':'reference.fits'}
    prepared = {'config':cfg, 'fields':[field]}
    monkeypatch.setattr(pipeline, 'run', lambda *a: dict(skipped={STEM:'failed'}, n_exposures=0))
    with pytest.raises(RuntimeError, match='incomplete pane coverage'):
        w.image_fields(prepared)
    assert not (tmp_path/'out/test/qpm/complete.json').exists()
