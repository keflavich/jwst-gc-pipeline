"""The m8 forced fill adopts the final phase's satstar products (#1284, #1379).

With ``--forced-fill-nbrsub`` the fill subtracts the m7 star-only model, which
was fitted against the m7 satstar model.  Before this change the m8 frame
preparation refitted the saturated stars whenever the m7 catalog predated the
#1331 ePSF-core stamp, so the fill subtracted a satstar model m7 never saw and
their difference stayed in every fit box near a saturated star.  On
gc-treasury 10678 o118 that put 70 firm forced F480M rows above 1.5x their
pre-nbrsub flux; with the m7 products adopted, 18.

The loader's own behaviour is pinned in
``reduction/tests/test_satstar_hybrid_psf.py``; these tests pin the wiring from
the forced fill's frame arguments to the loader.
"""
import inspect

import numpy as np
import pytest

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import forced_fill as ff

FRAME = 'jw10678118001_02101_00001_nrcalong_destreak_o118_crf.fits'


def _args(**kw):
    return ff._frame_args_from_filename(
        f'/x/gc-treasury/F480M/pipeline/{FRAME}', options=None, filt='F480M',
        field='118', basepath='/x/gc-treasury', proposal_id='10678',
        bg_boxsizes=None, pupil='clear', resbg_path=None,
        satstar_overrides=None, satstar_drops=None, module='merged', **kw)


@pytest.mark.parametrize('label', ['m7', 'm5'])
def test_forced_fill_frames_adopt_the_satstar_products(label):
    args = _args(satstar_label=label)
    assert args['adopt_satstar_products'] is True
    assert args['satstar_label'] == label


def test_every_frame_argument_is_a_frame_preparation_parameter():
    params = inspect.signature(C._prepare_frame_for_photometry).parameters
    assert set(_args()) <= set(params)
    assert params['adopt_satstar_products'].default is False


@pytest.mark.parametrize('adopt', [True, False, None])
def test_frame_preparation_hands_adoption_to_the_satstar_loader(
        tmp_path, monkeypatch, adopt):
    """``None``: the flag is not passed, as in every fitting phase."""
    from astropy.io import fits
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
    from jwst_gc_pipeline.photometry.tests.test_gate_reject_handoff import (
        SHAPE, _gaussian_grid_psf_fwhm, _prep_options, _write_crf)
    for k, v in (('NIRCAM_SATSTAR_TIGHT_BOUND', '0'),
                 ('SATSTAR_COMPONENT_OVERLAP_FRAC', '0'),
                 ('NIRCAM_SATSTAR_LOCK_POS', '0'),
                 ('NIRCAM_SATSTAR_RECOVERED_CAP', '0')):
        monkeypatch.setenv(k, v)
    d = tmp_path / 'F480M' / 'pipeline'
    d.mkdir(parents=True)
    fn = str(d / 'jw10678132001_02101_00004_nrcalong_destreak_o132_crf.fits')
    _write_crf(fn)
    fits.PrimaryHDU(np.zeros(SHAPE)).writeto(
        fn.replace('.fits', '_m7_satstar_model.fits'))
    monkeypatch.setattr(L, 'get_psf_model',
                        lambda *a, **k: (_gaussian_grid_psf_fwhm(2.574), None))
    seen = []

    def _loader(*a, **k):
        seen.append(k)

    monkeypatch.setattr(L, 'load_or_make_satstar_catalog', _loader)
    extra = {} if adopt is None else {'adopt_satstar_products': adopt}
    C._prepare_frame_for_photometry(
        _prep_options(), 'F480M', 'nrcalong', '132', str(tmp_path), fn,
        '10678', exposurenumber=4, visit_id=1, vgroup_id='02101',
        bg_boxsizes=None, use_webbpsf=True, pupil='clear', resbg_path=None,
        satstar_label='m7', **extra)
    assert len(seen) == 1
    assert seen[0]['adopt_existing'] is bool(adopt)
    assert seen[0]['file_suffix'].endswith('_m7')
