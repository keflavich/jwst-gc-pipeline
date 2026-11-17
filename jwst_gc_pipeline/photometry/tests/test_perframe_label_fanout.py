"""Two merge labels fit one frame at m7; each label's merge must read its own fit.

The gc-treasury m7 fan-out runs the ``merged`` pass and then the ``nrcb``
pass over the same nrcblong frames.  Their fits differ (different seed, a
different smoothed residual background), and before the merge-label token
both were written to one detector-keyed path, so the ``merged`` label's
per-filter merge read the ``nrcb`` pass's catalog (it finished last).

This runs the REAL writer (``cataloging._save_manual_pass`` ->
``save_photutils_results``) for both passes on one frame, then the REAL
per-filter merge (``merge_catalogs.merge_individual_frames``, its
``combine_singleframe`` stubbed to record what it was given) for both labels.
It imports nothing added by the fix, so it also runs, and fails, on the
commit before it.
"""
import os
import types

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import merge_catalogs as MC

SHAPE = (8, 8)
CRF = '/x/jw10678105001_02101_00004_nrcblong_destreak_o105_crf.fits'
#: the flux each pass "measures" -- the marker of whose fit a file holds
FLUX = {'merged': 111.0, 'nrcb': 222.0}
SEED = {'merged': 'crossband_merged_o105_m6_seed.fits',
        'nrcb': 'crossband_nrcb_o105_m6_seed.fits'}
RESBG = {'merged': 'f480m-merged_m6_smoothed_bg_i2d.fits',
         'nrcb': 'f480m-nrcb_m6_smoothed_bg_i2d.fits'}


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.crval = [266.4, -28.9]
    w.wcs.crpix = [4, 4]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    return w


def _fit(label):
    """One pass's fit result: three stars, flux = the pass's marker."""
    return Table({'x_fit': [2.0, 4.0, 6.0], 'y_fit': [2.0, 4.0, 6.0],
                  'flux_fit': [FLUX[label]] * 3, 'flux_err': [1.0] * 3,
                  'local_bkg': [0.0] * 3})


def _write_pass(basepath, label, written):
    """The real ``_save_manual_pass`` for one label's m7 fit of the frame."""
    im1 = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros(SHAPE))])
    ctx = types.SimpleNamespace(
        out_basepath=str(basepath), original_data=np.zeros(SHAPE),
        satstar_model_subtracted=None, dqarr=np.zeros(SHAPE, dtype=int),
        ww=_wcs(), filename=CRF, im1=im1, filtername='F480M',
        module='nrcblong', desat='', bgsub='_resbgsub', exposure_='_exp00004',
        visitid_='_visit001', vgroupid_='_vgroup02101', epsf_='', blur_='',
        group='', background_map=None, proposal_id='10678', field='105',
        inst_token='nircam', pupil='clear',
        # what do_photometry_step_manual sets for the pass
        prov_merge_label=label, prov_seed_catalog=f'/c/{SEED[label]}',
        prov_resbg_path=f'/p/{RESBG[label]}')
    options = types.SimpleNamespace(blur=False, each_exposure=True,
                                    proposal_id='10678', field='105')
    C._save_manual_pass(ctx, _fit(label), np.zeros(SHAPE), options, 'm7',
                        'nrcblong')
    return written


@pytest.fixture
def fanout(tmp_path, monkeypatch):
    monkeypatch.delenv('PSF_EPSF_CORE_DIR', raising=False)
    monkeypatch.delenv('SLURM_TMPDIR', raising=False)
    written = []
    monkeypatch.setattr(C, '_attach_residual_background',
                        lambda result, residual, ctx, options, label='': result)
    monkeypatch.setattr(C._L, 'save_residual_datamodel',
                        lambda filename, path, array: written.append(path))
    (tmp_path / 'F480M').mkdir()
    # the merged pass first, then the module pass -- the production order
    for label in ('merged', 'nrcb'):
        _write_pass(tmp_path, label, written)
    (tmp_path / 'catalogs').mkdir()
    return tmp_path, written


def _merge(basepath, module, monkeypatch):
    seen = []

    def _fake(tbls, offsets_table=None, filtername=None):
        seen.extend(sorted(float(f) for t in tbls for f in t['flux_fit']))
        return Table({'skycoord_avg': SkyCoord([266.4] * u.deg, [-28.9] * u.deg)})

    monkeypatch.setattr(MC, 'combine_singleframe', _fake)
    MC.merge_individual_frames(
        module=module, filtername='f480m', progid='10678', method='dao',
        suffix='_basic', target='gc-treasury', basepath=str(basepath),
        iteration_label='m7', field='105', resbgsub=True,
        do_replace_saturated=False)
    return seen


@pytest.mark.parametrize('label', ['merged', 'nrcb'])
def test_each_labels_merge_reads_its_own_passs_fit(fanout, monkeypatch, label):
    basepath, _ = fanout
    seen = _merge(basepath, label, monkeypatch)
    assert seen == [FLUX[label]] * 3, (
        f"the {label!r} merge read {sorted(set(seen))}, the other pass's fit "
        f"-- both passes wrote one per-frame path")


def test_the_two_passes_leave_two_catalogs_and_four_images(fanout):
    basepath, written = fanout
    cats = sorted(os.listdir(basepath / 'F480M'))
    cats = [c for c in cats if c.endswith('.fits')]
    assert len(cats) == 2, cats
    assert len(set(written)) == 4, written


def test_each_catalog_names_its_pass_and_inputs(fanout):
    basepath, _ = fanout
    by_label = {}
    for c in os.listdir(basepath / 'F480M'):
        if not c.endswith('.fits'):
            continue
        h = fits.getheader(basepath / 'F480M' / c, 1)
        t = Table.read(basepath / 'F480M' / c)
        by_label[h['MERGELBL']] = (h, float(t['flux_fit'][0]))
    assert set(by_label) == {'merged', 'nrcb'}
    for label, (h, flux) in by_label.items():
        assert flux == FLUX[label]
        assert h['SEEDCAT'] == SEED[label]
        assert h['RESBG'] == RESBG[label]
