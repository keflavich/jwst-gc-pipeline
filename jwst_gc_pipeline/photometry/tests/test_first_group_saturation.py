"""Regression test for the first-group SATURATED DQ correction.

The cal/crf SATURATED flag marks any pixel saturated in ANY ramp group; on
bright MIRI emission that floods huge regions and fuses real point sources into
one DQ blob (cloudc 2526 F770W: 62705 px -> 720 first-group). Only first-group-
saturated pixels are truly unrecoverable. correct_dq_first_group_saturation()
clears the SATURATED bit on later-group-only pixels, reading the sibling
_ramp.fits GROUPDQ. See project_cloudc_f770w_satstar_gate_miscalib.
"""
import os
import numpy as np
from astropy.io import fits
from jwst.datamodels import dqflags

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    first_group_saturation_mask, correct_dq_first_group_saturation)

SAT = dqflags.pixel['SATURATED']


def _write_pair(tmp_path):
    """Write a crf + sibling _ramp.fits with a SMALL first-group-saturated core
    and a LARGE later-group-only saturated region (the emission-blob analogue)."""
    stem = 'jw01234_001_00001_mirimage'
    ny = nx = 40
    # cal/crf DQ: SATURATED across the whole large region (any-group flag)
    dq = np.zeros((ny, nx), dtype=np.uint32)
    dq[5:35, 5:35] |= SAT                       # 900 px flagged in the cal flag
    crf = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((ny, nx)), name='SCI'),
                        fits.ImageHDU(dq, name='DQ')])
    crf[0].header['INSTRUME'] = 'MIRI'
    crf_path = os.path.join(tmp_path, f'{stem}_o001_crf.fits')
    crf.writeto(crf_path)
    # ramp GROUPDQ (nint, ngroup, ny, nx): first group saturated ONLY in a small
    # 4x4 core; the rest of the region saturates in a later group.
    ng = 5
    gdq = np.zeros((1, ng, ny, nx), dtype=np.uint32)
    gdq[0, 3, 5:35, 5:35] |= SAT                # late-group saturation, big region
    gdq[0, 0, 18:22, 18:22] |= SAT              # first-group saturation, small core
    ramp = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((1, ng, ny, nx)), name='SCI'),
                         fits.ImageHDU(gdq, name='GROUPDQ')])
    ramp.writeto(os.path.join(tmp_path, f'{stem}_ramp.fits'))
    return crf_path, dq


def test_first_group_mask_is_small_core(tmp_path):
    crf_path, _ = _write_pair(str(tmp_path))
    fg = first_group_saturation_mask(crf_path)
    assert fg is not None
    assert int(fg.sum()) == 16            # only the 4x4 first-group core
    assert fg[20, 20] and not fg[6, 6]


def test_correction_clears_later_group_only(tmp_path, monkeypatch):
    crf_path, dq = _write_pair(str(tmp_path))
    assert int(((dq & SAT) > 0).sum()) == 900
    monkeypatch.setenv('MIRI_FIRSTGROUP_SAT_DQ', '1')
    out = correct_dq_first_group_saturation(dq.copy(), crf_path, 'MIRI')
    assert int(((out & SAT) > 0).sum()) == 16     # collapsed to the genuine core


def test_noop_when_gated_off_or_nircam(tmp_path, monkeypatch):
    crf_path, dq = _write_pair(str(tmp_path))
    monkeypatch.setenv('MIRI_FIRSTGROUP_SAT_DQ', '0')
    assert int(((correct_dq_first_group_saturation(dq.copy(), crf_path, 'MIRI') & SAT) > 0).sum()) == 900
    monkeypatch.setenv('MIRI_FIRSTGROUP_SAT_DQ', '1')
    assert int(((correct_dq_first_group_saturation(dq.copy(), crf_path, 'NIRCAM') & SAT) > 0).sum()) == 900
