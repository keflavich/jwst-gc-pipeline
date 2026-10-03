"""The ZEROFRAME anchor must not rewrite pixels that saturate in group 0.

Under a multi-frame readout (wd2 F150W: SHALLOW4, 4 frames of 10.7 s per
group) the first group of the ramp saturates in the cores of 14-16 mag stars.
``zeroframe_recover_saturated`` recognised a saturated first read only by a
frame-wide ceiling, 0.9 x the 99th percentile of group 0 over SATURATED pixels.
The pile-up level varies from pixel to pixel, so a group-0-saturated pixel
clipped below that ceiling counted as clean and was rewritten as
R x (clipped value): on nrcb3, 3699 of 6488 such pixels, rewritten to about half
of the rate the ramp fit had measured from the ZEROFRAME.  The recovered-core
cap then cut the star's flux to that peak.

``remove_saturated_stars`` now reads those pixels from the ramp GROUPDQ
(``_find_group0_saturation_for``) and hands them to the anchor
(``group0_saturated=``), which keeps them in the deep core.  The ceiling is
still estimated from the raw first read.
"""
import builtins
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    satstar_fit_switches, zeroframe_recover_saturated)

SATBIT = 2
FRAME = 'jw03523005001_10101_00001_nrcb3_align_o005_crf.fits'
RAMP = 'jw03523005001_10101_00001_nrcb3_ramp.fits'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)


def _scene(r_true=0.17):
    """Dark frame with bright unsaturated calibration pixels and two saturated
    stars.  Star A's 3x3 core sits at the group-0 rail (48000 DN) and sets the
    ceiling.  Star B's 3x3 core also saturates in group 0 but piled up at
    40000 DN, below the ceiling; its crf core holds the rate the ramp fit took
    from the ZEROFRAME.  Both stars have a recoverable rim at 20000 DN."""
    ny = nx = 120
    g0 = np.random.default_rng(7).normal(10, 3, (ny, nx))
    data = g0 * r_true
    dq = np.zeros((ny, nx), dtype=np.uint32)
    gdq0 = np.zeros((ny, nx), dtype=np.uint8)
    vals = np.random.default_rng(9).uniform(21000, 40000, (8, 8))
    g0[100:108, 8:16] = vals
    data[100:108, 8:16] = vals * r_true
    yy, xx = np.mgrid[0:ny, 0:nx]
    cores, rims = [], []
    for (cx, cy), rail in (((30, 30), 48000.0), ((70, 60), 40000.0)):
        rr = np.hypot(xx - cx, yy - cy)
        core = rr < 1.5
        rim = (rr >= 1.5) & (rr < 4)
        g0[core] = rail
        g0[rim] = 20000.0
        dq[core | rim] = SATBIT
        gdq0[core] = SATBIT
        data[core] = 90000.0 * r_true         # ramp-fit rate from the ZEROFRAME
        data[rim] = 20000.0 * r_true
        cores.append(core)
        rims.append(rim)
    return data, dq, g0, gdq0, cores, rims


def _groupdq(gdq0, ngroup=2):
    gdq = np.zeros((1, ngroup) + gdq0.shape, dtype=np.uint8)
    gdq[0, :] = gdq0
    return gdq


def _write_pair(tmp_path, g0, gdq=None, sci_ndim=4):
    fn = tmp_path / FRAME
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
                      {'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCB3',
                       'FILTER': 'F150W'})),
                  fits.ImageHDU(np.zeros(g0.shape, 'float32'), name='SCI'),
                  fits.ImageHDU(np.zeros(g0.shape, 'int32'), name='DQ')]
                 ).writeto(fn)
    sci = np.stack([g0, g0 * 2])[None].astype('float32')
    if sci_ndim == 3:
        sci = sci[0]
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(sci, name='SCI')]
    if gdq is not None:
        hdus.append(fits.ImageHDU(gdq, name='GROUPDQ'))
    fits.HDUList(hdus).writeto(tmp_path / RAMP)
    return str(fn)


# --------------------------------------------------------------------------
# the anchor with and without the GROUPDQ mask
# --------------------------------------------------------------------------

def test_scene_exercises_the_clipped_core():
    """Without GROUPDQ the anchor rewrites star B's group-0-saturated core
    from its clipped value: R x 40000, below the rate in the crf."""
    data, dq, g0, gdq0, (core_a, core_b), _ = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert deep[core_a].all()
    assert rim_mask[core_b].all()
    assert np.all(rec[core_b] < 0.5 * data[core_b])


def test_group0_saturated_core_goes_to_the_deep_core():
    data, dq, g0, gdq0, (core_a, core_b), (rim_a, rim_b) = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=gdq0 != 0)
    assert deep[core_a].all() and deep[core_b].all()
    assert not rim_mask[core_b].any()
    assert np.array_equal(rec[core_b], data[core_b])
    # the rims below the rail are still recovered
    assert rim_mask[rim_a].all() and rim_mask[rim_b].all()
    assert np.allclose(rec[rim_b], R * 20000.0)


def test_mask_leaves_the_ceiling_and_the_rate_ratio_alone():
    """The flagged pixels still enter the ceiling estimate (they are the
    plateau that sets it), so every unflagged pixel is classified as before,
    and R is measured from the same clean pixels."""
    data, dq, g0, gdq0, (core_a, core_b), _ = _scene()
    rec0, rim0, deep0, R0 = zeroframe_recover_saturated(data, dq, g0)
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=gdq0 != 0)
    assert R1 == pytest.approx(R0)
    keep = gdq0 == 0
    assert np.array_equal(rim0[keep], rim1[keep])
    assert np.array_equal(deep0[keep], deep1[keep])
    assert np.array_equal(rec0[keep], rec1[keep])


def test_mask_of_the_wrong_shape_is_ignored():
    data, dq, g0, gdq0, (core_a, core_b), _ = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=np.ones((5, 5), bool))
    assert rim_mask[core_b].all()


def test_fit_anchor_masks_the_core_instead_of_rewriting_it():
    data, dq, g0, gdq0, (core_a, core_b), _ = _scene()
    out, deep, rim, _ = SSF.zeroframe_fit_anchor(data, dq, g0,
                                                 group0_saturated=gdq0 != 0)
    assert deep[core_b].all() and not rim[core_b].any()
    assert np.array_equal(out[core_b], data[core_b])


# --------------------------------------------------------------------------
# reading GROUPDQ
# --------------------------------------------------------------------------

def test_reader_returns_first_read_saturation(tmp_path, capsys):
    g0 = np.ones((4, 4))
    gdq = np.zeros((2, 3, 4, 4), dtype=np.uint8)
    gdq[0, 0, 1, 2] = SATBIT
    gdq[0, 0, 3, 3] = SATBIT | 4              # SATURATED with another bit
    gdq[0, 0, 0, 0] = 4                       # JUMP_DET only
    gdq[0, 1, 2, 1] = SATBIT                  # saturated only in group 1
    gdq[1, 0, 2, 2] = SATBIT                  # second integration only
    fn = _write_pair(tmp_path, g0, gdq)
    sat0 = SSF._find_group0_saturation_for(fn)
    expect = np.zeros((4, 4), bool)
    expect[1, 2] = expect[3, 3] = True
    assert np.array_equal(sat0, expect)
    assert '2 px saturated in the first read' in capsys.readouterr().out


def test_reader_without_groupdq_returns_none(tmp_path):
    fn = _write_pair(tmp_path, np.ones((4, 4)), gdq=None)
    assert SSF._find_group0_saturation_for(fn) is None


def test_reader_without_a_4d_ramp_returns_none(tmp_path):
    gdq = np.full((1, 2, 4, 4), SATBIT, np.uint8)
    fn = _write_pair(tmp_path, np.ones((4, 4)), gdq, sci_ndim=3)
    assert SSF._find_group0_saturation_for(fn) is None


def test_reader_without_a_ramp_returns_none(tmp_path):
    fn = tmp_path / FRAME
    fits.PrimaryHDU(np.zeros((4, 4))).writeto(fn)
    assert SSF._find_group0_saturation_for(str(fn)) is None


def test_zeroframe_loader_is_unchanged(tmp_path):
    """The first read handed to the anchor and the deblender is the raw
    group 0; the mask travels separately."""
    data, dq, g0, gdq0, (core_a, core_b), _ = _scene()
    fn = _write_pair(tmp_path, g0, _groupdq(gdq0))
    assert np.allclose(SSF._find_zeroframe_for(fn), g0.astype('float32'))


# --------------------------------------------------------------------------
# the switch and the writer
# --------------------------------------------------------------------------

def test_switch_default_is_on(monkeypatch):
    assert satstar_fit_switches()['g0_groupdq'] is True
    monkeypatch.setenv('SATSTAR_ZF_G0_GROUPDQ', 'off')
    assert satstar_fit_switches()['g0_groupdq'] is False
    monkeypatch.setenv('SATSTAR_ZF_G0_GROUPDQ', 'of')
    with pytest.raises(ValueError, match='SATSTAR_ZF_G0_GROUPDQ'):
        satstar_fit_switches()


@pytest.mark.parametrize('env, zf_on, hands_mask', [
    ({}, True, True),
    ({'SATSTAR_ZF_G0_GROUPDQ': '0'}, True, False),
    ({'SATSTAR_ZEROFRAME_FIT': '0'}, False, False),
])
def test_remove_saturated_stars_hands_the_mask_to_the_fit(tmp_path,
                                                          monkeypatch, env,
                                                          zf_on, hands_mask):
    fn = _write_pair(tmp_path, np.ones((8, 8)), _groupdq(np.zeros((8, 8))))
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_model', 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mask = np.zeros((8, 8), bool)
    mask[3, 3] = True
    reads = []
    monkeypatch.setattr(SSF, '_find_zeroframe_for',
                        lambda filename: np.ones((8, 8)))

    def _read(filename):
        reads.append(filename)
        return mask
    monkeypatch.setattr(SSF, '_find_group0_saturation_for', _read)
    seen = {}

    def _fit(fh, **kw):
        seen.update(kw)
        return Table({'flux_fit': [1.0]})
    monkeypatch.setattr(SSF, 'get_saturated_stars', _fit)
    SSF.remove_saturated_stars(fn, recovery_signature='off')
    assert ('zeroframe' in seen) is zf_on
    assert ('zeroframe_group0_saturated' in seen) is hands_mask
    assert bool(reads) is hands_mask
    if hands_mask:
        assert seen['zeroframe_group0_saturated'] is mask
