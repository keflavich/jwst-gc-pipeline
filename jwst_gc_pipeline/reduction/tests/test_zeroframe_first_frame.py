"""The ZEROFRAME anchor reads the first frame where jwst flagged a
multi-frame group 0.

Under SHALLOW4 (wd2 F150W: 4 frames of 10.7 s per group) group 0 is the
average of four frames.  Once the later frames reach the pile-up level the
average flattens: group 0 / first frame falls from 2.66 at a predicted
last-frame fill of 0-0.2 to 2.44 at fill 1.0-1.1 and 1.69 at fill 2-3
(nrcb3).  The jwst saturation step flags those reads DO_NOT_USE (its
partial-saturation test for averaged groups) or SATURATED, and ``ramp_fit``
takes their rate from the first frame.  The anchor on main reads only the
SATURATED bit and rewrites the DO_NOT_USE pixels as ``R x group0``, up to 20%
low.

The ramp ZEROFRAME extension holds the first frame alone.  With
``SATSTAR_ZF_FIRST_FRAME`` on, ``remove_saturated_stars`` loads it
(``_find_first_frame_for``) together with the group-0 SATURATED | DO_NOT_USE
mask (``_find_group0_saturation_for(do_not_use=True)``), and the anchor
replaces a flagged group 0 by ``k x first frame`` (``first_frame_group0``),
with ``k`` a function of the first frame measured on the frame's own
unflagged pixels.
"""
import builtins
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    first_frame_group0, satstar_fit_switches, zeroframe_recover_saturated)

SATBIT = 2
K_TRUE = 2.6
FRAME = 'jw03523005001_10101_00001_nrcb3_align_o005_crf.fits'
RAMP = 'jw03523005001_10101_00001_nrcb3_ramp.fits'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)


def _scene(r_true=0.17):
    """Dark frame with bright unsaturated calibration pixels and three
    saturated stars, each with a SATURATED wing (4 <= r < 8 px) at 8000 DN that
    calibrates group 0 / first frame.  The first frame is group 0 / K_TRUE
    wherever group 0 did not clip.

    * Star A: 3x3 core at the group-0 rail (48000 DN, sets the ceiling) and
      flagged SATURATED, first frame saturated as well (jwst zeroes the
      ZEROFRAME there).
    * Star B: 3x3 core flagged SATURATED and piled up at 40000 DN, below the
      ceiling; its first frame (20000 DN) is valid.
    * Star C: rim (1.5 <= r < 4 px) clipped to 24000 DN, 20% below
      K_TRUE x first frame, and flagged DO_NOT_USE only.

    The crf holds the rate the ramp fit measured everywhere.  Returns
    ``data, dq, g0, ff, gsat, gdnu, masks``."""
    ny = nx = 120
    g0 = np.random.default_rng(7).normal(10, 3, (ny, nx))
    true = g0.copy()                          # unclipped group 0
    dq = np.zeros((ny, nx), dtype=np.uint32)
    gsat = np.zeros((ny, nx), dtype=bool)
    gdnu = np.zeros((ny, nx), dtype=bool)
    vals = np.random.default_rng(9).uniform(21000, 40000, (8, 8))
    g0[100:108, 8:16] = true[100:108, 8:16] = vals
    yy, xx = np.mgrid[0:ny, 0:nx]
    masks = {}
    for name, (cx, cy) in (('a', (30, 30)), ('b', (70, 60)), ('c', (40, 90))):
        rr = np.hypot(xx - cx, yy - cy)
        core = rr < 1.5
        rim = (rr >= 1.5) & (rr < 4)
        wing = (rr >= 4) & (rr < 8)
        g0[wing] = true[wing] = 8000.0
        g0[rim] = true[rim] = 20000.0
        dq[core | rim | wing] = SATBIT
        masks[name] = (core, rim, wing)
    core_a, _, _ = masks['a']
    g0[core_a] = true[core_a] = 48000.0
    gsat[core_a] = True
    core_b, _, _ = masks['b']
    g0[core_b] = 40000.0
    true[core_b] = K_TRUE * 20000.0
    gsat[core_b] = True
    _, rim_c, _ = masks['c']
    true[rim_c] = 30000.0
    g0[rim_c] = 0.8 * 30000.0
    gdnu[rim_c] = True
    ff = true / K_TRUE
    ff[core_a] = 0.0
    data = true * r_true
    return data, dq, g0, ff, gsat, gdnu, masks


# --------------------------------------------------------------------------
# first_frame_group0
# --------------------------------------------------------------------------

def _ffg(g0, ff, flagged, region=None, **kw):
    region = np.ones(g0.shape, bool) if region is None else region
    kw.setdefault('lo', 2000.0)
    kw.setdefault('hi', 43200.0)
    return first_frame_group0(g0, ff, region, group0_saturated=flagged, **kw)


def test_k_is_measured_on_unflagged_pixels():
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    _, _, k = _ffg(g0, ff, gsat | gdnu)
    assert k == pytest.approx(K_TRUE)


def test_flagged_pixels_read_the_scaled_first_frame():
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    out, rep, k = _ffg(g0, ff, gsat | gdnu)
    core_a, rim_a, wing_a = masks['a']
    core_b, rim_b, _ = masks['b']
    _, rim_c, _ = masks['c']
    assert rep[core_b].all() and rep[rim_c].all()
    assert np.allclose(out[core_b], K_TRUE * 20000.0)
    assert np.allclose(out[rim_c], 30000.0)
    # a saturated first frame (zeroed by jwst) keeps its group 0
    assert not rep[core_a].any()
    assert np.array_equal(out[core_a], g0[core_a])
    # unflagged pixels are untouched
    assert not rep[rim_a | rim_b | wing_a].any()
    assert np.array_equal(out[~rep], g0[~rep])


def test_an_unflagged_low_group0_is_kept():
    """The replacement follows the jwst flags only; an unflagged group 0 is
    what ``ramp_fit`` used as well."""
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    _, rim_c, _ = masks['c']
    _, _, wing_a = masks['a']
    iy, ix = np.argwhere(wing_a)[0]
    g0[iy, ix] = 0.7 * 8000.0               # 30% below k x first frame
    out, rep, k = _ffg(g0, ff, gsat | gdnu)
    assert k == pytest.approx(K_TRUE)
    assert not rep[iy, ix] and out[iy, ix] == g0[iy, ix]
    out, rep, k = _ffg(g0, ff, gsat)
    assert not rep[rim_c].any()
    assert np.array_equal(out[rim_c], g0[rim_c])


def _drift_scene(n=4000):
    """Unflagged pixels whose group 0 / first frame drifts with brightness,
    2.70 at a first frame of 1000 DN to 2.50 at 10000 DN, and flagged pixels
    at 1000, 5000, 10000 and 30000 DN whose group 0 is clipped to 0."""
    rng = np.random.default_rng(3)
    ff = np.exp(rng.uniform(np.log(1000.0), np.log(10000.0), n))
    kf = lambda f: 2.70 - 0.20 * np.log10(np.asarray(f) / 1000.0)
    g0 = kf(ff) * ff
    probe = np.array([1000.0, 5000.0, 10000.0, 30000.0])
    ff = np.concatenate([ff, probe])[None]
    g0 = np.concatenate([g0, np.zeros(probe.size)])[None]
    flagged = np.zeros(ff.shape, bool)
    flagged[0, n:] = True
    return g0, ff, flagged, kf, probe


def test_k_follows_the_first_frame_brightness():
    g0, ff, flagged, kf, probe = _drift_scene()
    out, rep, k = _ffg(g0, ff, flagged, hi=np.inf)
    assert rep[0, -4:].all() and rep.sum() == 4
    k_used = out[0, -4:] / probe
    # interpolated between the bin medians inside the calibrated range
    assert k_used[:3] == pytest.approx(kf(probe[:3]), abs=0.02)
    # held at the brightest bin beyond it
    assert k_used[3] == pytest.approx(k_used[2], abs=0.01)
    assert k == pytest.approx(kf(9000.0), abs=0.02)


def test_k_falls_back_to_a_single_median():
    g0, ff, flagged, kf, probe = _drift_scene()
    out, rep, k = _ffg(g0, ff, flagged, hi=np.inf, min_bin=10**6)
    ratio = g0[~flagged] / ff[~flagged]
    assert k == pytest.approx(np.median(ratio))
    assert np.allclose(out[0, -4:], k * probe)


def test_only_the_region_is_replaced():
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    _, rim_c, _ = masks['c']
    region = np.ones(g0.shape, bool)
    region[rim_c] = False
    out, rep, k = _ffg(g0, ff, gsat | gdnu, region=region)
    assert not rep[rim_c].any()
    assert np.array_equal(out[rim_c], g0[rim_c])


def test_flagged_pixels_stay_out_of_the_calibration():
    """A flagged pixel inside [lo, hi) has a clipped group 0; it must not
    pull k."""
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    _, _, wing_b = masks['b']
    g0[wing_b] = 0.5 * 8000.0
    _, _, k = _ffg(g0, ff, gsat | gdnu | wing_b)
    assert k == pytest.approx(K_TRUE)


@pytest.mark.parametrize('first_frame', [None, np.ones((5, 5))])
def test_missing_or_misshaped_first_frame_is_a_no_op(first_frame):
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    out, rep, k = _ffg(g0, first_frame, gsat | gdnu)
    assert out is g0 and not rep.any() and np.isnan(k)


def test_too_few_calibration_pixels_is_a_no_op():
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    out, rep, k = _ffg(g0, ff, gsat | gdnu, min_px=10**6)
    assert out is g0 and not rep.any() and np.isnan(k)


# --------------------------------------------------------------------------
# the anchor
# --------------------------------------------------------------------------

def test_scene_under_reads_the_clipped_rim_without_the_first_frame():
    """The anchor on main: SATURATED-only mask, raw group 0."""
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    core_b, _, _ = masks['b']
    _, rim_c, _ = masks['c']
    rec, rim, deep, R = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=gsat)
    assert rim[rim_c].all()
    assert np.allclose(rec[rim_c], 0.8 * data[rim_c])
    assert deep[core_b].all()


def test_anchor_rewrites_flagged_pixels_from_the_first_frame():
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    core_a, _, _ = masks['a']
    core_b, _, _ = masks['b']
    _, rim_c, _ = masks['c']
    rec, rim, deep, R = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=gsat | gdnu, first_frame=ff)
    assert rim[rim_c].all() and not deep[rim_c].any()
    assert np.allclose(rec[rim_c], data[rim_c])
    # the flagged core with a valid first frame joins the rim
    assert rim[core_b].all() and not deep[core_b].any()
    assert np.allclose(rec[core_b], data[core_b])
    # the core whose first frame saturated stays deep
    assert deep[core_a].all() and not rim[core_a].any()


def test_first_frame_leaves_r_and_the_other_pixels_alone():
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    core_b, _, _ = masks['b']
    _, rim_c, _ = masks['c']
    rec0, rim0, deep0, R0 = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=gsat)
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=gsat | gdnu, first_frame=ff)
    assert R1 == pytest.approx(R0)
    keep = ~(core_b | rim_c)
    assert np.array_equal(rim0[keep], rim1[keep])
    assert np.array_equal(deep0[keep], deep1[keep])
    assert np.array_equal(rec0[keep], rec1[keep])


def test_fit_anchor_passes_the_first_frame_on():
    data, dq, g0, ff, gsat, gdnu, masks = _scene()
    _, rim_c, _ = masks['c']
    out, deep, rim, _ = SSF.zeroframe_fit_anchor(
        data, dq, g0, group0_saturated=gsat | gdnu, first_frame=ff)
    assert np.allclose(out[rim_c], data[rim_c])


class _AnchorReached(Exception):
    pass


def test_get_saturated_stars_hands_the_first_frame_to_the_anchor(monkeypatch):
    n = 200
    sci = np.ones((n, n))
    dq = np.zeros((n, n), dtype=np.uint32)
    dq[98:103, 98:103] = SATBIT
    sci[dq != 0] = np.nan
    wcs_hdr = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                           'CRPIX1': 100, 'CRPIX2': 100, 'CRVAL1': 150.0,
                           'CRVAL2': 2.0, 'CDELT1': -1.7e-5, 'CDELT2': 1.7e-5,
                           'BUNIT': 'MJy/sr'})
    fh = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({
            'INSTRUME': 'NIRCAM', 'FILTER': 'F150W', 'PUPIL': 'CLEAR',
            'DETECTOR': 'NRCB3', 'MODULE': 'B', 'CHANNEL': 'SHORT'})),
        fits.ImageHDU(sci, header=wcs_hdr, name='SCI'),
        fits.ImageHDU(np.full((n, n), 0.1), name='ERR'),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.01), name='VAR_POISSON')])
    first = np.ones((n, n))
    seen = {}

    def _anchor(data, dq, zeroframe, **kw):
        seen.update(kw)
        raise _AnchorReached
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    with pytest.raises(_AnchorReached):
        SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)),
                                zeroframe_first_frame=first, plot=False)
    assert seen['first_frame'] is first


# --------------------------------------------------------------------------
# reading the ZEROFRAME extension
# --------------------------------------------------------------------------

def _write_pair(tmp_path, shape=(4, 4), nframes=4, zeroframe=True,
                sci_ndim=4):
    fn = tmp_path / FRAME
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
                      {'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCB3',
                       'FILTER': 'F150W'})),
                  fits.ImageHDU(np.zeros(shape, 'float32'), name='SCI'),
                  fits.ImageHDU(np.zeros(shape, 'int32'), name='DQ')]
                 ).writeto(fn)
    g0 = np.arange(np.prod(shape), dtype='float32').reshape(shape) + 10
    sci = np.stack([g0, g0 * 2])[None]
    if sci_ndim == 3:
        sci = sci[0]
    hdr = fits.Header() if nframes is None else fits.Header({'NFRAMES':
                                                              nframes})
    hdus = [fits.PrimaryHDU(header=hdr), fits.ImageHDU(sci, name='SCI')]
    zf = (g0 / K_TRUE)[None]
    if zeroframe:
        hdus.append(fits.ImageHDU(zf, name='ZEROFRAME'))
    gdq = np.zeros(sci.shape, 'uint8')
    if sci_ndim == 4:
        gdq[0, 0, 0, :3] = [2, 1, 3]     # SATURATED, DO_NOT_USE, both
        gdq[0, 1, 1, :] = 2              # later group: ignored
    hdus.append(fits.ImageHDU(gdq, name='GROUPDQ'))
    fits.HDUList(hdus).writeto(tmp_path / RAMP)
    return str(fn), zf[0]


def test_loader_returns_the_zeroframe_extension(tmp_path, capsys):
    fn, zf = _write_pair(tmp_path)
    ff = SSF._find_first_frame_for(fn)
    assert np.allclose(ff, zf)
    assert 'NFRAMES=4' in capsys.readouterr().out


@pytest.mark.parametrize('kw', [
    {'nframes': 1},              # group 0 is the first frame already
    {'nframes': None},           # NFRAMES missing -> 1
    {'zeroframe': False},
    {'sci_ndim': 3},             # group-0 loader falls back to the ZEROFRAME
])
def test_loader_returns_none(tmp_path, kw):
    fn, _ = _write_pair(tmp_path, **kw)
    assert SSF._find_first_frame_for(fn) is None


def test_loader_without_a_ramp_returns_none(tmp_path):
    fn = tmp_path / FRAME
    fits.PrimaryHDU(np.zeros((4, 4))).writeto(fn)
    assert SSF._find_first_frame_for(str(fn)) is None


def test_group0_loader_adds_do_not_use_on_request(tmp_path, capsys):
    fn, _ = _write_pair(tmp_path)
    sat = SSF._find_group0_saturation_for(fn)
    both = SSF._find_group0_saturation_for(fn, do_not_use=True)
    expect_sat = np.zeros((4, 4), bool)
    expect_sat[0, [0, 2]] = True
    expect_both = np.zeros((4, 4), bool)
    expect_both[0, :3] = True
    assert np.array_equal(sat, expect_sat)
    assert np.array_equal(both, expect_both)
    assert 'saturated or DO_NOT_USE' in capsys.readouterr().out


# --------------------------------------------------------------------------
# the switch and the wiring
# --------------------------------------------------------------------------

def test_switch_default_follows_the_recovered_cap(monkeypatch):
    assert satstar_fit_switches()['first_frame'] is False
    monkeypatch.setenv('NIRCAM_SATSTAR_RECOVERED_CAP', '1')
    assert satstar_fit_switches()['first_frame'] is True
    monkeypatch.setenv('SATSTAR_ZF_FIRST_FRAME', 'off')
    assert satstar_fit_switches()['first_frame'] is False
    monkeypatch.setenv('NIRCAM_SATSTAR_RECOVERED_CAP', '0')
    monkeypatch.setenv('SATSTAR_ZF_FIRST_FRAME', 'on')
    assert satstar_fit_switches()['first_frame'] is True
    monkeypatch.setenv('SATSTAR_ZF_FIRST_FRAME', 'of')
    with pytest.raises(ValueError, match='SATSTAR_ZF_FIRST_FRAME'):
        satstar_fit_switches()


_CAP = {'NIRCAM_SATSTAR_RECOVERED_CAP': '1'}


@pytest.mark.parametrize('env, zf_on, hands_ff, g0_bits', [
    (_CAP, True, True, 'sat|dnu'),
    ({}, True, False, None),
    ({**_CAP, 'SATSTAR_ZF_FIRST_FRAME': '0'}, True, False, 'sat'),
    ({'SATSTAR_ZF_FIRST_FRAME': '1'}, True, True, 'sat|dnu'),
    ({**_CAP, 'SATSTAR_ZEROFRAME_FIT': '0'}, False, False, None),
])
def test_remove_saturated_stars_hands_the_first_frame_to_the_fit(
        tmp_path, monkeypatch, env, zf_on, hands_ff, g0_bits):
    """With a first frame the group-0 mask carries DO_NOT_USE as well, and it
    is loaded even when SATSTAR_ZF_G0_GROUPDQ is off."""
    fn, _ = _write_pair(tmp_path, shape=(8, 8))
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_model', 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    first = np.ones((8, 8))
    masks = {'sat': np.zeros((8, 8), bool), 'sat|dnu': np.ones((8, 8), bool)}
    reads, g0_reads = [], []
    monkeypatch.setattr(SSF, '_find_zeroframe_for',
                        lambda filename: np.ones((8, 8)))

    def _g0(filename, do_not_use=False):
        bits = 'sat|dnu' if do_not_use else 'sat'
        g0_reads.append(bits)
        return masks[bits]
    monkeypatch.setattr(SSF, '_find_group0_saturation_for', _g0)

    def _read(filename):
        reads.append(filename)
        return first
    monkeypatch.setattr(SSF, '_find_first_frame_for', _read)
    seen = {}

    def _fit(fh, **kw):
        seen.update(kw)
        return Table({'flux_fit': [1.0]})
    monkeypatch.setattr(SSF, 'get_saturated_stars', _fit)
    SSF.remove_saturated_stars(fn, recovery_signature='off')
    assert ('zeroframe' in seen) is zf_on
    assert ('zeroframe_first_frame' in seen) is hands_ff
    assert bool(reads) is hands_ff
    if hands_ff:
        assert seen['zeroframe_first_frame'] is first
    assert g0_reads == ([] if g0_bits is None else [g0_bits])
    if g0_bits is not None:
        assert seen['zeroframe_group0_saturated'] is masks[g0_bits]


def test_without_a_first_frame_the_mask_stays_saturated_only(tmp_path,
                                                             monkeypatch):
    """NFRAMES = 1 (or no ZEROFRAME): the first-frame loader returns None and
    the group-0 mask is the SATURATED bit alone."""
    fn, _ = _write_pair(tmp_path, shape=(8, 8))
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_model', 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)
    monkeypatch.setenv('NIRCAM_SATSTAR_RECOVERED_CAP', '1')
    g0_reads = []
    monkeypatch.setattr(SSF, '_find_zeroframe_for',
                        lambda filename: np.ones((8, 8)))
    monkeypatch.setattr(SSF, '_find_first_frame_for', lambda filename: None)

    def _g0(filename, do_not_use=False):
        g0_reads.append(do_not_use)
        return np.zeros((8, 8), bool)
    monkeypatch.setattr(SSF, '_find_group0_saturation_for', _g0)
    monkeypatch.setattr(SSF, 'get_saturated_stars',
                        lambda fh, **kw: Table({'flux_fit': [1.0]}))
    SSF.remove_saturated_stars(fn, recovery_signature='off')
    assert g0_reads == [False]
