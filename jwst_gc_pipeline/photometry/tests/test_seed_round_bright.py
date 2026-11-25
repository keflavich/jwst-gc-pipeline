"""i2d residual seed: a detection whose roundness lies beyond the loose window
is admitted up to ``round_bright_max`` only when it is bright (ERR S/N),
prominent (annulus prominence) and still in the residual (its peak over the
peak of the data co-add).  Off by default.

The synthetic stars below are Gaussians of FWHM 1.99 px (F182M) elongated
along y.  On flat unit-variance noise an elongation of 2.5 gives roundness1 +0.7
and roundness2 +1.2: outside the tight (0.5) and loose (0.8) windows and
inside the bright window (1.5).  An elongation of 1.7 gives roundness2 +0.74,
inside the loose window.
"""
import ast
import inspect
import os
import textwrap
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.band_option import band_mopt
from jwst_gc_pipeline.photometry.cataloging import (_build_i2d_augmented_seed,
                                                   _minor_axis_sigma, _near_large_hole,
                                                   _peak_fraction)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

N = 140
FWHM = 1.99            # F182M

BRIGHT = (30.0, 30.0)      # elongation 2.5, amplitude 300: ERR S/N 300, prominence ~300
FAINT = (30.0, 70.0)       # elongation 2.5, amplitude 20: ERR S/N 20, prominence ~20
LOOSE = (70.0, 30.0)       # elongation 1.7, amplitude 300: roundness2 +0.74, inside the loose window
ROUND = (70.0, 70.0)       # round star, amplitude 300: inside the tight window
KNOT = (108.0, 108.0)      # elongation 2.5, amplitude 300 inside PSF-scale structure of rms 20
LOBE = (30.0, 110.0)       # elongation 2.5, amplitude 300 in the residual; the data co-add has a
                           # round star of amplitude 3000 there (what is left of a subtracted star)
RIM = (110.0, 30.0)        # elongation 2.5, amplitude 300, 2 px left of a block of masked pixels
                           # (a hole of ``hole`` x ``hole`` pixels, only when asked for)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [266.5, -28.8]
    w.wcs.crpix = [N / 2, N / 2]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    return w


def _blob(x0, y0, amp, q):
    """Gaussian of FWHM ``FWHM`` across x and ``q`` x FWHM along y."""
    yy, xx = np.mgrid[0:N, 0:N]
    s = FWHM / 2.3548
    return amp * np.exp(-((xx - x0) ** 2 / (2 * s ** 2) + (yy - y0) ** 2 / (2 * (s * q) ** 2)))


def _image(hole=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(11)
    img = rng.normal(0, 1, (N, N))
    img += _blob(*BRIGHT, 300.0, 2.5)
    img += _blob(*FAINT, 20.0, 2.5)
    img += _blob(*LOOSE, 300.0, 1.7)
    img += _blob(*ROUND, 300.0, 1.0)
    struct = gaussian_filter(rng.normal(0, 1, (N, N)), 1.2)
    struct *= 20.0 / np.std(struct)
    box = np.zeros((N, N))
    box[90:126, 90:126] = 1
    img += struct * box
    img += _blob(*KNOT, 300.0, 2.5)
    img += _blob(*LOBE, 300.0, 2.5)
    if hole:
        img += _blob(*RIM, 300.0, 2.5)
        img[int(RIM[1]) - hole // 2:int(RIM[1]) - hole // 2 + hole,
            int(RIM[0]) + 2:int(RIM[0]) + 2 + hole] = np.nan
    return img


def _data(img):
    """The data co-add: the detection image plus the star that the lobe is left of."""
    return img + _blob(*LOBE, 3000.0, 1.0)


def _write(tmp_path, img, err=True, data=None):
    hdr = _wcs().to_header()
    if data is not None:
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header=hdr, name='SCI')]).writeto(
            os.path.join(tmp_path, 'data_i2d.fits'))
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(img, header=hdr, name='SCI')]
    if err:
        hdus.append(fits.ImageHDU(np.ones_like(img), header=hdr, name='ERR'))
    hdus.append(fits.ImageHDU(np.ones_like(img), header=hdr, name='WHT'))
    det = os.path.join(tmp_path, 'det_i2d.fits')
    fits.HDUList(hdus).writeto(det)
    prev = Table({'skycoord': _wcs().pixel_to_world([5.0], [5.0]), 'flux': [100.0]})
    ppath = os.path.join(tmp_path, 'prev_m6_vetted.fits')
    prev.write(ppath)
    return det, ppath


def _run(tmp_path, *, err=True, loose=0.8, data='coadd', hole=0, **bright):
    """``data``: 'coadd' (the data co-add on the detection grid), 'none' (no path), 'missing'
    (a path with no file), 'shape' (a co-add of another shape)."""
    os.makedirs(tmp_path, exist_ok=True)
    img = _image(hole)
    coadd = {'coadd': _data(img), 'shape': np.zeros((N + 5, N + 5))}.get(data)
    det, prev = _write(str(tmp_path), img, err=err, data=coadd)
    path = {'coadd': 'data_i2d.fits', 'shape': 'data_i2d.fits', 'missing': 'no_such_i2d.fits',
            'none': None}[data]
    out = _build_i2d_augmented_seed(det, prev, 'F182M', local_snr_min=5.0,
                                    roundlo=-0.5, roundhi=0.5,
                                    round_loose_max=loose, round_loose_prom_min=5.0,
                                    data_i2d_path=None if path is None else os.path.join(str(tmp_path), path),
                                    **bright)
    t = Table.read(out)
    x, y = _wcs().world_to_pixel(t['skycoord'])
    return t, np.c_[x, y]


def _row(xy, pos, r=1.5):
    """Index of the seed within ``r`` px of ``pos``, or None."""
    d = np.hypot(xy[:, 0] - pos[0], xy[:, 1] - pos[1])
    return int(np.argmin(d)) if d.min() < r else None


def test_defaults_are_off():
    assert MANUAL_DEFAULTS['manual_seed_round_bright_max'] == 0.0
    assert MANUAL_DEFAULTS['manual_seed_round_bright_snr_min'] == 100.0
    assert MANUAL_DEFAULTS['manual_seed_round_bright_prom_min'] == 50.0
    assert MANUAL_DEFAULTS['manual_seed_round_bright_frac_min'] == 0.9
    assert MANUAL_DEFAULTS['manual_seed_round_bright_width_min'] == 0.60


def test_options_are_wired_to_the_canonical_defaults():
    import inspect
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as CL
    src = inspect.getsource(CL)
    for name in ('max', 'snr_min', 'prom_min', 'frac_min', 'width_min'):
        flag = '--manual-seed-round-bright-' + name.replace('_', '-')
        key = f'manual_seed_round_bright_{name}'
        assert f'"{flag}"' in src
        assert f'dest="{key}"' in src
        assert f"default=MANUAL_DEFAULTS['{key}']" in src


def test_window_off_leaves_the_seed_table_unchanged(tmp_path):
    a, xy = _run(tmp_path / 'a')
    b, _ = _run(tmp_path / 'b', round_bright_max=0.0)
    assert 'seed_round_bright' not in a.colnames and 'seed_round_bright' not in b.colnames
    assert 'RBSNRMIN' not in a.meta and 'RBSNRMIN' not in b.meta
    assert _row(xy, BRIGHT) is None
    assert a.colnames == b.colnames
    assert len(a) == len(b)
    assert np.all(np.asarray(a['skycoord'].ra) == np.asarray(b['skycoord'].ra))


def test_window_admits_the_bright_star(tmp_path):
    t, xy = _run(tmp_path, round_bright_max=1.5)
    i = _row(xy, BRIGHT)
    assert i is not None
    assert bool(t['seed_round_bright'][i]) and not bool(t['seed_round_loose'][i])
    assert str(t['seed_origin'][i]) == 'i2d'
    # every other row, the carried-over one included, is False
    assert int(np.sum(t['seed_round_bright'])) == 1


def test_classes_are_disjoint(tmp_path):
    """The star inside the loose window carries only seed_round_loose, the star inside the
    tight window neither flag."""
    t, xy = _run(tmp_path, round_bright_max=1.5)
    i, j = _row(xy, LOOSE), _row(xy, ROUND)
    assert i is not None and j is not None
    assert bool(t['seed_round_loose'][i]) and not bool(t['seed_round_bright'][i])
    assert not bool(t['seed_round_loose'][j]) and not bool(t['seed_round_bright'][j])


def test_window_works_without_the_loose_window(tmp_path):
    """With the loose window off, the bright window admits what the loose one would not have, and the
    star the loose window would have taken needs the bright gates as well."""
    t, xy = _run(tmp_path, loose=0.0, round_bright_max=1.5)
    i, j = _row(xy, BRIGHT), _row(xy, LOOSE)
    assert i is not None and j is not None      # both are bright and prominent
    assert bool(t['seed_round_bright'][i]) and bool(t['seed_round_bright'][j])
    assert not np.any(t['seed_round_loose'])


def test_snr_gate_rejects_the_faint_star(tmp_path):
    t, xy = _run(tmp_path / 'hi', round_bright_max=1.5)
    assert _row(xy, FAINT) is None
    # the S/N gate alone: with it relaxed and the prominence floor kept, the star enters
    t, xy = _run(tmp_path / 'lo', round_bright_max=1.5, round_bright_snr_min=10.0,
                 round_bright_prom_min=10.0)
    i = _row(xy, FAINT)
    assert i is not None and bool(t['seed_round_bright'][i])


def test_snr_floor_is_the_value_of_the_band(tmp_path):
    """``--manual-seed-round-bright-snr-min F182M=10,F212N=30`` gives each band its own
    floor, and a band that the option does not name the default 100.  The faint star
    (ERR S/N 20, prominence floor lowered to 10) enters only at the floor of 10, and the
    seed table records the floor of its band."""
    name = 'manual_seed_round_bright_snr_min'
    opts = types.SimpleNamespace(**{name: 'F182M=10,F212N=30'})
    seen = {}
    for band in ('F182M', 'f212n', 'F480M'):
        floor = band_mopt(opts, name, band)
        t, xy = _run(tmp_path / band, round_bright_max=1.5, round_bright_snr_min=floor,
                     round_bright_prom_min=10.0)
        seen[band] = (floor, _row(xy, FAINT) is not None, t.meta['RBSNRMIN'])
    assert seen == {'F182M': (10.0, True, 10.0), 'f212n': (30.0, False, 30.0),
                    'F480M': (MANUAL_DEFAULTS[name], False, MANUAL_DEFAULTS[name])}
    # a plain number is the floor of every band
    for band in ('F182M', 'F480M'):
        assert band_mopt(types.SimpleNamespace(**{name: '10'}), name, band) == 10.0
    # the floor is recorded with the window on only
    t, _ = _run(tmp_path / 'default', round_bright_max=1.5)
    assert t.meta['RBSNRMIN'] == 100.0


def test_pipeline_reads_the_floor_of_the_band_before_the_try():
    """run_manual_pipeline hands the builder the floor of the band it builds the seed for,
    read outside the try whose except falls back to the seed without i2d detections (a
    malformed value must stop the run)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cataloging.run_manual_pipeline)))
    (call,) = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
               and getattr(c.func, 'id', None) == '_build_i2d_augmented_seed']
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kw['round_bright_snr_min'] == '_rb_snr' and kw['round_bright_max'] == '_rb_max'
    (read,) = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(getattr(t, 'id', None) == '_rb_snr' for t in n.targets)]
    assert ast.unparse(read.value.func) == 'band_mopt'
    assert [ast.unparse(a) for a in read.value.args] == [
        'opts_phase', "'manual_seed_round_bright_snr_min'", 'filt']
    (guard,) = [t for t in ast.walk(tree) if isinstance(t, ast.Try)
                and 'i2d-augmented seed failed' in ast.unparse(t.handlers[0])]
    assert any(n is call for stmt in guard.body for n in ast.walk(stmt))
    assert not any(n is read for stmt in guard.body for n in ast.walk(stmt))


def test_prominence_gate_rejects_the_knot(tmp_path):
    t, xy = _run(tmp_path / 'hi', round_bright_max=1.5)
    assert _row(xy, KNOT) is None
    # the prominence gate alone: opened, the knot enters at its ERR S/N of ~300
    t, xy = _run(tmp_path / 'open', round_bright_max=1.5, round_bright_prom_min=0.0)
    i = _row(xy, KNOT)
    assert i is not None and bool(t['seed_round_bright'][i])


def test_fraction_gate_rejects_the_residual_lobe(tmp_path):
    """A lobe with S/N 300 and prominence ~300 whose data peak is ten times higher is the
    residual of a subtracted star.  The fraction gate alone rejects it."""
    t, xy = _run(tmp_path / 'hi', round_bright_max=1.5)
    assert _row(xy, LOBE) is None and _row(xy, BRIGHT) is not None
    t, xy = _run(tmp_path / 'open', round_bright_max=1.5, round_bright_frac_min=0.0)
    i = _row(xy, LOBE)
    assert i is not None and bool(t['seed_round_bright'][i])


@pytest.mark.parametrize('data', ['none', 'missing', 'shape'])
def test_no_usable_data_coadd_gives_no_bright_seed(tmp_path, data):
    t, xy = _run(tmp_path, round_bright_max=1.5, data=data)
    assert _row(xy, BRIGHT) is None
    assert not np.any(t['seed_round_bright'])
    # the loose window is not affected
    assert _row(xy, LOOSE) is not None


def test_peak_fraction():
    resid = np.zeros((20, 20))
    data = np.zeros((20, 20))
    resid[10, 10], data[10, 10] = 3.0, 4.0
    resid[5, 5], data[5, 5] = 1.0, 0.0               # a data maximum that is not positive
    data[15, 15] = 5.0
    resid[14:17, 14:17] = np.nan                     # no finite residual
    out = _peak_fraction(resid, data, [10.2, 5, 15, -3, np.nan, 10], [9.8, 5, 15, 2, 4, 30])
    assert out[0] == pytest.approx(0.75)
    assert np.all(np.isnan(out[1:]))


def test_window_edge_is_the_bright_max(tmp_path):
    """A window below the star's roundness (about 1.2) does not admit it."""
    t, xy = _run(tmp_path, round_bright_max=1.0)
    assert _row(xy, BRIGHT) is None


def test_window_below_the_loose_edge_is_inert(tmp_path):
    a, xy = _run(tmp_path / 'a')
    b, _ = _run(tmp_path / 'b', round_bright_max=0.6)
    assert 'seed_round_bright' not in b.colnames
    assert len(a) == len(b)


def test_no_err_plane_gives_no_bright_seed(tmp_path):
    t, xy = _run(tmp_path, err=False, round_bright_max=1.5)
    assert _row(xy, BRIGHT) is None
    assert not np.any(t['seed_round_bright'])


@pytest.mark.parametrize('hole, kept', [(1, True), (4, True), (5, False), (8, False)])
def test_hole_guard_drops_the_rim_of_a_large_hole(tmp_path, hole, kept):
    """The star 2 px from a block of masked pixels: dropped when the block has 20 or more pixels
    (5 x 5 and 8 x 8), kept for one pixel or 16 (4 x 4).  The other bright star is unaffected."""
    t, xy = _run(tmp_path, round_bright_max=1.5, hole=hole)
    i = _row(xy, RIM)
    if kept:
        assert i is not None and bool(t['seed_round_bright'][i])
    else:
        assert i is None
    assert _row(xy, BRIGHT) is not None


def test_near_large_hole():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:15, 10:15] = True               # 25 pixels
    mask[30, 30] = True                     # 1 pixel
    mask[20:22, 20:22] = True               # 4 pixels, joined diagonally to the next one
    mask[22:25, 22:25] = True               # 9 pixels: the diagonal pair is one 8-connected blob of 13
    x = [15.0, 18.0, 19.0, 30.0, 26.0, np.nan, 12.0, -50.0]
    y = [12.0, 12.0, 12.0, 30.0, 26.0, 5.0, np.nan, 12.0]
    out = _near_large_hole(mask, x, y, min_area=20, radius=4.0)
    # (15, 12): 1 px from the 5 x 5 block; (18, 12): 4 px, not < 4; (19, 12): 5 px; the single pixel and the
    # 13-pixel blob are below 20; NaN positions and positions off the image are False
    assert out.tolist() == [True, False, False, False, False, False, False, False]
    assert _near_large_hole(mask, x, y, min_area=10, radius=4.0).tolist() == [True, False, False, False, True,
                                                                              False, False, False]
    assert not _near_large_hole(np.zeros((9, 9), dtype=bool), [4.0], [4.0]).any()


def _gauss(n, fwhm, x0, y0, amp=1000.0):
    yy, xx = np.mgrid[0:n, 0:n]
    s = fwhm / 2.355
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))


def test_minor_axis_sigma_of_a_star():
    """A Gaussian star of the F212N PSF width (FWHM 2.34 px, standard deviation 0.99 px) measures 0.65-0.95
    of that standard deviation at every sub-pixel position: the pixels at a quarter of the peak and above
    are a disk smaller than the Gaussian's own spread."""
    s = 2.34 / 2.355
    w = [_minor_axis_sigma(_gauss(31, 2.34, 15 + dx, 15 + dy), [15.0], [15.0])[0]
         for dx in (-0.5, -0.25, 0.0, 0.25, 0.5) for dy in (-0.5, -0.25, 0.0, 0.25, 0.5)]
    assert min(w) >= 0.65 * s and max(w) <= 0.95 * s


def test_minor_axis_sigma_of_a_pair_is_not_thin():
    """Two equal stars 3 px apart are elongated, not thin: the width across the pair's short axis is a
    star's."""
    s = 2.34 / 2.355
    img = _gauss(31, 2.34, 13.5, 15.0) + _gauss(31, 2.34, 16.5, 15.0)
    assert _minor_axis_sigma(img, [15.0], [15.0])[0] >= 0.65 * s


def test_minor_axis_sigma_of_a_streak_and_a_spike_is_zero():
    horizontal = np.zeros((31, 31))
    horizontal[15, 10:21] = 1000.0
    diagonal = np.zeros((31, 31))
    diagonal[np.arange(10, 21), np.arange(10, 21)] = 1000.0
    spike = np.zeros((31, 31))
    spike[15, 15] = 1000.0
    pair = np.zeros((31, 31))
    pair[15, 15:17] = 1000.0
    for img in (horizontal, diagonal, spike, pair):
        assert _minor_axis_sigma(img, [15.0], [15.0])[0] == pytest.approx(0.0, abs=1e-6)


def test_minor_axis_sigma_is_nan_where_it_cannot_be_measured():
    img = _gauss(31, 2.34, 15.0, 15.0)
    neg = -img
    blank = img.copy()
    blank[14:17, 14:17] = np.nan
    out = _minor_axis_sigma(img, [3.0, 27.0, 15.0, np.nan, 15.0], [15.0, 15.0, 3.0, 15.0, np.nan])
    assert np.all(np.isnan(out))                      # a window off the image on each side, not finite
    assert np.isnan(_minor_axis_sigma(neg, [15.0], [15.0])[0])       # the 3x3 maximum is not positive
    assert np.isnan(_minor_axis_sigma(blank, [15.0], [15.0])[0])     # no finite pixel at the peak


def test_minor_axis_sigma_ignores_a_masked_pixel_beside_the_peak():
    img = _gauss(31, 2.34, 15.0, 15.0)
    ref = _minor_axis_sigma(img, [15.0], [15.0])[0]
    img[15, 17] = np.nan
    w = _minor_axis_sigma(img, [15.0], [15.0])[0]
    assert np.isfinite(w) and w == pytest.approx(ref, rel=0.15)


def test_width_gate_is_in_units_of_the_psf_standard_deviation(tmp_path):
    """The bright star of the fixture is 0.73 of the PSF standard deviation wide across x.  A floor
    just below that keeps it, a floor just above drops it, no floor keeps it."""
    w = _minor_axis_sigma(_image(), [BRIGHT[0]], [BRIGHT[1]])[0] / (FWHM / 2.355)
    assert 0.6 < w < 0.9
    _, xy = _run(tmp_path / 'below', round_bright_max=1.5, round_bright_width_min=0.9 * w)
    assert _row(xy, BRIGHT) is not None
    _, xy = _run(tmp_path / 'above', round_bright_max=1.5, round_bright_width_min=1.1 * w)
    assert _row(xy, BRIGHT) is None
    _, xy = _run(tmp_path / 'off', round_bright_max=1.5, round_bright_width_min=0.0)
    assert _row(xy, BRIGHT) is not None
    # the default floor of 0.60 keeps it
    _, xy = _run(tmp_path / 'default', round_bright_max=1.5)
    assert _row(xy, BRIGHT) is not None
