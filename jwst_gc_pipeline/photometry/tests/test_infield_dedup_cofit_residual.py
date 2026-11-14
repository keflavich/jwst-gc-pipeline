"""``SATSTAR_INFIELD_DEDUP='cofit_residual'``: the residual test and the
edge-only keeper preference on top of the 'cofit' in-field dedup.

The satstar finder fits ONE star per saturated DQ component, so two stars
sharing a component in every frame are never fit side by side and 'cofit'
collapses them (gc-treasury 10678 m7 o078/o043 F480M: about 21 of 32
collapsed rows).  'cofit_residual' keeps such a pair when a frame that
fits only the better-ranked member leaves the other member's light in its
satstar residual: >= 0.5 of its peak AND >= 0.25 of its fitted flux within
3 px, both in that frame.  The flux condition is what rejects F212N rows
whose every fit is at a detector edge (peak ratio ~1 wherever the
neighbour's model is negligible, < 0.1 of the flux).

Small synthetic exposures (Gaussian stars on a 400x400 frame) with real-
format satstar files pin:

* a real pair is kept as two stars (and 'cofit' alone collapses it);
* a shoulder fit on a bright star's wing still collapses;
* a row with a peak but not its flux at its position still collapses, and
  the flux threshold is overridable from the environment;
* an edge-truncated duplicate collapses onto the INTERIOR member, not onto
  the edge-only row with more support;
* a missing, unreadable, truncated or wrongly shaped residual (or frame)
  falls back to 'cofit' for the pair, with a log line;
* the lower member is measured only >= 20 px inside the frame, on all four
  sides; edge-only means every fit <= 14 px from the array edge (10 px from
  the science area past 4 reference pixels);
* the flux fraction is signed (a spike over a dip collapses), and the peak
  and flux tests must pass in the SAME run (a NaN or inf run measures
  nothing, and does not hide a passing run);
* satstar ``x_fit``/``y_fit`` are stamp coordinates: the edge test reads the
  frame-pixel ``xcentroid``/``ycentroid``;
* the settings, the env thresholds and the run_manual_pipeline wiring.
"""
import inspect
import os
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging as C

PIXSCALE = 0.063 / 3600.0       # deg/px, NIRCam LW
SHAPE = (400, 400)
RA0, DEC0 = 266.70, -28.55
SIGMA = 1.2                     # px, Gaussian PSF (FWHM ~0.18")
BG = 10.0

_ENVS = ('SATSTAR_INFIELD_DEDUP', 'SATSTAR_INFIELD_DEDUP_MIRI',
         'SATSTAR_INFIELD_DEDUP_NIRISS', 'GC_INSTRUMENT_OVERRIDE',
         'SATSTAR_INFIELD_RESID_RATIO_MIN', 'SATSTAR_INFIELD_RESID_FLUX_MIN')


@pytest.fixture
def clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _wcs(shift=(0.0, 0.0)):
    """Reference WCS; a frame with ``shift`` puts reference pixel (x, y) at
    frame pixel (x + dx, y + dy)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [200.5 + shift[0], 200.5 + shift[1]]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


REF = _wcs()


def _sky(xy):
    xy = np.atleast_2d(np.asarray(xy, float))
    return REF.pixel_to_world(xy[:, 0], xy[:, 1])


def _gauss(shape, x, y, flux):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (flux / (2 * np.pi * SIGMA ** 2)
            * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA ** 2)))


def _data_i2d(tmp_path):
    path = tmp_path / 'data_i2d.fits'
    if path.exists():
        return str(path)
    hdu = fits.ImageHDU(np.zeros(SHAPE, dtype='float32'), name='SCI')
    hdu.header.update(REF.to_header())
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return str(path)


def _write_exposure(pdir, expnum, stars, fits_acc, fits_rej=(), shift=(0, 0),
                    residual=True, seed=0, shape=SHAPE):
    """One frame plus its m7 satstar catalog / rejected / residual files.

    ``stars``: true sources ``(x, y, flux)`` in REFERENCE pixels.
    ``fits_acc``: accepted satstar fits ``(x, y, flux)`` (reference pixels);
    their Gaussians are the satstar model, so the residual is data minus
    them.  ``fits_rej``: gate-rejected fits ``(x, y)`` (not subtracted).
    The catalog carries frame-pixel ``xcentroid``/``ycentroid`` and stamp
    ``x_fit``/``y_fit`` (7.0: the middle of a 15 px stamp), as written by
    the satstar finder.
    """
    w = _wcs(shift)
    dx, dy = shift
    rng = np.random.default_rng(seed)
    data = np.full(shape, BG) + rng.normal(0, 0.5, shape)
    for x, y, f in stars:
        data += _gauss(shape, x + dx, y + dy, f)
    model = np.zeros(shape)
    for x, y, f in fits_acc:
        model += _gauss(shape, x + dx, y + dy, f)
    stem = pdir / f'jw10678001001_02101_{expnum:05d}_nrcalong_destreak_o001_crf'
    frame = f'{stem}.fits'
    sci = fits.ImageHDU(data.astype('float32'), name='SCI')
    sci.header.update(w.to_header())
    fits.HDUList([fits.PrimaryHDU(), sci]).writeto(frame)
    for kind, rows in (('catalog', [(x, y, f) for x, y, f in fits_acc]),
                       ('rejected', [(x, y, 1e3) for x, y in fits_rej])):
        if kind == 'rejected' and not rows:
            continue
        xs = np.array([r[0] + dx for r in rows], float)
        ys = np.array([r[1] + dy for r in rows], float)
        tbl = Table({'xcentroid': xs, 'ycentroid': ys,
                     'x_fit': np.full(len(xs), 7.0),
                     'y_fit': np.full(len(xs), 7.0),
                     'flux_fit': np.array([r[2] for r in rows], float)})
        tbl['skycoord_fit'] = w.pixel_to_world(xs, ys)
        tbl.write(f'{stem}_resbgsub_m7_satstar_{kind}.fits')
    resid_path = f'{stem}_resbgsub_m7_satstar_residual.fits'
    if residual:
        fits.PrimaryHDU((data - model).astype('float32')).writeto(resid_path)
    return resid_path


def _runs(pdir):
    return C._satstar_cofit_runs(str(pdir), proposal_id='10678', field='001',
                                 phase='m7')


def _merged(xy, flux, support):
    return Table({'skycoord': _sky(xy), 'flux': np.asarray(flux, float),
                  'satstar_nframes': np.asarray(support, float),
                  'replaced_saturated': np.ones(len(xy), bool),
                  'rowid': np.arange(len(xy))})


def _clean(tbl, tmp_path, runs, residual_mode=True):
    out, ndup, _ = C._clean_offfov_dups_and_offfield(
        tbl.copy(), 'F480M', _data_i2d(tmp_path), str(tmp_path),
        cofit_positions=[r['positions'] for r in runs],
        cofit_runs=(runs if residual_mode else None))
    return sorted(np.asarray(out['rowid']).tolist()), ndup


# K: a bright star fit in exposures 1 and 3; D: a second star 6 px (0.38")
# away that shares K's DQ component, so the finder fits it alone in
# exposure 2 and never beside K.  D ranks lower (support 1).
K = (150.0, 150.0, 2.0e5)
D = (156.0, 150.0, 6.0e4)


def _pair_scene(pdir, d_true_flux=D[2], residual=True):
    stars = [K, (D[0], D[1], d_true_flux)]
    _write_exposure(pdir, 1, stars, [K], residual=residual, seed=1)
    _write_exposure(pdir, 2, stars, [D], residual=residual, seed=2)
    _write_exposure(pdir, 3, stars, [K], residual=residual, seed=3)


def test_real_pair_is_kept_as_two_stars(tmp_path, clean_env, capsys):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _pair_scene(pdir)
    runs = _runs(pdir)
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs, residual_mode=False) == ([0], 1)
    capsys.readouterr()
    assert _clean(tbl, tmp_path, runs) == ([0, 1], 0)
    log = capsys.readouterr().out
    assert 'kept as two stars:' in log
    assert '1 never-co-fit pair(s) tested: 1 kept as two stars' in log


def test_peak_without_the_flux_collapses_and_the_threshold_is_env(tmp_path,
                                                                  clean_env):
    # The F212N edge-only case: there IS light at D in the frames that fit
    # only K (peak ratio ~1, K's model is negligible 6 px away), but only 4%
    # of D's catalogued flux -- not the star the D row claims.
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _pair_scene(pdir, d_true_flux=0.04 * D[2])
    runs = _runs(pdir)
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    dist, stats = C._satstar_residual_distinct_pairs(
        _sky([K[:2], D[:2]]), [2, 1], [K[2], D[2]], runs)
    assert dist([0], [1], [0.378]).tolist() == [False]
    assert stats['explained'] == 1 and stats['distinct'] == 0
    assert _clean(tbl, tmp_path, runs) == ([0], 1)
    # a looser flux threshold from the environment keeps it
    clean_env.setenv('SATSTAR_INFIELD_RESID_FLUX_MIN', '0.01')
    assert _clean(tbl, tmp_path, runs) == ([0, 1], 0)
    # and a peak threshold above 1 collapses it again
    clean_env.setenv('SATSTAR_INFIELD_RESID_RATIO_MIN', '1.5')
    assert _clean(tbl, tmp_path, runs) == ([0], 1)


def test_shoulder_fit_on_a_bright_star_collapses(tmp_path, clean_env):
    # One bright star S.  Exposure 2 fits a shoulder row W 4 px out on S's
    # wing (17% of S's flux) and not S; exposures 1 and 3 fit S alone and
    # subtract it cleanly, so nothing is left at W.
    S = (150.0, 150.0, 4.0e5)
    W = (154.0, 150.0, 0.17 * 4.0e5)
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _write_exposure(pdir, 1, [S], [S], seed=1)
    _write_exposure(pdir, 2, [S], [W], seed=2)
    _write_exposure(pdir, 3, [S], [S], seed=3)
    runs = _runs(pdir)
    tbl = _merged([S[:2], W[:2]], [S[2], W[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0], 1)


def test_edge_duplicate_keeps_the_interior_member(tmp_path, clean_env, capsys):
    # A star E at reference x=150 is fit in exposure 1 (40+ px inside).
    # Exposures 2 and 3 are shifted so the star falls off the left edge;
    # their truncated fits T land 0.4" (6.3 px) inside, at frame x = 5,
    # with more support and a larger flux.  'cofit' keeps T; the edge
    # preference keeps E.  E projects 11 px inside exposures 2/3 (< 20 px),
    # so the residual test cannot separate them and the pair collapses.
    E = (150.0, 150.0, 1.0e5)
    T = (156.3, 150.0, 1.3e5)
    shift = (5.0 - 156.3, 0.0)                # T at frame x = 5
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _write_exposure(pdir, 1, [E], [E], seed=1)
    _write_exposure(pdir, 2, [E], [T], shift=shift, seed=2)
    _write_exposure(pdir, 3, [E], [T], shift=shift, seed=3)
    runs = _runs(pdir)
    edge = C._satstar_edge_only_rows(_sky([E[:2], T[:2]]), runs)
    assert edge.tolist() == [False, True]
    tbl = _merged([E[:2], T[:2]], [E[2], T[2]], [1, 2])
    assert _clean(tbl, tmp_path, runs, residual_mode=False) == ([1], 1)
    assert _clean(tbl, tmp_path, runs) == ([0], 1)
    assert '1 of 2 in-field row(s) are edge-only' in capsys.readouterr().out


def test_missing_residual_falls_back_to_cofit(tmp_path, clean_env, capsys):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _pair_scene(pdir, residual=False)
    runs = _runs(pdir)
    assert all(r['residual'] and not os.path.exists(r['residual'])
               for r in runs)
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0], 1)          # as 'cofit'
    log = capsys.readouterr().out
    assert "fall back to 'cofit'" in log and 'not on disk' in log
    assert '1 fell back to' in log


def test_unreadable_residual_falls_back_to_cofit(tmp_path, clean_env, capsys):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _pair_scene(pdir)
    runs = _runs(pdir)
    for r in runs:
        with open(r['residual'], 'wb') as fh:
            fh.write(b'not a fits file' * 10)
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0], 1)
    log = capsys.readouterr().out
    assert "fall back to 'cofit'" in log and '1 fell back to' in log


def test_edge_test_reads_frame_pixels_not_stamp_coordinates(tmp_path,
                                                            clean_env):
    # x_fit/y_fit = 7 (stamp middle) would put every fit 3 px from the edge
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _pair_scene(pdir)
    runs = _runs(pdir)
    for r in runs:
        assert np.all(r['xy'] > 100)                       # frame pixels
    assert not C._satstar_edge_only_rows(_sky([K[:2], D[:2]]), runs).any()


def test_runs_carry_frame_residual_and_positions(tmp_path, clean_env):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _write_exposure(pdir, 1, [K, D], [K], fits_rej=[D[:2]], seed=1)
    runs = _runs(pdir)
    assert len(runs) == 1
    run = runs[0]
    assert len(run['positions']) == 2 and len(run['accepted']) == 1
    assert run['shape'] == SHAPE and run['frame'].endswith('_crf.fits')
    assert run['residual'].endswith('_m7_satstar_residual.fits')
    # the 'cofit' reader returns the same positions and skips frame headers
    pos = C._satstar_cofit_positions(str(pdir), proposal_id='10678',
                                     field='001', phase='m7')
    assert len(pos) == 1
    assert np.allclose(pos[0].ra.deg, run['positions'].ra.deg)


def test_residual_left_on_a_gaussian():
    data = BG + _gauss((61, 61), 30.3, 29.8, 5e4)
    pk, ff = C._satstar_residual_left(data, data, 30.3, 29.8, 5e4)
    assert pk == pytest.approx(1.0)
    assert 0.9 < ff < 1.0                    # r <= 3 px of a sigma=1.2 px PSF
    pk, ff = C._satstar_residual_left(np.full((61, 61), BG), data, 30.3, 29.8,
                                      5e4)
    assert pk == pytest.approx(0.0) and ff == pytest.approx(0.0)
    assert np.isnan(C._satstar_residual_left(data, data, 30.3, 29.8, 0.0)[1])


def test_hooks_of_never_cofit_duplicates():
    sc = _sky([K[:2], D[:2]])
    runs = [_sky([K[:2]]), _sky([D[:2]])]
    assert C._never_cofit_duplicates(sc, [2, 1], [1, 1], runs).tolist() == [
        False, True]
    seen = []

    def apart(i1, i2, sep):
        seen.append((list(i1), list(i2)))
        return np.ones(len(i1), bool)
    assert C._never_cofit_duplicates(sc, [2, 1], [1, 1], runs,
                                     distinct_pairs=apart).tolist() == [
        False, False]
    assert seen == [([0], [1])]
    # demote changes only the survivor
    assert C._never_cofit_duplicates(sc, [2, 1], [1, 1], runs,
                                     demote=[True, False]).tolist() == [
        True, False]


# ---- damaged residual files -------------------------------------------------

def _truncate(path, frac):
    raw = open(path, 'rb').read()
    with open(path, 'wb') as fh:
        fh.write(raw[:int(len(raw) * frac)])


@pytest.mark.parametrize('damage', ['truncated_half', 'truncated_to_90pct',
                                    'one_d', 'wrong_2d_shape', 'both_one_d',
                                    'frame_truncated'])
def test_damaged_residual_falls_back_to_cofit(tmp_path, clean_env, capsys,
                                              damage):
    # A file cut off inside its data raises TypeError ("buffer is too small
    # for requested array") under memmap, and a 1-D residual would
    # IndexError in the cutout: both must skip the run, not abort.
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _pair_scene(pdir)
    runs = _runs(pdir)
    for r in runs:
        if damage == 'truncated_half':
            _truncate(r['residual'], 0.5)
        elif damage == 'truncated_to_90pct':
            _truncate(r['residual'], 0.9)
        elif damage == 'one_d':
            os.remove(r['residual'])
            fits.PrimaryHDU(np.full(SHAPE[0] * SHAPE[1], BG, 'float32')
                            ).writeto(r['residual'])
        elif damage == 'wrong_2d_shape':
            os.remove(r['residual'])
            fits.PrimaryHDU(np.full((200, 200), BG, 'float32')
                            ).writeto(r['residual'])
        elif damage == 'both_one_d':                # same shape, not 2-D
            flat = np.full(SHAPE[0] * SHAPE[1], BG, 'float32')
            os.remove(r['residual'])
            fits.PrimaryHDU(flat).writeto(r['residual'])
            os.remove(r['frame'])
            fits.HDUList([fits.PrimaryHDU(),
                          fits.ImageHDU(flat, name='SCI')]).writeto(r['frame'])
        else:
            _truncate(r['frame'], 0.6)
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0], 1)          # as 'cofit'
    log = capsys.readouterr().out
    assert "fall back to 'cofit'" in log and '1 fell back to' in log
    if damage.startswith('truncated') or damage == 'frame_truncated':
        assert 'TypeError' in log
    elif damage in ('one_d', 'wrong_2d_shape', 'both_one_d'):
        assert 'is not the frame SCI shape' in log


# ---- the 20 px inside cutoff ------------------------------------------------

@pytest.mark.parametrize('edge', ['left', 'right', 'bottom', 'top'])
@pytest.mark.parametrize('inset, kept', [(18, False), (22, True)])
def test_lower_member_is_measured_only_20px_inside(tmp_path, clean_env, edge,
                                                   inset, kept):
    # D sits ``inset`` px from one array edge (left: x = inset; right:
    # x = nx - inset), K 6 px further in.  Every frame has the same WCS
    # (frame = reference pixels) and is NOT square, (ny, nx) = (300, 400),
    # so x and y limits differ.  At 22 px the residual test measures D and
    # keeps it; at 18 px it is not measured and 'cofit' collapses the pair.
    shape = (300, 400)
    ny, nx = shape
    inward = {'left': (1, 0), 'right': (-1, 0), 'bottom': (0, 1),
              'top': (0, -1)}[edge]
    dxy = {'left': (inset, 150.0), 'right': (nx - inset, 150.0),
           'bottom': (200.0, inset), 'top': (200.0, ny - inset)}[edge]
    d = (float(dxy[0]), float(dxy[1]), D[2])
    k = (d[0] + 6 * inward[0], d[1] + 6 * inward[1], K[2])
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    _write_exposure(pdir, 1, [k, d], [k], seed=1, shape=shape)
    _write_exposure(pdir, 2, [k, d], [d], seed=2, shape=shape)
    _write_exposure(pdir, 3, [k, d], [k], seed=3, shape=shape)
    runs = _runs(pdir)
    assert all(r['shape'] == shape for r in runs)
    assert not C._satstar_edge_only_rows(_sky([k[:2], d[:2]]), runs).any()
    tbl = _merged([k[:2], d[:2]], [k[2], d[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == (([0, 1], 0) if kept else ([0], 1))
    dist, stats = C._satstar_residual_distinct_pairs(
        _sky([k[:2], d[:2]]), [2, 1], [k[2], d[2]], runs)
    assert dist(np.array([0]), np.array([1]), np.array([0.378])).tolist() == [
        kept]
    assert stats['untestable'] == (0 if kept else 1)


# ---- edge-only rows: the 10 px + 4 reference-pixel boundary -----------------

def test_edge_only_boundary_is_14px_from_the_array_edge():
    # A non-square frame (ny, nx) = (300, 400) so that x and y sides differ.
    # Edge distance = min(x, y, nx-1-x, ny-1-y) - 4 <= 10: x or y <= 14, or
    # x >= 385 / y >= 285.
    shape = (300, 400)
    xy = np.array([[14.0, 150], [15.0, 150],        # left
                   [385.0, 150], [384.0, 150],      # right (nx - 15)
                   [200, 14.0], [200, 15.0],        # bottom
                   [200, 285.0], [200, 284.0],      # top (ny - 15)
                   [10.0, 150],                     # + an interior fit
                   [200, 150]])                     # interior
    expect = [True, False, True, False, True, False, True, False, False,
              False]
    sc = _sky([(20 + 25 * i, 60) for i in range(len(xy))])
    runs = [{'positions': sc, 'xy': xy, 'shape': shape},
            {'positions': sc[8:9], 'xy': np.array([[100.0, 150]]),
             'shape': shape}]
    assert C._satstar_edge_only_rows(sc, runs).tolist() == expect
    # a row with no fit at all is not edge-only
    far = _sky([(380, 380)])
    assert C._satstar_edge_only_rows(far, runs).tolist() == [False]


# ---- residual_left sign, and per-run combination of the two tests -----------

def _gauss_s(shape, x, y, flux, sigma):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (flux / (2 * np.pi * sigma ** 2)
            * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))


#: D's peak above the background in the data (K's wing 6 px away is ~0)
D_PEAK = D[2] / (2 * np.pi * SIGMA ** 2)


def _residual(kind, rng_seed=7):
    """A K-only-run residual with something specific left at D."""
    rng = np.random.default_rng(rng_seed)
    res = np.full(SHAPE, BG) + rng.normal(0, 0.5, SHAPE)
    if kind == 'star':                               # all of D: both pass
        res += _gauss(SHAPE, D[0], D[1], D[2])
    elif kind == 'spike':                            # peak 0.9, flux ~0.15
        res += _gauss_s(SHAPE, D[0], D[1],
                        0.9 * D_PEAK * 2 * np.pi * 0.5 ** 2, 0.5)
    elif kind == 'bump':                             # peak ~0.2, flux ~0.46
        res += _gauss_s(SHAPE, D[0], D[1], 0.9 * D[2], 2.5)
    elif kind == 'spike_and_dip':                    # peak ~0.58, flux < 0
        res += _gauss_s(SHAPE, D[0], D[1],
                        0.9 * D_PEAK * 2 * np.pi * 0.5 ** 2, 0.5)
        res -= _gauss_s(SHAPE, D[0], D[1], 0.9 * D[2], 2.0)
    elif kind == 'nan':
        res[int(D[1]) - 20:int(D[1]) + 21, int(D[0]) - 20:int(D[0]) + 21] = (
            np.nan)
    elif kind == 'inf':                              # star + one inf pixel
        res += _gauss(SHAPE, D[0], D[1], D[2])
        res[int(D[1]), int(D[0])] = np.inf
    else:
        raise ValueError(kind)
    return res.astype('float32')


def _set_residual(run, kind):
    os.remove(run['residual'])
    fits.PrimaryHDU(_residual(kind)).writeto(run['residual'])


def _nan_frame_at_d(run):
    with fits.open(run['frame'], mode='update') as fh:
        fh['SCI'].data[int(D[1]) - 1:int(D[1]) + 2,
                       int(D[0]) - 1:int(D[0]) + 2] = np.nan


def _k_only_runs(pdir, kinds):
    """The pair scene with the residual of exposures 1 and 3 (which fit only
    K) replaced by ``kinds[0]`` / ``kinds[1]``; returns (all runs, [run 1,
    run 3])."""
    _pair_scene(pdir)
    runs = _runs(pdir)
    konly = [r for r in runs if '_00002_' not in r['stem']]
    assert len(konly) == 2 and '_00001_' in konly[0]['stem']
    for r, kind in zip(konly, kinds):
        _set_residual(r, kind)
    return runs, konly


def _distinct(runs):
    dist, stats = C._satstar_residual_distinct_pairs(
        _sky([K[:2], D[:2]]), [2, 1], [K[2], D[2]], runs)
    out = dist(np.array([0]), np.array([1]), np.array([0.378]))
    return bool(out[0]), stats


@pytest.mark.parametrize('kind, peak_ok, flux_ok', [
    ('star', True, True), ('spike', True, False), ('bump', False, True),
    ('spike_and_dip', True, False)])
def test_residual_kinds_measure_as_designed(tmp_path, kind, peak_ok,
                                            flux_ok):
    data = (np.full(SHAPE, BG) + _gauss(SHAPE, *K) + _gauss(SHAPE, *D))
    pk, ff = C._satstar_residual_left(_residual(kind), data, D[0], D[1], D[2])
    assert (pk >= 0.5) == peak_ok and (ff >= 0.25) == flux_ok
    if kind == 'spike_and_dip':
        # light left at the peak, but a net DEFICIT in the aperture: the
        # flux fraction is signed, not an absolute value
        assert 0.5 < pk < 0.7 and ff < -0.3


def test_spike_over_a_dip_collapses(tmp_path, clean_env):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    runs, _ = _k_only_runs(pdir, ['spike_and_dip', 'spike_and_dip'])
    kept, stats = _distinct(runs)
    assert not kept and stats['explained'] == 1
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0], 1)


@pytest.mark.parametrize('kinds', [('nan', 'star'), ('star', 'nan')])
def test_one_nan_run_does_not_hide_a_passing_run(tmp_path, clean_env, kinds):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    runs, _ = _k_only_runs(pdir, kinds)
    kept, stats = _distinct(runs)
    assert kept and stats['distinct'] == 1
    # the run order does not matter
    assert _distinct(runs[::-1])[0]
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0, 1], 0)


def test_all_nan_runs_are_untestable_and_collapse(tmp_path, clean_env):
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    runs, _ = _k_only_runs(pdir, ['nan', 'nan'])
    kept, stats = _distinct(runs)
    assert not kept
    assert (stats['untestable'], stats['explained'], stats['fallback']) == (
        1, 0, 0)
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0], 1)


@pytest.mark.parametrize('kinds', [('bump', 'spike'), ('spike', 'bump'),
                                   ('nanpeak', 'spike'), ('spike', 'nanpeak'),
                                   ('inf', 'inf')])
def test_both_tests_must_pass_in_the_same_run(tmp_path, clean_env, kinds):
    # bump: flux passes, peak fails.  spike: peak passes, flux fails.
    # nanpeak: the frame's SCI is NaN at D (peak fraction NaN) while the
    # residual holds all of D (flux passes).  inf: all of D plus one inf
    # pixel.  No single run passes both, so the pair collapses -- although
    # the best peak and the best flux, taken over runs separately, pass.
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    runs, konly = _k_only_runs(
        pdir, ['star' if k == 'nanpeak' else k for k in kinds])
    for r, k in zip(konly, kinds):
        if k == 'nanpeak':
            _nan_frame_at_d(r)
    kept, stats = _distinct(runs)
    # an inf in the residual is no measurement at all
    counted = 'untestable' if kinds == ('inf', 'inf') else 'explained'
    assert not kept and stats[counted] == 1 and stats['distinct'] == 0
    assert _distinct(runs[::-1])[0] is False
    tbl = _merged([K[:2], D[:2]], [K[2], D[2]], [2, 1])
    assert _clean(tbl, tmp_path, runs) == ([0], 1)


def test_a_run_with_only_a_flux_measurement_counts_as_measured(tmp_path,
                                                                clean_env):
    # exposure 1 alone: NaN peak (frame NaN at D), finite flux -> the pair
    # is measured ('explained'), not untestable
    pdir = tmp_path / 'pipeline'
    pdir.mkdir()
    runs, konly = _k_only_runs(pdir, ['star', 'star'])
    _nan_frame_at_d(konly[0])
    one = [r for r in runs if '_00003_' not in r['stem']]
    kept, stats = _distinct(one)
    assert not kept
    assert (stats['explained'], stats['untestable']) == (1, 0)


# ---- settings and wiring ----------------------------------------------------

@pytest.mark.parametrize('value', ['cofit_residual', ' COFIT_RESIDUAL\n'])
def test_settings_parse_cofit_residual(clean_env, value):
    clean_env.setenv('SATSTAR_INFIELD_DEDUP', value)
    assert C._infield_dedup_settings('gc-treasury', 'merged', 'F480M') == (
        'cofit_residual', 'SATSTAR_INFIELD_DEDUP', True, True)


def test_default_mode_is_still_cofit(clean_env):
    assert C._infield_dedup_settings('gc-treasury', 'merged', 'F212N')[0] == (
        'cofit')
    assert C._validate_infield_dedup_env()['SATSTAR_INFIELD_DEDUP'] == 'cofit'


@pytest.mark.parametrize('target', C._EXTENDED_EMISSION_TARGETS)
def test_extended_targets_ignore_cofit_residual(clean_env, target):
    clean_env.setenv('SATSTAR_INFIELD_DEDUP', 'cofit_residual')
    _, _, offfov_only, use_cofit = C._infield_dedup_settings(
        target, 'merged', 'F480M')
    assert offfov_only is True and use_cofit is False


def test_thresholds_default_and_env(clean_env):
    assert C._resid_dedup_thresholds() == (0.5, 0.25)
    assert (C._RESID_DEDUP_RATIO_MIN, C._RESID_DEDUP_FLUX_MIN) == (0.5, 0.25)
    clean_env.setenv('SATSTAR_INFIELD_RESID_RATIO_MIN', ' 0.6 ')
    clean_env.setenv('SATSTAR_INFIELD_RESID_FLUX_MIN', '')
    assert C._resid_dedup_thresholds() == (0.6, 0.25)


@pytest.mark.parametrize('env', ['SATSTAR_INFIELD_RESID_RATIO_MIN',
                                 'SATSTAR_INFIELD_RESID_FLUX_MIN'])
@pytest.mark.parametrize('value', ['half', 'nan', 'inf'])
def test_bad_threshold_raises_naming_the_variable(clean_env, env, value):
    clean_env.setenv(env, value)
    with pytest.raises(ValueError, match=rf'^{env}='):
        C._resid_dedup_thresholds()


class _ReachedFirstStep(Exception):
    pass


@pytest.mark.parametrize('env', ['SATSTAR_INFIELD_RESID_RATIO_MIN',
                                 'SATSTAR_INFIELD_RESID_FLUX_MIN'])
def test_bad_threshold_stops_run_manual_pipeline_at_entry(clean_env, env):
    reached = []

    def _first_step(*args, **kwargs):
        reached.append(True)
        raise _ReachedFirstStep

    clean_env.setattr(C._L, '_cutout_out_basepath', _first_step)
    args = (types.SimpleNamespace(), ['nrcb'], ['F480M'],
            {'10678': {'gc-treasury': 1}}, '10678', 'gc-treasury', '111',
            '/nonexistent-basepath', {}, {})
    clean_env.setenv(env, '0.3')
    with pytest.raises(_ReachedFirstStep):
        C.run_manual_pipeline(*args)
    reached.clear()
    clean_env.setenv(env, 'x')
    with pytest.raises(ValueError, match=rf'^{env}='):
        C.run_manual_pipeline(*args)
    assert reached == []


def test_run_manual_pipeline_passes_the_runs_only_in_cofit_residual():
    src = inspect.getsource(C.run_manual_pipeline)
    flat = ' '.join(src.split())
    assert "if _want_cofit: if _infield_mode == 'cofit_residual':" in flat
    assert ('_cofit_runs = _satstar_cofit_runs( os.path.join(cut_bp, filt, '
            "'pipeline'), proposal_id=proposal_id, field=field, "
            'phase=merge_label)') in flat
    assert "_cofit = [_r['positions'] for _r in _cofit_runs]" in flat
    start = src.index('_clean_offfov_dups_and_offfield(')
    call = ' '.join(src[start:src.index(')', src.index('cofit_runs=', start))
                        + 1].split())
    assert 'cofit_positions=_cofit, cofit_runs=_cofit_runs)' in call
