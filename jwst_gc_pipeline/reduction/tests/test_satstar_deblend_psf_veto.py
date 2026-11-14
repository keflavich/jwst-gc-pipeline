"""The ZEROFRAME deblend vetoes secondary peaks that a PSF model predicts (#1195).

One bright star puts local maxima on its PSF rings and diffraction spikes.
Where a catalogue source sits on such a bead, or the bead passes the
compactness test, ``deblend_blob_zeroframe`` read it as a companion.  With a
``psf_stamp`` it fits the accepted stars with PSF stamps and keeps only the
peaks of the residual that exceed ``psf_margin`` times the model there.

The PSF here is analytic (an Airy pattern plus beaded axis spikes), so the
tests need no stpsf data.  The model PSF equals the true one except in the
mismatch tests, which scale the true spike amplitude.
"""
import numpy as np
import pytest
from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel
from scipy import ndimage, special

from jwst_gc_pipeline.reduction import satstar_deblend as sd
from jwst_gc_pipeline.reduction.saturated_star_finding import deblend_psf_stamp

SHAPE = (101, 101)
FWHM = 2.2          # px
CEIL = 1.0e4        # ZEROFRAME clip level
SAT_DQ_FRAC = 0.02  # the DQ-saturated component reaches down to this x CEIL


def _psf(yy, xx, yc, xc, spike=1.0):
    dx, dy = xx - xc, yy - yc
    r = np.hypot(dx, dy) + 1e-6
    x = np.pi * r * 1.029 / FWHM
    airy = (2 * special.j1(x) / x) ** 2
    d = np.minimum(np.abs(dx), np.abs(dy))
    beads = (spike * 3e-3 * np.exp(-d ** 2 / (2 * 0.7 ** 2))
             * (1 + np.cos(2 * np.pi * r / 2.1)) / (1 + r / 6.0))
    return airy + beads


def _model(yy, xx, yc, xc):
    return _psf(yy, xx, yc, xc)


def _frame(stars, seed, spike=1.0, noise=5.0):
    """ZEROFRAME clipped at CEIL, and the DQ-saturated component holding the
    first star.  ``stars`` is a list of (y, x, flux in units of CEIL)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    zfu = 100.0 + rng.normal(0, noise, SHAPE)
    for (y, x, f) in stars:
        zfu += f * CEIL * _psf(yy, xx, y, x, spike)
    zf = np.minimum(zfu, CEIL)
    sat = ndimage.binary_fill_holes(zfu >= SAT_DQ_FRAC * CEIL)
    lab, _ = ndimage.label(sat)
    sat = lab == lab[50, 50]
    sources, _ = ndimage.label(sat)
    return zf, sources


def _deblend(zf, sources, **kw):
    sl = ndimage.find_objects(sources)[0]
    return sd.deblend_blob_zeroframe(
        zf, np.zeros(SHAPE), sources, 1, sl, FWHM, sat_ceiling=0.98 * CEIL, **kw)


def _single(nsat, seed):
    rng = np.random.default_rng(1000 + seed)
    return (50 + rng.uniform(-.5, .5), 50 + rng.uniform(-.5, .5), nsat)


def _bead_xy(star):
    """Catalogue positions (x, y) on the spike beads, as a detection there
    would give."""
    y, x, _ = star
    return np.array([(x + dx * 2.1 * k, y + dy * 2.1 * k)
                     for k in range(3, 7)
                     for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))])


SEEDS = range(8)


def test_beads_split_a_single_star_without_the_veto():
    """Guard on the scene: the plain peak search splits a bright star here."""
    n = []
    for s in SEEDS:
        star = _single(400, s)
        zf, src = _frame([star], s)
        n.append(len(_deblend(zf, src, confirm_xy=_bead_xy(star))[0]))
    assert np.mean(np.array(n) > 1) > 0.5


@pytest.mark.parametrize('nsat', [3, 30, 100, 400])
def test_single_star_stays_one_star(nsat):
    for s in SEEDS:
        star = _single(nsat, s)
        zf, src = _frame([star], s)
        centers, info = _deblend(zf, src, confirm_xy=_bead_xy(star),
                                 psf_stamp=_model)
        assert info['psf_veto'] == 'applied'
        assert len(centers) == 1, (nsat, s, centers)


@pytest.mark.parametrize('spike', [0.8, 1.2])
def test_single_star_with_a_20_percent_spike_mismatch(spike):
    """The truth differs from the model by 20% in spike amplitude; the 25%
    margin absorbs it."""
    for s in SEEDS:
        star = _single(400, s)
        zf, src = _frame([star], s, spike=spike)
        centers, _ = _deblend(zf, src, confirm_xy=_bead_xy(star),
                              psf_stamp=_model)
        assert len(centers) == 1, (spike, s, centers)


@pytest.mark.parametrize('ratio', [0.3, 0.1, 0.03])
@pytest.mark.parametrize('confirmed', [True, False])
def test_companion_is_kept(ratio, confirmed):
    """A saturated star with a fainter star 8 px away: both are recovered
    down to a flux ratio of 0.03, with or without a catalogue source."""
    for s in SEEDS:
        rng = np.random.default_rng(s)
        ang = rng.uniform(0, 2 * np.pi)
        a = (50.0, 50.0, 30)
        b = (50 + 8 * np.sin(ang), 50 + 8 * np.cos(ang), ratio * 30)
        zf, src = _frame([a, b], s)
        conf = np.array([[b[1], b[0]]]) if confirmed else None
        centers, info = _deblend(zf, src, confirm_xy=conf, psf_stamp=_model)
        assert info['psf_veto'] == 'applied'
        assert any(np.hypot(c[0] - b[0], c[1] - b[1]) < 1.5 for c in centers), \
            (ratio, s, centers)


def test_no_stamp_or_no_margin_keeps_the_plain_search():
    star = _single(400, 0)
    zf, src = _frame([star], 0)
    base, binfo = _deblend(zf, src, confirm_xy=_bead_xy(star))
    assert binfo['psf_veto'] == 'off'
    for kw in (dict(psf_stamp=None), dict(psf_stamp=_model, psf_margin=0.0)):
        centers, info = _deblend(zf, src, confirm_xy=_bead_xy(star), **kw)
        assert info['psf_veto'] == 'off'
        assert centers == base


def test_giant_component_falls_back(monkeypatch):
    star = _single(400, 0)
    zf, src = _frame([star], 0)
    base, _ = _deblend(zf, src, confirm_xy=_bead_xy(star))
    monkeypatch.setattr(sd, 'PSF_VETO_MAX_AREA', 10)
    centers, info = _deblend(zf, src, confirm_xy=_bead_xy(star), psf_stamp=_model)
    assert info['psf_veto'] == 'fallback_area'
    assert centers == base


def test_too_few_fit_pixels_falls_back(monkeypatch):
    star = _single(400, 0)
    zf, src = _frame([star], 0)
    base, _ = _deblend(zf, src, confirm_xy=_bead_xy(star))
    monkeypatch.setattr(sd, 'PSF_VETO_MIN_FIT_PX', 10 ** 6)
    centers, info = _deblend(zf, src, confirm_xy=_bead_xy(star), psf_stamp=_model)
    assert info['psf_veto'] == 'fallback_few_px'
    assert centers == base


def test_failed_fit_falls_back(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError('too many iterations')
    star = _single(400, 0)
    zf, src = _frame([star], 0)
    base, _ = _deblend(zf, src, confirm_xy=_bead_xy(star))
    monkeypatch.setattr(sd, 'nnls', _boom)
    centers, info = _deblend(zf, src, confirm_xy=_bead_xy(star), psf_stamp=_model)
    assert info['psf_veto'] == 'fallback_fit_failed'
    assert centers == base


# --- build_deblended_source_records: failure handling and stats ----------

def _records(zf, src, **kw):
    sl = ndimage.find_objects(src)[0]
    area = int((src == 1).sum())
    com = tuple(float(c) for c in ndimage.center_of_mass(src == 1))
    return sd.build_deblended_source_records(
        src > 0, src, [com], [area], zf, np.zeros(SHAPE), FWHM, **kw), com


def test_failure_is_logged_and_counted(monkeypatch, capsys):
    def _boom(*a, **k):
        raise ValueError('bad crop')
    star = _single(30, 0)
    zf, src = _frame([star], 0)
    monkeypatch.setattr(sd, 'deblend_blob_zeroframe', _boom)
    stats = {}
    recs, com = _records(zf, src, stats=stats)
    assert [r['com'] for r in recs] == [com]
    assert stats['n_failed'] == 1
    assert stats['first_error'] == 'ValueError: bad crop'
    assert 'ValueError: bad crop' in capsys.readouterr().out


def test_stats_count_components_and_splits(monkeypatch):
    """``n_split`` counts the components that returned more than one star."""
    star = _single(30, 0)
    zf, src = _frame([star], 0)
    monkeypatch.setattr(sd, 'deblend_blob_zeroframe',
                        lambda *a, **k: ([(30.0, 30.0), (30.0, 40.0)],
                                         dict(psf_veto='off')))
    stats = {}
    recs, _ = _records(zf, src, stats=stats)
    assert len(recs) == 2
    assert (stats['n_components'], stats['n_split']) == (1, 1)
    assert 'n_failed' not in stats
    monkeypatch.setattr(sd, 'deblend_blob_zeroframe',
                        lambda *a, **k: ([(30.0, 30.0)], dict(psf_veto='off')))
    stats = {}
    _records(zf, src, stats=stats)
    assert (stats['n_components'], stats.get('n_split', 0)) == (1, 0)


def test_missing_skimage_raises(monkeypatch):
    """Without scikit-image the secondary peak search is skipped inside
    ``deblend_blob_zeroframe``; the entry point refuses instead of returning
    unsplit centroids."""
    star = _single(30, 0)
    zf, src = _frame([star], 0)
    monkeypatch.setattr(sd, '_HAVE_SKIMAGE', False)
    with pytest.raises(ImportError, match='scikit-image'):
        _records(zf, src)


def test_unexpected_error_propagates(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError('unrelated bug')
    star = _single(30, 0)
    zf, src = _frame([star], 0)
    monkeypatch.setattr(sd, 'deblend_blob_zeroframe', _boom)
    with pytest.raises(RuntimeError, match='unrelated bug'):
        _records(zf, src)


def test_stats_count_veto_status():
    star = _single(30, 0)
    zf, src = _frame([star], 0)
    stats = {}
    _records(zf, src, stats=stats, psf_stamp=_model)
    assert stats['psf_veto'] == {'applied': 1}


# --- the production PSF stamp -------------------------------------------

def _grid(oversampling=2, half=30):
    n = 2 * half * oversampling + 1
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * (1.0 * oversampling) ** 2))
    data = np.array([g / g.sum() * oversampling ** 2] * 4)
    xypos = [(0.5, 0.5), (2047.5, 0.5), (0.5, 2047.5), (2047.5, 2047.5)]
    return GriddedPSFModel(NDData(data, meta={'grid_xypos': xypos,
                                              'oversampling': oversampling}))


def test_deblend_psf_stamp_from_a_gridded_psf():
    stamp = deblend_psf_stamp(_grid())
    yy, xx = np.mgrid[1000:1021, 500:521].astype(float)
    img = stamp(yy, xx, 1010.3, 510.6)
    assert img.shape == yy.shape and np.all(np.isfinite(img))
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    assert (1000 + iy, 500 + ix) == (1010, 511)
    # evaluated at arbitrary-shaped pixel arrays too
    assert stamp(yy.ravel(), xx.ravel(), 1010.3, 510.6).shape == (yy.size,)


def test_deblend_psf_stamp_needs_a_gridded_psf():
    assert deblend_psf_stamp(object()) is None
