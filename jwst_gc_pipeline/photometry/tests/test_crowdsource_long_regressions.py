"""Regression tests for the long-wavelength cataloging code
(crowdsource_catalogs_long.py).

Each test pins a specific bug / defensive path documented by a comment block
in the module.  These complement the existing
``test_seed_skycoord_resolution.py`` (which already covers the Star B masked
(0,0) sentinel and SeededFinder silent-drop logging).

Covered regressions
--------------------
* ``overlap_slices`` monkeypatch (crowdsource_catalogs_long.py:54-91).
  photutils.make_model_image passes ``small_array_shape`` as an ndarray;
  an out-of-frame source with ``e_max == 0`` then triggers
  ``ValueError: truth value of an array ... is ambiguous`` inside
  astropy's overlap_slices.  The patch coerces the shape to a tuple.

* ``_strip_chunk`` (crowdsource_catalogs_long.py:693-707).  ``_chunkXXofYY``
  suffixes must be stripped so semantic checks like ``is_iter3`` still fire
  when image-chunking is on.

* ``_resolve_seed_skycoords`` per-filter snap precedence
  (crowdsource_catalogs_long.py:1180-1188).  When a ``preferred_skycoord_col``
  (e.g. ``skycoord_f480m``) is requested it must win over plain ra/dec
  columns; otherwise the iter3 per-filter seed snap is silently ignored
  (sickle F480M source 55, 2026-06-03).

* ``_filter_near_saturation`` Star B regression
  (crowdsource_catalogs_long.py:1456-1507 + threshold note ~4657).  The
  saturation-proximity radius was cut 5.0 -> 1.0 px so that a real star
  ~2.24 px from a saturated neighbour survives while a fit whose centre
  sits on the saturated pixel is still dropped.

* ``forced_psf_photometry`` closed-form flux solve
  (crowdsource_catalogs_long.py:199-296).  The linear least-squares flux
  must equal the injected flux for a noise-free single source.
"""
import types

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u

from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L


def _make_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [50.5, 50.5]
    w.wcs.crval = [266.55, -28.80]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    w.wcs.cunit = ['deg', 'deg']
    return w


# ---------------------------------------------------------------------------
# overlap_slices monkeypatch
# ---------------------------------------------------------------------------
class TestOverlapSlicesPatch:
    # A (15,15) cutout centred at pixel -7.63 makes astropy's internal
    # ``indices_max`` land on exactly 0 (round(-7.63 - 7.5) + 15 =
    # round(-15.13) + 15 = -15 + 15 = 0), which hits the branch
    # ``if e_max == 0 and small_array_shape != (0, 0)``.  When
    # ``small_array_shape`` is an ndarray that comparison returns an array
    # and the surrounding ``and`` raises the ambiguous-truth ValueError.
    LARGE = (100, 100)
    SMALL_NDARRAY = np.array([15, 15])     # ndarray, as photutils passes it
    POSITION = (-7.63, -7.63)

    def test_patch_is_installed(self):
        import astropy.nddata.utils as ndu
        import photutils.utils.cutouts as pc
        assert ndu.overlap_slices is L._overlap_slices_tuple_shape
        assert pc.overlap_slices is L._overlap_slices_tuple_shape

    def test_unpatched_raises_ambiguous_truth_value(self):
        # Demonstrate the original bug: ndarray small_array_shape + e_max==0
        # raises "truth value of an array ... is ambiguous".
        with pytest.raises(ValueError, match="ambiguous"):
            L._original_overlap_slices(self.LARGE, self.SMALL_NDARRAY,
                                       self.POSITION, mode='partial')

    def test_patched_coerces_shape_and_raises_catchable_nooverlap(self):
        # After coercion the comparison is a plain tuple comparison, so the
        # function raises the *catchable* NoOverlapError that
        # make_model_image expects (out-of-frame source), NOT a ValueError
        # about array truth values.
        from astropy.nddata.utils import NoOverlapError
        with pytest.raises(NoOverlapError):
            L._overlap_slices_tuple_shape(self.LARGE, self.SMALL_NDARRAY,
                                          self.POSITION, mode='partial')


# ---------------------------------------------------------------------------
# _strip_chunk
# ---------------------------------------------------------------------------
class TestBgsubToken:
    """`--use-iter3-residual-bg` must add a `_resbgsub` filename token so its
    catalogs/residuals are namespaced apart from plain runs, and `_bgsub`
    must never be a substring of `_resbgsub` (exact-token glob matching in
    mosaic_each_exposure_residuals relies on this)."""

    def _opts(self, bgsub, residbg):
        return types.SimpleNamespace(bgsub=bgsub, use_iter3_residual_bg=residbg)

    @pytest.mark.parametrize("bgsub, residbg, expected", [
        (False, False, ''),
        (True, False, '_bgsub'),
        (False, True, '_resbgsub'),
        (True, True, '_bgsub_resbgsub'),
    ])
    def test_token(self, bgsub, residbg, expected):
        assert L._bgsub_token(self._opts(bgsub, residbg)) == expected

    def test_missing_attr_defaults_off(self):
        # options object without the attr at all -> treated as not set.
        assert L._bgsub_token(types.SimpleNamespace(bgsub=False)) == ''

    def test_bgsub_not_substring_of_resbgsub(self):
        assert '_bgsub' not in '_resbgsub'


class TestStripChunk:
    @pytest.mark.parametrize("label, expected", [
        ('iter3_chunk03of08', 'iter3'),
        ('iter3_chunk3of8', 'iter3'),
        ('iter3_chunk10of12', 'iter3'),
        ('iter2', 'iter2'),               # no token -> unchanged
        ('', ''),
        # iter4resbgrefit (final residual-bg refit) must survive chunk
        # stripping so its iter3-like xy_bounds / seed path still fire.
        ('iter4resbgrefit', 'iter4resbgrefit'),
        ('iter4resbgrefit_chunk01of04', 'iter4resbgrefit'),
    ])
    def test_strip(self, label, expected):
        assert L._strip_chunk(label) == expected

    def test_none_passthrough(self):
        assert L._strip_chunk(None) is None

    def test_is_iter3_check_survives_chunking(self):
        # The reason the helper exists: 'is iter3' semantic checks must still
        # fire on a chunk-suffixed label.
        assert L._strip_chunk('iter3_chunk03of08') == 'iter3'
        assert 'iter3' == L._strip_chunk('iter3_chunk03of08')


# ---------------------------------------------------------------------------
# _resolve_seed_skycoords: per-filter preferred column must beat ra/dec.
# ---------------------------------------------------------------------------
class TestResolveSeedPreferredColumn:
    def test_preferred_skycoord_col_overrides_radec(self):
        """sickle F480M source 55 (2026-06-03): a union row carries SW-only
        ra/dec AND a per-filter ``skycoord_f480m`` snap.  When the snap col
        is requested it must take precedence over the plain ra/dec."""
        t = Table()
        t['flux_fit'] = np.array([1.0, 2.0])
        # plain ra/dec = the *unsnapped* SW astrometry
        t['ra'] = np.array([266.5500, 266.5600])
        t['dec'] = np.array([-28.8000, -28.7900])
        # per-filter snap = the *correct* LW position, deliberately different
        snap = SkyCoord(ra=[266.5510, 266.5610] * u.deg,
                        dec=[-28.8010, -28.7910] * u.deg, frame='icrs')
        t['skycoord_f480m'] = snap

        out = L._resolve_seed_skycoords(Table(t, copy=True),
                                        preferred_skycoord_col='skycoord_f480m')
        sk = out['skycoord']
        np.testing.assert_allclose(np.asarray(sk.ra.deg), snap.ra.deg,
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(sk.dec.deg), snap.dec.deg,
                                   rtol=0, atol=1e-9)

    def test_radec_used_when_no_preferred_col(self):
        # Sanity: without a preferred column the plain ra/dec path is fine.
        t = Table()
        t['flux_fit'] = np.array([1.0, 2.0])
        t['ra'] = np.array([266.55, 266.56])
        t['dec'] = np.array([-28.80, -28.79])
        out = L._resolve_seed_skycoords(Table(t, copy=True))
        np.testing.assert_allclose(np.asarray(out['skycoord'].ra.deg),
                                   [266.55, 266.56])


# ---------------------------------------------------------------------------
# _filter_near_saturation: Star B regression (5.0 px -> 1.0 px).
# ---------------------------------------------------------------------------
class _FakePhot:
    """Minimal stand-in for a PSFPhotometry object: only ``.results``."""
    def __init__(self, results):
        self.results = results


def _make_phot_results(xy_flux):
    t = Table()
    t['id'] = np.arange(1, len(xy_flux) + 1)
    t['x_fit'] = np.array([p[0] for p in xy_flux], dtype=float)
    t['y_fit'] = np.array([p[1] for p in xy_flux], dtype=float)
    t['flux_fit'] = np.array([p[2] for p in xy_flux], dtype=float)
    return t


def _dq_with_saturated_pixel(shape, sat_yx):
    dq = np.zeros(shape, dtype=np.uint32)
    sat_bit = L.dqflags.pixel['SATURATED']
    for (y, x) in sat_yx:
        dq[y, x] |= sat_bit
    return dq


class TestFilterNearSaturation:
    def _scenario(self):
        # Saturated pixel at (y=50, x=50).
        # "Star B": real star at (x=52, y=51) -> dist sqrt(5) ~ 2.236 px.
        # "on-sat" fit: centre at (x=50, y=50) -> dist 0 px (bogus, drop).
        dq = _dq_with_saturated_pixel((100, 100), [(50, 50)])
        results = _make_phot_results([
            (52.0, 51.0, 3500.0),   # Star B
            (50.0, 50.0, 9e4),      # fit centred on the saturated pixel
            (10.0, 10.0, 500.0),    # far-away control
        ])
        return dq, results

    def test_tight_radius_keeps_starB_drops_on_sat(self):
        dq, results = self._scenario()
        phot = _FakePhot(results)
        n_drop = L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0,
                                           label='test_basic')
        kept_ids = set(np.asarray(phot.results['id']))
        assert n_drop == 1, "only the on-saturated-pixel fit should drop"
        assert 1 in kept_ids, "Star B (2.24 px away) must survive at 1.0 px"
        assert 2 not in kept_ids, "the on-sat fit must be dropped"
        assert 3 in kept_ids

    def test_legacy_5px_radius_would_kill_starB(self):
        # Demonstrates *why* the radius was tightened: the old 5.0 px value
        # drops Star B along with the bogus on-sat fit.
        dq, results = self._scenario()
        phot = _FakePhot(results)
        n_drop = L._filter_near_saturation(phot, dq, max_sat_dist_pix=5.0,
                                           label='test_legacy')
        kept_ids = set(np.asarray(phot.results['id']))
        assert 1 not in kept_ids, \
            "regression demo: 5.0 px radius incorrectly drops Star B"
        assert n_drop == 2

    def test_noop_without_saturated_pixels(self):
        dq = np.zeros((100, 100), dtype=np.uint32)
        phot = _FakePhot(_make_phot_results([(50.0, 50.0, 1.0)]))
        n_drop = L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0,
                                           label='test_noop')
        assert n_drop == 0
        assert len(phot.results) == 1

    def test_noop_when_dq_is_none(self):
        phot = _FakePhot(_make_phot_results([(50.0, 50.0, 1.0)]))
        assert L._filter_near_saturation(phot, None, max_sat_dist_pix=1.0,
                                         label='test_none') == 0


# ---------------------------------------------------------------------------
# forced_psf_photometry: closed-form flux solve.
# ---------------------------------------------------------------------------
class _GaussianPSF:
    """A simple un-normalised Gaussian PSF exposing ``evaluate``.

    NOT normalised on purpose: forced_psf_photometry only requires that the
    stamp it evaluates matches the model used to build the image pixel-for-
    pixel, which holds for any pointwise function.
    """
    def __init__(self, sigma=1.6):
        self.sigma = sigma

    def evaluate(self, x, y, flux, x_0, y_0):
        g = np.exp(-((x - x_0) ** 2 + (y - y_0) ** 2) / (2.0 * self.sigma ** 2))
        return flux * g


class TestForcedPsfPhotometry:
    def test_recovers_injected_flux_noise_free(self):
        psf = _GaussianPSF(sigma=1.6)
        ny = nx = 41
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        x0, y0 = 20.0, 20.0
        true_flux = 1234.5
        image = psf.evaluate(xx, yy, true_flux, x0, y0)

        init = Table({'x_init': [x0], 'y_init': [y0]})
        out = L.forced_psf_photometry(image, psf, init, fit_shape=(5, 5))

        assert np.isfinite(out['flux_fit'][0])
        np.testing.assert_allclose(out['flux_fit'][0], true_flux, rtol=1e-6)
        # x_fit/y_fit are fixed at the init positions (forced photometry).
        np.testing.assert_allclose(out['x_fit'][0], x0)
        np.testing.assert_allclose(out['y_fit'][0], y0)
        assert out['flux_err'][0] > 0

    def test_source_off_frame_returns_nan(self):
        psf = _GaussianPSF()
        image = np.zeros((41, 41))
        init = Table({'x_init': [1.0], 'y_init': [1.0]})  # stamp falls off edge
        out = L.forced_psf_photometry(image, psf, init, fit_shape=(5, 5))
        assert np.isnan(out['flux_fit'][0])


# ---------------------------------------------------------------------------
# resolve_max_group_size (crowdsource_catalogs_long.py:449-477)
# 0 is REJECTED as ambiguous (used to mean "no cap", reads like "no grouping");
# 'unlimited' -> None; positive int passes through.
# ---------------------------------------------------------------------------
class TestResolveMaxGroupSize:
    @pytest.mark.parametrize('word', ['unlimited', 'inf', 'infinite',
                                      'nocap', 'none', 'UNLIMITED'])
    def test_unlimited_words_map_to_none(self, word):
        assert L.resolve_max_group_size(word) is None

    @pytest.mark.parametrize('raw,expected', [(10, 10), ('15', 15), (1, 1)])
    def test_positive_int_passthrough(self, raw, expected):
        assert L.resolve_max_group_size(raw) == expected

    def test_zero_is_rejected(self):
        with pytest.raises(SystemExit):
            L.resolve_max_group_size(0)

    def test_negative_is_rejected(self):
        with pytest.raises(SystemExit):
            L.resolve_max_group_size(-5)

    def test_none_is_rejected(self):
        # No implicit default: must be set explicitly.
        with pytest.raises(SystemExit):
            L.resolve_max_group_size(None)

    def test_garbage_is_rejected(self):
        with pytest.raises(SystemExit):
            L.resolve_max_group_size('big')


# ---------------------------------------------------------------------------
# normalize_vgroup_id (crowdsource_catalogs_long.py:888-903)
# Returns (token, int_id); token is always '_vgroup<value>', int extracted from
# the first run of digits (None if none).  Idempotent on an already-prefixed id.
# ---------------------------------------------------------------------------
class TestNormalizeVgroupId:
    def test_empty_returns_blank(self):
        assert L.normalize_vgroup_id('') == ('', None)
        assert L.normalize_vgroup_id(None) == ('', None)

    def test_plain_digits(self):
        assert L.normalize_vgroup_id('7') == ('_vgroup7', 7)
        assert L.normalize_vgroup_id(3) == ('_vgroup3', 3)

    def test_idempotent_on_prefixed(self):
        # Already-prefixed -> prefix stripped before re-tokenizing, int recovered.
        assert L.normalize_vgroup_id('_vgroup12') == ('_vgroup12', 12)

    def test_mixed_token_extracts_first_int(self):
        tok, n = L.normalize_vgroup_id('a4b')
        assert tok == '_vgroupa4b'
        assert n == 4

    def test_no_digits_gives_none_id(self):
        tok, n = L.normalize_vgroup_id('abc')
        assert tok == '_vgroupabc'
        assert n is None
