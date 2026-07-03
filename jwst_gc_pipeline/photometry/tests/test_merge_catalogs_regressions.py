"""Regression tests for catalog *merging* (merge_catalogs.py).

Each test class pins a specific bug or defensive code path documented by a
comment block in ``merge_catalogs.py`` so that parameter tuning / refactors
cannot silently re-introduce the failure.

Covered regressions
--------------------
* ``nanaverage_numpy`` / ``nanaverage_dask`` all-zero-weight handling
  (merge_catalogs.py:141-162).  All-zero / all-NaN-weight rows must return
  NaN (not 0, not a crash); otherwise downstream averages are corrupted.

* ``combine_singleframe`` NaN-flux_err position loss
  (merge_catalogs.py:440-465).  ~31k of 45k merged-exposure rows were
  silently dropped because sources whose every frame had NaN ``flux_err``
  got all-zero inverse-variance weights -> NaN averaged position -> dropped
  by the caller's NaN-sky reject.  Position averaging must NOT depend on
  flux_err; flux averaging must fall back to uniform weights.

* ``merge_catalogs`` ref-filter validation (merge_catalogs.py:564-572).
  A requested ``ref_filter`` absent from the input tables must raise rather
  than silently switching the astrometric reference of the merge.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import merge_catalogs as MC


# ---------------------------------------------------------------------------
# nanaverage: all-zero / all-NaN weight rows must yield NaN, not 0.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nanaverage", [MC.nanaverage_numpy, MC.nanaverage_dask],
                         ids=["numpy", "dask"])
class TestNanaverage:
    def test_basic_weighted_average(self, nanaverage):
        data = np.array([[1.0, 3.0],
                         [10.0, 30.0]])
        weights = np.array([[1.0, 1.0],
                            [3.0, 1.0]])
        out = nanaverage(data, axis=1, weights=weights)
        np.testing.assert_allclose(out, [2.0, (10 * 3 + 30) / 4.0])

    def test_nan_data_entry_is_excluded(self, nanaverage):
        # A NaN datum in a row must be dropped (weight zeroed), not poison
        # the whole row's average.
        data = np.array([[2.0, np.nan],
                         [4.0, 6.0]])
        weights = np.array([[1.0, 1.0],
                            [1.0, 1.0]])
        out = nanaverage(data, axis=1, weights=weights)
        np.testing.assert_allclose(out, [2.0, 5.0])

    def test_all_zero_weight_row_returns_nan(self, nanaverage):
        # The load-bearing behavior the combine_singleframe fix relies on:
        # a row whose weights are all zero must come back NaN, so the source
        # is *visibly* flagged rather than silently averaged to 0.
        data = np.array([[1.0, 2.0],
                         [5.0, 7.0]])
        weights = np.array([[0.0, 0.0],
                            [1.0, 1.0]])
        out = nanaverage(data, axis=1, weights=weights)
        assert np.isnan(out[0]), "all-zero-weight row should be NaN"
        np.testing.assert_allclose(out[1], 6.0)

    def test_all_nan_weight_row_returns_nan(self, nanaverage):
        # NaN inverse-variance weights (1/NaN**2) must be treated as zero
        # and the all-NaN-weight row returned as NaN.
        data = np.array([[1.0, 2.0],
                         [5.0, 7.0]])
        weights = np.array([[np.nan, np.nan],
                            [2.0, 2.0]])
        out = nanaverage(data, axis=1, weights=weights)
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1], 6.0)


# ---------------------------------------------------------------------------
# combine_singleframe: NaN flux_err must not lose sources with valid positions.
# ---------------------------------------------------------------------------
def _make_crowdsource_frame(skycoords, flux, dflux, qf, fracflux,
                            exposure, filename):
    """Build a minimal crowdsource-style single-exposure catalog.

    combine_singleframe picks the crowdsource column convention when a
    ``qf`` column is present (flux_colname='flux', flux_err_colname='dflux',
    skycoord_colname='skycoord').
    """
    t = Table()
    t['id'] = np.arange(1, len(flux) + 1)
    t['skycoord'] = skycoords
    t['flux'] = np.asarray(flux, dtype='float32')
    t['dflux'] = np.asarray(dflux, dtype='float32')
    t['qf'] = np.asarray(qf, dtype='float32')
    t['fracflux'] = np.asarray(fracflux, dtype='float32')
    t.meta['exposure'] = exposure
    t.meta['MODULE'] = 'nrca'
    t.meta['filter'] = 'f405n'
    t.meta['ra_offset'] = 0.0 * u.arcsec
    t.meta['dec_offset'] = 0.0 * u.arcsec
    t.meta['FILENAME'] = str(filename)
    return t


@pytest.fixture
def _sci_fits(tmp_path):
    """A minimal FITS file with a 'SCI' HDU (no RAOFFSET -> assumed zero).

    combine_singleframe opens ``tbl.meta['FILENAME']`` and reads
    ``fh['SCI'].header`` during its re-matching loop.
    """
    path = tmp_path / "dummy_sci.fits"
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(data=np.zeros((4, 4)), name='SCI')])
    hdul.writeto(path, overwrite=True)
    return path


def test_combine_singleframe_keeps_sources_with_nan_fluxerr(_sci_fits):
    """Regression for the ~31k silently-dropped-rows bug (2026-06-04).

    A source whose every matched frame has NaN ``flux_err`` but valid
    positions must come out with a FINITE averaged position and flux.
    Before the fix, inverse-variance weighting drove its position weights
    to all-zero -> NaN avg_ra/dec -> the source was dropped downstream.
    """
    rng = np.random.default_rng(42)
    n_src = 5
    n_frame = 4

    # 5 sources ~1 arcsec apart (>> min_offset=0.1").  Source index 2 is the
    # "faint target" that will carry NaN dflux in every frame.
    base_ra = 266.5 + np.arange(n_src) * (1.0 / 3600.0)
    base_dec = np.full(n_src, -28.8)
    faint = 2

    tbls = []
    for f in range(n_frame):
        # sub-mas per-frame jitter: well inside max_offset (100 mas), enough
        # to avoid degenerate exact-zero separations in the offset fit.
        ra = base_ra + rng.normal(scale=1e-6, size=n_src)
        dec = base_dec + rng.normal(scale=1e-6, size=n_src)
        sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

        flux = np.full(n_src, 1000.0)
        dflux = np.full(n_src, 10.0)
        dflux[faint] = np.nan          # the regression trigger
        qf = np.full(n_src, 1.0)       # pass the >0.95 offset-match gate
        fracflux = np.full(n_src, 0.9)  # pass the >0.85 gate
        tbls.append(_make_crowdsource_frame(sc, flux, dflux, qf, fracflux,
                                            exposure=f + 1, filename=_sci_fits))

    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy)

    # All 5 sources should survive (no new sources added; frames coincide).
    assert len(out) == n_src

    sc_avg = out['skycoord_avg']
    ra = np.asarray(sc_avg.ra.deg, dtype=float)
    dec = np.asarray(sc_avg.dec.deg, dtype=float)
    flux_avg = np.asarray(out['flux_avg'], dtype=float)

    # The fix: every source with valid positions keeps a finite avg position
    # and a finite avg flux, regardless of NaN flux_err.
    assert np.all(np.isfinite(ra)), f"NaN avg RA at rows {np.where(~np.isfinite(ra))[0]}"
    assert np.all(np.isfinite(dec)), f"NaN avg Dec at rows {np.where(~np.isfinite(dec))[0]}"
    assert np.all(np.isfinite(flux_avg)), \
        f"NaN flux_avg at rows {np.where(~np.isfinite(flux_avg))[0]}"


# ---------------------------------------------------------------------------
# merge_catalogs: missing ref_filter must raise (no silent reference switch).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bgsub, resbgsub, expected", [
    (False, False, ''),
    (True, False, '_bgsub'),
    (False, True, '_resbgsub'),
    (True, True, '_bgsub_resbgsub'),
])
def test_bgsub_token(bgsub, resbgsub, expected):
    """merge_catalogs filename token must match the producer's convention
    (crowdsource_catalogs_long._bgsub_token) so the merge globs find the
    _resbgsub catalogs; _bgsub must not be a substring of _resbgsub."""
    assert MC._bgsub_token(bgsub, resbgsub) == expected
    assert '_bgsub' not in '_resbgsub'


def test_merge_catalogs_missing_ref_filter_raises():
    """merge_catalogs.py:564-572 -- requesting a ref_filter that is not in
    the input tables must raise ValueError, not silently pick another."""
    t1 = Table({'flux': [1.0]})
    t1.meta['filter'] = 'f405n'
    t2 = Table({'flux': [2.0]})
    t2.meta['filter'] = 'f410m'

    with pytest.raises(ValueError, match="not present"):
        MC.merge_catalogs([t1, t2], ref_filter='f187n')


class TestSatstarStaleFileExclusion:
    """load_satstar_catalog must EXCLUDE stale/old-scheme per-exposure satstar
    files that lack an iteration token (_m<N>_).  A too-broad
    ``*satstar_catalog.fits`` glob swept in w51 Oct-2025 files
    (jw06151-o002_..._<N>_o002_crf_satstar_catalog.fits, ~2600 ungated
    emission-phantom rows each) and ballooned the consolidated catalog ~40x.
    See project_miri_partialsat_divot.
    """

    def _write(self, path, nrows):
        from astropy.table import Table
        import numpy as np
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        t = Table()
        t['x_fit'] = np.arange(nrows, dtype=float)
        t['y_fit'] = np.arange(nrows, dtype=float)
        t['flux_fit'] = np.full(nrows, 1000.0)
        t['skycoord_fit'] = SkyCoord(np.linspace(266.5, 266.6, nrows) * u.deg,
                                     np.linspace(-28.6, -28.7, nrows) * u.deg)
        t.write(path, overwrite=True)

    def test_stale_untokened_files_excluded(self, tmp_path, monkeypatch):
        import os
        from jwst_gc_pipeline.photometry import merge_catalogs as M
        pdir = tmp_path / 'F770W' / 'pipeline'
        pdir.mkdir(parents=True)
        # current-scheme tokened files (2 frames x m6) = real satstars
        self._write(pdir / 'jw06151002001_02103_00001_mirimage_o002_crf_resbgsub_m6_satstar_catalog.fits', 3)
        self._write(pdir / 'jw06151002001_02103_00002_mirimage_o002_crf_resbgsub_m6_satstar_catalog.fits', 3)
        # stale old-scheme untokened file (emission phantoms)
        self._write(pdir / 'jw06151-o002_t001_miri_f770w_0_o002_crf_satstar_catalog.fits', 500)
        (tmp_path / 'catalogs').mkdir()
        out = M.load_satstar_catalog('f770w', target='w51', basepath=str(tmp_path) + '/')
        assert out is not None
        # the 500-row stale file is excluded; only the tokened frames feed the
        # consolidation (the 2 frames share positions -> dedup to 3 unique).
        assert len(out) == 3, f'expected 3 (tokened+deduped), got {len(out)} -- stale not excluded?'

    def test_only_stale_returns_none(self, tmp_path):
        from jwst_gc_pipeline.photometry import merge_catalogs as M
        pdir = tmp_path / 'F770W' / 'pipeline'
        pdir.mkdir(parents=True)
        self._write(pdir / 'jw06151-o002_t001_miri_f770w_0_o002_crf_satstar_catalog.fits', 500)
        (tmp_path / 'catalogs').mkdir()
        out = M.load_satstar_catalog('f770w', target='w51', basepath=str(tmp_path) + '/')
        assert out is None  # do not pollute from stale-only dirs
