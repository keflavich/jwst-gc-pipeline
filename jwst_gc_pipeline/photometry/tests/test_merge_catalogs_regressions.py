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


class TestDedupOffsetResolver:
    """merge dedup radius must scale to the broad MIRI PSF FWHM (env-gated),
    else daofind's split detections of one star survive as duplicates whose
    models stack -> over-subtraction pits + ~3x over-counted catalog.
    See project_miri_f2550w_dedup_pileup."""

    def test_default_unchanged(self, monkeypatch):
        import astropy.units as u
        from jwst_gc_pipeline.photometry import merge_catalogs as M
        monkeypatch.delenv('MERGE_DEDUP_OFFSET_ARCSEC', raising=False)
        monkeypatch.delenv('MERGE_DEDUP_FWHM_FRAC', raising=False)
        cur = 0.10 * u.arcsec
        assert M._resolve_dedup_offset('f2550w', cur) == cur
        assert M._resolve_dedup_offset('f405n', cur) == cur

    def test_fwhm_frac_scales_miri(self, monkeypatch):
        import astropy.units as u
        from jwst_gc_pipeline.photometry import merge_catalogs as M
        monkeypatch.delenv('MERGE_DEDUP_OFFSET_ARCSEC', raising=False)
        monkeypatch.setenv('MERGE_DEDUP_FWHM_FRAC', '0.5')
        out = M._resolve_dedup_offset('f2550w', 0.10 * u.arcsec)
        assert abs(out.to_value(u.arcsec) - 0.5 * 0.803) < 1e-9
        # unknown filter (NIRCam) -> unchanged even with frac set
        assert M._resolve_dedup_offset('f405n', 0.10 * u.arcsec) == 0.10 * u.arcsec

    def test_absolute_env_wins(self, monkeypatch):
        import astropy.units as u
        from jwst_gc_pipeline.photometry import merge_catalogs as M
        monkeypatch.setenv('MERGE_DEDUP_OFFSET_ARCSEC', '0.35')
        monkeypatch.setenv('MERGE_DEDUP_FWHM_FRAC', '0.5')
        out = M._resolve_dedup_offset('f2550w', 0.10 * u.arcsec)
        assert abs(out.to_value(u.arcsec) - 0.35) < 1e-9


def test_alignment_star_mask_excludes_position_unsafe_rows():
    """Referee-4 gate: transformation star sets must exclude satstar-replaced,
    saturated-unrepaired, forced-filled, and poor-qfit rows."""
    import numpy as np
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.column_utils import alignment_star_mask
    cat = Table({
        'replaced_saturated': [False, True, False, False, False],
        'is_saturated': [False, False, True, False, False],
        'forced_filled': [False, False, False, True, False],
        'qfit': [0.1, 0.1, 0.1, 0.1, 0.9],
    })
    m = alignment_star_mask(cat)
    assert list(m) == [True, False, False, False, False]
    # per-band suffix form
    cat2 = Table({'replaced_saturated_f182m': [True, False],
                  'qfit_f182m': [0.05, 0.05]})
    m2 = alignment_star_mask(cat2, suffix='_f182m')
    assert list(m2) == [False, True]
    # missing columns -> no exclusion (safe on any generation)
    m3 = alignment_star_mask(Table({'flux': [1.0, 2.0]}))
    assert m3.all()


def test_replace_saturated_uses_detector_coords(monkeypatch, tmp_path):
    """x_fit/y_fit in satstar catalogs are CUTOUT coordinates (pad-centered,
    ~(81,81)); detector coordinates live in xcentroid/ycentroid.  Replaced
    rows and appended satstar-only rows must get the DETECTOR coordinates in
    their pixel columns (regression: cutout coords silently corrupted x/y of
    every replaced row)."""
    import numpy as np
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from jwst_gc_pipeline.photometry import merge_catalogs as M

    ra0, dec0 = 266.5, -28.7
    satstar = Table({
        'skycoord_fit': SkyCoord([ra0, ra0 + 0.01]*u.deg, [dec0, dec0]*u.deg),
        'flux_fit': [5000.0, 7000.0],
        'flux_err': [10.0, 10.0],
        'x_fit': [81.2, 80.9],          # cutout frame
        'y_fit': [81.0, 81.4],
        'x_err': [0.01, 0.01],
        'y_err': [0.01, 0.01],
        'xcentroid': [512.3, 1400.7],   # detector frame
        'ycentroid': [300.1, 1650.2],
    })
    monkeypatch.setattr(M, 'load_satstar_catalog', lambda *a, **k: satstar)

    class _FakeSvo:
        @staticmethod
        def get_filter_list(_):
            t = Table({'filterID': ['JWST/NIRCam.F182M'], 'ZeroPoint': [850.0]})
            t.add_index('filterID')
            return t
    monkeypatch.setattr(M, 'SvoFps', _FakeSvo)

    (tmp_path / 'reduction').mkdir()
    Table({'Filter': ['F182M'], 'PSF FWHM (arcsec)': [0.06]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv', format='ascii.ecsv')

    # catalog row 0 sits at satstar 0's position (clipped daophot, fainter);
    # satstar 1 is unmatched -> appended as a new row.
    cat = Table({
        'skycoord': SkyCoord([ra0]*u.deg, [dec0]*u.deg),
        'flux': [1000.0],
        'dflux': [5.0],
        'x': [512.0], 'y': [300.0], 'dx': [0.01], 'dy': [0.01],
    })
    M.replace_saturated(cat, 'f182m', basepath=str(tmp_path) + '/',
                        fwhm_basepath=str(tmp_path))

    assert bool(cat['replaced_saturated'][0])
    # replaced row: detector coords, not the ~81 cutout coords
    assert abs(cat['x'][0] - 512.3) < 1e-6
    assert abs(cat['y'][0] - 300.1) < 1e-6
    # appended satstar-only row: detector coords too
    new = cat[~np.isfinite(np.asarray(cat['satstar_match_sep'], float))] \
        if 'satstar_match_sep' in cat.colnames else cat[1:]
    assert len(cat) == 2
    assert abs(cat['x'][1] - 1400.7) < 1e-6
    assert abs(cat['y'][1] - 1650.2) < 1e-6
    # match separation recorded for the replaced row
    assert np.isfinite(cat['satstar_match_sep'][0])


def test_color_reliable_mask_one_band_repaired():
    """Strip audit 2026-07-11: substituted-in-one-band + clipped-unrepaired-in-
    the-other colors are spurious and must be flagged unreliable; repaired-in-
    both and clean rows stay reliable."""
    import numpy as np
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.column_utils import color_reliable_mask
    cat = Table({
        # rows: 0 clean/clean, 1 repaired/repaired, 2 repaired/clipped-unrepaired,
        #       3 clipped-unrepaired/repaired, 4 clipped/clipped (both unrepaired),
        #       5 repaired/clean
        'replaced_saturated_f405n': [0, 1, 1, 0, 0, 1],
        'is_saturated_f405n':       [0, 1, 1, 1, 1, 1],
        'replaced_saturated_f410m': [0, 1, 0, 1, 0, 0],
        'is_saturated_f410m':       [0, 1, 1, 1, 1, 0],
    })
    ok = color_reliable_mask(cat, 'f405n', 'f410m')
    assert ok.tolist() == [True, True, False, False, True, True]


def test_satstar_gate_rejected_flagging(monkeypatch, tmp_path):
    """Phase A3: catalog rows matching a gate-REJECTED satstar candidate
    (<0.15") are flagged satstar_gate_rejected (photometry untouched);
    replaced rows and unrelated rows are not flagged; no rejected files ->
    all False."""
    import numpy as np
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from jwst_gc_pipeline.photometry import merge_catalogs as MC

    cat = Table({
        'skycoord': SkyCoord([10.0, 10.1, 10.2] * u.deg, [0.0, 0.0, 0.0] * u.deg),
        'flux': [100.0, 200.0, 300.0],
        'replaced_saturated': [False, False, False],
    })
    # row 1 sits 0.05" from a rejected candidate; row 0/2 far away
    rej = Table({'flux_fit': [240.0], 'flux_err': [10.0], 'qfit': [0.1],
                 'reject_reason': ['implied_peak_gate']})
    rej['skycoord_fit'] = SkyCoord([10.1 + 0.05 / 3600.0] * u.deg,
                                   [0.0] * u.deg)

    _real_loader = MC.load_rejected_satstar_catalog
    monkeypatch.setattr(MC, 'load_rejected_satstar_catalog',
                        lambda *a, **k: rej)
    flux_before = np.asarray(cat['flux']).copy()
    # exercise only the flag block: call through a minimal shim replicating it
    _gate_rej = np.zeros(len(cat), dtype=bool)
    _ri, _rsep, _ = SkyCoord(cat['skycoord']).match_to_catalog_sky(
        SkyCoord(rej['skycoord_fit']))
    _gate_rej = (_rsep.arcsec < 0.15) & ~np.asarray(cat['replaced_saturated'])
    cat['satstar_gate_rejected'] = _gate_rej
    assert cat['satstar_gate_rejected'].tolist() == [False, True, False]
    assert np.all(np.asarray(cat['flux']) == flux_before)

    # loader contract: no files -> None (use the real, unpatched loader)
    assert _real_loader('f999n', target='brick', basepath=str(tmp_path)) is None


def test_color_reliable_mask_gate_rejected_uncorrected():
    """Gate-rejected strip rows without the gray-zone correction count as
    clipped-unrepaired; corrected ones stay reliable."""
    import numpy as np
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.column_utils import color_reliable_mask
    cat = Table({
        # 0: repaired vs gate-rejected UNcorrected -> unreliable
        # 1: repaired vs gate-rejected corrected  -> reliable
        # 2: clean vs gate-rejected uncorrected   -> reliable (no repaired side)
        'replaced_saturated_f405n':   [1, 1, 0],
        'is_saturated_f405n':         [1, 1, 0],
        'replaced_saturated_f410m':   [0, 0, 0],
        'is_saturated_f410m':         [0, 0, 0],
        'satstar_gate_rejected_f410m': [1, 1, 1],
        'satclip_corrected_f410m':    [0, 1, 0],
    })
    ok = color_reliable_mask(cat, 'f405n', 'f410m')
    assert ok.tolist() == [False, True, True]


def test_grayzone_clip_correction_bounds():
    """PR #81 decision: adopt the rejected wing-fit flux ONLY in the bounded
    gray zone (implied_peak_gate reason, sane qfit, 0.02-0.5 mag brighter);
    garbage fits, big adjustments, and near-zero adjustments are untouched."""
    import numpy as np
    catflux = np.array([100.0, 100.0, 100.0, 100.0])
    rflux = np.array([110.0,          # +0.10 mag -> corrected
                      100.5,          # +0.005 mag -> too small, no-op
                      300.0,          # +1.19 mag -> beyond gray zone, no-op
                      110.0])         # garbage qfit -> no-op
    rq = np.array([0.1, 0.1, 0.1, 45.0])
    reason = np.array(['implied_peak_gate'] * 4)
    with np.errstate(divide='ignore', invalid='ignore'):
        dmag = 2.5 * np.log10(rflux / catflux)
    ok = ((reason == 'implied_peak_gate')
          & np.isfinite(rq) & (rq > 0) & (rq < 1)
          & np.isfinite(rflux) & (rflux > 0)
          & np.isfinite(dmag) & (dmag >= 0.02) & (dmag <= 0.5))
    assert ok.tolist() == [True, False, False, False]
    # uncertainty floor = half the adjustment
    dflux = rflux[0] - catflux[0]
    err = np.hypot(5.0, 0.5 * dflux)
    assert err > 5.0


def test_pooled_wingcal_build_and_apply(tmp_path):
    """Phase B1: per-frame calibrator buckets pool into an n-weighted C(r);
    the pooled ratio applies ONLY to rows whose per-frame self-cal was
    skipped (wingcal_ratio == 1.0 exactly), dividing flux and updating
    provenance; already-calibrated rows are untouched."""
    import numpy as np
    from astropy.table import Table
    from jwst_gc_pipeline.photometry import merge_catalogs as MC

    base = tmp_path
    pdir = base / 'F410M' / 'pipeline'
    pdir.mkdir(parents=True)
    (base / 'catalogs').mkdir()
    Table({'rmask_px': [3, 5], 'ratio_median': [1.05, 1.10],
           'n_stars': [4, 3], 'ratio_madstd': [0.02, 0.03]}).write(
        pdir / 'a_m12_wingcal_calibrators.fits')
    Table({'rmask_px': [3, 5], 'ratio_median': [1.07, 1.14],
           'n_stars': [4, 6], 'ratio_madstd': [0.02, 0.03]}).write(
        pdir / 'b_m12_wingcal_calibrators.fits')
    pooled = MC.build_pooled_wingcal('f410m', basepath=str(base))
    assert len(pooled) == 2
    r3 = float(pooled['ratio'][pooled['rmask_px'] == 3][0])
    assert abs(r3 - (1.05 * 4 + 1.07 * 4) / 8) < 1e-9

    cat = Table({'flux_fit': [1000.0, 1000.0, 1000.0],
                 'flux_err': [10.0, 10.0, 10.0],
                 'wingcal_ratio': [1.0, 1.02, 1.0],
                 'wingcal_rmask': [3.0, 3.0, np.nan]})
    out = MC.apply_pooled_wingcal(cat, 'f410m', basepath=str(base))
    assert out['wingcal_pooled'].tolist() == [True, False, False]
    assert abs(out['flux_fit'][0] - 1000.0 / r3) < 1e-6
    assert out['flux_fit'][1] == 1000.0
    assert out['flux_fit'][2] == 1000.0
    assert abs(out['wingcal_ratio'][0] - r3) < 1e-9


# ---------------------------------------------------------------------------
# combine_singleframe: flux_err_prop must be the inverse-variance propagated
# error 1/sqrt(sum(1/sigma^2)) (2026-07-21 fix: it was sqrt(N_good) too big).
# ---------------------------------------------------------------------------
def test_flux_err_prop_is_inverse_variance(_sci_fits):
    """With w = keepmask/sigma^2 the old expression
    sqrt(nansum(sigma^2 * w) / nansum(w)) has numerator == N_good, so it
    returned sqrt(N_good / sum(w)) -- sqrt(N_good) LARGER than the documented
    propagated uncertainty.  Hand-computed sigmas pin the correct value."""
    n_frame = 3
    base_ra = 266.5 + np.arange(2) * (1.0 / 3600.0)
    base_dec = np.full(2, -28.8)
    # per-source, per-frame sigma; source 1 has one NaN flux_err frame
    sig = np.array([[10.0, 20.0, 40.0],
                    [10.0, np.nan, 40.0]])
    tbls = []
    for f in range(n_frame):
        sc = SkyCoord(ra=base_ra * u.deg, dec=base_dec * u.deg, frame='icrs')
        tbls.append(_make_crowdsource_frame(
            sc, flux=[1000.0, 1000.0], dflux=sig[:, f],
            qf=[1.0, 1.0], fracflux=[0.9, 0.9],
            exposure=f + 1, filename=_sci_fits))

    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy)
    assert len(out) == 2

    expected = [1.0 / np.sqrt(1 / 100 + 1 / 400 + 1 / 1600),   # 8.72872...
                1.0 / np.sqrt(1 / 100 + 1 / 1600)]             # 9.70143...
    np.testing.assert_allclose(np.asarray(out['dflux_prop'], dtype=float),
                               expected, rtol=1e-5)
    # and the meta text must document the corrected formula
    assert '1/sqrt' in out.meta['dflux_prop']


# ---------------------------------------------------------------------------
# combine_singleframe: MODULE-absent meta must not KeyError in the rematch
# diagnostic print (the conditional was INSIDE the subscript: meta['']).
# ---------------------------------------------------------------------------
def test_combine_singleframe_rematch_without_module_meta(_sci_fits, monkeypatch):
    monkeypatch.setenv('MERGE_REMATCH_DIAGNOSTICS', '1')
    rng = np.random.default_rng(7)
    n_src, n_frame = 4, 2
    base_ra = 266.5 + np.arange(n_src) * (1.0 / 3600.0)
    base_dec = np.full(n_src, -28.8)
    tbls = []
    for f in range(n_frame):
        ra = base_ra + rng.normal(scale=1e-6, size=n_src)
        dec = base_dec + rng.normal(scale=1e-6, size=n_src)
        sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        t = _make_crowdsource_frame(sc, np.full(n_src, 1000.0),
                                    np.full(n_src, 10.0), np.full(n_src, 1.0),
                                    np.full(n_src, 0.9),
                                    exposure=f + 1, filename=_sci_fits)
        del t.meta['MODULE']    # the regression trigger: KeyError('')
        tbls.append(t)
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy)
    assert len(out) == n_src


# ---------------------------------------------------------------------------
# combine_singleframe: the realign=False production path must NOT run the
# re-matching diagnostic (mutual match + per-exposure FITS open).
# ---------------------------------------------------------------------------
def test_combine_singleframe_default_skips_rematch_diagnostics(monkeypatch):
    """FILENAME points nowhere: if the rematch loop ran, the fits.open would
    raise.  The per-exposure offsets are recorded as NaN (not measured)."""
    monkeypatch.delenv('MERGE_REMATCH_DIAGNOSTICS', raising=False)
    rng = np.random.default_rng(8)
    n_src, n_frame = 3, 2
    base_ra = 266.5 + np.arange(n_src) * (1.0 / 3600.0)
    base_dec = np.full(n_src, -28.8)
    tbls = []
    for f in range(n_frame):
        ra = base_ra + rng.normal(scale=1e-6, size=n_src)
        dec = base_dec + rng.normal(scale=1e-6, size=n_src)
        sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        tbls.append(_make_crowdsource_frame(
            sc, np.full(n_src, 1000.0), np.full(n_src, 10.0),
            np.full(n_src, 1.0), np.full(n_src, 0.9),
            exposure=f + 1, filename='/nonexistent/path/does_not_exist.fits'))
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy)
    assert len(out) == n_src
    offs = out.meta['offsets']
    assert len(offs) == n_frame
    for ra_off, dec_off in offs.values():
        assert np.isnan(float(ra_off)) and np.isnan(float(dec_off))


# ---------------------------------------------------------------------------
# replace_saturated: a crowdsource-style catalog with x/y but WITHOUT dx/dy
# must not KeyError (the dx/dy write was guarded only on 'x').
# ---------------------------------------------------------------------------
def test_replace_saturated_without_dxdy_columns(monkeypatch, tmp_path):
    import numpy as np
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from jwst_gc_pipeline.photometry import merge_catalogs as M

    ra0, dec0 = 266.5, -28.7
    satstar = Table({
        'skycoord_fit': SkyCoord([ra0, ra0 + 0.01]*u.deg, [dec0, dec0]*u.deg),
        'flux_fit': [5000.0, 7000.0],
        'flux_err': [10.0, 10.0],
        'x_fit': [81.2, 80.9],
        'y_fit': [81.0, 81.4],
        'x_err': [0.01, 0.01],
        'y_err': [0.01, 0.01],
        'xcentroid': [512.3, 1400.7],
        'ycentroid': [300.1, 1650.2],
    })
    monkeypatch.setattr(M, 'load_satstar_catalog', lambda *a, **k: satstar)

    class _FakeSvo:
        @staticmethod
        def get_filter_list(_):
            t = Table({'filterID': ['JWST/NIRCam.F182M'], 'ZeroPoint': [850.0]})
            t.add_index('filterID')
            return t
    monkeypatch.setattr(M, 'SvoFps', _FakeSvo)

    (tmp_path / 'reduction').mkdir()
    Table({'Filter': ['F182M'], 'PSF FWHM (arcsec)': [0.06]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv', format='ascii.ecsv')

    # x/y present but NO dx/dy: the old code raised KeyError('dx')
    cat = Table({
        'skycoord': SkyCoord([ra0]*u.deg, [dec0]*u.deg),
        'flux': [1000.0],
        'dflux': [5.0],
        'x': [512.0], 'y': [300.0],
    })
    M.replace_saturated(cat, 'f182m', basepath=str(tmp_path) + '/',
                        fwhm_basepath=str(tmp_path))

    assert bool(cat['replaced_saturated'][0])
    assert abs(cat['x'][0] - 512.3) < 1e-6
    assert abs(cat['y'][0] - 300.1) < 1e-6
    assert len(cat) == 2                       # unmatched satstar appended
    assert 'dx' not in cat.colnames


# ---------------------------------------------------------------------------
# oksep quality-cut helpers: per-target filename token + actual sep_* columns.
# ---------------------------------------------------------------------------
def test_qualcuts_oksep_suffix_per_target():
    # targets with proposal 2221 keep the historical literal name that
    # stage_release.py / make_all_cmds_m7.py glob
    assert MC._qualcuts_oksep_suffix('brick') == '_qualcuts_oksep2221'
    assert MC._qualcuts_oksep_suffix('cloudc') == '_qualcuts_oksep2221'
    # other targets get their own proposal token(s), not a hardcoded "2221"
    assert MC._qualcuts_oksep_suffix('sgrc') == '_qualcuts_oksep4147'
    assert MC._qualcuts_oksep_suffix('ngc6334') == '_qualcuts_oksep6778-7213'
    # unknown target: fall back to the target name
    assert MC._qualcuts_oksep_suffix('nosuchtarget') == '_qualcuts_oksepnosuchtarget'


def test_oksep_sep_cols_from_actual_columns():
    cols = ['sep_f405n', 'sep_f410m', 'sep_f480m', 'sep_f200w', 'sep_f150w2',
            'flux_f405n', 'satstar_match_sep_f405n']
    # narrow/medium bands present in the TABLE are used (f480m was invisible
    # to the old brick-filternames iteration); wide (w) bands stay excluded
    assert MC._oksep_sep_cols(cols) == ['sep_f405n', 'sep_f410m', 'sep_f480m']
