"""Regression tests for reference-catalog construction
(make_reference_from_pipeline_catalogs.py).

Covered regressions / defensive paths
--------------------------------------
* ``summarize_offsets`` empty-input guard (lines 596-610).  Zero matches must
  return a NaN-filled summary with ``n_matches == 0`` instead of crashing on
  ``nanmean`` / ``mad_std`` of an empty array.

* ``initial_spatial_photometric_match`` MAD fallback (lines 546-551).  When
  the photometric residuals have zero spread (``mad == 0``) or non-finite
  MAD, sigma-clipping is undefined; the code must fall back to keeping all
  finite residuals rather than rejecting every source.

* ``_normalize_reference_ks_magnitude`` Ks stacking (lines 442-461).  Multiple
  candidate Ks columns with partial NaN coverage must be collapsed with a
  per-source ``nanmedian``; sources with no finite Ks (or non-finite
  coordinates) are dropped, not propagated as NaN.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import make_reference_from_pipeline_catalogs as R


# ---------------------------------------------------------------------------
# summarize_offsets: empty input.
# ---------------------------------------------------------------------------
class TestSummarizeOffsets:
    def test_empty_input_returns_nan_summary(self):
        out = R.summarize_offsets(np.array([]), np.array([]), np.array([]))
        assert out["n_matches"] == 0
        nan_keys = [k for k in out if k != "n_matches"]
        assert nan_keys, "summary should contain statistic fields"
        for k in nan_keys:
            assert np.isnan(out[k]), f"{k} should be NaN for empty input"

    def test_nonempty_values(self):
        # dra/ddec are in mas; outputs converted to arcsec.
        dra = np.array([100.0, 200.0, 300.0])   # mas
        ddec = np.array([0.0, 0.0, 0.0])
        sep = np.array([100.0, 200.0, 300.0])
        out = R.summarize_offsets(dra, ddec, sep)
        assert out["n_matches"] == 3
        np.testing.assert_allclose(out["median_dra_arcsec"], 0.2)
        np.testing.assert_allclose(out["mean_dra_arcsec"], 0.2)
        np.testing.assert_allclose(out["median_sep_mas"], 200.0)
        np.testing.assert_allclose(out["vector_median_offset_arcsec"], 0.2)


# ---------------------------------------------------------------------------
# initial_spatial_photometric_match: MAD == 0 fallback.
# ---------------------------------------------------------------------------
def _coords(ra, dec):
    return SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg,
                    frame='icrs')


class TestInitialSpatialPhotometricMatchMadFallback:
    def _matched_tables(self, n=20, degenerate=False):
        """Build co-located catalog + reference tables.

        ``degenerate=True`` gives every source an identical flux and an
        identical reference magnitude so the least-squares residuals are
        *exactly* equal -> ``mad_std == 0`` (the fallback trigger).  A
        perfect log-linear relation only yields ~1e-15 residuals, which is
        not exactly zero and would exercise the normal sigma-clip path.
        """
        ra = 266.5 + np.arange(n) * (2.0 / 3600.0)
        dec = np.full(n, -28.8)
        if degenerate:
            flux = np.full(n, 1000.0)
            refmag = np.full(n, 15.0)
        else:
            flux = np.linspace(100.0, 10000.0, n)
            a, b = 25.0, -2.5
            refmag = a + b * np.log10(flux)

        cat = Table()
        cat['skycoord'] = _coords(ra, dec)
        cat['flux'] = flux

        ref = Table()
        ref['skycoord'] = _coords(ra, dec)   # co-located -> mutual matches
        ref['Ks_refmag'] = refmag
        return cat, ref

    def test_zero_mad_keeps_all_finite(self):
        cat, ref = self._matched_tables(n=20, degenerate=True)
        keep, stats = R.initial_spatial_photometric_match(
            cat, ref, reference_mag_column='Ks_refmag',
            reference_label='TEST', max_sep=0.2 * u.arcsec, nsigma=3.0)
        # mad == 0 -> fall back to all finite residuals; nothing dropped.
        assert stats["photometric_residual_mad"] == 0.0
        assert int(stats["initial_matches"]) == int(stats["selected_matches"])
        assert keep.sum() == len(cat)

    def test_too_few_matches_raises(self):
        cat, ref = self._matched_tables(n=5)   # < 10 -> must raise
        with pytest.raises(ValueError, match=">=10"):
            R.initial_spatial_photometric_match(
                cat, ref, reference_mag_column='Ks_refmag',
                reference_label='TEST', max_sep=0.2 * u.arcsec, nsigma=3.0)


# ---------------------------------------------------------------------------
# _normalize_reference_ks_magnitude: nanmedian stacking of Ks columns.
# ---------------------------------------------------------------------------
class TestNormalizeReferenceKsMagnitude:
    def test_partial_nan_columns_nanmedian(self):
        # 4 sources, two candidate Ks columns with different NaN patterns.
        ref = Table()
        # Real VVV/GNS catalogs carry degree-unit coordinate columns; the
        # function builds a fk5 SkyCoord from them, which requires units.
        ref['RAJ2000'] = np.array([266.50, 266.51, 266.52, 266.53]) * u.deg
        ref['DEJ2000'] = np.array([-28.80, -28.81, -28.82, -28.83]) * u.deg
        #              src0    src1     src2     src3
        ref['Ksmag'] = np.array([10.0, np.nan, 12.0, np.nan])
        ref['Ksmag3'] = np.array([np.nan, 11.0, 14.0, np.nan])

        out, magcol = R._normalize_reference_ks_magnitude(
            ref, candidates=('Ksmag', 'Ksmag3'))
        assert magcol == 'Ks_refmag'

        # src3 has no finite Ks -> dropped; survivors keep 3 rows.
        assert len(out) == 3
        km = np.asarray(out['Ks_refmag'])
        # src0 -> 10.0 (only Ksmag), src1 -> 11.0 (only Ksmag3),
        # src2 -> median(12, 14) = 13.0.
        np.testing.assert_allclose(sorted(km), [10.0, 11.0, 13.0])
        assert np.all(np.isfinite(km))
        assert 'skycoord' in out.colnames

    def test_missing_all_candidates_raises(self):
        ref = Table()
        ref['RAJ2000'] = [266.5]
        ref['DEJ2000'] = [-28.8]
        with pytest.raises(ValueError, match="missing all supported Ks"):
            R._normalize_reference_ks_magnitude(ref, candidates=('Ksmag', 'Ks'))

    def test_nonfinite_coordinates_dropped(self):
        ref = Table()
        ref['RAJ2000'] = np.array([266.50, np.nan]) * u.deg
        ref['DEJ2000'] = np.array([-28.80, -28.81]) * u.deg
        ref['Ksmag'] = np.array([10.0, 11.0])
        out, _ = R._normalize_reference_ks_magnitude(ref, candidates=('Ksmag',))
        assert len(out) == 1
        np.testing.assert_allclose(out['Ks_refmag'][0], 10.0)
