"""Regression tests for cataloging.py (manual-iteration PSF photometry path).

Pins the MIRI-vs-NIRCam divergence in ``_filter_extended_emission`` so the
Phase-4 untangle (splitting the two keep-logics) can't change behavior:

* NIRCam path (``min_prominence == 0``): keep = star_like (qfit/flags/peakSB)
  AND local-SNR cut.  A high-qfit low-SNR source is dropped.
* MIRI path (``min_prominence > 0``): the deep-i2d prominence is the SOLE
  discriminator; star_like/SNR are BYPASSED (a bright real MIRI star on
  extended emission has bad qfit but high prominence -> kept; a flat-emission
  bump has low/NaN prominence -> dropped, including off-i2d sources).

Importing cataloging pulls crowdsource_catalogs_long (webbpsf) -> slow cold.
"""
import numpy as np
import pytest

from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging as C


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [30.5, 30.5]
    w.wcs.crval = [266.55, -28.80]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    w.wcs.cunit = ['deg', 'deg']
    return w


class TestStructureNoiseKeepRobust:
    """The mean-of-squares structure_noise is contaminated by the point sources
    themselves: a bright neighbour spikes the 51px window and inflates the
    rejection threshold for a faint source nearby, dropping real faint stars on
    the bg-subtracted m5/m6 residual (cloudc F770W m6: 2520->583).  robust=True
    measures the structure noise with a spike-immune sliding MAD instead.
    """

    @staticmethod
    def _scene():
        # flat near-zero background (bright stars already subtracted), one bright
        # residual blob and a genuine faint point source ~12 px away from it.
        rng = np.random.RandomState(0)
        data = rng.normal(0.0, 1.0, (121, 121)).astype(float)
        yy, xx = np.mgrid[0:121, 0:121]
        bright = 400.0 * np.exp(-(((xx - 54)**2 + (yy - 60)**2) / (2 * 2.0**2)))
        faint = 18.0 * np.exp(-(((xx - 66)**2 + (yy - 60)**2) / (2 * 2.0**2)))
        data += bright + faint
        err = np.ones_like(data)
        return data, err, np.array([66.0]), np.array([60.0])  # faint-source xy

    def test_meansq_drops_faint_neighbour_robust_keeps(self):
        data, err, x, y = self._scene()
        old = C._structure_noise_keep(data, err, xpix=x, ypix=y, struct_x=3.0,
                                      struct_y=4.0, robust=False)
        new = C._structure_noise_keep(data, err, xpix=x, ypix=y, struct_x=3.0,
                                      struct_y=8.0, robust=True)
        # the bright neighbour's spike makes the mean-of-squares prune reject the
        # genuine faint point source; the robust MAD prune keeps it.
        assert not bool(old[0])
        assert bool(new[0])

    def test_robust_still_rejects_diffuse_emission(self):
        # a broad emission plateau (no compact peak) must still be pruned by the
        # robust metric -- it is genuine structure, not a point source.
        rng = np.random.RandomState(1)
        data = rng.normal(0.0, 1.0, (121, 121)).astype(float)
        yy, xx = np.mgrid[0:121, 0:121]
        data += 12.0 * np.exp(-(((xx - 60)**2 + (yy - 60)**2) / (2 * 18.0**2)))
        err = np.ones_like(data)
        keep = C._structure_noise_keep(data, err, xpix=np.array([60.0]),
                                       ypix=np.array([60.0]), struct_x=3.0,
                                       struct_y=8.0, robust=True)
        assert not bool(keep[0])


class TestFilterExtendedEmissionNircam:
    def test_keeps_starlike_drops_high_qfit_low_snr(self):
        # No i2d image -> peakSB/prominence skipped; NIRCam keep-logic only.
        t = Table({
            'qfit': [0.1, 0.5],            # row0 star-like, row1 not
            'flags': [0, 0],
            'flux': [100.0, 100.0],
            'flux_err': [10.0, 10.0],      # snr = 10 for both (>= local_snr_min)
        })
        out = C._filter_extended_emission(t, min_prominence=0.0,
                                          qfit_max=0.2, local_snr_min=5.0,
                                          label='nircam-test')
        assert len(out) == 1
        np.testing.assert_allclose(out['qfit'][0], 0.1)

    def test_low_snr_dropped_if_starlike_but_NOT_qfit_confident(self):
        # The snr floor still drops a low-snr star_like source PROVIDED it is not
        # qfit-confident.  (Mechanism-1 fix `_emission_keep_nircam` KEEPS qfit<=
        # qfit_max sources regardless of snr -- their flux_err is inflated by
        # group-fit covariance -- so a qfit-confident low-snr source is NOT a
        # valid drop case; star_like here comes via keep_flags with qfit>qfit_max.)
        t = Table({
            'qfit': [0.30],               # ABOVE qfit_max -> not qfit-confident
            'flags': [1],                 # star_like via keep_flags=(1,)
            'flux': [10.0],
            'flux_err': [10.0],           # snr = 1 < local_snr_min
        })
        out = C._filter_extended_emission(t, min_prominence=0.0,
                                          qfit_max=0.2, local_snr_min=5.0,
                                          label='nircam-test')
        assert len(out) == 0

    def test_qfit_confident_kept_despite_broken_snr(self):
        # Mechanism 1: a qfit-confident source (group-fit covariance inflates
        # flux_err -> S/N ~0) must be KEPT regardless of the formal S/N.
        t = Table({
            'qfit': [0.016],
            'flags': [0],
            'flux': [6200.0],
            'flux_err': [8076.0],          # snr = 0.77 (broken by group degeneracy)
        })
        out = C._filter_extended_emission(t, min_prominence=0.0,
                                          qfit_max=0.2, local_snr_min=5.0,
                                          label='nircam-test')
        assert len(out) == 1

    def test_bright_isolated_keeps_borderline_qfit_star(self):
        # Mechanism 2 (sickle F480M star3: flux 6381, S/N 201, qfit 0.282): a
        # high-S/N SINGLETON (group_size==1 -> trustworthy S/N) with a still-
        # PSF-like qfit just above qfit_max is a real star -> KEPT.
        t = Table({'qfit': [0.282], 'flags': [0],
                   'flux': [6381.0], 'flux_err': [31.7],   # snr ~ 201
                   'group_size': [1.0]})
        out = C._filter_extended_emission(t, min_prominence=0.0, qfit_max=0.2,
                                          local_snr_min=5.0, label='nircam-test')
        assert len(out) == 1

    def test_bright_isolated_keeps_star2_faint_on_emission(self):
        # sickle F480M star2: real, concentrated+symmetric by eye, faint on BRIGHT
        # emission -> qfit 0.31 (>qfit_max), S/N ~23.6 singleton.  At the default
        # snr_high_keep=20 it is a >20sigma point source -> KEPT.
        t = Table({'qfit': [0.31], 'flags': [0],
                   'flux': [560.0], 'flux_err': [23.7],    # snr ~ 23.6
                   'group_size': [1.0]})
        out = C._filter_extended_emission(t, min_prominence=0.0, qfit_max=0.2,
                                          local_snr_min=5.0, label='nircam-test')
        assert len(out) == 1

    def test_bright_isolated_does_not_admit_grouped_or_faint_or_emission(self):
        # group_size>1 (S/N untrustworthy), faint (S/N<snr_high_keep), and
        # high-qfit (emission knot) variants must all be DROPPED.
        t = Table({'qfit': [0.30, 0.30, 0.60],
                   'flags': [0, 0, 0],
                   'flux': [6381.0, 360.0, 6381.0],
                   'flux_err': [31.7, 30.0, 31.7],         # snr 201, 12, 201
                   'group_size': [2.0, 1.0, 1.0]})         # grouped, faint, emission
        out = C._filter_extended_emission(t, min_prominence=0.0,
                                          qfit_max=0.2, local_snr_min=5.0,
                                          snr_high_keep=20.0,
                                          qfit_high_keep_max=0.4,
                                          label='nircam-test')
        assert len(out) == 0


class TestFilterExtendedEmissionMiri:
    def _image_and_catalog(self):
        ww = _wcs()
        ny = nx = 60
        yy, xx = np.mgrid[0:ny, 0:nx]
        img = np.ones((ny, nx)) + 0.01 * np.sin(xx) * np.cos(yy)  # texture -> MAD>0
        # bright narrow peak at (30, 30) -> high prominence
        img += 200.0 * np.exp(-(((xx - 30) ** 2 + (yy - 30) ** 2) / 2.0))
        # source A on the peak, source B on flat texture far from any peak
        sc = SkyCoord([ww.pixel_to_world(30, 30), ww.pixel_to_world(20, 20)])
        t = Table({
            'qfit': [0.9, 0.9],            # both bad qfit: NIRCam path would drop both
            'flags': [0, 0],
            'flux': [1.0, 1.0],
            'flux_err': [1.0, 1.0],        # snr 1: NIRCam path would drop both
            'skycoord': sc,
        })
        return img, ww, t

    def test_prominence_alone_keeps_real_star_bypasses_starlike(self):
        img, ww, t = self._image_and_catalog()
        out = C._filter_extended_emission(t, data_i2d_image=img, ww_i2d=ww,
                                          min_prominence=10.0, label='miri-test')
        # Only the prominent source A survives; star_like/snr are bypassed.
        assert len(out) == 1
        # surviving source is the one at the peak (RA/Dec of pixel 30,30)
        kept = SkyCoord(out['skycoord'])
        peak = ww.pixel_to_world(30, 30)
        assert kept[0].separation(peak).arcsec < 0.5

    def test_off_i2d_nan_prominence_dropped(self):
        # A source whose prominence cannot be measured (NaN) must be dropped
        # by the MIRI gate (off-i2d / other-obs footprint).
        img, ww, t = self._image_and_catalog()
        out = C._filter_extended_emission(t, data_i2d_image=img, ww_i2d=ww,
                                          min_prominence=10.0, label='miri-test')
        kept = SkyCoord(out['skycoord'])
        flat = ww.pixel_to_world(20, 20)
        assert all(k.separation(flat).arcsec > 0.5 for k in kept)


class TestZeroframeRecoverSaturated:
    """#2 ZEROFRAME saturated-rim recovery: de-inflate the brighter-fatter rim
    of the most-saturated stars from the ramp first read."""

    def _frame(self):
        from jwst.datamodels import dqflags
        ny = nx = 40
        yy, xx = np.mgrid[0:ny, 0:nx]
        r = np.hypot(xx - 20, yy - 20)
        # group-0 (DN): bright star profile, clipped at the saturation ceiling.
        g0 = 100.0 + 60000.0 * np.exp(-(r ** 2) / (2 * 3.0 ** 2))
        ceiling = 50000.0
        g0 = np.where(g0 > ceiling, ceiling, g0)
        Rtrue = 1.10
        cal = Rtrue * g0.copy()                 # cal = R*group0 at unsaturated px
        sat = r < 5                             # DQ-SATURATED core+rim
        dq = np.zeros((ny, nx), dtype=np.int32)
        dq[sat] = dqflags.pixel['SATURATED']
        # inflate the rim (saturated, group-0 not at ceiling) -> brighter-fatter
        rim_true = sat & (g0 < ceiling)
        cal[rim_true] = cal[rim_true] * 1.3
        return cal, dq, g0, r, Rtrue

    def test_deinflates_rim_flags_deepcore_leaves_far_untouched(self):
        from jwst_gc_pipeline.reduction.saturated_star_finding import (
            zeroframe_recover_saturated)
        cal, dq, g0, r, Rtrue = self._frame()
        rec, rim, deep, R = zeroframe_recover_saturated(
            cal, dq, g0, R_g0_min=2000.0, sat_dilate=3)
        # R recovered ~ the true ratio
        assert np.isfinite(R) and abs(R - Rtrue) < 0.05
        # rim pixels de-inflated (recovered below the inflated cal)
        assert rim.sum() > 0
        assert np.nanmedian(rec[rim]) < np.nanmedian(cal[rim])
        # deep core (group-0 also saturated) flagged, not recovered
        assert deep.sum() > 0
        # pixels far from the star are untouched
        far = r > 10
        assert np.allclose(rec[far], cal[far])

    def test_noop_without_group0(self):
        from jwst_gc_pipeline.reduction.saturated_star_finding import (
            zeroframe_recover_saturated)
        cal, dq, g0, r, Rtrue = self._frame()
        rec, rim, deep, R = zeroframe_recover_saturated(cal, dq, None)
        assert np.array_equal(rec, cal) and rim.sum() == 0 and deep.sum() == 0


# ---------------------------------------------------------------------------
# _dedup_combined_vetted: per-obs vetted combine dedup (cos(dec) + no-flux)
# ---------------------------------------------------------------------------
class TestDedupCombinedVetted:
    DEC0 = -28.8  # GC-ish declination, cos(dec) ~ 0.876

    def test_cosdec_scaling_finds_ra_duplicates(self):
        """Two rows 0.10" apart in TRUE angular distance along RA: the raw
        RA-degree Euclidean distance is 0.10/cos(dec) ~ 0.114" > the 0.11"
        radius, so the old (unscaled) call MISSED the duplicate."""
        cosd = np.cos(np.deg2rad(self.DEC0))
        dra = 0.10 / 3600.0 / cosd   # raw RA offset == 0.10" true separation
        t = Table()
        t['skycoord'] = SkyCoord([266.5, 266.5 + dra] * u.deg,
                                 [self.DEC0] * 2 * u.deg)
        t['flux'] = [100.0, 50.0]
        out = C._dedup_combined_vetted(t)
        assert len(out) == 1
        assert out['flux'][0] == 100.0    # brighter wins

    def test_distinct_sources_kept(self):
        cosd = np.cos(np.deg2rad(self.DEC0))
        dra = 0.20 / 3600.0 / cosd   # 0.20" true separation: NOT a duplicate
        t = Table()
        t['skycoord'] = SkyCoord([266.5, 266.5 + dra] * u.deg,
                                 [self.DEC0] * 2 * u.deg)
        t['flux'] = [100.0, 50.0]
        out = C._dedup_combined_vetted(t)
        assert len(out) == 2

    def test_no_flux_column_does_not_crash(self):
        """flux=None used to crash in _dedup_close_sources
        (np.isfinite(None)); without a flux column the dedup ranks
        uniformly and must still work."""
        t = Table()
        t['skycoord'] = SkyCoord([266.5, 266.5, 266.6] * u.deg,
                                 [self.DEC0] * 3 * u.deg)
        out = C._dedup_combined_vetted(t)
        assert len(out) == 2   # coincident pair deduped; distant row kept


# ---------------------------------------------------------------------------
# annotate_independent_detection: n_filt_independent across modules
# ---------------------------------------------------------------------------
def _annotate_setup(tmp_path, modules):
    from types import SimpleNamespace
    (tmp_path / 'catalogs').mkdir(exist_ok=True)
    ra0, dec0 = 266.5, -28.8
    merged = Table()
    merged['skycoord_ref'] = SkyCoord(
        [ra0, ra0 + 1e-3, ra0 + 2e-3] * u.deg, [dec0] * 3 * u.deg)
    merged_path = str(tmp_path / 'merged.fits')
    merged.write(merged_path, overwrite=True)

    def write_m6(module, idx):
        t = Table()
        t['skycoord'] = SkyCoord(merged['skycoord_ref'][idx])
        t.write(f'{tmp_path}/catalogs/f405n_{module}_indivexp_merged'
                f'_resbgsub_m6_dao_basic_vetted.fits', overwrite=True)

    options = SimpleNamespace(desaturated=False, bgsub=False, blur=False,
                              proposal_id='2221', field='001', modules=modules)
    return merged_path, write_m6, options


def test_annotate_independent_two_modules_counts_filter_once(tmp_path):
    """Regression (2026-07-21): the module-outer loop added ``indep`` to
    ``n_filt_independent`` once PER MODULE (source 0, seen by both modules,
    counted 2 for ONE filter) while the flag column kept only the LAST
    module's match.  Independence must be OR-ed across modules and counted
    once per filter."""
    merged_path, write_m6, options = _annotate_setup(tmp_path, 'nrca,nrcb')
    write_m6('nrca', [0])        # module A sees source 0
    write_m6('nrcb', [0, 1])     # module B sees sources 0 and 1

    C.annotate_independent_detection(merged_path, str(tmp_path), ['F405N'],
                                     options)
    out = Table.read(merged_path)
    # flag = OR over modules (source 1 was seen ONLY by nrca's partner nrcb;
    # before the fix the flag was last-module-only, which happened to work,
    # but a source seen only by the FIRST module lost its flag)
    assert list(np.asarray(out['independently_detected_f405n'], dtype=bool)) \
        == [True, True, False]
    # count: one filter -> at most 1, never 2
    assert list(np.asarray(out['n_filt_independent'])) == [1, 1, 0]


def test_annotate_independent_single_module_unchanged(tmp_path):
    merged_path, write_m6, options = _annotate_setup(tmp_path, 'nrca')
    write_m6('nrca', [0, 2])
    C.annotate_independent_detection(merged_path, str(tmp_path), ['F405N'],
                                     options)
    out = Table.read(merged_path)
    assert list(np.asarray(out['independently_detected_f405n'], dtype=bool)) \
        == [True, False, True]
    assert list(np.asarray(out['n_filt_independent'])) == [1, 0, 1]


def test_annotate_independent_first_module_only_flag_kept(tmp_path):
    """A source seen ONLY by the first module must keep its flag (the old
    code overwrote the column with the last module's matches)."""
    merged_path, write_m6, options = _annotate_setup(tmp_path, 'nrca,nrcb')
    write_m6('nrca', [1])        # ONLY module A sees source 1
    write_m6('nrcb', [2])        # module B sees source 2
    C.annotate_independent_detection(merged_path, str(tmp_path), ['F405N'],
                                     options)
    out = Table.read(merged_path)
    assert list(np.asarray(out['independently_detected_f405n'], dtype=bool)) \
        == [False, True, True]
    assert list(np.asarray(out['n_filt_independent'])) == [0, 1, 1]
