"""Regression tests for issue #159: ``--cutout-region`` runs could not write
their mergedcat ``_i2d`` mosaics.

A cutout / finite-crop i2d grid used to be built by composing a pixel ``Shift``
in front of the parent i2d's forward transform.  Modern ``gwcs`` writes a
rectified i2d WCS as a single atomic ``FITSImagingWCSTransform``, whose
projection is an ATTRIBUTE rather than a node of a model tree -- so the
composition satisfied neither of the two shapes ``jwst.resample`` accepts for a
custom ``output_wcs``, and every mergedcat residual / model resample died with::

    Custom 'output_wcs' does not match expected GWCS structure for an imaging
    WCS: could not find a Projection model in the output WCS transform.

The fix re-derives the cutout grid as a SUB-GRID of the parent grid (same
projection / cdelt / pc, CRPIX moved by the crop origin), which keeps the
FITS-imaging structure intact.

Everything here is hermetic: the synthetic frames are 120x120 px, so the whole
WCS-structure bug reproduces with no survey data.
"""
import os

import numpy as np
import pytest

from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L

SHAPE = (120, 120)
CRVAL = (266.5, -28.8)
CDELT = 1.0e-5          # deg/px == 0.036"/px


def _fits_imaging_gwcs(crpix=(60.0, 60.0), shape=SHAPE):
    """A rectified-i2d-like GWCS: one ``FITSImagingWCSTransform``, TAN."""
    import astropy.units as u
    from astropy.coordinates import ICRS
    from astropy.modeling import models
    from gwcs import wcs as gwcs_wcs, coordinate_frames as cf
    fitswcs = pytest.importorskip('gwcs.fitswcs')
    tr = fitswcs.FITSImagingWCSTransform(
        models.Pix2Sky_TAN(), crpix=list(crpix), crval=list(CRVAL),
        cdelt=[-CDELT, CDELT], pc=[[1.0, 0.0], [0.0, 1.0]])
    det = cf.Frame2D(name='detector', axes_names=('x', 'y'), unit=(u.pix, u.pix))
    # jwst.resample looks the output frame up by the name 'world'
    sky = cf.CelestialFrame(reference_frame=ICRS(), name='world',
                            unit=(u.deg, u.deg))
    w = gwcs_wcs.WCS([(det, tr), (sky, None)])
    w.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
    return w


def _write_frame(path, dither_x=0.0, seed=0):
    """A minimal but resample-able single-exposure datamodel.

    Only the central 60x60 box carries finite data, so the coadd lands on an
    over-allocated canvas and ``_crop_datamodel_to_finite`` fires -- the exact
    situation a ``--cutout-region`` run produces.
    """
    from stcal.alignment import util as align_util
    from stdatamodels.jwst.datamodels import ImageModel
    m = ImageModel(SHAPE)
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.1, SHAPE).astype('float32')
    data[SHAPE[0] // 2, SHAPE[1] // 2] += 100.0
    off_box = np.ones(SHAPE, dtype=bool)
    off_box[30:90, 30:90] = False
    data[off_box] = np.nan
    m.data = data
    m.err = np.full(SHAPE, 0.1, dtype='float32')
    m.dq = np.zeros(SHAPE, dtype='uint32')
    m.var_poisson = np.full(SHAPE, 0.01, dtype='float32')
    m.var_rnoise = np.full(SHAPE, 0.01, dtype='float32')
    m.var_flat = np.zeros(SHAPE, dtype='float32')
    m.meta.wcs = _fits_imaging_gwcs(crpix=(60.0 + dither_x, 60.0))
    m.meta.telescope = 'JWST'
    m.meta.filename = os.path.basename(path)
    m.meta.instrument.name = 'NIRCAM'
    m.meta.instrument.detector = 'NRCALONG'
    m.meta.instrument.channel = 'LONG'
    m.meta.instrument.filter = 'F480M'
    m.meta.instrument.pupil = 'CLEAR'
    m.meta.instrument.module = 'A'
    m.meta.exposure.type = 'NRC_IMAGE'
    m.meta.exposure.start_time = 60000.0
    m.meta.exposure.mid_time = 60000.0
    m.meta.exposure.end_time = 60000.1
    m.meta.exposure.exposure_time = 100.0
    m.meta.exposure.integration_time = 100.0
    m.meta.exposure.elapsed_exposure_time = 100.0
    m.meta.exposure.nints = 1
    m.meta.exposure.ngroups = 5
    m.meta.exposure.frame_time = 10.0
    m.meta.exposure.group_time = 10.0
    m.meta.observation.date = '2024-01-01'
    m.meta.observation.time = '00:00:00.000'
    m.meta.observation.date_beg = '2024-01-01T00:00:00.000'
    m.meta.observation.observation_number = '001'
    m.meta.observation.visit_number = '001'
    m.meta.observation.program_number = '06151'
    m.meta.target.ra, m.meta.target.dec = CRVAL
    m.meta.photometry.pixelarea_steradians = 1e-13
    m.meta.photometry.pixelarea_arcsecsq = (CDELT * 3600.0) ** 2
    m.meta.wcsinfo.v2_ref = 0.0
    m.meta.wcsinfo.v3_ref = 0.0
    m.meta.wcsinfo.roll_ref = 0.0
    m.meta.wcsinfo.ra_ref, m.meta.wcsinfo.dec_ref = CRVAL
    m.meta.wcsinfo.v3yangle = 0.0
    m.meta.wcsinfo.vparity = -1
    m.meta.wcsinfo.s_region = align_util.compute_s_region_imaging(m.meta.wcs,
                                                                  shape=SHAPE)
    m.meta.cal_step.assign_wcs = 'COMPLETE'
    m.save(path)
    m.close()
    return path


class TestShiftGwcsStructure:
    """``_shift_gwcs`` must keep a FITS-imaging WCS a FITS-imaging WCS."""

    def test_fits_imaging_transform_survives_the_shift(self):
        fitswcs = pytest.importorskip('gwcs.fitswcs')
        parent = _fits_imaging_gwcs()
        shifted = L._shift_gwcs(parent, 25, 26)
        assert isinstance(shifted.forward_transform,
                          fitswcs.FITSImagingWCSTransform), (
            'the cutout grid must remain a FITSImagingWCSTransform, else '
            'jwst.resample rejects it as a custom output_wcs (#159)')

    def test_shift_is_astrometrically_identical_to_the_old_composition(self):
        """The CRPIX sub-grid must reproduce the old Shift-composition exactly
        (it is the same transform, not an approximation of it)."""
        from astropy.modeling.models import Shift
        parent = _fits_imaging_gwcs()
        x0, y0 = 25, 26
        shifted = L._shift_gwcs(parent, x0, y0)
        composed = (Shift(float(x0)) & Shift(float(y0))) | parent.forward_transform
        for xy in [(0.0, 0.0), (13.5, 41.25), (68.0, 70.0)]:
            np.testing.assert_allclose(np.asarray(shifted(*xy)),
                                       np.asarray(composed(*xy)),
                                       rtol=0, atol=1e-12)

    def test_non_fits_imaging_wcs_falls_back_to_composition(self):
        """A detector-frame (distorted) WCS has no CRPIX-only sub-grid; it must
        still shift, and still expose a Projection node to resample."""
        import astropy.units as u
        from astropy.coordinates import ICRS
        from astropy.modeling import models
        from gwcs import wcs as gwcs_wcs, coordinate_frames as cf
        det2sky = ((models.Shift(0) & models.Shift(0))
                   | (models.Scale(CDELT) & models.Scale(CDELT))
                   | models.Pix2Sky_TAN()
                   | models.RotateNative2Celestial(CRVAL[0], CRVAL[1], 180))
        det = cf.Frame2D(name='detector', axes_names=('x', 'y'),
                         unit=(u.pix, u.pix))
        sky = cf.CelestialFrame(reference_frame=ICRS(), name='world',
                                unit=(u.deg, u.deg))
        parent = gwcs_wcs.WCS([(det, det2sky), (sky, None)])
        shifted = L._shift_gwcs(parent, 7, 9)
        np.testing.assert_allclose(np.asarray(shifted(0.0, 0.0)),
                                   np.asarray(parent(7.0, 9.0)),
                                   rtol=0, atol=1e-12)
        L._check_resample_output_wcs(shifted)   # must not raise


class TestResampleOutputWcsGuard:
    def test_accepts_a_fits_imaging_wcs(self):
        L._check_resample_output_wcs(_fits_imaging_gwcs())

    def test_rejects_a_shift_wrapped_fits_imaging_wcs(self):
        """The pre-fix shape.  Caught up front instead of after the drizzle."""
        import gwcs as _gwcs
        from astropy.modeling.models import Shift
        parent = _fits_imaging_gwcs()
        broken = _gwcs.WCS(
            forward_transform=((Shift(3.0) & Shift(4.0))
                               | parent.forward_transform),
            input_frame=parent.input_frame, output_frame=parent.output_frame)
        with pytest.raises(ValueError, match='not a valid resample target'):
            L._check_resample_output_wcs(broken, context='data_i2d')


@pytest.mark.crds
class TestCutoutMergedcatMosaicsAreWritten:
    """End-to-end: crop a coadd to its finite data, then land a second mosaic
    on that cropped grid -- the mergedcat residual/model path that failed."""

    def test_residual_and_model_i2d_land_on_the_cropped_data_grid(self, tmp_path,
                                                                  caplog):
        import logging
        from stdatamodels.jwst.datamodels import ImageModel
        pytest.importorskip('jwst.resample')
        caplog.set_level(logging.WARNING, logger='jwst.resample.resample')
        d = str(tmp_path)
        files = [_write_frame(os.path.join(d, f'frame{i}_cal.fits'),
                              dither_x=2.0 * i, seed=i) for i in range(2)]

        data_i2d = L._resample_to_i2d(files, d, 'synth_data', crop_to_data=True)
        assert data_i2d and os.path.exists(data_i2d)
        with ImageModel(data_i2d) as m:
            data_shape = m.data.shape
            data_corner = np.asarray(m.meta.wcs(0.0, 0.0))
        # the crop must actually have happened, else the bug is not exercised
        assert data_shape[0] < SHAPE[0] and data_shape[1] < SHAPE[1]

        shared = L._i2d_grid_output_wcs(data_i2d, os.path.join(d, 'grid.asdf'))
        assert shared is not None

        for product in ('synth_mergedcat_residual', 'synth_mergedcat_model'):
            out = L._resample_to_i2d(files, d, product, crop_to_data=False,
                                     output_wcs=shared)
            assert out and os.path.exists(out), (
                f'{product}_i2d.fits was not written -- issue #159')
            with ImageModel(out) as m:
                assert m.data.shape == data_shape
                np.testing.assert_allclose(np.asarray(m.meta.wcs(0.0, 0.0)),
                                           data_corner, rtol=0, atol=1e-12)

        # resample took the FITSImagingWCSTransform fast path: no linear-WCS
        # refit, so no "may not produce correct WCS parameters" caveat either.
        assert not [r for r in caplog.records
                    if "not using 'FITSImagingWCSTransform'" in r.getMessage()]

    def test_cropped_gwcs_matches_the_cropped_fits_header(self, tmp_path):
        """``_crop_datamodel_to_finite`` shifts the GWCS and the SCI FITS WCS by
        independent code paths; they must agree, or catalog sky positions and
        the displayed image drift apart."""
        from astropy.io import fits
        from astropy import wcs as astropy_wcs
        from stdatamodels.jwst.datamodels import ImageModel
        pytest.importorskip('jwst.resample')
        d = str(tmp_path)
        files = [_write_frame(os.path.join(d, f'frame{i}_cal.fits'),
                              dither_x=2.0 * i, seed=i) for i in range(2)]
        data_i2d = L._resample_to_i2d(files, d, 'synth_data', crop_to_data=True)
        with ImageModel(data_i2d) as m, fits.open(data_i2d) as h:
            fw = astropy_wcs.WCS(h['SCI'].header)
            ny, nx = m.data.shape
            for xy in [(0.0, 0.0), (nx / 2.0, ny / 2.0), (nx - 1.0, ny - 1.0)]:
                np.testing.assert_allclose(
                    np.asarray(m.meta.wcs(*xy)),
                    np.asarray(fw.pixel_to_world_values(*xy)),
                    rtol=0, atol=1e-9)


class TestPixelScaleSanityCheck:
    """The "CDELT is wrong" spam that accompanied every cutout drizzle."""

    @staticmethod
    def _cd_matrix_header():
        from astropy.io import fits
        h = fits.Header()
        h['NAXIS'] = 2
        h['NAXIS1'] = h['NAXIS2'] = 100
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        h['CRPIX1'] = h['CRPIX2'] = 50.0
        h['CRVAL1'], h['CRVAL2'] = CRVAL
        # CD-matrix form: wcslib normalises this to PC with CDELT == 1, which is
        # what made the old `cdelt[1] != 1` assertion cry wolf on every frame.
        h['CD1_1'] = -CDELT
        h['CD1_2'] = 0.0
        h['CD2_1'] = 0.0
        h['CD2_2'] = CDELT
        return h

    def test_cd_matrix_header_is_not_flagged(self, capsys):
        from astropy import wcs as astropy_wcs
        from jwst_gc_pipeline.reduction.saturated_star_finding import (
            _has_plausible_pixel_scale)
        ww = astropy_wcs.WCS(self._cd_matrix_header())
        assert ww.wcs.cdelt[1] == 1, (
            'precondition: a CD-matrix header really does report cdelt == 1')
        assert _has_plausible_pixel_scale(ww), (
            'a 0.036"/px CD-matrix WCS is valid; flagging it produced the '
            '"CDELT is wrong" spam of #159')

    def test_unset_identity_wcs_is_still_flagged(self):
        from astropy import wcs as astropy_wcs
        from jwst_gc_pipeline.reduction.saturated_star_finding import (
            _has_plausible_pixel_scale)
        h = self._cd_matrix_header()
        for key in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
            del h[key]
        assert not _has_plausible_pixel_scale(astropy_wcs.WCS(h)), (
            'a 1 deg/px identity WCS must still be reported'
        )


class TestMergedcatMosaicFailIsLoud:
    """A run that cannot write its mergedcat mosaics must not exit 0."""

    def test_failure_raises_by_default(self, monkeypatch):
        from jwst_gc_pipeline.photometry import cataloging
        monkeypatch.delenv(cataloging.MERGEDCAT_MOSAIC_OVERRIDE_ENV,
                           raising=False)
        with pytest.raises(cataloging.MergedcatMosaicError, match='m6'):
            cataloging._handle_mergedcat_mosaic_failure(
                'm6', 'nrcb', 'F480M', RuntimeError('boom'))

    def test_original_exception_is_chained(self, monkeypatch):
        from jwst_gc_pipeline.photometry import cataloging
        monkeypatch.delenv(cataloging.MERGEDCAT_MOSAIC_OVERRIDE_ENV,
                           raising=False)
        original = RuntimeError(
            "Custom 'output_wcs' does not match expected GWCS structure")
        with pytest.raises(cataloging.MergedcatMosaicError) as excinfo:
            cataloging._handle_mergedcat_mosaic_failure(
                'm6', 'nrcb', 'F480M', original)
        assert excinfo.value.__cause__ is original

    def test_override_env_restores_limp_through(self, monkeypatch):
        from jwst_gc_pipeline.photometry import cataloging
        monkeypatch.setenv(cataloging.MERGEDCAT_MOSAIC_OVERRIDE_ENV, '1')
        # returns instead of raising -- opt-in, never the default
        assert cataloging._handle_mergedcat_mosaic_failure(
            'm6', 'nrcb', 'F480M', RuntimeError('boom')) is None


class TestCutoutFitsWcsHeader:
    def test_old_linear_wcs_representation_is_replaced_not_merged(self):
        from astropy.io import fits
        from astropy import wcs as astropy_wcs
        old = TestPixelScaleSanityCheck._cd_matrix_header()
        new = fits.Header()
        new['CTYPE1'] = 'RA---TAN'
        new['CTYPE2'] = 'DEC--TAN'
        new['CRPIX1'] = new['CRPIX2'] = 10.0
        new['CRVAL1'], new['CRVAL2'] = CRVAL
        new['PC1_1'] = -1.0
        new['PC1_2'] = new['PC2_1'] = 0.0
        new['PC2_2'] = 1.0
        new['CDELT1'] = new['CDELT2'] = CDELT
        merged = L._update_fits_wcs_header(old.copy(), new)
        assert not any(k in merged for k in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')), (
            'stale CD keywords alongside the new PC/CDELT are a FITS-standard '
            'violation and make astropy silently ignore CDELT')
        w = astropy_wcs.WCS(merged)
        assert tuple(w.wcs.crpix) == (10.0, 10.0)
        np.testing.assert_allclose(np.abs(w.wcs.cdelt), [CDELT, CDELT])
