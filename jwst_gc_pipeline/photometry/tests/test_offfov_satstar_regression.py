"""Regression tests for out-of-field (forced) saturated-star handling and the
model/residual i2d background, pinning the 2026-06-17 fixes.

Covered regressions
-------------------
* Box artifact (runaway corner over-contribution): an off-FOV satstar fit in
  the spike-less corner of far frames runs away to 1e9-1e11 and, if not
  suppressed, paints a huge fake background ("box") into the model mosaic.
  reconcile_outside_fov_satstar_fluxes must pin such a discordant group to a
  ROBUST-LOW flux (reject the runaway outliers), so the box never returns.

* Off-FOV star wrongly DROPPED: an earlier "drop when no detector diversity"
  rule deleted real, fittable off-FOV stars from the model (sickle 17:46:12.67
  in F480M/F470N).  reconcile must NEVER drop a present source -- it must be
  included (overridden) and fitted.

* Negative model/residual i2d pedestal: per-frame model/residual datamodels
  inherit the input frame's meta.background.level with subtracted=False, so
  ResampleStep re-subtracts that sky level into a spurious uniform NEGATIVE
  pedestal.  save_residual_datamodel must zero meta.background.
"""
import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    reconcile_outside_fov_satstar_fluxes)

_STAR = SkyCoord(266.5658 * u.deg, -28.8041 * u.deg)


def _frame(flux, ddec, star=_STAR):
    t = QTable()
    t['flux_fit'] = [float(flux)]
    t['outside_fov_seed'] = [True]
    t['skycoord_fit'] = SkyCoord([star])
    t.meta['DET_RA'] = 266.5658
    t.meta['DET_DEC'] = -28.8041 + ddec
    return t


class TestBoxArtifactSuppressed:
    def test_runaway_corner_fits_pinned_low(self):
        """1e9/1e11 corner mis-fits must be rejected; pin tracks the ~1e5 cluster."""
        frames = [_frame(1.0e5, 0.02), _frame(1.2e5, 0.05), _frame(1.1e5, 0.20),
                  _frame(1.0e9, 0.30), _frame(1.0e11, 0.40)]
        ovr, drp = reconcile_outside_fov_satstar_fluxes(
            [(f'F{i}', t) for i, t in enumerate(frames)],
            match_radius=1.0 * u.arcsec, disagree_factor=2.0)
        assert len(drp) == 0
        assert len(ovr) == 1
        _, flux = ovr[0]
        # robust-low must be near the sane cluster, NOT the runaway values
        assert flux < 1.0e6, f"runaway not suppressed: pinned {flux:.3e}"
        assert flux >= 9.0e4

    def test_pinned_flux_never_exceeds_group_minimum_cluster(self):
        frames = [_frame(2.0e8, 0.0), _frame(2.2e8, 0.1), _frame(9.8e10, 0.3)]
        ovr, drp = reconcile_outside_fov_satstar_fluxes(
            [(f'F{i}', t) for i, t in enumerate(frames)],
            match_radius=1.0 * u.arcsec, disagree_factor=2.0)
        assert len(drp) == 0 and len(ovr) == 1
        assert ovr[0][1] <= 2.2e8 * 1.01


class TestOffFovStarAlwaysIncluded:
    def test_single_detector_disagreement_not_dropped(self):
        """No detector diversity must NOT cause a drop; star stays, pinned low."""
        d = 0.0005  # ~1.8" dithers, one detector
        frames = [_frame(5.1e8, 0 * d), _frame(3.8e8, 1 * d), _frame(0.9e8, 2 * d)]
        ovr, drp = reconcile_outside_fov_satstar_fluxes(
            [(f'F{i}', t) for i, t in enumerate(frames)],
            match_radius=1.0 * u.arcsec, disagree_factor=2.0)
        assert len(drp) == 0, "present off-FOV star must never be dropped"
        assert len(ovr) == 1, "discordant group must be pinned (included)"

    def test_no_drops_ever_returned(self):
        """drops list is always empty across assorted inputs."""
        for frames in (
            [_frame(1e5, 0.02), _frame(1e11, 0.4)],
            [_frame(2e8, 0.0), _frame(3e8, 0.001), _frame(8e10, 0.002)],
            [_frame(1e5, 0.02)],
        ):
            _, drp = reconcile_outside_fov_satstar_fluxes(
                [(f'F{i}', t) for i, t in enumerate(frames)],
                match_radius=1.0 * u.arcsec, disagree_factor=2.0)
            assert drp == []

    def test_consistent_family_left_untouched(self):
        frames = [_frame(1.0e5, 0.02), _frame(1.3e5, 0.20), _frame(0.8e5, 0.30)]
        ovr, drp = reconcile_outside_fov_satstar_fluxes(
            [(f'F{i}', t) for i, t in enumerate(frames)],
            match_radius=1.0 * u.arcsec, disagree_factor=2.0)
        assert ovr == [] and drp == []


class TestModelResidualBackgroundZeroed:
    @staticmethod
    def _trivial_gwcs():
        import astropy.units as u
        from astropy.coordinates import ICRS
        from astropy.modeling import models
        from gwcs import wcs as gwcs_wcs, coordinate_frames as cf
        det2sky = ((models.Shift(0) & models.Shift(0))
                   | (models.Scale(1e-5) & models.Scale(1e-5))
                   | models.Pix2Sky_TAN()
                   | models.RotateNative2Celestial(266.5, -28.8, 180))
        det = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
        sky = cf.CelestialFrame(reference_frame=ICRS(), name="icrs",
                                unit=(u.deg, u.deg))
        return gwcs_wcs.WCS([(det, det2sky), (sky, None)])

    def test_save_residual_datamodel_zeros_meta_background(self, tmp_path):
        from stdatamodels.jwst.datamodels import ImageModel
        from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
            save_residual_datamodel)
        # input frame carries a sky background level with subtracted=False
        m = ImageModel((16, 16))
        m.data = np.ones((16, 16), dtype='float32')
        m.meta.wcs = self._trivial_gwcs()
        m.meta.background.level = 12.3
        m.meta.background.subtracted = False
        inp = str(tmp_path / 'inp.fits')
        m.save(inp)
        m.close()
        outp = str(tmp_path / 'model.fits')
        save_residual_datamodel(inp, outp, np.ones((16, 16), dtype='float32'))
        with ImageModel(outp) as out:
            assert out.meta.background.subtracted is True
            assert (out.meta.background.level or 0.0) == 0.0
