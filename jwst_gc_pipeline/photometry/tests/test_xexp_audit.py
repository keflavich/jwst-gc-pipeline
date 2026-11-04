"""Cross-exposure audit: which exposures cover a merged source, which carry it.

``xexp_reconcile`` adds ``ncover`` (exposures whose array holds the position),
``nfound`` (exposures with a daophot or satstar counterpart) and
``single_exposure`` (bright or saturated, covered by two or more, carried by
exactly one) to the m2 merged catalog.  These tests pin the counting rules on
small synthetic frames and the placement of the call in ``cataloging.py``; the
numbers on real data are in the PR.
"""
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import xexp_reconcile as X
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

SCALE = 0.031                      # arcsec / pixel
# The fixtures hold a handful of rows, so the rank cut (brightest 5% of the
# multi-covered rows) would decide which of them is bright; it has its own tests.
PAR = X.XexpParams(single_top_pct=100.0)
SHAPE = (200, 200)
QUIET = dict(log=lambda *a, **k: None)
CUTOUT_PAD = 40.0                  # where a satstar fit sits in its stamp


def _usable(bad=None):
    u = np.ones(SHAPE, bool)
    if bad is not None:
        u[bad] = False
    return u


def _disk(cy, cx, r, shape=SHAPE):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _match(rows, cat=(), sat=(), usable=None, is_sat=None):
    """``rows`` = [(x, y, flux)] merged rows; ``cat`` = [(x, y, flux)] daophot
    rows of the frame; ``sat`` = [(x, y)] satstar rows.  Returns (covered, found)."""
    rows = np.asarray(rows, float).reshape(-1, 3)
    cat = np.asarray(cat, float).reshape(-1, 3)
    sat = np.asarray(sat, float).reshape(-1, 2)
    if is_sat is None:
        is_sat = np.zeros(len(rows), bool)
    return X.frame_cover_and_match(
        rows[:, 0], rows[:, 1], _usable() if usable is None else usable,
        cat[:, 0], cat[:, 1], cat[:, 2], sat[:, 0], sat[:, 1], rows[:, 2],
        np.asarray(is_sat, bool), SCALE, PAR)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def test_frame_catalog_names_in_merge_order_and_case_insensitive():
    meta = {'FN0': '/a/x0.fits', 'FN10': 'x10.fits', 'FN2': 'x2.fits',
            'fn1': 'x1.fits', 'FNAME': 'no.fits', 'FN': 'no.fits',
            'FILENAME': 'no.fits'}
    assert X.frame_catalog_names(meta) == ['x0.fits', 'x1.fits', 'x2.fits',
                                           'x10.fits']


def test_split_catalog_name():
    name = ('f212n_nrcb1_o078_visit001_vgroup02101_exp00001_hybpsf_m2_'
            'daophot_basic.fits')
    assert X.split_catalog_name(name, 'F212N') == (
        'nrcb1', '078', '_visit001_vgroup02101_exp00001_hybpsf_m2_daophot_basic')
    assert X.split_catalog_name(name, 'F480M') is None
    assert X.split_catalog_name('f770w_mirimage_o139_visit001_exp00002_m2_'
                                'daophot_basic.fits', 'F770W')[:2] == (
        'mirimage', '139')
    assert X.split_catalog_name('not_a_catalog.fits', 'F212N') is None


def test_usable_pixels_uses_the_instrument_bad_bits():
    dq = np.array([[0, 1, 2, 4]])
    np.testing.assert_array_equal(X.usable_pixels(dq, 3),
                                  [[True, False, False, True]])
    np.testing.assert_array_equal(X.usable_pixels(dq, 4),
                                  [[True, True, True, False]])


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_inside_the_array_is_covered_the_border_margin_is_not():
    cov, _ = _match([(100, 100, 1e3), (3, 100, 1e3), (100, 196, 1e3),
                     (np.nan, 100, 1e3)])
    assert cov.tolist() == [True, False, False, False]


def test_the_array_bounds_are_x_by_columns_and_y_by_rows():
    u = np.ones((80, 120), bool)                    # 80 rows, 120 columns
    rows = [(110, 40, 1e3), (116, 40, 1e3), (60, 76, 1e3), (3, 40, 1e3),
            (60, 3, 1e3)]
    cov, _ = _match(rows, usable=u)
    assert cov.tolist() == [True, False, False, False, False]


def test_a_saturated_core_is_a_hole_in_a_frame_that_holds_the_star():
    # a radius-10 core (317 px) around the star.  Real o078 F212N cores run to
    # ~380 px; a rule of "distance to a usable pixel <= 6" would drop these.
    u = _usable()
    u[_disk(100, 100, 10)] = False
    assert (~u).sum() > 300
    cov, _ = _match([(100, 100, 1e3)], usable=u)
    assert cov.tolist() == [True]


def test_hole_size_limit_is_max_hole_pix_inclusive():
    for rows, expect in ((slice(50, 90), True),     # 40 x 50 = 2000 px
                         (slice(50, 91), False)):   # 41 x 50 = 2050 px
        u = _usable()
        u[rows, 50:100] = False
        cov, _ = _match([(75, 60, 1e3)], usable=u)
        assert cov.tolist() == [expect]


def test_a_masked_strip_is_not_covered_usable_pixels_around_it_are():
    u = _usable()
    u[120:180, 20:80] = False                        # rows 120-179, cols 20-79
    # (x, y) = (50, 150) is inside the strip; (150, 50) is its mirror image
    cov, _ = _match([(50, 150, 1e3), (150, 50, 1e3)], usable=u)
    assert cov.tolist() == [False, True]


def test_a_core_near_the_edge_does_not_join_the_border_into_one_region():
    # a 6-px bad border ring (4656 px) with a radius-8 core 12 px from the top
    # edge: the core reaches into the ring.  Labelled with the ring, the two are
    # one region of ~4800 px and the star would count as not covered.
    u = _usable()
    u[:6, :] = u[-6:, :] = u[:, :6] = u[:, -6:] = False
    u[_disk(12, 100, 8)] = False
    cov, _ = _match([(100, 12, 1e3)], usable=u)
    assert cov.tolist() == [True]


@pytest.mark.parametrize('edge', ['low-y', 'high-y', 'low-x', 'high-x'])
def test_a_core_beside_each_edge_band_stays_its_own_region(edge):
    # A 6-px unusable band on all four edges and a 3 x 3 core touching the inner
    # side of one band.  The margin ring is cleared on every side before the
    # labelling, so the core is a 9-px region.  Left in on any one side, that
    # band (~1100 px) and the core are one region over max_hole_pix and the star
    # reads as not covered.
    ny, nx = SHAPE
    par = X.XexpParams(single_top_pct=100.0, max_hole_pix=100)
    u = _usable()
    u[:6, :] = u[-6:, :] = u[:, :6] = u[:, -6:] = False
    cy, cx = {'low-y': (7, 100), 'high-y': (ny - 8, 100),
              'low-x': (100, 7), 'high-x': (100, nx - 8)}[edge]
    u[cy - 1:cy + 2, cx - 1:cx + 2] = False
    assert X.covered_positions([cx], [cy], u, par).tolist() == [True]
    # the same core in the middle of the array is a hole of 9 px: covered too
    u2 = _usable()
    u2[99:102, 99:102] = False
    assert X.covered_positions([100], [100], u2, par).tolist() == [True]


def test_a_core_over_max_hole_pix_beside_an_edge_is_still_a_hole():
    # the clearing must not hide a real hole: a 12 x 12 core (144 px > 100)
    # touching the high-y band stays a region of its own and is not covered
    ny, nx = SHAPE
    par = X.XexpParams(single_top_pct=100.0, max_hole_pix=100)
    u = _usable()
    u[:6, :] = u[-6:, :] = u[:, :6] = u[:, -6:] = False
    u[ny - 18:ny - 6, 94:106] = False
    assert X.covered_positions([100], [ny - 12], u, par).tolist() == [False]


def test_all_bad_frame_covers_nothing():
    cov, found = _match([(100, 100, 1e3)], usable=np.zeros(SHAPE, bool))
    assert not cov.any() and not found.any()


# ---------------------------------------------------------------------------
# counterparts
# ---------------------------------------------------------------------------

def test_daophot_counterpart_needs_position_and_flux():
    r = 0.10 / SCALE                                  # 3.2 px
    cat = [(100.0 + 0.5 * r, 100.0, 900.0),           # close, flux x0.9: found
           (140.0 + 1.5 * r, 100.0, 1e3),             # too far
           (160.0, 100.0, 1e4),                       # close, flux x10: not it
           (30.0, 100.0, 100.0)]                      # close, flux x0.1: not it
    rows = [(100, 100, 1e3), (140, 100, 1e3), (160, 100, 1e3), (30, 100, 1e3)]
    cov, found = _match(rows, cat=cat)
    assert cov.all()
    assert found.tolist() == [True, False, False, False]


def test_a_nearer_row_with_the_wrong_flux_does_not_hide_the_counterpart():
    cat = [(100.5, 100, 1e5), (101.5, 100, 2e3), (99, 101, 1e5), (101, 99, 1e5),
           (100, 102, 1e5), (98, 100, 1e5), (100, 98, 1e5), (102, 102, 1e5)]
    _, found = _match([(100, 100, 1e3)], cat=cat)
    assert found.tolist() == [True]


def test_flux_ratio_bounds_are_the_factor_both_ways():
    for ratio, expect in ((2.9, True), (3.2, False), (1 / 2.9, True),
                          (1 / 3.2, False)):
        _, found = _match([(100, 100, 1e3)], cat=[(100.2, 100, 1e3 * ratio)])
        assert bool(found[0]) is expect, ratio


def test_saturated_row_matches_a_wider_unsaturated_daophot_row():
    # 0.2" (6.5 px) off: outside the daophot radius, inside the satstar one;
    # flux x10 and x0.5 both match (no upper bound), x0.1 does not
    off = 0.2 / SCALE
    for flux, expect in ((1e4, True), (500.0, True), (100.0, False)):
        _, found = _match([(100, 100, 1e3)], cat=[(100 + off, 100, flux)],
                          is_sat=[True])
        assert bool(found[0]) is expect, flux
    # the same offset does not match an unsaturated row
    _, found = _match([(100, 100, 1e3)], cat=[(100 + off, 100, 1e3)])
    assert not found[0]


def test_satstar_row_is_a_counterpart_without_a_daophot_row():
    off = 0.25 / SCALE
    cov, found = _match([(100, 100, 1e3), (160, 100, 1e3)],
                        sat=[(100 + off, 100), (160 + 2 * off, 100)])
    assert cov.all()
    assert found.tolist() == [True, False]


def test_found_implies_covered():
    # a catalog row sits on the source but the source is in the border margin
    cov, found = _match([(3, 100, 1e3)], cat=[(3, 100, 1e3)])
    assert not cov[0] and not found[0]


def test_empty_frame_catalogs_do_not_crash():
    cov, found = _match([(100, 100, 1e3)])
    assert cov.tolist() == [True] and found.tolist() == [False]


def test_non_positive_merged_flux_matches_nothing():
    _, found = _match([(100, 100, 0.0), (150, 150, -5.0)],
                      cat=[(100, 100, 1e3), (150, 150, 1e3)])
    assert found.tolist() == [False, False]


def test_matching_is_not_one_to_one():
    # Each merged row looks for a counterpart on its own, so one daophot row is
    # the counterpart of every merged row within the radius.  In a close pair
    # that makes `found` lenient (two rows, one detection, both found), and a row
    # whose own star the frame did not detect is found through its neighbour's
    # row.  `nfound` can therefore exceed the number of distinct detections.
    half = 0.025 / SCALE                              # rows 0.05" apart
    _, found = _match([(100 - half, 100, 1e3), (100 + half, 100, 1e3)],
                      cat=[(100, 100, 1e3)])
    assert found.tolist() == [True, True]
    _, found = _match([(100, 100, 1e3)], cat=[(100 + 0.08 / SCALE, 100, 1e3)])
    assert found.tolist() == [True]


# ---------------------------------------------------------------------------
# footprint guard
# ---------------------------------------------------------------------------

class _RotatedWCS:
    """Affine sky <-> pixel, rotated by 33 deg about the array centre."""
    RA0, DEC0, DEG, THETA = 266.0, -29.0, 1e-5, np.deg2rad(33.0)

    def _wrap(self, ra):
        return ra

    def _dra(self, ra):
        return np.asarray(ra, float) - self.RA0

    def pixel_to_world_values(self, x, y):
        c, s = np.cos(self.THETA), np.sin(self.THETA)
        xp, yp = np.asarray(x, float) - 100.0, np.asarray(y, float) - 100.0
        cosd = np.cos(np.deg2rad(self.DEC0))
        return (self._wrap(self.RA0 + (c * xp - s * yp) * self.DEG / cosd),
                self.DEC0 + (s * xp + c * yp) * self.DEG)

    def world_to_pixel_values(self, ra, dec):
        c, s = np.cos(self.THETA), np.sin(self.THETA)
        cosd = np.cos(np.deg2rad(self.DEC0))
        u = self._dra(ra) * cosd / self.DEG
        v = (np.asarray(dec, float) - self.DEC0) / self.DEG
        return c * u + s * v + 100.0, -s * u + c * v + 100.0


class _WrappedWCS(_RotatedWCS):
    """The same frame centred on RA = 0, RA returned in [0, 360): half of its
    edge samples sit near 0 and half near 360, so their plain mean is ~180."""
    RA0 = 0.0

    def _wrap(self, ra):
        return np.asarray(ra, float) % 360.0

    def _dra(self, ra):
        return ((np.asarray(ra, float) - self.RA0 + 180.0) % 360.0) - 180.0


@pytest.mark.parametrize('wcs_class', [_RotatedWCS, _WrappedWCS])
def test_footprint_guard_never_drops_a_row_that_lands_on_the_array(wcs_class):
    rng = np.random.default_rng(3)
    w = wcs_class()
    # a sky box four times the frame on a side
    x = rng.uniform(-300, 500, 20000)
    y = rng.uniform(-300, 500, 20000)
    ra, dec = w.pixel_to_world_values(x, y)
    near = X.footprint_candidates(w, SHAPE, ra, dec)
    px, py = w.world_to_pixel_values(ra, dec)
    on_array = (px >= 0) & (px <= SHAPE[1] - 1) & (py >= 0) & (py <= SHAPE[0] - 1)
    assert on_array.sum() > 500
    assert near[on_array].all()
    # and it prunes: most of the box is far from the frame
    assert near.sum() < 0.35 * len(ra)


@pytest.mark.parametrize('wcs_class', [_RotatedWCS, _WrappedWCS])
def test_footprint_guard_excludes_rows_far_beyond_each_side(wcs_class):
    w = wcs_class()
    cosd = np.cos(np.deg2rad(w.DEC0))
    far = 600 * w.DEG                                # three frame widths
    ra = (w.RA0 + np.array([0.0, far / cosd, -far / cosd, 0.0, 0.0])) % 360.0
    dec = w.DEC0 + np.array([0.0, 0.0, 0.0, far, -far])
    near = X.footprint_candidates(w, SHAPE, ra, dec)
    assert near.tolist() == [True, False, False, False, False]


def test_wrapped_frame_really_straddles_ra_zero():
    # the fixture only tests the wrap if its edges fall on both sides of RA = 0
    t = np.linspace(0.0, 199.0, 50)
    ra, _ = _WrappedWCS().pixel_to_world_values(
        np.r_[t, t], np.r_[np.zeros_like(t), np.full_like(t, 199.0)])
    assert (ra < 1.0).any() and (ra > 359.0).any()


def test_footprint_guard_drops_non_finite_rows():
    w = _RotatedWCS()
    ra, dec = w.pixel_to_world_values([100.0, 100.0], [100.0, 100.0])
    near = X.footprint_candidates(w, SHAPE, np.array([ra[0], np.nan]),
                                  np.array([dec[0], dec[1]]))
    assert near.tolist() == [True, False]


# ---------------------------------------------------------------------------
# per-frame files
# ---------------------------------------------------------------------------

class _LinearWCS:
    """sky = pixel * 1e-5 deg about (RA0, DEC0), flat; enough for the audit."""
    RA0, DEC0, DEG = 266.0, -29.0, 1e-5

    def world_to_pixel_values(self, ra, dec):
        return ((np.asarray(ra) - self.RA0) / self.DEG,
                (np.asarray(dec) - self.DEC0) / self.DEG)

    def pixel_to_world_values(self, x, y):
        return (self.RA0 + np.asarray(x, float) * self.DEG,
                self.DEC0 + np.asarray(y, float) * self.DEG)


def _sky(x, y):
    return _LinearWCS.RA0 + x * _LinearWCS.DEG, _LinearWCS.DEC0 + y * _LinearWCS.DEG


def _write_frame(tmp_path, tag, rows, sat_rows=(), bad=None, shape=SHAPE):
    """crf with a DQ plane, a per-frame catalog (x_fit, y_fit, flux_fit) and an
    optional satstar catalog.  ``rows`` = [(x, y, flux)]."""
    crf = str(tmp_path / f'{tag}_crf.fits')
    dq = np.zeros(shape, np.uint32)
    if bad is not None:
        dq[bad] = 1
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros(shape, np.float32), name='SCI'),
                  fits.ImageHDU(dq, name='DQ')]).writeto(crf)
    arr = np.asarray(rows, float).reshape(-1, 3)
    cat = Table({'x_fit': arr[:, 0], 'y_fit': arr[:, 1], 'flux_fit': arr[:, 2]})
    cat.meta['FILENAME'] = crf
    cat.meta['PIXSCALE'] = SCALE
    cpath = str(tmp_path / f'f212n_nrcb1_o078_{tag}_m2_daophot_basic.fits')
    cat.write(cpath)
    spath = ''
    if len(sat_rows):
        sarr = np.asarray(sat_rows, float).reshape(-1, 3)
        spath = crf.replace('.fits', '_m12_satstar_catalog.fits')
        # as the pipeline writes it: x_fit/y_fit are cutout-frame coordinates
        # (the fit runs in a pad-centred stamp), the detector pixels are
        # xcentroid/ycentroid
        Table({'x_fit': np.full(len(sarr), CUTOUT_PAD),
               'y_fit': np.full(len(sarr), CUTOUT_PAD),
               'flux_fit': sarr[:, 2],
               'xcentroid': sarr[:, 0], 'ycentroid': sarr[:, 1]}).write(spath)
    return X.FrameFiles(name=Path(cpath).name, detector='nrcb1', catalog=cpath,
                        crf=crf, satstar=spath)


def test_resolve_frame_files_follows_the_catalog_header(tmp_path):
    ff = _write_frame(tmp_path, 'a', [(50, 50, 5e3)], sat_rows=[(60, 60, 1e5)])
    got = X.resolve_frame_files(ff.name, catalog_dirs=['/nowhere', str(tmp_path)],
                                filt='F212N', satstar_suffix='_m12')
    assert got == ff
    # no satstar catalog for this suffix -> ''
    got = X.resolve_frame_files(ff.name, catalog_dirs=[str(tmp_path)],
                                filt='F212N', satstar_suffix='_m99')
    assert got.satstar == '' and got.crf == ff.crf


def test_resolve_frame_files_errors(tmp_path):
    ff = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])
    with pytest.raises(X.XexpError, match='not found'):
        X.resolve_frame_files(ff.name, catalog_dirs=['/nowhere'], filt='F212N',
                              satstar_suffix='')
    with pytest.raises(X.XexpError, match='not a F480M per-frame catalog'):
        X.resolve_frame_files(ff.name, catalog_dirs=[str(tmp_path)], filt='F480M',
                              satstar_suffix='')
    Path(ff.crf).unlink()
    with pytest.raises(X.XexpError, match='does not exist'):
        X.resolve_frame_files(ff.name, catalog_dirs=[str(tmp_path)],
                              filt='F212N', satstar_suffix='')


def test_resolve_frames_keeps_merge_order_and_marks_failures(tmp_path):
    a = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])
    c = _write_frame(tmp_path, 'c', [(50, 50, 5e3)])
    missing = 'f212n_nrcb1_o078_b_m2_daophot_basic.fits'
    frames = X.resolve_frames([a.name, missing, c.name],
                              catalog_dirs=[str(tmp_path)], filt='F212N',
                              satstar_suffix='_m12')
    assert [f.name for f in frames] == [a.name, missing, c.name]
    assert [bool(f.error) for f in frames] == [False, True, False]
    assert 'not found' in frames[1].error


def _age(path, mtime):
    os.utime(path, (mtime, mtime))


def _reference(tmp_path, mtime=2000):
    ref = str(tmp_path / 'merged.fits')
    Path(ref).write_text('merged')
    _age(ref, mtime)
    return ref


def test_frames_newer_than_counts_frames_rewritten_after_the_reference(tmp_path):
    a = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])
    b = _write_frame(tmp_path, 'b', [(50, 50, 5e3)])
    ref = _reference(tmp_path)
    for ff in (a, b):
        _age(ff.catalog, 1000)
        _age(ff.crf, 5000)                  # an input: its age is not the run's
    assert X.frames_newer_than([a, b], ref) == 0
    _age(b.catalog, 3000)
    assert X.frames_newer_than([a, b], ref) == 1
    _age(a.catalog, 3000)
    assert X.frames_newer_than([a, b], ref) == 2


def test_frames_newer_than_counts_a_frame_once_whichever_catalog_is_newer(tmp_path):
    both = _write_frame(tmp_path, 'a', [(50, 50, 5e3)], sat_rows=[(60, 60, 1e5)])
    sat_only = _write_frame(tmp_path, 'b', [(50, 50, 5e3)], sat_rows=[(60, 60, 1e5)])
    ref = _reference(tmp_path)
    _age(both.catalog, 3000)
    _age(both.satstar, 3000)
    _age(sat_only.catalog, 1000)
    _age(sat_only.satstar, 3000)
    assert X.frames_newer_than([both, sat_only], ref) == 2


def test_frames_newer_than_needs_a_strictly_later_write(tmp_path):
    a = _write_frame(tmp_path, 'a', [(50, 50, 5e3)], sat_rows=[(60, 60, 1e5)])
    ref = _reference(tmp_path)
    _age(a.catalog, 2000)
    _age(a.satstar, 2000)
    assert X.frames_newer_than([a], ref) == 0


def test_frames_newer_than_skips_a_frame_without_a_satstar_catalog_or_files(tmp_path):
    a = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])        # satstar == ''
    ref = _reference(tmp_path)
    _age(a.catalog, 1000)
    unresolved = X.FrameFiles(name='f212n_nrcb1_o078_b_m2_daophot_basic.fits',
                              detector='', catalog='', crf='', satstar='',
                              error='XexpError: not found')
    assert X.frames_newer_than([a, unresolved], ref) == 0
    _age(a.catalog, 3000)
    assert X.frames_newer_than([a, unresolved], ref) == 1


def test_frames_newer_than_of_no_frames_is_zero(tmp_path):
    assert X.frames_newer_than([], _reference(tmp_path)) == 0


# ---------------------------------------------------------------------------
# end to end on written files
# ---------------------------------------------------------------------------

def _reconcile(frames, merged, is_sat, params=PAR, wcs_loader=None):
    xy = np.asarray([(x, y) for x, y, _ in merged], float)
    ra, dec = _sky(xy[:, 0], xy[:, 1])
    flux = np.asarray([f for _, _, f in merged], float)
    return X.reconcile(ra, dec, flux, flux / 100.0, np.asarray(is_sat, bool),
                       frames, params,
                       wcs_loader=wcs_loader or (lambda crf: _LinearWCS()), **QUIET)


def test_reconcile_counts_and_flags(tmp_path):
    #            0: carried by all    1: bright, one frame of three
    #            2: bright, one       3: bright, one
    #            4: saturated, one frame (satstar row)
    merged = [(50, 50, 5e3), (100, 100, 8e3), (150, 150, 2e3),
              (60, 60, 5e3), (170, 60, 2e5)]
    is_sat = [False, False, False, False, True]
    f0 = _write_frame(tmp_path, 'a', [(50, 50, 5e3), (100, 100, 8e3),
                                      (150, 150, 2e3), (60, 60, 5e3)],
                      sat_rows=[(170, 60, 2e5)])
    f1 = _write_frame(tmp_path, 'b', [(50, 50, 5e3)])
    f2 = _write_frame(tmp_path, 'c', [(50, 50, 5e3)])
    res = _reconcile([f0, f1, f2], merged, is_sat)
    assert res.ncover.tolist() == [3, 3, 3, 3, 3]
    assert res.nfound.tolist() == [3, 1, 1, 1, 1]
    # S/N = flux / (flux/100) = 100 for all rows: all bright; row 0 is carried
    # by every frame, so it is the only unflagged multi-frame row
    assert res.single_exposure.tolist() == [False, True, True, True, True]
    assert res.frames['status'].tolist() == ['ok', 'ok', 'ok']
    assert res.frames['n_cover'].tolist() == [5, 5, 5]
    assert res.frames['n_found'].tolist() == [5, 1, 1]


def test_a_satstar_row_is_found_by_its_detector_position_not_its_stamp_position(tmp_path):
    # Real satstar catalogs put x_fit/y_fit in the fit stamp, so every row reads
    # about (pad, pad); the first real-data run matched on those and found no
    # satstar counterpart anywhere.
    merged = [(120, 130, 2e5), (CUTOUT_PAD, CUTOUT_PAD, 2e5)]
    f0 = _write_frame(tmp_path, 'a', [], sat_rows=[(120, 130, 2e5)])
    f1 = _write_frame(tmp_path, 'b', [])
    res = _reconcile([f0, f1], merged, [True, True])
    assert res.ncover.tolist() == [2, 2]
    assert res.nfound.tolist() == [1, 0]
    assert res.single_exposure.tolist() == [True, False]


def test_read_xyf_of_a_satstar_catalog_takes_the_detector_frame_columns(tmp_path):
    ff = _write_frame(tmp_path, 'a', [(50, 50, 5e3)],
                      sat_rows=[(120, 60, 1e5), (30, 140, 2e5)])
    x, y, f = X._read_xyf(ff.satstar, detector_frame=True)
    assert (x.tolist(), y.tolist(), f.tolist()) == (
        [120.0, 30.0], [60.0, 140.0], [1e5, 2e5])
    x, y, _ = X._read_xyf(ff.satstar)                 # a daophot catalog's columns
    assert x.tolist() == [CUTOUT_PAD] * 2 and y.tolist() == [CUTOUT_PAD] * 2
    x, y, _ = X._read_xyf(ff.catalog, detector_frame=True)
    assert (x.tolist(), y.tolist()) == ([50.0], [50.0])


@pytest.mark.parametrize('present', [(), ('xcentroid',), ('ycentroid',)])
def test_a_satstar_catalog_without_both_centroids_falls_back_to_x_fit(tmp_path, present):
    ff = _write_frame(tmp_path, 'a', [], sat_rows=[(120, 60, 1e5)])
    cols = {'x_fit': [120.0], 'y_fit': [60.0], 'flux_fit': [1e5]}
    cols.update({c: [7.0] for c in present})
    Table(cols).write(ff.satstar, overwrite=True)
    x, y, _ = X._read_xyf(ff.satstar, detector_frame=True)
    assert (x.tolist(), y.tolist()) == ([120.0], [60.0])


def test_single_exposure_needs_brightness_and_two_covering_frames(tmp_path):
    merged = [(50, 50, 5e3), (100, 100, 5e3), (60, 60, 5e3)]
    # frame b is 80 rows x 120 columns, so (100, 100) is off its array
    f0 = _write_frame(tmp_path, 'a', [(50, 50, 5e3), (100, 100, 5e3), (60, 60, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [], shape=(80, 120))
    xy = np.asarray([(x, y) for x, y, _ in merged], float)
    ra, dec = _sky(xy[:, 0], xy[:, 1])
    flux = np.array([5e3, 5e3, 5e3])
    err = np.array([5e2, 5e1, 5e1])                      # S/N 10, 100, 100
    res = X.reconcile(ra, dec, flux, err, np.zeros(3, bool), [f0, f1], PAR,
                      wcs_loader=lambda crf: _LinearWCS(), **QUIET)
    assert res.ncover.tolist() == [2, 1, 2]
    assert res.nfound.tolist() == [1, 1, 1]
    # row 0: faint; row 1: one covering frame; row 2: bright, two cover, one carries
    assert res.single_exposure.tolist() == [False, False, True]


def test_the_brightness_cut_is_single_snr_min_inclusive(tmp_path):
    f0 = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [])
    ra, dec = _sky(np.array([100.0]), np.array([100.0]))
    out = {}
    for snr_min in (49.0, 50.0, 51.0):
        res = X.reconcile(ra, dec, np.array([5e3]), np.array([100.0]),   # S/N 50
                          np.zeros(1, bool), [f0, f1],
                          X.XexpParams(single_snr_min=snr_min, single_top_pct=100.0),
                          wcs_loader=lambda crf: _LinearWCS(), **QUIET)
        out[snr_min] = bool(res.single_exposure[0])
    assert out == {49.0: True, 50.0: True, 51.0: False}


def test_a_row_no_frame_carries_is_not_flagged(tmp_path):
    # covered by both frames, carried by neither (an orphan of the vetting
    # steps): there is no exposure that "has" it, so nothing to flag
    f0 = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [(50, 50, 5e3)])
    res = _reconcile([f0, f1], [(50, 50, 5e3), (120, 120, 5e3)], [False, False])
    assert res.ncover.tolist() == [2, 2] and res.nfound.tolist() == [2, 0]
    assert res.single_exposure.tolist() == [False, False]


def test_failed_fits_in_a_frame_are_not_counterparts(tmp_path):
    # NaN or non-positive flux, or a NaN position, is a failed fit, in the
    # daophot rows as in the satstar rows (which match on position alone)
    bad = [(100, 100, np.nan), (120, 120, -3.0), (np.nan, 140, 5e3),
           (140, np.nan, 5e3)]
    f0 = _write_frame(tmp_path, 'a', bad, sat_rows=bad)
    f1 = _write_frame(tmp_path, 'b', [(100, 100, 5e3)])
    rows = [(100, 100, 5e3), (120, 120, 5e3), (140, 140, 5e3)]
    res = _reconcile([f0, f1], rows, [True, True, True])
    assert res.ncover.tolist() == [2, 2, 2]
    assert res.nfound.tolist() == [1, 0, 0]


def test_pixel_scale_comes_from_the_header_or_else_from_the_wcs(tmp_path):
    ff = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])
    assert X._pixel_scale_arcsec(ff.catalog, _LinearWCS(), SHAPE) == pytest.approx(SCALE)
    tab = Table.read(ff.catalog)
    tab.meta.pop('PIXSCALE')
    bare = str(tmp_path / 'bare.fits')
    tab.write(bare)
    # 1e-5 deg of RA per pixel, at the declination of the array centre (row 100)
    want = 1e-5 * np.cos(np.deg2rad(-29.0 + 100 * 1e-5)) * 3600.0
    assert X._pixel_scale_arcsec(bare, _LinearWCS(), SHAPE) == pytest.approx(want)


def test_a_hole_around_a_bright_star_in_one_frame_still_counts_as_covered(tmp_path):
    # frame b has a radius-10 saturated/NaN core on the star and no daophot row
    # for it; frame a carries it.  Covered twice, carried once: flagged.
    merged = [(100, 100, 5e3)]
    f0 = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [], bad=_disk(100, 100, 10))
    res = _reconcile([f0, f1], merged, [False])
    assert res.ncover.tolist() == [2] and res.nfound.tolist() == [1]
    assert res.single_exposure.tolist() == [True]


def test_unreadable_frame_is_skipped_and_listed(tmp_path):
    merged = [(100, 100, 5e3)]
    ok = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    gone = X.FrameFiles(name='gone.fits', detector='nrcb1',
                        catalog=str(tmp_path / 'gone_cat.fits'),
                        crf=str(tmp_path / 'gone_crf.fits'), satstar='')
    res = _reconcile([ok, gone], merged, [False])
    assert res.ncover.tolist() == [1] and res.nfound.tolist() == [1]
    assert res.frames['status'].tolist() == ['ok', 'skipped']
    assert 'gone_crf' in res.frames['note'][1]


def _corrupt(path, how):
    raw = Path(path).read_bytes()
    Path(path).write_bytes({'empty': b'', 'text': b'not a fits file\n',
                            'cut-in-data': raw[:-2880]}[how])


@pytest.mark.parametrize('how', ['empty', 'text', 'cut-in-data'])
@pytest.mark.parametrize('which', ['catalog', 'satstar'])
def test_a_damaged_frame_table_skips_that_frame_and_nothing_else(tmp_path, how, which):
    # 500 rows span several FITS blocks, so cutting the last block keeps the header
    rows = [(20 + i % 150, 20 + 10 * (i // 150), 1e5) for i in range(500)]
    merged = [(100, 100, 1e5)]
    ok = _write_frame(tmp_path, 'a', [(100, 100, 1e5)])
    bad = _write_frame(tmp_path, 'b', rows, sat_rows=rows if which == 'satstar' else ())
    _corrupt(bad.catalog if which == 'catalog' else bad.satstar, how)
    res = _reconcile([ok, bad], merged, [False])
    assert res.ncover.tolist() == [1] and res.nfound.tolist() == [1]
    assert res.frames['status'].tolist() == ['ok', 'skipped']
    assert res.frames['note'][1]


def test_a_damaged_crf_skips_that_frame(tmp_path):
    merged = [(100, 100, 1e5)]
    ok = _write_frame(tmp_path, 'a', [(100, 100, 1e5)])
    bad = _write_frame(tmp_path, 'b', [(100, 100, 1e5)])
    Path(bad.crf).write_bytes(Path(bad.crf).read_bytes()[:3000])
    res = _reconcile([ok, bad], merged, [False])
    assert res.ncover.tolist() == [1]
    assert res.frames['status'].tolist() == ['ok', 'skipped']


def test_a_gwcs_that_cannot_be_read_skips_that_frame(tmp_path):
    # frame_wcs raises GwcsReadError (a RuntimeError) for a product whose ASDF
    # extension is unreadable; that frame is dropped, the rest are still audited
    from jwst_gc_pipeline.frame_wcs import GwcsReadError
    merged = [(100, 100, 1e5)]
    ok = _write_frame(tmp_path, 'a', [(100, 100, 1e5)])
    bad = _write_frame(tmp_path, 'b', [(100, 100, 1e5)])

    def loader(crf):
        if crf == bad.crf:
            raise GwcsReadError(f'{crf} has an ASDF extension but its GWCS could not be read')
        return _LinearWCS()

    res = _reconcile([ok, bad], merged, [False], wcs_loader=loader)
    assert res.ncover.tolist() == [1] and res.nfound.tolist() == [1]
    assert res.frames['status'].tolist() == ['ok', 'skipped']
    assert 'GwcsReadError' in res.frames['note'][1]


def test_a_damaged_catalog_is_listed_by_the_merged_audit_not_fatal(tmp_path):
    f0 = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [(100, 100, 5e3)])
    tab = _merged_table([f0, f1], [(100, 100, 5e3)])
    _corrupt(f1.catalog, 'empty')
    msgs = []
    _audit(tab, tmp_path, msgs)
    assert tab['ncover'].tolist() == [1]
    assert tab.meta['XEXPFRM'] == 1 and tab.meta['XEXPSKP'] == 1
    assert any('skipped' in m for m in msgs)


def test_unresolved_frame_is_skipped_with_its_reason(tmp_path):
    merged = [(100, 100, 5e3)]
    ok = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    bad = X.FrameFiles(name='b.fits', detector='', catalog='', crf='', satstar='',
                       error='XexpError: b.fits: not found in []')
    res = _reconcile([ok, bad], merged, [False])
    assert res.ncover.tolist() == [1]
    assert res.frames['status'].tolist() == ['ok', 'skipped']
    assert res.frames['note'][1] == 'XexpError: b.fits: not found in []'


def test_add_audit_columns_and_header(tmp_path):
    merged = [(100, 100, 5e3), (150, 150, 5e3)]
    f0 = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [(100, 100, 5e3), (150, 150, 5e3)])
    res = _reconcile([f0, f1], merged, [False, False])
    tab = Table({'flux': [5e3, 5e3]})
    X.add_audit_columns(tab, res, PAR)
    assert tab['ncover'].dtype == np.int16 and tab['nfound'].dtype == np.int16
    assert tab['single_exposure'].dtype == bool
    assert tab.meta['XEXPVER'] == X.SCHEMA
    assert tab.meta['XEXPFRM'] == 2 and tab.meta['XEXPSKP'] == 0
    assert tab.meta['XEXPSNR'] == 50.0 and tab.meta['XEXPTOP'] == 100.0
    assert tab['ncover'].tolist() == [2, 2]
    assert tab['nfound'].tolist() == [2, 1]


def test_reconcile_rank_cut_is_taken_among_the_rows_two_frames_cover(tmp_path):
    # rows 0-3 are covered by all three frames and carried by frame a only; row 4,
    # the brightest, is covered by frame a alone (a 30 px hole of bad pixels in b
    # and c), so it is not in the pool the rank is taken from
    merged = [(40, 40, 2e3), (40, 80, 5e3), (80, 40, 6e3), (80, 80, 8e3), (150, 150, 1e7)]
    hole = _disk(150, 150, 30)
    f0 = _write_frame(tmp_path, 'a', merged)
    f1 = _write_frame(tmp_path, 'b', [], bad=hole)
    f2 = _write_frame(tmp_path, 'c', [], bad=hole)
    res = _reconcile([f0, f1, f2], merged, [False] * 5, params=X.XexpParams(single_top_pct=40.0))
    assert res.ncover.tolist() == [3, 3, 3, 3, 1]
    assert res.nfound.tolist() == [1, 1, 1, 1, 1]
    # pool 2e3..8e3: the top 40% is flux >= 5.8e3; with row 4 in the pool it would be 6.8e3
    assert res.single_exposure.tolist() == [False, False, True, True, False]
    res = _reconcile([f0, f1, f2], merged, [False] * 5, params=PAR)
    assert res.single_exposure.tolist() == [True, True, True, True, False]


def test_reconcile_flags_a_faint_saturated_row_whatever_the_rank_cut(tmp_path):
    merged = [(40, 40, 2e3), (80, 80, 8e3)]
    f0 = _write_frame(tmp_path, 'a', [(80, 80, 8e3)], sat_rows=[(40, 40, 2e3)])
    f1 = _write_frame(tmp_path, 'b', [(80, 80, 8e3)])
    res = _reconcile([f0, f1], merged, [True, False], params=X.XexpParams(single_top_pct=5.0))
    assert res.ncover.tolist() == [2, 2] and res.nfound.tolist() == [1, 2]
    assert res.single_exposure.tolist() == [True, False]


def test_header_records_the_rank_cut_it_was_made_with(tmp_path):
    f0 = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    res = _reconcile([f0], [(100, 100, 5e3)], [False])
    tab = Table({'flux': [5e3]})
    X.add_audit_columns(tab, res, X.XexpParams(single_snr_min=30.0, single_top_pct=7.5))
    assert tab.meta['XEXPSNR'] == 30.0 and tab.meta['XEXPTOP'] == 7.5


def test_audit_merged_catalog_applies_the_rank_cut_of_its_params(tmp_path):
    rows = [(60, 60, 1e5), (120, 120, 9e5)]
    f0 = _write_frame(tmp_path, 'a', rows)
    f1 = _write_frame(tmp_path, 'b', [rows[1]])
    f2 = _write_frame(tmp_path, 'c', [rows[1]])
    for params, expect in ((PAR, [True, False]),
                           (X.XexpParams(single_top_pct=5.0), [False, False])):
        tab = _merged_table([f0, f1, f2], rows)
        X.audit_merged_catalog(
            tab, filt='F212N', catalog_dirs=[str(tmp_path)], satstar_suffix='_m12',
            params=params, wcs_loader=lambda crf: _LinearWCS(), bad_dq_mask=3,
            log=lambda *a, **k: None)
        assert tab['single_exposure'].tolist() == expect
        assert tab.meta['XEXPTOP'] == params.single_top_pct


def _merged_table(frames, rows, is_sat=None):
    xy = np.asarray([(x, y) for x, y, _ in rows], float)
    ra, dec = _sky(xy[:, 0], xy[:, 1])
    flux = np.asarray([f for _, _, f in rows], float)
    tab = Table({'skycoord': SkyCoord(ra, dec, unit='deg'), 'flux': flux,
                 'flux_err': flux / 100.0,
                 'is_saturated': np.zeros(len(rows), bool) if is_sat is None
                 else np.asarray(is_sat, bool)})
    for i, ff in enumerate(frames):
        tab.meta[f'fn{i}'] = ff.name
    return tab


def _audit(tab, tmp_path, msgs):
    return X.audit_merged_catalog(
        tab, filt='F212N', catalog_dirs=[str(tmp_path)], satstar_suffix='_m12',
        params=PAR, wcs_loader=lambda crf: _LinearWCS(), bad_dq_mask=3,
        log=msgs.append)


def test_audit_merged_catalog_adds_the_columns(tmp_path):
    f0 = _write_frame(tmp_path, 'a', [(50, 50, 5e3), (100, 100, 8e3)],
                      sat_rows=[(170, 60, 2e5)])
    f1 = _write_frame(tmp_path, 'b', [(50, 50, 5e3)])
    tab = _merged_table([f0, f1],
                        [(50, 50, 5e3), (100, 100, 8e3), (170, 60, 2e5)],
                        is_sat=[False, False, True])
    msgs = []
    assert _audit(tab, tmp_path, msgs) is not None
    assert tab['ncover'].tolist() == [2, 2, 2]
    assert tab['nfound'].tolist() == [2, 1, 1]
    assert tab['single_exposure'].tolist() == [False, True, True]
    assert tab.meta['XEXPFRM'] == 2 and tab.meta['XEXPSKP'] == 0
    assert any('flagged single_exposure 2' in m for m in msgs)


def test_audit_merged_catalog_keeps_the_rows_and_columns_it_was_given(tmp_path):
    f0 = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [])
    tab = _merged_table([f0, f1], [(50, 50, 5e3), (100, 100, 8e3)])
    before = {c: np.asarray(tab[c]).copy() for c in ('flux', 'flux_err')}
    names = list(tab.colnames)
    _audit(tab, tmp_path, [])
    assert len(tab) == 2
    assert tab.colnames == names + ['ncover', 'nfound', 'single_exposure']
    for c, v in before.items():
        np.testing.assert_array_equal(np.asarray(tab[c]), v)


def test_audit_merged_catalog_lists_an_unresolvable_frame(tmp_path):
    f0 = _write_frame(tmp_path, 'a', [(50, 50, 5e3)])
    tab = _merged_table([f0], [(50, 50, 5e3)])
    tab.meta['fn1'] = 'f212n_nrcb1_o078_zz_m2_daophot_basic.fits'
    msgs = []
    _audit(tab, tmp_path, msgs)
    assert tab.meta['XEXPFRM'] == 1 and tab.meta['XEXPSKP'] == 1
    assert tab['ncover'].tolist() == [1]
    assert any('WARNING' in m and 'zz_m2' in m for m in msgs)


def test_audit_merged_catalog_without_fn_cards_leaves_the_table_alone(tmp_path):
    tab = Table({'skycoord': SkyCoord([266.0], [-29.0], unit='deg'),
                 'flux': [1.0]})
    msgs = []
    assert _audit(tab, tmp_path, msgs) is None
    assert tab.colnames == ['skycoord', 'flux']
    assert any('no fn cards' in m for m in msgs)


def test_is_saturated_or_replaced_saturated_marks_a_row_saturated(tmp_path):
    # a faint row (S/N 10) the merge replaced from a satstar fit counts as bright
    f0 = _write_frame(tmp_path, 'a', [(100, 100, 5e3)])
    f1 = _write_frame(tmp_path, 'b', [])
    tab = _merged_table([f0, f1], [(100, 100, 5e3)])
    tab['flux_err'] = [5e2]                                  # S/N 10
    _audit(tab, tmp_path, [])
    assert tab['single_exposure'].tolist() == [False]
    del tab['ncover'], tab['nfound'], tab['single_exposure']
    tab['replaced_saturated'] = [True]
    _audit(tab, tmp_path, [])
    assert tab['single_exposure'].tolist() == [True]


# ---------------------------------------------------------------------------
# brightness and saturation
# ---------------------------------------------------------------------------

def test_bright_rows_is_saturated_or_snr_at_the_cut_inclusive():
    flux = np.array([100., 99., 100., 10., np.nan, 100., -100., 100., np.nan, np.inf, -100.])
    err = np.array([2., 2., 0., 2., 2., np.nan, 2., -2., 2., 2., -2.])
    sat = [False, False, False, True, False, False, False, False, True, False, False]
    got = X.bright_rows(flux, err, sat, 50.0)
    assert got.dtype == bool
    assert got.tolist() == [
        True,    # S/N 50.0 at a cut of 50.0
        False,   # S/N 49.5
        False,   # no error estimate
        True,    # saturated, faint
        False,   # no flux
        False,   # error is NaN
        False,   # negative flux
        False,   # negative error
        True,    # saturated, no flux
        False,   # infinite S/N is a failed fit, not a bright star
        False,   # negative flux over a negative error is not S/N +50
    ]


def test_bright_rows_follows_the_cut():
    flux, err = np.array([100., 100.]), np.array([2., 4.])
    assert X.bright_rows(flux, err, [False, False], 25.0).tolist() == [True, True]
    assert X.bright_rows(flux, err, [False, False], 26.0).tolist() == [True, False]


def _snr100(flux):
    """errors that put every row at S/N 100."""
    return np.asarray(flux, float) / 100.0


def test_bright_rows_rank_cut_keeps_the_brightest_percent():
    flux = np.arange(1., 101.)
    got = X.bright_rows(flux, _snr100(flux), np.zeros(100, bool), 50.0, top_pct=10.0)
    assert got.sum() == 10 and got[90:].all() and not got[:90].any()   # np.quantile(flux, 0.9) = 90.1


def test_bright_rows_keeps_the_row_at_the_rank_cut():
    flux = np.array([1., 2., 3., 4., 5.])
    got = X.bright_rows(flux, _snr100(flux), np.zeros(5, bool), 50.0, top_pct=50.0)
    assert got.tolist() == [False, False, True, True, True]            # np.quantile(flux, 0.5) = 3


def test_bright_rows_needs_both_the_snr_and_the_rank():
    flux = np.arange(1., 11.)
    err = _snr100(flux)
    err[9] = 1e3                                                        # brightest row: S/N 0.01
    got = X.bright_rows(flux, err, np.zeros(10, bool), 50.0, top_pct=30.0)
    # the top 30% is flux >= 7.3 = rows 8, 9, 10; the last one fails the S/N
    assert got.tolist() == [False] * 7 + [True, True, False]
    err = _snr100(flux)
    err[7] = 1e3                                                        # a top-30% row with S/N 0.008
    got = X.bright_rows(flux, err, np.zeros(10, bool), 50.0, top_pct=30.0)
    assert got.tolist() == [False] * 7 + [False, True, True]


def test_bright_rows_keeps_saturated_rows_whatever_their_rank():
    flux = np.array([1., 2., 3., 4.])
    got = X.bright_rows(flux, _snr100(flux), [True, False, False, False], 50.0, top_pct=25.0)
    assert got.tolist() == [True, False, False, True]                   # np.quantile(flux, 0.75) = 3.25


def test_bright_rows_rank_is_taken_among_the_multi_covered_rows_only():
    flux = np.array([1., 2., 3., 4., 5., 6., 7., 8., 1e6, 1e6])
    err = _snr100(flux)
    nosat = np.zeros(10, bool)
    multi = np.array([True] * 8 + [False] * 2)
    pooled = X.bright_rows(flux, err, nosat, 50.0, top_pct=25.0, multi=multi)
    assert pooled.tolist() == [False] * 6 + [True] * 4                 # cut 6.25 from 1..8
    every = X.bright_rows(flux, err, nosat, 50.0, top_pct=25.0)
    assert every.tolist() == [False] * 7 + [True] * 3                  # cut 7.75 from all ten rows


def test_bright_rows_rank_ignores_rows_without_a_finite_flux():
    flux = np.array([1., 2., 3., 4., np.nan, np.inf])
    err = np.full(6, 0.01)
    got = X.bright_rows(flux, err, np.zeros(6, bool), 50.0, top_pct=25.0)
    assert got.tolist() == [False, False, False, True, False, False]   # np.quantile(1..4, 0.75) = 3.25


def test_bright_rows_with_an_empty_pool_keeps_only_the_saturated_rows():
    flux, err = np.array([1., 2., 3.]), np.full(3, 0.01)
    got = X.bright_rows(flux, err, [False, True, False], 50.0, top_pct=50.0, multi=[False] * 3)
    assert got.tolist() == [False, True, False]
    got = X.bright_rows(np.full(3, np.nan), err, [True, False, False], 50.0, top_pct=50.0)
    assert got.tolist() == [True, False, False]


@pytest.mark.parametrize('pct', [100.0, 250.0])
def test_bright_rows_rank_cut_is_off_at_100_percent_and_above(pct):
    flux = np.array([1., 2., 3., 4.])
    got = X.bright_rows(flux, _snr100(flux), np.zeros(4, bool), 50.0, top_pct=pct)
    assert got.tolist() == [True] * 4


def test_bright_rows_at_100_percent_is_the_snr_cut_alone_for_rows_outside_the_pool():
    # a row two frames do not cover is outside the rank pool; with the rank cut off its brightness is the
    # S/N cut alone, whatever its flux is next to the pooled rows
    flux = np.array([10., 20., 30., 1.])
    got = X.bright_rows(flux, _snr100(flux), np.zeros(4, bool), 50.0, top_pct=100.0,
                        multi=[True, True, True, False])
    assert got.tolist() == [True, True, True, True]


def test_saturated_rows_reads_both_flags_and_tolerates_absent_or_masked_columns():
    t = Table({'flux': [1., 2., 3., 4.]})
    assert X.saturated_rows(t).tolist() == [False] * 4
    t['is_saturated'] = [True, False, False, False]
    assert X.saturated_rows(t).tolist() == [True, False, False, False]
    t['replaced_saturated'] = [False, False, True, False]
    assert X.saturated_rows(t).tolist() == [True, False, True, False]
    t['is_saturated'] = np.ma.array([True, True, False, False],
                                    mask=[False, True, False, False])
    assert X.saturated_rows(t).tolist() == [True, False, True, False]


def test_saturated_rows_alone_is_enough_without_is_saturated():
    t = Table({'flux': [1., 2.], 'replaced_saturated': [False, True]})
    assert X.saturated_rows(t).tolist() == [False, True]


# ---------------------------------------------------------------------------
# bad-pixel mask
# ---------------------------------------------------------------------------

_LONG = 'jwst_gc_pipeline.photometry.crowdsource_catalogs_long'


def test_bad_dq_mask_for_asks_the_pipeline_for_the_instrument_of_the_filter(monkeypatch):
    # a stand-in module keeps the 19 s import of the real one out of this test
    import jwst_gc_pipeline.photometry as pkg
    asked = []
    fake = types.ModuleType(_LONG)

    def inst(filt):
        asked.append(filt)
        return 'MIRI' if filt == 'F770W' else 'NIRCAM'

    fake._instrument_from_filter = inst
    fake._bad_dq_bitmask = lambda instrument: {'NIRCAM': 3, 'MIRI': 547}[instrument]
    monkeypatch.setitem(sys.modules, _LONG, fake)
    monkeypatch.setattr(pkg, 'crowdsource_catalogs_long', fake, raising=False)
    assert X.bad_dq_mask_for('F212N') == 3
    assert X.bad_dq_mask_for('F770W') == 547
    assert asked == ['F212N', 'F770W']


def test_bad_dq_mask_for_nircam_is_the_default_the_audit_assumes():
    assert X.bad_dq_mask_for('F212N') == X.DEFAULT_BAD_DQ
    assert X.bad_dq_mask_for('F770W') != X.DEFAULT_BAD_DQ


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_audit_summary_partitions_the_multi_covered_rows():
    rng = np.random.default_rng(1)
    ncover = rng.integers(0, 6, 4000)
    nfound = np.minimum(ncover, rng.integers(0, 6, 4000))
    bright = rng.random(4000) < 0.3
    single = (ncover >= 2) & (nfound == 1) & bright
    s = X.audit_summary(ncover, nfound, single, bright)
    assert s['rows'] == 4000 and s['single_exposure'] == int(single.sum())
    for tag in ('', 'bright_'):
        assert (s[tag + 'all'] + s[tag + 'some'] + s[tag + 'one']
                + s[tag + 'none']) == s[tag + 'multi']
    assert s['multi'] == int((ncover >= 2).sum())
    assert s['bright_multi'] == int(((ncover >= 2) & bright).sum())
    assert s['bright_one'] == s['single_exposure']


def test_audit_summary_small_example():
    s = X.audit_summary([3, 3, 3, 3, 1, 0], [3, 2, 1, 0, 1, 0], [0, 0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1])
    assert (s['all'], s['some'], s['one'], s['none']) == (1, 1, 1, 1)
    assert s['multi'] == 4 and s['rows'] == 6


# ---------------------------------------------------------------------------
# options and defaults
# ---------------------------------------------------------------------------

def test_params_follow_the_defaults_and_options():
    p = X.params_from_options(object())
    assert p.single_snr_min == MANUAL_DEFAULTS['manual_xexp_single_snr'] == 50.0
    assert p.single_top_pct == MANUAL_DEFAULTS['manual_xexp_single_top_pct'] == 5.0

    class Snr:
        manual_xexp_single_snr = 80.0

    class Top:
        manual_xexp_single_top_pct = 12.5

    class Both(Snr, Top):
        pass
    p = X.params_from_options(Snr())
    assert (p.single_snr_min, p.single_top_pct) == (80.0, 5.0)
    p = X.params_from_options(Top())
    assert (p.single_snr_min, p.single_top_pct) == (50.0, 12.5)
    p = X.params_from_options(Both())
    assert (p.single_snr_min, p.single_top_pct) == (80.0, 12.5)


def test_dataclass_default_matches_manual_defaults():
    assert X.XexpParams().single_snr_min == MANUAL_DEFAULTS['manual_xexp_single_snr']
    assert X.XexpParams().single_top_pct == MANUAL_DEFAULTS['manual_xexp_single_top_pct']
    assert MANUAL_DEFAULTS['manual_xexp_audit'] is True


@pytest.mark.parametrize('pct', [0.0, -5.0, 100.5, float('nan')])
def test_params_reject_a_rank_cut_outside_zero_to_100(pct):
    with pytest.raises(ValueError, match='single_top_pct'):
        X.XexpParams(single_top_pct=pct)


@pytest.mark.parametrize('pct', [1e-6, 5.0, 100.0])
def test_params_accept_a_rank_cut_in_zero_to_100(pct):
    assert X.XexpParams(single_top_pct=pct).single_top_pct == pct


# ---------------------------------------------------------------------------
# where the call sits in run_manual_pipeline (no test drives that call site)
# ---------------------------------------------------------------------------

_SRC = (Path(__file__).resolve().parents[1] / 'cataloging.py').read_text()


def test_audit_runs_after_the_checkpoint_and_before_the_catalog_is_written():
    ckpt = re.search(r'^\s+_run_astrometry_stage_checkpoint\(\n\s+merge_label',
                     _SRC, re.M).start()
    hook = _SRC.index('_xexp.audit_merged_catalog(')
    iter_found = _SRC.index("merged['iter_found'] = _ifound")
    write = _SRC.index('merged.write(merged_path, overwrite=True)', iter_found)
    stamp = _SRC.index('_stamp_catalog_provenance(merged_path, phase, options,',
                       iter_found)
    assert ckpt < hook < iter_found < write < stamp


def test_audit_is_gated_on_the_m12_full_frame_option():
    opt = "mopt(opts_phase, 'manual_xexp_audit')"
    assert _SRC.count(opt) == 1
    g = _SRC.index(opt)
    hook = _SRC.index('_xexp.audit_merged_catalog(')
    # the `if` that holds the option is the one directly above the call
    assert g < hook and hook - g < 300
    cond = _SRC[g - 160:g]
    assert "phase == 'm12'" in cond
    assert "getattr(options, 'cutout_region'" in cond
    # the satstar suffix it passes is the one defined for the m12 phase
    assert 'satstar_suffix=_ss_suffix' in _SRC[hook:hook + 800]
    # and the bad-pixel mask comes from the helper the QA script also uses
    assert 'bad_dq_mask=_xexp.bad_dq_mask_for(filt)' in _SRC[hook:hook + 800]
    # the cuts come from the run's options, not from the dataclass defaults
    assert 'params=_xexp.params_from_options(opts_phase)' in _SRC[hook:hook + 800]
