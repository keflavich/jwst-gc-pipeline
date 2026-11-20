"""Anchoring / dataset-gain decision rules; mosaic merge priority, SOURCE codes and FLAGS on a tiny TAN grid."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from jwst_gc_pipeline.miri_qpm import mosaic, offsets, qa


def test_anchor_rules():
    mk = np.r_[np.full(10, 50.0), np.full(7, 60.0), np.full(20, 500.0), np.full(9, 20.0), np.full(9, 100.0)]
    g = np.r_[np.full(10, 1), np.full(7, 2), np.full(20, 3), np.full(9, 3), np.full(9, 2)].astype(np.int8)
    q = mk - 2.0
    q[10:17] = mk[10:17] + 1.0
    a = offsets.anchor_offsets(q, mk, g)
    assert a[1] == (10, -2.0)                          # >= 8 blocks in 30-450 MJy/sr
    assert a[2] == (16, -2.0)                          # 7 blocks at +1 and 9 at -2: median of 16 = -2
    assert a[3][0] == 0 and np.isnan(a[3][1])          # 500 and 20 MJy/sr are outside the range
    a = offsets.anchor_offsets(q[:17], mk[:17], g[:17])
    assert a[2][0] == 7 and np.isnan(a[2][1])          # < 8 blocks: no anchor


def test_dataset_gain_rules():
    rng = np.random.default_rng(4)
    n = 3000
    mk = rng.uniform(40, 400, n)
    g = rng.integers(1, 4, n).astype(np.int8)
    q = 0.97 * mk * (1 + 0.003 * rng.standard_normal(n))
    few = np.array([f"jw00001{v:03d}001" for v in rng.integers(1, 5, n)])     # 4 visits
    gain, rep = offsets.dataset_gain(q, mk, g, few)
    assert not rep["adopted"] and gain == {1: 1.0, 2: 1.0, 3: 1.0}
    many = np.array([f"jw00001{v:03d}001" for v in rng.integers(1, 9, n)])    # 8 visits
    gain, rep = offsets.dataset_gain(q, mk, g, many)
    assert rep["adopted"] and rep["cv_cell_rms_Lg"] < rep["cv_cell_rms_L0"]
    assert all(abs(gain[p] - 0.97) < 1e-3 for p in (1, 2, 3))
    gain, rep = offsets.dataset_gain(mk * (1 + 0.003 * rng.standard_normal(n)), mk, g, many)
    assert all(abs(gain[p] - 1) < 2e-3 for p in (1, 2, 3))   # adopted or not, a null gain stays ~1


def test_groups_of():
    vis5 = [f"jw00001{v:03d}001_02101_00001_mirimage" for v in range(1, 6)]
    assert offsets.groups_of(vis5)[1] == "visit"
    one = [f"jw00001001001_0210{a}_00001_mirimage" for a in range(1, 4)]
    grp, unit = offsets.groups_of(one)
    assert unit == "visit+activity" and len(set(grp)) == 3


def _ref():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [266.4, -28.9]
    w.wcs.crpix = [25.0, 20.0]
    w.wcs.cdelt = [-3e-5, 3e-5]
    img = (np.arange(40 * 50, dtype=np.float32).reshape(40, 50) / 7).astype(">f4")
    img[:, :10] = np.nan               # outside the science footprint
    img[20:23, 30:33] = np.nan         # a hole inside it, pane below 450 MJy/sr
    img[10:13, 25:28] = np.nan         # a hole whose pane data are partly > 450 MJy/sr (saturated core)
    return w.to_header(), img


def _merged(tmp_path, fill_bright_holes=False):
    hdr, img = _ref()
    pts = np.array([[-10.0, -3.0], [45.0, 30.0]])
    whdr, shape, x0, y0 = mosaic.output_grid(hdr, img.shape, pts, "full", margin=0)
    mm = mosaic.create(tmp_path / "m.fits", fits.Header([("QPMNEXP", 1)]), whdr, shape)
    # one 'exposure': a 30 x 40 pane image on reference pixels x -5..34, y 5..34
    yy, xx = np.mgrid[0:30, 0:40].astype(float)
    pixmap = np.dstack([xx - 5 - x0, yy + 5 - y0])
    Q = np.full(yy.shape, 7.0, np.float32)
    above = np.zeros(Q.shape, bool)
    above[:, :2] = True                                   # ref x -5, -4
    above[5, 30] = True                                   # ref (y 10, x 25): one pixel of the second hole
    cut = mosaic.drizzle_local(Q, np.ones_like(Q), {mosaic.FLAG_ABOVE: above}, np.ones_like(Q), pixmap, shape)
    mosaic.accumulate(mm, cut, mosaic.FLAG_WARN)
    mosaic.finalize(mm, img, None, x0, y0, fill_bright_holes=fill_bright_holes)
    return hdr, img, whdr, shape, x0, y0, mm


def test_mosaic_merge(tmp_path):
    hdr, img, whdr, shape, x0, y0, mm = _merged(tmp_path)
    assert (x0, y0) == (-10, -3) and shape == (43, 60)
    assert whdr["CRPIX1"] == hdr["CRPIX1"] + 10 and whdr["REFX0"] == -10
    sci, src, flg = np.array(mm["SCI"]), np.array(mm["SOURCE"]), np.array(mm["FLAGS"])
    ref_on_out = np.full(shape, np.nan, np.float32)
    ref_on_out[-y0:-y0 + 40, -x0:-x0 + 50] = img
    fin = np.isfinite(ref_on_out)
    native = np.asarray(sci[fin], np.float32)              # mosaic is big-endian on disk
    assert (src[fin] == 1).all() and np.array_equal(native.view(np.uint32), ref_on_out[fin].view(np.uint32))
    hole = np.zeros(shape, bool)
    hole[20 - y0:23 - y0, 30 - x0:33 - x0] = True
    assert (src[hole] == 3).all() and np.allclose(sci[hole], 7.0)
    assert (src[8 - y0, 0 - x0] == 2) and np.isclose(sci[8 - y0, 0 - x0], 7.0)      # pane only, no science
    assert src[1, 1] == 0 and np.isnan(sci[1, 1])                                    # no data
    bright = np.zeros(shape, bool)                       # the whole hole stays NaN, not only the > 450 pixel
    bright[10 - y0:13 - y0, 25 - x0:28 - x0] = True
    assert (src[bright] == 4).all() and np.isnan(sci[bright]).all() and (np.array(mm["WHT"])[bright] == 0).all()
    assert flg[10 - y0, 25 - x0] & mosaic.FLAG_ABOVE and (flg[bright] & mosaic.FLAG_WARN).all()
    assert qa.pixel_stats(mm)["counts"]["hole_unfilled_above_450"] == 9
    assert (flg[src == 1] == 0).all() and (flg[src >= 2] & mosaic.FLAG_WARN).all()
    assert flg[8 - y0, -5 - x0] & mosaic.FLAG_ABOVE and not flg[8 - y0, 0 - x0] & mosaic.FLAG_ABOVE
    ident = qa.science_identity(mm, img, x0, y0)
    assert ident["bit_identical"] and ident["n_reference_finite"] == int(np.isfinite(img).sum())
    del mm
    with fits.open(tmp_path / "m.fits") as h:                # the skeleton is a valid FITS file
        assert [e.name for e in h[1:]] == ["SCI", "ERR", "WHT", "SOURCE", "FLAGS"]
        assert h["SOURCE"].data.dtype == np.uint8 and WCS(h["SCI"].header).wcs.crpix[0] == hdr["CRPIX1"] + 10
        before = h["SCI"].data.copy()
    fits.setval(tmp_path / "m.fits", "QPMNEXP", value=3)       # the pipeline's pass-B failure path
    with fits.open(tmp_path / "m.fits") as h:
        assert h[0].header["QPMNEXP"] == 3 and np.array_equal(h["SCI"].data, before, equal_nan=True)


def test_mosaic_fill_bright_holes(tmp_path):
    *_, shape, x0, y0, mm = _merged(tmp_path, fill_bright_holes=True)
    sl = (slice(10 - y0, 13 - y0), slice(25 - x0, 28 - x0))
    assert (np.array(mm["SOURCE"])[sl] == 3).all() and np.allclose(np.array(mm["SCI"])[sl], 7.0)


def test_science_mask_makes_pixels_non_science(tmp_path):
    """Reference pixels where the mask is 0 read NaN: they get pane data (not the reference value) and the identity
    check runs against the masked reference."""
    hdr, img = _ref()
    mask = np.ones(img.shape, np.uint8)
    mask[25:35, 15:25] = 0                                 # a 'fill field' inside the pane footprint
    mref = mosaic.MaskedRef(img, mask)
    assert mref.shape == img.shape and np.isnan(mref[25:35, 15:25]).all() and mref[0:5, 30:35].dtype == np.float32
    np.testing.assert_array_equal(mref[0:5, 30:35], img[0:5, 30:35])
    pts = np.array([[-10.0, -3.0], [45.0, 30.0]])
    whdr, shape, x0, y0 = mosaic.output_grid(hdr, img.shape, pts, "full", margin=0)
    mm = mosaic.create(tmp_path / "m.fits", fits.Header(), whdr, shape)
    yy, xx = np.mgrid[0:30, 0:40].astype(float)
    Q = np.full(yy.shape, 7.0, np.float32)
    cut = mosaic.drizzle_local(Q, np.ones_like(Q), {}, np.ones_like(Q), np.dstack([xx - 5 - x0, yy + 5 - y0]), shape)
    mosaic.accumulate(mm, cut, 0)
    mosaic.finalize(mm, mref, None, x0, y0)
    sl = (slice(25 - y0, 35 - y0), slice(15 - x0, 25 - x0))
    assert np.allclose(np.array(mm["SCI"])[sl], 7.0) and (np.array(mm["SOURCE"])[sl] >= 2).all()
    assert qa.science_identity(mm, mref, x0, y0)["bit_identical"]
    assert not qa.science_identity(mm, img, x0, y0)["bit_identical"]
