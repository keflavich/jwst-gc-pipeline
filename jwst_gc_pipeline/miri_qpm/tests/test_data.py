"""Packaged reference data: checksums, contents and the documented constants."""
import hashlib
import json

import numpy as np
from astropy.io import fits

from jwst_gc_pipeline.miri_qpm import calib

D = calib.DATA / "F770W"


def test_checksums():
    sums = dict(line.split()[::-1] for line in (D / "SHA256SUMS").read_text().splitlines() if line.strip())
    assert set(sums) == {"qpm_R1_F770W.fits", "law_R1_F770W.json", "kernels_F770W.npz", "geometry_F770W.json"}
    for name, h in sums.items():
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == h, name


def test_refdata_contents():
    ref = calib.refdata()
    assert (ref.x0, ref.y0) == (0, 10)
    ys, xs = np.nonzero(ref.region)
    assert (xs.min() + ref.x0, xs.max() + ref.x0) == (9, 231)
    for g, (lo, hi) in {1: (22, 241), 2: (244, 462), 3: (466, 682)}.items():
        yg = np.nonzero(ref.region == g)[0] + ref.y0
        assert (yg.min(), yg.max()) == (lo, hi)
    pane = ref.region > 0
    assert np.isfinite(ref.flat[pane]).all() and np.isfinite(ref.flat_err[pane]).all()
    assert set(np.unique(ref.badpix[pane])) <= {0, 1, 2} and not ref.badpix[~pane].any()
    np.testing.assert_allclose(ref.a[1:], [0.597459, 0.602438, 0.250064])
    np.testing.assert_allclose(ref.b[1:], [1.042792, 1.126051, 0.461483])
    assert ref.law["point_source_bias"]["s_common"] == 0.942
    for kid in ("production", "refit10678"):
        r = calib.refdata(kernels=kid)
        for g in (1, 2, 3):
            assert r.kernels[g].shape == (5, 21) and abs(r.kernels[g].sum() - 1) < 1e-12
    with fits.open(D / "qpm_R1_F770W.fits") as h:
        assert h[0].header["RECIPE"] == "R1"


def test_full_region():
    reg = calib.full_region(calib.refdata())
    assert reg.shape == calib.SHAPE
    assert (reg[calib.LYOT] == 4).all()
    assert [int((reg == g).sum()) for g in (1, 2, 3)] == [int((calib.refdata().region == g).sum()) for g in (1, 2, 3)]


def test_geometry_json_formula():
    geo = json.load(open(D / "geometry_F770W.json"))
    assert set(geo["panes"]) == {"1", "2", "3"} and all(len(geo["panes"][g]["cx"]) == 3 for g in "123")
