"""`crossfilter_residual_field.load` resolves an upper-case band (#690).

Catalog filenames on disk are lower-case (``f212n_merged_...``), while every
other entry point in the repo takes upper-case filter names.  The loader must
fold the band before building the path.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "analysis" / "crossfilter_residual_field.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("_xf_resid_field", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("band", ["F212N", "f212n"])
def test_load_accepts_either_band_case(tmp_path, monkeypatch, band):
    mod = _load_module()
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    catdir = tmp_path / "brick" / "catalogs"
    catdir.mkdir(parents=True)
    n = 5
    tbl = Table()
    tbl["skycoord"] = SkyCoord(np.linspace(266.5, 266.6, n) * u.deg,
                               np.full(n, -28.7) * u.deg)
    tbl["flux"] = np.full(n, 100.0)
    tbl.write(catdir / "f212n_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits")

    sc, mag = mod.load("brick", band, "resbgsub_m7")
    assert len(sc) == n
    assert np.all(np.isfinite(mag))
