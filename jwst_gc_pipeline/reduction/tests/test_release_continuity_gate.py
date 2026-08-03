"""The release photometric-continuity gate (scripts/release/stage_release.py)
must refuse to ship a merged catalog whose degenerate-pair colors drift or
whose saturation classes are photometrically discontinuous, and pass a clean
one.  The gate function is imported from the script by path (scripts/ is not a
package)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

_SPEC = importlib.util.spec_from_file_location(
    "stage_release",
    Path(__file__).resolve().parents[3] / "scripts" / "release" / "stage_release.py")
stage_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_release)


def _merged_catalog(strip_offset=0.0, n=20000, seed=0):
    """Flat F405N-F410M color -0.10 with an optional suppression strip at the
    bright end of F410M (12.2-13.3 in a 12-18 catalog)."""
    rng = np.random.default_rng(seed)
    mB = rng.uniform(12.0, 18.0, n)
    color = np.full(n, -0.10) + rng.normal(0, 0.05, n)
    strip = (mB > 12.2) & (mB < 13.3)
    color[strip] += strip_offset
    return Table({'mag_vega_f405n': mB + color, 'mag_vega_f410m': mB})


def _items_for(tmp_path, cat):
    src = tmp_path / "basic_merged_test_m7.fits"
    cat.write(src)
    return [{"category": "catalog", "kind": "catalog_full", "filter": None,
             "iteration": "m7", "observation": None, "src": str(src)}]


def test_clean_catalog_passes(tmp_path):
    fails = stage_release.check_photometric_continuity(
        _items_for(tmp_path, _merged_catalog(strip_offset=0.0)))
    assert fails == []


def test_suppression_strip_refused(tmp_path):
    fails = stage_release.check_photometric_continuity(
        _items_for(tmp_path, _merged_catalog(strip_offset=-0.35)))
    assert fails, "a 0.35-mag degenerate-pair drift must fail the gate"
    assert any("f405n-f410m" in f for f in fails)


def _sparse_onset_catalog(seed=4):
    """Flat locus (-0.10) with populated faint + mid bins and a SPARSE bright bin
    (50 stars at F410M~12.5-12.7) thrown -0.5 off-locus -- the shape of a real
    saturation-onset bin, where few unsaturated stars remain and a handful should
    not decide a release.  No saturation flags (science subset == all rows)."""
    rng = np.random.default_rng(seed)
    mB = np.concatenate([rng.uniform(15.0, 19.0, 40000),   # faint plateau
                         rng.uniform(13.0, 15.0, 8000),    # populated mid, flat
                         rng.uniform(12.5, 12.7, 50)])     # sparse bright bin
    color = np.full(len(mB), -0.10) + rng.normal(0, 0.03, len(mB))
    color[-50:] += -0.5
    return Table({'mag_vega_f405n': mB + color, 'mag_vega_f410m': mB})


def test_sparse_onset_bin_does_not_block(tmp_path):
    """A large offset confined to a SPARSE bright bin (fewer than min_n stars)
    must not decide the release: at min_n=200 the n=50 bin is excluded and the
    populated bins are flat -> pass."""
    cat = _sparse_onset_catalog()
    fails = stage_release.check_photometric_continuity(
        _items_for(tmp_path, cat), flatness_min_n=200)
    assert fails == []


def test_sparse_onset_bin_would_block_at_low_min_n(tmp_path):
    """Control for the test above: the SAME catalog fails at min_n=20, proving the
    pass is min_n suppressing a sparse bin -- not an accidentally-flat catalog."""
    cat = _sparse_onset_catalog()
    fails = stage_release.check_photometric_continuity(
        _items_for(tmp_path, cat), flatness_min_n=20)
    assert any("f405n-f410m" in f for f in fails)


def test_no_merged_table_returns_none(tmp_path):
    assert stage_release.check_photometric_continuity([]) is None
    # ecsv-only shipment: gate reads only the fits combined table
    items = [{"kind": "catalog_full", "src": str(tmp_path / "x.ecsv")}]
    assert stage_release.check_photometric_continuity(items) is None


def test_missing_bands_is_not_a_failure(tmp_path):
    cat = Table({'mag_vega_f090w': np.linspace(12.0, 18.0, 500)})
    fails = stage_release.check_photometric_continuity(_items_for(tmp_path, cat))
    assert fails == []
