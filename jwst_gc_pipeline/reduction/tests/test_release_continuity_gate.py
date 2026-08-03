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


def _known_limit_catalog(drift_lo, drift_hi, drift=-0.35, sat_faint=14.0,
                         n=40000, seed=2):
    """F405N-F410M (a KNOWN_CONTINUITY_LIMITS pair) with saturation flags that
    reach down to ``sat_faint`` and a color offset ``drift`` applied to the
    UNSATURATED rows in mag_f410m [drift_lo, drift_hi).  A drift inside the
    saturation-onset regime (brighter than sat_faint) is the deep-core floor
    (WARN); a drift fainter than sat_faint is a real defect (blocks)."""
    rng = np.random.default_rng(seed)
    mB = rng.uniform(9.0, 19.0, n)
    color = np.full(n, -0.10) + rng.normal(0, 0.04, n)
    sat = mB < (sat_faint + rng.normal(0, 0.2, n))     # fuzzy saturation edge
    hit = (~sat) & (mB >= drift_lo) & (mB < drift_hi)
    color[hit] += drift
    return Table({
        'mag_vega_f405n': mB + color,
        'mag_vega_f410m': mB,
        'is_saturated_f405n': sat,
        'is_saturated_f410m': sat,
        'replaced_saturated_f405n': np.zeros(n, bool),
        'replaced_saturated_f410m': np.zeros(n, bool),
    })


def test_known_limit_onset_floor_warns(tmp_path):
    # drift confined to the bright unsaturated onset sliver (brighter than the
    # saturation faint edge) -> deep-core floor -> WARN, does not block.
    cat = _known_limit_catalog(drift_lo=12.0, drift_hi=13.5, sat_faint=14.0)
    fails = stage_release.check_photometric_continuity(_items_for(tmp_path, cat))
    assert fails == []


def test_known_limit_deep_drift_still_blocks(tmp_path):
    # same known-limit pair, but the drift is FAINTER than saturation: no floor
    # excuse -> must block even for a KNOWN_CONTINUITY_LIMITS pair.
    cat = _known_limit_catalog(drift_lo=15.0, drift_hi=16.5, sat_faint=14.0)
    fails = stage_release.check_photometric_continuity(_items_for(tmp_path, cat))
    assert fails, "a drift below the saturation floor must block a known-limit pair"
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
