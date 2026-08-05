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


def _strip_catalog(n, drift=-0.35, seed=1):
    """N-row F405N-F410M table, flat locus -0.10 with a -drift suppression strip
    over F410M 12.2-13.3, no saturation flags (science subset == all rows)."""
    rng = np.random.default_rng(seed)
    mB = rng.uniform(12.0, 18.0, n)
    color = np.full(n, -0.10) + rng.normal(0, 0.05, n)
    color[(mB > 12.2) & (mB < 13.3)] += drift
    return Table({'mag_vega_f405n': mB + color, 'mag_vega_f410m': mB})


def test_small_catalog_with_strip_still_blocks(tmp_path):
    """R1: a catalog too small for the min_n=200 science floor (nan) but with a
    real strip the flag-inclusive metric CAN measure must not pass -- it fails
    NOT-CERTIFIED, not 'ok'."""
    cat = _strip_catalog(1500)          # < 10*200 rows -> science nan
    fails = stage_release.check_photometric_continuity(_items_for(tmp_path, cat))
    assert any("f405n-f410m" in f and "unmeasurable" in f for f in fails)


def test_both_unmeasurable_strip_blocks(tmp_path):
    """R1: when BOTH science and flag-inclusive are unmeasurable (very small
    catalog), the pair still blocks -- a gate never passes a population it
    declined to measure."""
    cat = _strip_catalog(150)           # both metrics nan
    fails = stage_release.check_photometric_continuity(_items_for(tmp_path, cat))
    assert any("f405n-f410m" in f and "unmeasurable" in f for f in fails)


def test_informational_metric_uses_default_min_n(tmp_path, capsys):
    """R2: the LOGGED flag-inclusive drift is a satstar flux-scale diagnostic and
    must be computed at the default min_n, not min_n=200 -- a satstar offset lives
    in exactly the sparse bins min_n=200 suppresses. Science subset is flat (the
    flagged sparse offset is cut) so the gate passes, but the log must still show
    the offset."""
    rng = np.random.default_rng(7)
    mB = np.concatenate([rng.uniform(15.0, 19.0, 40000),
                         rng.uniform(13.0, 15.0, 8000),
                         rng.uniform(12.4, 12.6, 50)])   # sparse bright bin
    color = np.full(len(mB), -0.10) + rng.normal(0, 0.03, len(mB))
    color[-50:] += -0.6
    rs = np.zeros(len(mB), bool)
    rs[-50:] = True                                      # flagged recovered-satstar
    cat = Table({'mag_vega_f405n': mB + color, 'mag_vega_f410m': mB,
                 'replaced_saturated_f405n': rs, 'replaced_saturated_f410m': rs})
    fails = stage_release.check_photometric_continuity(_items_for(tmp_path, cat))
    assert fails == []                                   # science subset is flat
    line = [l for l in capsys.readouterr().out.splitlines()
            if "degenerate-pair f405n-f410m" in l][-1]
    # flag-inclusive at the default min_n sees the sparse offset (~0.6); at
    # min_n=200 it would read ~0.00 and hide it.
    full = float(line.split("full-inclusive")[1].split("mag")[0])
    assert full > 0.3


def test_no_merged_table_returns_none(tmp_path):
    assert stage_release.check_photometric_continuity([]) is None
    # ecsv-only shipment: gate reads only the fits combined table
    items = [{"kind": "catalog_full", "src": str(tmp_path / "x.ecsv")}]
    assert stage_release.check_photometric_continuity(items) is None


def test_missing_bands_is_not_a_failure(tmp_path):
    cat = Table({'mag_vega_f090w': np.linspace(12.0, 18.0, 500)})
    fails = stage_release.check_photometric_continuity(_items_for(tmp_path, cat))
    assert fails == []


# ---------------------------------------------------------------------------
# Saturation-BOUNDARY known limit: the F410M-F405N railed-deep-core floor is
# WARN (not FAIL) only when TRIPLY scoped -- the F410M readout is NGROUPS<=2 AND
# the jump is below the deep-core-floor ceiling. Fails closed on the unknown.
# ---------------------------------------------------------------------------
from astropy.io import fits   # noqa: E402


def _boundary_cat(jump, n=6000, seed=5):
    """F410M-F405N catalog with a boundary jump: bright stars (mag_f405n < 13)
    are replaced_saturated in F410M and carry a `jump` color offset, so the
    saturation_continuity metric reads ~|jump|."""
    rng = np.random.default_rng(seed)
    mref = rng.uniform(10.5, 17.0, n)                 # mag_vega_f405n (band_ref)
    sat = mref < 13.0 + rng.normal(0, 0.3, n)         # replaced_saturated_f410m
    color = -0.10 + rng.normal(0, 0.05, n) + np.where(sat, jump, 0.0)
    return Table({
        'mag_vega_f410m': mref + color, 'mag_vega_f405n': mref,
        'replaced_saturated_f410m': sat,
        'is_saturated_f410m': np.zeros(n, bool),
        'forced_filled_f410m': np.zeros(n, bool),
        'replaced_saturated_f405n': np.zeros(n, bool),
        'forced_filled_f405n': np.zeros(n, bool),
        'is_saturated_f405n': np.zeros(n, bool),
        'independently_detected_f405n': np.ones(n, bool)})


def _mosaic_item(tmp_path, filt, ngroups):
    p = tmp_path / f"{filt.lower()}_i2d.fits"
    hdu = fits.PrimaryHDU()
    if ngroups is not None:
        hdu.header['NGROUPS'] = ngroups
    hdu.writeto(p, overwrite=True)
    return {"category": "image", "kind": "science", "filter": filt.upper(),
            "src": str(p)}


def _boundary_items(tmp_path, cat, ngroups):
    items = _items_for(tmp_path, cat)
    if ngroups is not None:
        items.append(_mosaic_item(tmp_path, "F410M", ngroups))
    return items


def test_boundary_floor_warns_when_2group_and_below_ceiling(tmp_path):
    # brick-like: ~0.17 mag jump, NGROUPS=2 -> documented railed-core floor, WARN
    items = _boundary_items(tmp_path, _boundary_cat(0.17), ngroups=2)
    assert stage_release.check_photometric_continuity(items) == []


def test_boundary_gross_break_blocks_even_at_2group(tmp_path):
    # cloudc-like: a gross jump on a 2-group field is NOT the deep-core floor
    items = _boundary_items(tmp_path, _boundary_cat(0.8), ngroups=2)
    fails = stage_release.check_photometric_continuity(items)
    assert any("f410m-f405n continuity" in f for f in fails)


def test_boundary_blocks_when_not_2group(tmp_path):
    # w51-like: same ~0.17 jump but NGROUPS=5 -> not the railed regime, blocks
    items = _boundary_items(tmp_path, _boundary_cat(0.17), ngroups=5)
    fails = stage_release.check_photometric_continuity(items)
    assert any("f410m-f405n continuity" in f for f in fails)


def test_boundary_fails_closed_when_ngroups_unknown(tmp_path):
    # no F410M science mosaic shipped -> readout cannot be verified -> block
    items = _boundary_items(tmp_path, _boundary_cat(0.17), ngroups=None)
    fails = stage_release.check_photometric_continuity(items)
    assert any("f410m-f405n continuity" in f for f in fails)


def test_boundary_exemption_does_not_cover_other_pairs(tmp_path):
    # F182M-F187N is not a known-limit pair: a boundary break blocks regardless
    # of readout. Build the jump on that pair with an F182M mosaic at NGROUPS=2.
    rng = np.random.default_rng(6)
    n = 6000
    mref = rng.uniform(10.5, 17.0, n)
    sat = mref < 13.0 + rng.normal(0, 0.3, n)
    color = 0.15 + rng.normal(0, 0.05, n) + np.where(sat, 0.17, 0.0)
    cat = Table({
        'mag_vega_f182m': mref + color, 'mag_vega_f187n': mref,
        'replaced_saturated_f182m': sat,
        'is_saturated_f182m': np.zeros(n, bool),
        'forced_filled_f182m': np.zeros(n, bool),
        'replaced_saturated_f187n': np.zeros(n, bool),
        'forced_filled_f187n': np.zeros(n, bool),
        'is_saturated_f187n': np.zeros(n, bool),
        'independently_detected_f187n': np.ones(n, bool)})
    items = _items_for(tmp_path, cat) + [_mosaic_item(tmp_path, "F182M", 2)]
    fails = stage_release.check_photometric_continuity(items)
    assert any("f182m-f187n continuity" in f for f in fails)


def _c2_boundary_cat(offset=0.15, n=8000, seed=3):
    """SHARP saturation boundary (no mixed bins) -> saturation_continuity falls
    back to the C2-locus-offset kind rather than a C1 boundary jump."""
    rng = np.random.default_rng(seed)
    mref = rng.uniform(10.5, 17.0, n)
    sat = mref < 13.0                                  # sharp, no fuzz
    color = -0.10 + rng.normal(0, 0.03, n) + np.where(sat, offset, 0.0)
    return Table({
        'mag_vega_f410m': mref + color, 'mag_vega_f405n': mref,
        'replaced_saturated_f410m': sat,
        'is_saturated_f410m': np.zeros(n, bool),
        'forced_filled_f410m': np.zeros(n, bool),
        'replaced_saturated_f405n': np.zeros(n, bool),
        'forced_filled_f405n': np.zeros(n, bool),
        'is_saturated_f405n': np.zeros(n, bool),
        'independently_detected_f405n': np.ones(n, bool)})


def test_boundary_waiver_is_recorded_on_the_catalog_item(tmp_path):
    # the exemption must leave a machine-readable trace (items -> MANIFEST.json),
    # not only a stdout line.
    items = _boundary_items(tmp_path, _boundary_cat(0.17), ngroups=2)
    assert stage_release.check_photometric_continuity(items) == []
    cat_item = next(it for it in items if it.get("kind") == "catalog_full")
    waivers = cat_item.get("continuity_waivers")
    assert waivers and waivers[0]["pair"] == "f410m-f405n"
    assert waivers[0]["ngroups"] == 2
    assert waivers[0]["kind"] == "C1-boundary-jump"
    assert 0.10 <= waivers[0]["metric"] < 0.25


def test_boundary_tightened_ceiling_blocks_a_regression(tmp_path):
    # a jump of 0.30 (above the measured 0.170 floor) is a regression, not the
    # floor -- the 0.25 ceiling must block it even at NGROUPS=2.
    items = _boundary_items(tmp_path, _boundary_cat(0.30), ngroups=2)
    fails = stage_release.check_photometric_continuity(items)
    assert any("f410m-f405n continuity" in f for f in fails)


def test_boundary_c2_locus_offset_is_not_exempted(tmp_path):
    # the exemption is for the C1 railed-core boundary jump only; a C2-locus-offset
    # (a different defect) under the ceiling at NGROUPS=2 must still block.
    items = _boundary_items(tmp_path, _c2_boundary_cat(0.15), ngroups=2)
    fails = stage_release.check_photometric_continuity(items)
    assert any("f410m-f405n continuity" in f for f in fails)


@pytest.mark.parametrize("order", [[2, 8], [8, 2]])
def test_band_ngroups_uses_deepest_mosaic(tmp_path, order):
    # a field shipping F410M mosaics at NGROUPS 2 AND 8 must be judged by the
    # DEEPEST (8) -> not the railed regime -> blocks, regardless of item order.
    items = _items_for(tmp_path, _boundary_cat(0.17))
    for i, ng in enumerate(order):
        src = str(tmp_path / f"f410m_{i}_i2d.fits")
        fits.PrimaryHDU(header=fits.Header({"NGROUPS": ng})).writeto(src, overwrite=True)
        items.append({"category": "image", "kind": "science", "filter": "F410M",
                      "src": src})
    fails = stage_release.check_photometric_continuity(items)
    assert any("f410m-f405n continuity" in f for f in fails)


def test_ngroups_noninteger_header_fails_closed(tmp_path):
    # a header whose NGROUPS is not int-convertible must not crash and must not
    # grant an exemption (fail closed -> block).
    items = _items_for(tmp_path, _boundary_cat(0.17))
    p = tmp_path / "f410m_bad_i2d.fits"
    fits.PrimaryHDU(header=fits.Header({"NGROUPS": "BRIGHT2"})).writeto(p, overwrite=True)
    items.append({"category": "image", "kind": "science", "filter": "F410M",
                  "src": str(p)})
    fails = stage_release.check_photometric_continuity(items)
    assert any("f410m-f405n continuity" in f for f in fails)
