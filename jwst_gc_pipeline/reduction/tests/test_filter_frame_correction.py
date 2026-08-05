import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction import filter_frame_correction as ffc


def _table(rows, anchor="F212N", frame="instrument"):
    t = Table(names=ffc.filter_frame_table_schema(),
              dtype=("U12", "U10", "U10", "U12", float, float, int, float))
    for det, filt, dx, dy in rows:
        t.add_row((det, filt, anchor, frame, dx, dy, 8, 0.05))
    return t


def test_anchor_is_the_filter_closest_to_two_micron():
    assert ffc.anchor_filter(["F115W", "F182M", "F212N", "F405N"]) == "F212N"
    assert ffc.anchor_filter(["F162M", "F210M", "F360M", "F480M"]) == "F210M"
    assert ffc.anchor_filter(["F187N", "F200W", "F444W"]) == "F200W"


def test_anchor_tie_break_is_deterministic():
    # F187N (1.87) and F212N (2.12) are 0.13/0.12 from 2.0 -- F212N wins on
    # distance; make the tie explicit with two equidistant names.
    a = ffc.anchor_filter(["F190N", "F210N"])
    assert a == ffc.anchor_filter(["F210N", "F190N"])


def test_anchor_rejects_empty():
    with pytest.raises(ffc.FilterFrameError):
        ffc.anchor_filter([])


def test_module_mean_gauge_is_enforced_per_module():
    """The gauge is what keeps this orthogonal to the offsets table."""
    raw = {"NRCA1": (10.0, 0.0), "NRCA2": (12.0, 0.0),
           "NRCA3": (14.0, 0.0), "NRCA4": (16.0, 0.0),
           "NRCB1": (-5.0, 1.0), "NRCB2": (-5.0, 3.0),
           "NRCB3": (-5.0, 5.0), "NRCB4": (-5.0, 7.0)}
    out = ffc._remove_module_means(raw)
    a = np.array([out[d] for d in ("NRCA1", "NRCA2", "NRCA3", "NRCA4")])
    b = np.array([out[d] for d in ("NRCB1", "NRCB2", "NRCB3", "NRCB4")])
    assert np.allclose(a.mean(axis=0), 0.0)
    assert np.allclose(b.mean(axis=0), 0.0)
    # shape survives: A was a pure ramp in dRA, B a pure ramp in dDec
    assert np.allclose(a[:, 0], [-3.0, -1.0, 1.0, 3.0])
    assert np.allclose(b[:, 1], [-3.0, -1.0, 1.0, 3.0])
    # a module that is a pure common shift corrects to nothing
    assert np.allclose([out[d][1] for d in ("NRCA1", "NRCA4")], 0.0)


def test_a_pure_module_shift_produces_no_correction():
    """The offsets table already owns the module mean; we must not re-apply it."""
    raw = {d: (7.0, -3.0) for d in ("NRCA1", "NRCA2", "NRCA3", "NRCA4")}
    out = ffc._remove_module_means(raw)
    assert all(np.allclose(v, 0.0) for v in out.values())


def test_table_offsets_are_regauged_not_trusted():
    """A table written with no gauge must not inject a module-level shift."""
    t = _table([("NRCA1", "F187N", 5.0, 0.0), ("NRCA2", "F187N", 7.0, 0.0),
                ("NRCA3", "F187N", 9.0, 0.0), ("NRCA4", "F187N", 11.0, 0.0)])
    frame, out = ffc.table_filter_offsets(t, "F187N", "F212N")
    assert frame == "instrument"
    arr = np.array(list(out.values()))
    assert np.allclose(arr.mean(axis=0), 0.0)
    assert np.allclose(sorted(arr[:, 0]), [-3.0, -1.0, 1.0, 3.0])


def test_table_offsets_missing_filter_raises():
    t = _table([("NRCA1", "F187N", 1.0, 0.0), ("NRCA2", "F187N", -1.0, 0.0)])
    with pytest.raises(ffc.FilterFrameError):
        ffc.table_filter_offsets(t, "F480M", "F212N")


def test_load_table_rejects_a_missing_column(tmp_path):
    t = _table([("NRCA1", "F187N", 1.0, 0.0), ("NRCA2", "F187N", -1.0, 0.0)])
    t.remove_column("sem_mas")
    p = tmp_path / "bad.ecsv"
    t.write(p)
    with pytest.raises(ffc.FilterFrameError, match="sem_mas"):
        ffc.load_filter_frame_table(str(p))


def test_instrument_to_sky_round_trips():
    off = {"NRCA1": (1.0, 0.0), "NRCB1": (0.0, 2.0)}
    for roll in (0.0, 89.13, 141.01, 275.46):
        sky = ffc.instrument_to_sky(off, roll)
        back = ffc.instrument_to_sky(sky, -roll)
        for d in off:
            assert np.allclose(back[d], off[d], atol=1e-9)
    # a 90 deg roll takes +x to -y
    sky = ffc.instrument_to_sky({"NRCA1": (1.0, 0.0)}, 90.0)
    assert np.allclose(sky["NRCA1"], (0.0, -1.0), atol=1e-9)


def test_correction_enabled_is_off_by_default(monkeypatch):
    monkeypatch.delenv("FILTER_FRAME_CORRECTION", raising=False)
    assert not ffc.correction_enabled()
    monkeypatch.setenv("FILTER_FRAME_CORRECTION", "1")
    assert ffc.correction_enabled()
    monkeypatch.setenv("FILTER_FRAME_CORRECTION", "0")
    assert not ffc.correction_enabled()


def test_gdc_offsets_match_the_documented_amplitudes():
    """The module docstring quotes 1.18-2.11 mas; regenerate and check.

    Skipped where the STDGDC library is not on disk.
    """
    pytest.importorskip("jwst_gc_pipeline.astrometry_gdc.stdgdc")
    expected = {"F115W": 1.18, "F150W": 1.65, "F182M": 1.94,
                "F200W": 2.11, "F210M": 1.93}
    for filt, want in expected.items():
        try:
            off = ffc.gdc_filter_offsets(filt, "F212N")
        except (ffc.FilterFrameError, OSError):
            pytest.skip(f"STDGDC library unavailable for {filt}")
        arr = np.array(list(off.values()))
        rms = float(np.sqrt((arr ** 2).sum(axis=1).mean()))
        assert abs(rms - want) < 0.15, (filt, rms, want)
        # gauge holds on the returned values
        for mod in ("NRCA", "NRCB"):
            m = np.array([v for d, v in off.items() if d.startswith(mod)])
            assert np.allclose(m.mean(axis=0), 0.0, atol=1e-9)


def test_gdc_offsets_refuse_an_uncovered_filter():
    pytest.importorskip("jwst_gc_pipeline.astrometry_gdc.stdgdc")
    with pytest.raises(ffc.FilterFrameError, match="STDGDC"):
        ffc.gdc_filter_offsets("F187N", "F212N")
