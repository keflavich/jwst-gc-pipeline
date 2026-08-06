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


# ---------------------------------------------------------------------------
# the applying half: sign, frame, gauge-safety, crash protocol
# ---------------------------------------------------------------------------

SHIPPED = "jwst_gc_pipeline/reduction/data/filter_frame_offsets.ecsv"

#: Measured SKY residuals (F182M - F212N) after a per-(exposure, module)
#: median, brick jw02221-o001, ROLL_REF 89.13.  The CORRECTION must point the
#: other way; a test that only checked magnitude would pass either sign, which
#: is exactly how the sign error survived review round one.
BRICK_ROLL = 89.13
BRICK_F182M_SKY_RESIDUAL = {
    "NRCA1": (+0.37, +1.03), "NRCA2": (+0.19, +0.73),
    "NRCA3": (-0.12, -0.70), "NRCA4": (-0.28, -1.36),
    "NRCB1": (-1.33, -1.66), "NRCB2": (+1.49, -1.36),
    "NRCB3": (-1.92, +1.44), "NRCB4": (+1.42, +1.45),
}


def _shipped():
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    return ffc.load_filter_frame_table(os.path.join(here, SHIPPED))


def test_shipped_table_is_the_correction_not_the_residual():
    """Rotate the shipped rows onto brick's sky and they must OPPOSE the
    measured residual, detector by detector.  Negating the table fails this."""
    frame, off = ffc.table_filter_offsets(_shipped(), "F182M", "F212N")
    assert frame == "instrument"
    sky = ffc.instrument_to_sky(off, BRICK_ROLL)
    resid = ffc._remove_module_means(BRICK_F182M_SKY_RESIDUAL)
    dots = []
    for det, r in resid.items():
        c = np.asarray(sky[det])
        r = np.asarray(r)
        dots.append(float(c @ r) / (np.linalg.norm(c) * np.linalg.norm(r)))
        # each correction opposes its own residual
        assert dots[-1] < -0.5, (det, c, r)
    assert np.mean(dots) < -0.85, dots
    # and applying it shrinks the residual rather than growing it
    before = np.array(list(resid.values()))
    after = before + np.array([sky[d] for d in resid])
    assert np.sqrt((after ** 2).sum(1).mean()) < 0.35 * np.sqrt(
        (before ** 2).sum(1).mean())


def test_shipped_table_honours_its_own_gauge():
    tbl = _shipped()
    for filt in ("F182M", "F187N"):
        rows = [r for r in tbl if str(r["filter"]).upper() == filt]
        assert rows
        for mod in ("NRCA", "NRCB"):
            m = np.array([[r["dx_mas"], r["dy_mas"]] for r in rows
                          if str(r["detector"]).upper().startswith(mod)])
            assert np.allclose(m.mean(axis=0), 0.0, atol=1e-3), (filt, mod, m)


def test_sky_shift_sign_and_cos_dec():
    """dRA is an on-sky offset divided by cos(dec); dropping that is a 14%
    error at the Galactic centre, and negating it inverts the correction."""
    off = {"NRCA1": (100.0, 0.0)}
    dra, ddec = ffc.sky_shift_deg(off, "NRCA1", "sky", None, -28.7)
    assert ddec == 0.0
    assert dra > 0                                   # sign preserved
    assert dra == pytest.approx(100.0 / 3.6e6 / np.cos(np.radians(-28.7)))
    assert dra > 1.13 * (100.0 / 3.6e6)              # cos(dec) actually applied


def test_sky_shift_requires_a_declination():
    with pytest.raises(ffc.FilterFrameError, match="declination"):
        ffc.sky_shift_deg({"NRCA1": (1.0, 0.0)}, "NRCA1", "sky", None, None)


def test_sky_shift_requires_roll_for_instrument_frame():
    with pytest.raises(ffc.FilterFrameError, match="ROLL_REF"):
        ffc.sky_shift_deg({"NRCA1": (1.0, 0.0)}, "NRCA1", "instrument", None, -28.7)


def test_sky_shift_refuses_the_detector_frame():
    """gdc_filter_offsets output is mirrored relative to the instrument frame."""
    with pytest.raises(ffc.FilterFrameError, match="not applicable"):
        ffc.sky_shift_deg({"NRCA1": (1.0, 0.0)}, "NRCA1", "detector", 89.13, -28.7)


def test_sky_shift_refuses_an_unknown_frame():
    with pytest.raises(ffc.FilterFrameError, match="unknown frame"):
        ffc.sky_shift_deg({"NRCA1": (1.0, 0.0)}, "NRCA1", "v2v3", 89.13, -28.7)


def test_preconditions_skip_when_already_applied_onto_the_same_anchor():
    hdr = {ffc.MARKER: True, "FFRAMANC": "F212N"}
    assert ffc.check_apply_preconditions(hdr, "NRCA1", {}, anchor="F212N") == "skip"


def test_preconditions_refuse_a_different_anchor():
    hdr = {ffc.MARKER: True, "FFRAMANC": "F212N"}
    with pytest.raises(ffc.FilterFrameError, match="already corrected onto"):
        ffc.check_apply_preconditions(hdr, "NRCA1", {}, anchor="F200W")


def test_preconditions_refuse_a_crashed_previous_apply():
    with pytest.raises(ffc.FilterFrameError, match="pending"):
        ffc.check_apply_preconditions({ffc.PENDING: True}, "NRCA1",
                                      {"NRCA1": (1.0, 0.0)})


def test_preconditions_refuse_a_partial_coefficient_set():
    """Skipping one detector of a module is the partial correction that
    gdc_filter_offsets refuses -- the applier must not do it either."""
    with pytest.raises(ffc.FilterFrameError, match="gauge"):
        ffc.check_apply_preconditions({}, "NRCA3", {"NRCA1": (1.0, 0.0)})


def test_preconditions_pass_a_clean_frame():
    assert ffc.check_apply_preconditions({}, "NRCA1", {"NRCA1": (1.0, 0.0)}) == "go"


def test_table_offsets_refuse_mixed_frames():
    t = _table([("NRCA1", "F187N", 1.0, 0.0), ("NRCA2", "F187N", -1.0, 0.0)])
    t["frame"][1] = "sky"
    with pytest.raises(ffc.FilterFrameError, match="mix frames"):
        ffc.table_filter_offsets(t, "F187N", "F212N")


def test_miri_names_do_not_masquerade_as_two_micron():
    """F1000W is 10 microns, not 1.0 -- a three-digit parse makes a MIRI
    filter the best 2-micron anchor."""
    assert ffc.filter_wavelength_um("F1000W") == pytest.approx(10.0)
    assert ffc.filter_wavelength_um("F2100W") == pytest.approx(21.0)
    assert ffc.filter_wavelength_um("F212N") == pytest.approx(2.12)
    assert ffc.anchor_filter(["F1000W", "F1500W", "F2100W"]) == "F1000W"
    assert ffc.anchor_filter(["F1000W", "F212N"]) == "F212N"
