"""The binding defaults and run-setting checks, input pairing, and the de-smear / level order (no external data)."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from jwst_gc_pipeline.miri_qpm import calib, pipeline, psf

REF = calib.refdata()


def _tan_ref(path, sip=False):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"] if sip else ["RA---TAN", "DEC--TAN"]
    w.wcs.crval, w.wcs.crpix, w.wcs.cdelt = [266.8, -28.4], [5, 5], [-3e-5, 3e-5]
    h = w.to_header()
    if sip:
        h.update(A_ORDER=2, B_ORDER=2, A_2_0=1e-6, B_0_2=1e-6)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((10, 10), np.float32), h, name="SCI")]).writeto(path, overwrite=True)
    return str(path)


def _cfg(tmp_path, **kw):
    return dict(sci="x", rate_dir=str(tmp_path), ref=_tan_ref(tmp_path / "ref.fits"), out_dir=str(tmp_path), **kw)


def test_defaults_are_off():
    d = pipeline.DEFAULTS
    assert d["anchor"] == "off" and d["dataset_gain"] == "off" and d["register"] is None
    assert d["fill_bright_holes"] is True           # saturated reference cores are filled by default (user, 2026-10-11)
    assert d["edge_mode"] == "flag" and d["science_mask"] is None    # STScI-like: keep and flag, do not leave holes


def test_input_groups(tmp_path):
    rd = str(tmp_path)
    cfg = dict(ref=_tan_ref(tmp_path / "ref.fits"), out_dir=rd, inputs=[
        dict(name="10678", sci="a*", rate_dir=rd, register=False),
        dict(name="ext", sci="b*", rate_dir=rd, gc_like="true", search_px=[25])])
    c = pipeline.resolve_settings(cfg, "10678")
    g0, g1 = c["inputs"]
    assert (g0["register"], g1["register"], c["register"]) == (False, False, False)    # mode default for 'ext'
    assert g1["gc_like"] is True and g1["search_px"] == [25] and g0["search_px"] == pipeline.DEFAULTS["search_px"]
    assert pipeline.resolve_settings(cfg, "external")["inputs"][1]["register"] is True
    one = pipeline.resolve_settings(_cfg(tmp_path), "external")["inputs"]            # single-input configs
    assert len(one) == 1 and one[0]["sci"] == "x" and one[0]["register"] is True
    for bad, msg in ((dict(cfg, sci="x"), "not both"), (dict(cfg, dataset_gain="auto"), "several input groups"),
                     (dict(cfg, inputs=[dict(name="a", sci="x", rate_dir=rd, anchor="on")]), "unknown keys"),
                     (dict(cfg, inputs=[dict(name="a", sci="x", rate_dir=rd)] * 2), "unique"),
                     (dict(cfg, inputs=[dict(name="a", sci="x")]), "rate_dir")):
        with pytest.raises(SystemExit, match=msg):
            pipeline.resolve_settings(bad, "external")


def test_science_mask_setting(tmp_path):
    m = tmp_path / "mask.fits"
    fits.PrimaryHDU(np.ones((10, 10), np.uint8)).writeto(m)
    assert pipeline.resolve_settings(_cfg(tmp_path, science_mask=str(m)), "10678")["science_mask"] == str(m)
    fits.PrimaryHDU(np.ones((9, 10), np.uint8)).writeto(tmp_path / "bad.fits")
    with pytest.raises(SystemExit, match="reference grid"):
        pipeline.resolve_settings(_cfg(tmp_path, science_mask=str(tmp_path / "bad.fits")), "10678")


def test_resolve_settings(tmp_path):
    assert pipeline.resolve_settings(_cfg(tmp_path), "10678")["register"] is False
    c = pipeline.resolve_settings(_cfg(tmp_path), "external")
    assert c["register"] is True and c["anchor"] == "off" and c["dataset_gain"] == "off"
    with pytest.raises(SystemExit, match="not available in the 10678"):
        pipeline.resolve_settings(dict(_cfg(tmp_path), dataset_gain="auto"), "10678")
    assert pipeline.resolve_settings(dict(_cfg(tmp_path), dataset_gain="auto"), "external")["dataset_gain"] == "auto"
    with pytest.raises(SystemExit, match="cannot read the reference"):          # no hang: checked before the Pool
        pipeline.resolve_settings(dict(_cfg(tmp_path), ref=str(tmp_path / "none.fits")), "external")
    with pytest.raises(SystemExit, match="rate_dir"):
        pipeline.resolve_settings(dict(_cfg(tmp_path), rate_dir=str(tmp_path / "none")), "external")
    with pytest.raises(SystemExit, match="SIP"):
        pipeline.resolve_settings(dict(_cfg(tmp_path), ref=_tan_ref(tmp_path / "sip.fits", True)), "external")


def test_skip_reason():
    cfg = dict(pipeline.DEFAULTS)
    meta = dict(s_bkdsub="COMPLETE", k_npix=500000)
    assert "S_BKDSUB" in pipeline.skip_reason(meta, 0.26, cfg)
    assert pipeline.skip_reason(meta, 0.26, dict(cfg, allow_bkdsub=True)) is None
    assert "imager pixels" in pipeline.skip_reason(dict(s_bkdsub="", k_npix=10), 0.26, cfg)


def test_find_inputs_dedupes_and_filters(tmp_path):
    def mk(name, filt="F770W"):
        fits.PrimaryHDU(header=fits.Header([("FILTER", filt), ("SUBARRAY", "FULL")])).writeto(tmp_path / name)
    for s in ("cal", "crf"):
        mk(f"jw10678001001_02101_00001_mirimage_{s}.fits")
    mk("jw10678001001_02101_00002_mirimage_cal.fits", "F1000W")
    mk("jw10678001001_02101_00003_mirimage_cal.fits")                       # no rate file
    mk("jw10678001001_02101_00001_mirimage_rate.fits")
    rows, notes = pipeline.find_inputs(dict(sci=str(tmp_path / "*_c*.fits"), rate_dir=str(tmp_path), limit=0))
    assert [r[0] for r in rows] == ["jw10678001001_02101_00001_mirimage"] and rows[0][1].endswith("_cal.fits")
    assert any("F1000W" in n for n in notes) and any("no rate file" in n for n in notes)


def test_merge_groups_first_group_wins(monkeypatch):
    found = {"a": [("s1", "a/s1", "ra/s1"), ("s2", "a/s2", "ra/s2")], "b": [("s2", "b/s2", "rb/s2"), ("s3", "b/s3", "rb/s3")]}
    monkeypatch.setattr(pipeline, "find_inputs", lambda gi: (found[gi["name"]], [f"note {gi['name']}"]))
    rows, notes = pipeline.merge_groups([dict(name="a"), dict(name="b")])
    assert rows == [("s1", "a/s1", "ra/s1", 0), ("s2", "a/s2", "ra/s2", 0), ("s3", "b/s3", "rb/s3", 1)]
    assert notes[:2] == ["note a", "note b"] and "s2: also in input group 'b'" in notes[2]


def test_fallback_similarity_same_group_only():
    p = lambda v: np.array([1.0, v, 0, 0])
    ds_a, ds_b = "jw10678001001_02101_", "jw10678001001_02201_"          # stem[:20] = dither set
    ok = [dict(i=0, stem=ds_a + "1", p=p(1)), dict(i=1, stem=ds_b + "1", p=p(3)),
          dict(i=2, stem=ds_a + "2", p=p(100))]                           # i = 2: other input group, same dither set
    gof = [0, 0, 1, 0, 0, 1]
    np.testing.assert_array_equal(pipeline.fallback_similarity(dict(i=3, stem=ds_a + "3"), ok, gof), p(1))
    np.testing.assert_array_equal(pipeline.fallback_similarity(                       # no dither-set mate: group median
        dict(i=4, stem="jw10678001001_02301_3"), ok, gof), p(2))
    assert pipeline.fallback_similarity(dict(i=5, stem="jw02092001001_02101_1"), ok[:2], gof) is None


def test_desmear_then_subtract_level():
    """RL sees the cal-level image: a flat sky L with a level > L returns exactly L / gain - level (no point-source
    or other factor), whatever the sign of the result."""
    reg = REF.region
    Q = np.where(reg > 0, 20.0, np.nan)
    var = np.where(reg > 0, 1.0, np.nan)
    gain = {1: 1.0, 2: 1.25, 3: 1.0}
    level = {1: 35.0, 2: 5.0, 3: 60.0}
    out, v, vrn = pipeline.desmear_and_subtract(Q, var, var.copy(), reg, REF.kernels, psf.RL_ITER, gain, level)
    for g in (1, 2, 3):
        m = reg == g
        np.testing.assert_allclose(out[m], 20.0 / gain[g] - level[g], atol=2e-4)
        np.testing.assert_allclose(v[m], psf.RL_VAR_FACTOR[g] / gain[g] ** 2)
    assert np.isnan(Q[reg == 0]).all() and Q[reg > 0].max() == 20.0          # inputs untouched


def test_point_source_bias_only_in_headers():
    h = pipeline._common_header(fits.Header(), pipeline.DEFAULTS, REF)
    assert h["QPMPSB"] == REF.law["point_source_bias"]["s_common"] and "NOT applied" in h.comments["QPMPSB"]


def test_no_epoch_warning():
    """No time drift was detected robustly: an exposure years from the training epoch gets no warning (the offset is
    an informational column); NGROUPS outside 5-8 still warns (ramp length matters at the 1-2 % level)."""
    wcs = lambda x, y, with_bounding_box=False: (266.8, -28.4)                # inside the GC box
    meta = dict(ngroups=6, mjd=calib.refdata().law["validity"]["training_mjd"][0] - 3 * 365.25, imager_satfrac=0.0, outlier_frac=0.0)
    assert pipeline._exposure_warnings(meta, wcs, dict(gc_like="auto")) == []
    assert abs(meta["epoch_offset_yr"] - 3) < 1e-9
    assert "NGROUPS" in pipeline._exposure_warnings(dict(meta, ngroups=15), wcs, dict(gc_like="auto"))[0]
