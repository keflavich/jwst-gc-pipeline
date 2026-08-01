"""The per-exposure satstar catalog is cached skip-if-exists, but
--satstar-zeroframe-recover / --satstar-ramp-recover change the FIT.  Without
keying the cache on that config, a re-catalog toggling recovery silently reuses
the pre-recovery catalog and the recovery never reaches the merged photometry
(brick 2026-07: a full recovery re-run left the m8 satstar mags byte-identical
because the pre-recovery caches were reused).  The recovery signature stamped in
meta['SATRECOV'] must force a rebuild on mismatch and a cache-hit on match."""
import types

from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
from jwst_gc_pipeline.photometry.cataloging import _satstar_recovery_signature


def _opts(**kw):
    o = types.SimpleNamespace(satstar_zeroframe_recover=False,
                              satstar_ramp_recover=False,
                              satstar_zeroframe_dilate=3)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_recovery_signature_distinguishes_configs():
    base = _satstar_recovery_signature(_opts())
    assert base == "off"                                   # no recovery -> "off"
    # dilation is irrelevant when recovery is off
    assert _satstar_recovery_signature(_opts(satstar_zeroframe_dilate=5)) == "off"
    zf = _satstar_recovery_signature(_opts(satstar_zeroframe_recover=True))
    assert zf != base
    assert _satstar_recovery_signature(_opts(satstar_ramp_recover=True)) != base
    assert _satstar_recovery_signature(_opts(satstar_ramp_recover=True)) != zf
    # dilation DOES matter once recovery is on
    assert _satstar_recovery_signature(
        _opts(satstar_zeroframe_recover=True, satstar_zeroframe_dilate=5)) != zf
    assert _satstar_recovery_signature(_opts()) == base    # stable


def _write_cache(path, sig):
    t = Table({"flux": [1.0, 2.0]})
    if sig is not None:
        t.meta["SATRECOV"] = sig
    t.write(path, overwrite=True)


def _patch_rebuild(monkeypatch, sig_written):
    """Stub remove_saturated_stars: record it ran and (re)write the cache with
    the requested recovery_signature so the post-build read returns it."""
    calls = {"n": 0}

    def _stub(filename, overwrite=True, recovery_signature=None, **kw):
        calls["n"] += 1
        sig_written["sig"] = recovery_signature
        out = filename.replace(".fits", "_satstar_catalog.fits")
        _write_cache(out, recovery_signature)
    monkeypatch.setattr(CL, "remove_saturated_stars", _stub)
    return calls


def test_cache_hit_when_signature_matches(tmp_path, monkeypatch):
    fn = str(tmp_path / "frame.fits")
    _write_cache(tmp_path / "frame_satstar_catalog.fits", "zf1_ramp0_dil3")
    calls = _patch_rebuild(monkeypatch, {"sig": None})
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                          recovery_signature="zf1_ramp0_dil3")
    assert out is not None
    assert calls["n"] == 0            # cache reused, no refit


def test_cache_rebuilds_when_signature_differs(tmp_path, monkeypatch):
    fn = str(tmp_path / "frame.fits")
    _write_cache(tmp_path / "frame_satstar_catalog.fits", "off")  # pre-recovery
    written = {"sig": None}
    calls = _patch_rebuild(monkeypatch, written)
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                          recovery_signature="zf1_ramp0_dil3")
    assert out is not None
    assert calls["n"] == 1                        # refit forced
    assert written["sig"] == "zf1_ramp0_dil3"     # rebuilt with the new config
    assert str(out.meta.get("SATRECOV")) == "zf1_ramp0_dil3"


def test_legacy_cache_without_stamp_rebuilds_when_recovery_requested(tmp_path, monkeypatch):
    fn = str(tmp_path / "frame.fits")
    _write_cache(tmp_path / "frame_satstar_catalog.fits", None)  # legacy, no SATRECOV
    calls = _patch_rebuild(monkeypatch, {"sig": None})
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                          recovery_signature="zf1_ramp0_dil3")
    assert out is not None
    assert calls["n"] == 1            # legacy (=="off") vs recovery request -> rebuild


def test_legacy_cache_kept_for_nonrecovery_run(tmp_path, monkeypatch):
    """A legacy unstamped cache reads as "off", so a plain non-recovery re-run
    MUST reuse it -- otherwise merging this fix would force every field to rebuild
    all its satstar catalogs on the next (non-recovery) run."""
    fn = str(tmp_path / "frame.fits")
    _write_cache(tmp_path / "frame_satstar_catalog.fits", None)  # legacy, no SATRECOV
    calls = _patch_rebuild(monkeypatch, {"sig": None})
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                          recovery_signature="off")
    assert out is not None
    assert calls["n"] == 0            # legacy == "off" == request -> cache hit


def test_no_signature_is_backcompat_cache_hit(tmp_path, monkeypatch):
    fn = str(tmp_path / "frame.fits")
    _write_cache(tmp_path / "frame_satstar_catalog.fits", None)
    calls = _patch_rebuild(monkeypatch, {"sig": None})
    out = CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                          recovery_signature=None)
    assert out is not None
    assert calls["n"] == 0            # signature not requested -> old behavior
