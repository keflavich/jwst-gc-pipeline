"""Tests for scripts/logs/thin_pipeline_logs.py (log thinning tool)."""
import importlib.util
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "logs",
                       "thin_pipeline_logs.py")
_spec = importlib.util.spec_from_file_location("thin_pipeline_logs", _SCRIPT)
tpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpl)


SPAM = """\
Source 3: center at (x, y) = (101.2, 202.3), forced=False
  mask_buffer=8  bkg=(12,20)  (sat_area=370)
Set x_0.bounds = (99.0, 103.0)
Set y_0.bounds = (200.0, 204.0)
Number of pixels above threshold (5000): 42
8837.408

 id group_id group_size     local_bkg      x_init y_init     flux_init           x_fit
--- -------- ---------- ------------------ ------ ------ ----------------- ---------------
  1        1          1 12.3   101 202 1000.0 2 3 4 5 6 7         8 9  nan 1     2            False
Accepting source 3 with flux=1e5, fluxerr=10, snr=100, sidelobe_resid_sigma=1, ssr_ratio=1.1
Skipping source 4: snr=2, fluxerr=5, qfit=0.5, sidelobe_resid_sigma=1, ssr_ratio=0.2
"""
KEEP = """\
manual [m12]: fitting 48 frames with 8 parallel workers
Satstar summary: 12/14 sources accepted, 2 rejected
ERROR: something real happened
Traceback (most recent call last):
  ValueError: kaboom
"""


def _write(tmp_path, text, name="x.log", age_days=10):
    p = tmp_path / name
    p.write_text(text)
    old = os.path.getmtime(p) - age_days * 86400
    os.utime(p, (old, old))
    return str(p)


def _thin(path, aggressive=True, execute=True):
    patterns = dict(tpl.DEFAULT_PATTERNS)
    if aggressive:
        patterns.update(tpl.AGGRESSIVE_PATTERNS)
    return tpl.thin_file(path, patterns, execute)


def test_spam_removed_keepers_intact(tmp_path):
    p = _write(tmp_path, SPAM + KEEP + SPAM)
    kept, removed, removed_bytes, _ = _thin(p)
    text = open(p).read()
    for line in KEEP.splitlines():
        assert line in text
    assert "Source 3: center" not in text
    assert "Set x_0.bounds" not in text
    assert "group_id" not in text          # table stripped
    assert removed > 0 and removed_bytes > 0
    # marker carries the per-template counts, incl. accept/skip
    assert tpl.MARKER_PREFIX in text
    assert "accept_srcx1" in text and "skip_srcx1" in text


def test_default_mode_keeps_accept_skip(tmp_path):
    p = _write(tmp_path, SPAM)
    _thin(p, aggressive=False)
    text = open(p).read()
    assert "Accepting source 3" in text
    assert "Skipping source 4" in text
    assert "Set x_0.bounds" not in text


def test_dry_run_touches_nothing(tmp_path):
    p = _write(tmp_path, SPAM + KEEP)
    before = open(p).read()
    mtime = os.path.getmtime(p)
    kept, removed, removed_bytes, projected = _thin(p, execute=False)
    assert open(p).read() == before
    assert os.path.getmtime(p) == mtime
    assert removed > 0
    assert projected < os.path.getsize(p)


def test_idempotent(tmp_path):
    p = _write(tmp_path, SPAM + KEEP)
    _thin(p)
    once = open(p).read()
    _thin(p)
    assert open(p).read() == once          # markers survive, nothing re-removed


def test_mtime_preserved_on_execute(tmp_path):
    p = _write(tmp_path, SPAM + KEEP, age_days=30)
    mtime = os.path.getmtime(p)
    _thin(p)
    assert abs(os.path.getmtime(p) - mtime) < 1


def test_short_blank_runs_get_no_marker(tmp_path):
    p = _write(tmp_path, "real line one\n\nreal line two\n42.0\nreal line three\n")
    _thin(p)
    text = open(p).read()
    assert tpl.MARKER_PREFIX not in text
    assert "real line one\nreal line two\n" in text


def test_main_skips_young_files(tmp_path, capsys):
    p = _write(tmp_path, SPAM * 2000, age_days=0)
    rc = tpl.main([str(p), "--min-size-mb", "0.001"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SKIP (younger" in out
    assert "Source 3" in open(p).read()   # untouched


@pytest.mark.skipif(not tpl.shutil.which("zstd"), reason="zstd unavailable")
def test_compress_roundtrip(tmp_path):
    import subprocess
    p = _write(tmp_path, (SPAM + KEEP) * 50)
    _thin(p)
    thinned = open(p).read()
    zsize = tpl.compress_file(p, execute=True)
    assert zsize is not None and zsize > 0
    assert not os.path.exists(p)
    out = subprocess.run(["zstd", "-dq", p + ".zst", "-o", p], check=True)
    assert open(p).read() == thinned
