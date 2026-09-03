"""The product tree this gate globs comes from the environment (``JWST_BASE``).

``fields.yaml`` declares two roots -- ``orange`` (/orange/adamginsburg/jwst) and
``blue`` (/blue/adamginsburg/adamginsburg/jwst) -- and a ``root: blue`` field is
reachable under /orange only when a symlink was made for it.  gc-treasury (root
blue, #421) has none, so with the base hardcoded to /orange every glob in this
file matched nothing and the gate reported on an empty tree.  The sibling gates
``check_interframe_overlap.py`` and ``check_astrometry_checkpoints.py`` already
read ``JWST_BASE`` with the /orange default; these tests pin that this one now
does too, and that the default is unchanged.

The override is exercised through the two on-disk enumerators (``mosaic`` and
``field_bands``) rather than by reading ``BASE`` alone: those build their glob
patterns from the constant, so a test that only asserted on the constant would
pass on a version whose globs still pointed somewhere else.
"""
import importlib.util
from pathlib import Path

_SRC = Path(__file__).with_name("registration_failsafes.py")

MOSAIC = "jw10678-o088_t001_nircam_clear-f212n-merged_i2d.fits"


def _load():
    """Exec the script fresh so it re-reads the environment (module-level constant)."""
    spec = importlib.util.spec_from_file_location("registration_failsafes_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(root, field="gc-treasury", filt="F212N", name=MOSAIC):
    d = Path(root) / field / filt / "pipeline"
    d.mkdir(parents=True)
    (d / name).write_bytes(b"")      # the enumerators are name-only; nothing is opened
    return d / name


def test_default_base_is_orange(monkeypatch):
    """Unset environment -> the tree every existing field resolves under today."""
    monkeypatch.delenv("JWST_BASE", raising=False)
    assert _load().BASE == "/orange/adamginsburg/jwst"


def test_jwst_base_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("JWST_BASE", str(tmp_path))
    assert _load().BASE == str(tmp_path)


def test_globs_follow_the_override(monkeypatch, tmp_path):
    """A blue-root field's mosaic is found when the base points at the blue tree.

    This is the gc-treasury case: nothing of it exists under /orange.
    """
    expected = _tree(tmp_path)
    monkeypatch.setenv("JWST_BASE", str(tmp_path))
    rf = _load()
    assert rf.mosaic("gc-treasury", "F212N", "merged") == str(expected)
    assert rf.field_bands("gc-treasury") == ["F212N"]
    assert list(rf.field_band_mosaics("gc-treasury")) == ["F212N"]


def test_globs_find_nothing_off_the_override(monkeypatch, tmp_path):
    """The mutation guard: with the base pointed elsewhere, the same tree is invisible.

    Without this the override test could pass on a build that ignored the
    environment but happened to have the products under the default.
    """
    _tree(tmp_path / "blue")
    monkeypatch.setenv("JWST_BASE", str(tmp_path / "empty"))
    rf = _load()
    assert rf.mosaic("gc-treasury", "F212N", "merged") is None
    assert rf.field_bands("gc-treasury") == []
