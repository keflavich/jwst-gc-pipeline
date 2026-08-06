"""Meta-tests for test_residual_model_policy.py -- they must run ALWAYS.

pytest ORs skip conditions, so a function-level ``skipif(False)`` cannot cancel
a module-level ``pytestmark``.  The previous attempt put these beside the
policy tests with ``skipif(False)`` and they skipped with the rest -- on the
only tree where the divergence they guard actually crashes.  They live here,
outside that mark, instead.
"""
import glob

import pytest

from . import test_residual_model_policy as policy
from .test_residual_model_policy import _REQUIRED, f480m


def test_each_required_glob_matches_at_most_one_file():
    """The cross-run hazard in issue #266: a glob matching both ``resbgsub_m5``
    and a ``group`` variant lets `_latest` take numerator and denominator from
    different runs."""
    for what, pattern in _REQUIRED.items():
        assert len(glob.glob(pattern)) <= 1, (what, glob.glob(pattern))


def test_skip_predicate_covers_every_product_the_fixture_opens(monkeypatch):
    """The predicate and the fixture must glob the SAME patterns.

    They diverged for a month: the predicate checked
    `*mergedcat_residual_i2d.fits` while the fixture needed the `_m7_` variant,
    so a field with m2..m5 products but no m7 passed the predicate and then
    opened None -- three ERRORS on main since 2026-07-05 (issue #266).

    Checked by RECORDING what the fixture actually globs, not by inspecting its
    source: a source check only proves the label strings appear, and passes for
    a fixture that builds its paths inline or opens a fourth product.
    """
    seen = []
    real_latest = policy._latest

    def _record(pattern):
        seen.append(pattern)
        return real_latest(pattern)

    # patch the POLICY module's namespace -- the fixture resolves `_latest`
    # there, not here
    monkeypatch.setattr(policy, "_latest", _record)
    # ... and stub everything the fixture does BETWEEN the globs, so that a
    # tree with no products still reaches all three of them.  Unstubbed, on CI
    # -- the environment this whole change is about -- `_read_points` raised
    # before a single glob was recorded and the test went RED, and stubbing
    # only that one got as far as the first `_img(None)` skip and then failed
    # on `missing`.  Trading a silent skip for a red CI is not a fix.  What
    # this test checks is WHICH PATTERNS the fixture asks for; opening the
    # files is not part of that.
    monkeypatch.setattr(policy, "_read_points", lambda path: None)
    monkeypatch.setattr(policy, "_img", lambda path, what="": (None, None))
    monkeypatch.setattr(policy, "_peaks", lambda arr, w, stars, box=3: None)
    try:
        f480m.__wrapped__()
    except (pytest.skip.Exception, OSError, ValueError):
        # NAMED, not `BaseException`.  Widening to BaseException did catch the
        # `Skipped` that `except Exception` let escape -- `pytest.skip` raises
        # a BaseException subclass, which is what turned this guard into a skip
        # in the scenario it exists to catch -- but it also swallows
        # KeyboardInterrupt and SystemExit.  `Skipped` has a public handle, so
        # the specific form covers all the documented exits (the skip from
        # `_img(None)`, an unreadable FITS, a malformed region file) while a
        # NameError or TypeError from a future fixture edit still surfaces --
        # which is the breakage this test is for.  KeyError is deliberately
        # NOT here: the fixture indexes `_REQUIRED`, so a KeyError means the
        # predicate and the fixture have gone out of step -- the exact
        # condition under test.  Swallowing it let that mutant live.
        pass
    if not seen:
        pytest.skip("the fixture bailed before its first glob, so there is "
                    "nothing to compare against the skip predicate")
    missing = [p for p in _REQUIRED.values() if p not in seen]
    assert not missing, (
        "the fixture did not open every _REQUIRED product, so the predicate "
        "checks something the fixture does not need -- or stopped early:\n  "
        + "\n  ".join(missing))
    unknown = [p for p in seen if p not in set(_REQUIRED.values())]
    assert not unknown, (
        "the fixture opens product(s) the skip predicate does not check, so a "
        "tree missing them errors instead of skipping:\n  "
        + "\n  ".join(unknown))
