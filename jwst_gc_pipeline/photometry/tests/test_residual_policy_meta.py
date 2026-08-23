"""Meta-tests for test_residual_model_policy.py -- they must run ALWAYS.

pytest ORs skip conditions, so a function-level ``skipif(False)`` cannot cancel
a module-level ``pytestmark``.  The previous attempt put these beside the
policy tests with ``skipif(False)`` and they skipped with the rest -- on the
only tree where the divergence they guard actually crashes.  They live here,
outside that mark, instead.
"""
import glob

import numpy as np
import pytest

from . import test_residual_model_policy as policy
from .test_residual_model_policy import (
    _REQUIRED, f480m, over_rendered, over_subtracted,
    MODEL_PEAK_CEILING, RESID_CORE_FLOOR)


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
    monkeypatch.setattr(policy, "_troughs", lambda arr, w, stars, box=3: None)
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


class _FakeWCS:
    """Identity world->pixel, so the box arithmetic can be tested without a FITS."""

    def __init__(self, xy):
        self._xy = np.asarray(xy, float)

    def world_to_pixel(self, stars):
        return self._xy[:, 0], self._xy[:, 1]


def test_troughs_reads_the_core_minimum_and_peaks_the_maximum():
    """`_peaks` alone cannot see a crater: both a hole and a clean subtraction
    have a small positive maximum, which is why the over-subtraction assertions
    need their own reduction over the same box (issue #266 item 4)."""
    arr = np.zeros((21, 21))
    arr[10, 10] = -500.0          # crater
    arr[10, 11] = +3.0            # the max `_peaks` would report
    w = _FakeWCS([[10, 10]])
    assert policy._peaks(arr, w, None, box=3)[0] == pytest.approx(3.0)
    assert policy._troughs(arr, w, None, box=3)[0] == pytest.approx(-500.0)


def test_troughs_is_nan_off_image():
    arr = np.zeros((21, 21))
    w = _FakeWCS([[-5, -5]])
    assert np.isnan(policy._troughs(arr, w, None)[0])


def test_over_rendered_flags_only_above_the_ceiling():
    d = np.array([100.0, 100.0, 100.0, 100.0])
    m = np.array([146.0,             # the worst star measured 2026-08-21: kept
                  MODEL_PEAK_CEILING * 100.0 - 1,
                  MODEL_PEAK_CEILING * 100.0 + 1,
                  1000.0])
    assert list(over_rendered(d, m)) == [2, 3]


def test_over_subtracted_flags_only_below_the_floor():
    d = np.array([100.0, 100.0, 100.0, 100.0])
    rmin = np.array([-47.0,          # the worst star measured 2026-08-21: kept
                     RESID_CORE_FLOOR * 100.0 + 1,
                     RESID_CORE_FLOOR * 100.0 - 1,
                     -900.0])
    assert list(over_subtracted(d, rmin)) == [2, 3]


def test_two_sided_predicates_ignore_unusable_stars():
    """A star off the mosaic, or with a non-positive data peak, cannot be graded
    either way -- the same `ok` mask the one-sided assertions already use."""
    d = np.array([np.nan, 0.0, -5.0, 100.0])
    m = np.array([1e6, 1e6, 1e6, 1e6])
    rmin = np.array([-1e6, -1e6, -1e6, -1e6])
    assert list(over_rendered(d, m)) == [3]
    assert list(over_subtracted(d, rmin)) == [3]
