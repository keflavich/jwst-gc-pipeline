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
    try:
        f480m.__wrapped__()
    except BaseException:       # noqa: BLE001 -- see below
        # BaseException, not Exception.  `pytest.skip` raises `Skipped`, which
        # derives from BaseException, so `except Exception` let it escape and
        # turned THIS GUARD into a skip -- in exactly the scenario it exists to
        # catch (a product missing, the fixture calling _img(None)).  We only
        # care which globs were requested, so any exit is fine.
        pass
    assert seen, "the fixture globbed nothing"
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
