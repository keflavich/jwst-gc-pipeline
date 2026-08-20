"""cloudc claims three observations across two instruments and three programs.

Its NIRCam is 2221-o002; its MIRI is 2221-o001 (F2550W) and 2526-o021 (F770W),
two programs imaging NON-OVERLAPPING pointings of the same cloud.  Neither MIRI
observation is reachable from ``proposal_prefix``, so before they were declared
the overlap gate scoped cloudc to {02221-002}, filtered out every well-formed
F770W frame, and refused the field for "missing products".

The claim these tests protect is the one the entry makes possible.  MIRI
observation numbering is INVERTED with respect to NIRCam inside program 2221:

    cloudc NIRCam 2221-o002      cloudc MIRI 2221-o001
    brick  NIRCam 2221-o001      brick  MIRI 2221-o002

so cloudc now claims 02221-001, which is BRICK's NIRCam key.  ``_release_observations``'
docstring promises that listing a MIRI observation "can never re-admit a stray
NIRCam crf that shares its proposal-obs key" -- the per-directory derivation is
what keeps that true.  If a brick NIRCam frame is ever copied into a cloudc
filter directory, this entry is what would admit it, and nothing else watches
for that.
"""
import importlib.util
import os
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "stage_release",
    Path(__file__).resolve().parents[3] / "scripts" / "release" / "stage_release.py")
stage_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_release)


def test_cloudc_claims_both_miri_programs_and_its_nircam():
    """All three keys, or the gate cannot see a band it is asked to verify."""
    obs = stage_release._release_observations(stage_release.FIELDS["cloudc"])
    assert obs == {"02221-002", "02221-001", "02526-021"}, obs


def test_cloudc_miri_entries_name_both_programs():
    miri = {m["filter"]: m["src"] for m in stage_release.FIELDS["cloudc"]["miri"]}
    assert set(miri) == {"F770W", "F2550W"}
    assert "jw02526-o021" in miri["F770W"], miri["F770W"]
    assert "jw02221-o001" in miri["F2550W"], miri["F2550W"]
    # ...and each src lives under cloudc, not under the field whose name the
    # observation number invites (brick MIRI F2550W is 2221-o002).
    for src in miri.values():
        assert "/cloudc/" in src, src


@pytest.mark.localdata
def test_no_nircam_product_hides_under_the_newly_claimed_key():
    """The re-admission hazard, checked against the archive.

    cloudc now claims 02221-001 = brick's NIRCam key.  The per-directory
    derivation is what stops that admitting a brick NIRCam frame, and it only
    holds while cloudc's NIRCam directories contain nothing from 02221-001.
    A stray copied in later is exactly what this entry would let through.
    """
    import glob
    import re
    base = str(stage_release.FIELDS["cloudc"]["data_dir"])
    strays = []
    for filt in ("F182M", "F187N", "F212N", "F405N", "F410M", "F466N"):
        for path in glob.glob(f"{base}/{filt}/pipeline/jw*_crf.fits"):
            name = os.path.basename(path)
            m = re.match(r"^jw(?P<prop>\d{5})(?P<obs>\d{3})", name)
            if m and f"{m.group('prop')}-{m.group('obs')}" != "02221-002":
                strays.append(name)
    assert not strays, (
        f"cloudc NIRCam directories hold {len(strays)} frame(s) from another "
        f"observation; with 02221-001 now claimed, a brick NIRCam stray would "
        f"be admitted: {strays[:5]}")
