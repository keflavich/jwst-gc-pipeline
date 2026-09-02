"""Every band the registry declares has a saturated-star match radius.

`replace_saturated` indexes its radius table inline (`}[filtername]`), so an
unlisted band raised a bare `KeyError` -- and only after the fan-out had run, so
it cost hours to learn a band simply had no entry.  9438's F070W is the case:

    File ".../merge_catalogs.py", in replace_saturated
    KeyError: 'f070w'

The two tables in this module are NOT the same map and must not be merged:
`flag_near_saturated` uses 0.55" for every NIRCam band (a flagging radius),
`replace_saturated` uses 0.05" short-wave / 0.1" long-wave (a tight match
radius).  They are checked separately here.
"""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "merge_catalogs.py"


def _tables():
    """The two radius maps, in source order, as {band: arcsec}."""
    text = SRC.read_text()
    out = []
    for m in re.finditer(r"radius = \{# short-wave", text):
        seg = text[m.start():m.start() + 3000]
        out.append({b: float(v) for b, v in
                    re.findall(r"'([a-z0-9]+)':\s*([\d.]+)\*u\.arcsec", seg)})
    return out


def _declared_bands():
    from jwst_gc_pipeline import fields as fields_mod
    bands = set()
    for instrument in ("nircam", "miri", "niriss"):
        for by_proposal in fields_mod.obs_filters(instrument).values():
            for filters in by_proposal.values():
                bands.update(str(f).lower() for f in filters)
    return bands


def test_there_are_two_distinct_tables():
    """Guard the premise: if these ever became one map, the values are wrong."""
    tables = _tables()
    assert len(tables) == 2
    flag, replace = tables
    assert flag != replace, (
        "the flagging and replacement radius maps have become identical; they "
        "encode different conventions (0.55\" NIRCam vs 0.05\"/0.1\")")


@pytest.mark.parametrize("index,name", [(0, "flag_near_saturated"),
                                        (1, "replace_saturated")])
def test_every_declared_band_has_a_radius(index, name):
    table = _tables()[index]
    missing = sorted(_declared_bands() - set(table))
    assert not missing, (
        f"{name}'s radius map has no entry for {missing}; a band with no entry "
        f"raises inside the merge, after the fan-out has already run"
    )


def test_the_bands_that_were_missing_are_present_with_their_conventions():
    flag, replace = _tables()
    for band in ("f070w", "f430m"):
        assert band in flag and band in replace
    # flagging: one NIRCam radius for all
    assert flag["f070w"] == flag["f430m"] == 0.55
    # replacement: short-wave vs long-wave
    assert replace["f070w"] == 0.05
    assert replace["f430m"] == 0.1


def test_replace_saturated_names_the_band_it_cannot_find():
    """A bare KeyError after the fan-out is what this replaces."""
    text = SRC.read_text()
    assert "no saturated-star match radius for" in text
    assert "merge_catalogs.replace_saturated" in text
