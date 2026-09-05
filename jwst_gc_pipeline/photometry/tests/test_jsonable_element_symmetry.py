"""``_jsonable`` must treat a value the same on a key, in a list, and one list
deeper -- and the arms that do the stripping must be pinned.

Follow-up to PR #773 (issue #706).  #773 gave the list branch a numpy-scalar
arm and left three gaps behind, each verified against the code it landed:

* the dict-recursion arm of ``_jsonable_element`` was pinned by nothing --
  the only test covering it used the key ``"cells"``, which ``_jsonable``
  answers from its own dedicated branch and never reaches the helper.  On disk
  that arm is what serialises 68,508 ``windows`` and 222 ``probes`` dicts.
* the trap was closed one nesting level short: ``list``/``tuple`` was not a
  recordable element, so ``_jsonable({'keys': [['F212N', 'o001']]})`` returned
  ``{'keys': []}`` -- #706's symptom verbatim, on the shape
  ``module_antisymmetry['keys']`` and ``consensus['skipped']`` (220 on disk)
  are built in.
* the two filters disagreed: an ``np.bytes_`` on a key was dropped while the
  same value inside a list was kept and reached disk as the literal string
  ``"b'GaiaDR3'"``.
"""
import json

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    _jsonable, _jsonable_element, _json_default)


def _to_disk(obj):
    """What actually lands in the record file."""
    return json.loads(json.dumps(_jsonable(obj), default=_json_default))


# --------------------------------------------------------------------------
# the dict-recursion arm, on a key that does NOT take the "cells" branch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["probes", "windows", "corrections"])
def test_dicts_in_an_ordinary_list_key_are_recursed(key):
    """Delete the ``dict`` arm of ``_jsonable_element`` and this goes red.

    ``measure_offset`` puts its swept ``windows`` and ``agree_across_
    references`` its ``probes`` straight into the record through ``_jsonable``,
    and those dicts carry ``SkyCoord``/``ndarray`` members.  Without the arm
    they are handed back verbatim and the record file cannot be written.
    """
    out = _jsonable({key: [{"n": np.int64(4),
                            "coords": SkyCoord([1.0], [2.0], unit="deg")}]})
    assert out[key] == [{"n": 4}]
    assert _to_disk({key: [{"n": np.int64(4),
                            "coords": SkyCoord([1.0], [2.0],
                                               unit="deg")}]}) == {
        key: [{"n": 4}]}


def test_a_dict_in_a_list_is_recursed_not_stringified():
    """The tell if the arm is gone: ``_json_default``'s ``str(o)`` fallback
    turns the un-recursed dict into one long Python repr instead of an object.
    """
    disk = _to_disk({"probes": [{"coords": SkyCoord([1.0], [2.0],
                                                    unit="deg")}]})
    assert disk == {"probes": [{}]}
    assert not isinstance(disk["probes"][0], str)


# --------------------------------------------------------------------------
# one nesting level deeper -- #706's symptom on the `keys`/`skipped` shape
# --------------------------------------------------------------------------

def test_a_nested_list_is_not_recorded_as_empty():
    """The exact shape ``keys=[list(k) for k in sorted(antisym['keys'])]`` and
    ``skipped=[list(k) for k in cons['skipped']]`` build."""
    keys = [["F212N", "o001", "001", "nrca1", 1],
            ["F212N", "o001", "001", "nrcb1", 2]]
    assert _jsonable({"keys": keys})["keys"] == keys
    assert _to_disk({"keys": keys})["keys"] == keys


def test_the_m2_skipped_key_list_survives_serialisation():
    """The consequence, not just the shape.

    ``consensus['skipped']`` is a list of exposure-key lists, and
    ``_m2_skipped_exposures`` reads it back at a frozen stage to tell an
    exposure m2 DELIBERATELY excluded from one that appeared after the freeze.
    Emptied, an arches-style m2 skip reads as an unexplained new frame and m3
    raises ``AstrometryRegressionError`` over a defect m2 already worked
    around.  10 of the 638 records on disk carry a populated ``skipped``.
    """
    cons = {"consensus_ok": False,
            "median_scatter_mas": np.float64(19.2),
            "skipped": [["1", 4, "nrca1", "F212N", "0210d"],
                        ["1", 4, "nrcb1", "F212N", "0210d"]]}
    out = _jsonable({"consensus": cons})["consensus"]
    assert {tuple(k) for k in out["skipped"]} == {
        ("1", 4, "nrca1", "F212N", "0210d"),
        ("1", 4, "nrcb1", "F212N", "0210d")}


def test_a_nested_tuple_records_as_a_list():
    assert _jsonable({"keys": [("F212N", "o001")]})["keys"] == [["F212N",
                                                                 "o001"]]


def test_numpy_scalars_survive_one_list_deeper():
    assert _jsonable({"keys": [[np.int64(3), np.float32(1.5),
                                np.bool_(True)]]})["keys"] == [[3, 1.5, True]]


def test_dicts_survive_one_list_deeper():
    out = _jsonable({"groups": [[{"n": np.int64(4),
                                  "c": SkyCoord([1.0], [2.0], unit="deg")}]]})
    assert out["groups"] == [[{"n": 4}]]


# --------------------------------------------------------------------------
# the keyed filter and the element filter agree
# --------------------------------------------------------------------------

SYMMETRY_VALUES = [
    np.int64(2), np.int32(2), np.float64(1.5), np.float32(1.5),
    np.bool_(True), np.bytes_(b"GaiaDR3"), np.str_("VIRAC2"),
    2, 1.5, True, None, "s", b"raw",
]


@pytest.mark.parametrize("value", SYMMETRY_VALUES,
                         ids=lambda v: f"{type(v).__name__}:{v!r}")
def test_a_value_records_the_same_on_a_key_and_in_a_list(value):
    """One filter for both positions.  With two filters this parametrisation
    is what catches the next divergence, whichever direction it points."""
    keyed = _to_disk({"k": value})
    listed = _to_disk({"k": [value]})
    assert "k" in keyed, f"{value!r} kept in a list but dropped on a key"
    assert listed["k"] == [keyed["k"]]


def test_a_numpy_byte_string_is_decoded_not_repr_ed():
    """VIRAC2's ``source`` column is ``np.bytes_``.  Before this change it
    vanished on a key and landed in a list as the string ``"b'GaiaDR3'"``."""
    assert _to_disk({"src": np.bytes_(b"GaiaDR3")}) == {"src": "GaiaDR3"}
    assert _to_disk({"src": [np.bytes_(b"GaiaDR3")]}) == {"src": ["GaiaDR3"]}


# --------------------------------------------------------------------------
# what still gets stripped -- these go red if the strip is widened
# --------------------------------------------------------------------------

def test_skycoord_and_ndarray_on_a_key_are_still_stripped():
    out = _jsonable({"coords": SkyCoord([1.0], [2.0], unit="deg"),
                     "dra": np.array([1.0, 2.0]),
                     "off_mas": np.float64(4.0)})
    assert out == {"off_mas": 4.0}


def test_an_ndarray_inside_a_list_is_still_stripped():
    assert _jsonable({"k": [np.array([1.0, 2.0]), np.int64(3)]})["k"] == [3]


def test_an_ndarray_one_list_deeper_is_still_stripped():
    """The recursion added here must not smuggle arrays in behind it."""
    assert _jsonable({"k": [[np.array([1.0, 2.0]),
                             np.int64(3)]]})["k"] == [[3]]


def test_a_zero_d_ndarray_is_not_a_numpy_scalar():
    """``np.generic`` is the reason the strip survives the conversion arm."""
    zero_d = np.array(4.0)
    assert not isinstance(zero_d, np.generic)
    assert _jsonable({"k": zero_d}) == {}
    assert _jsonable({"k": [zero_d]})["k"] == []


def test_a_skycoord_inside_a_list_is_still_stripped():
    assert _jsonable({"k": [SkyCoord([1.0], [2.0], unit="deg"),
                            np.int64(3)]})["k"] == [3]


def test_a_skycoord_one_list_deeper_is_still_stripped():
    assert _jsonable({"k": [[SkyCoord([1.0], [2.0], unit="deg"),
                             np.int64(3)]]})["k"] == [[3]]


# --------------------------------------------------------------------------
# the helper is reachable on its own
# --------------------------------------------------------------------------

def test_jsonable_element_handles_a_bare_nested_structure():
    assert _jsonable_element([np.int64(1), [np.float32(2.5), "x"],
                              np.array([1.0])]) == [1, [2.5, "x"]]
