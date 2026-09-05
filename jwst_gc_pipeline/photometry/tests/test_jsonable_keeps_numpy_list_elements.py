"""``_jsonable`` must not delete numpy scalars out of a list (issue #706).

The helper converts a numpy scalar sitting directly on a key, but its
list/tuple branch keeps only elements that are *already* Python types.  Every
other numpy scalar -- ``np.int64``, ``np.bool_``, ``np.float32`` -- is dropped
from the list without a word, so a record field written from numpy arithmetic
reaches disk shorter than it was measured, or empty.  ``np.float64`` happens to
survive because it subclasses ``float``; a list holding both therefore keeps
the floats and silently loses the integers, which also breaks the positional
correspondence a reader relies on.
"""
import json

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_checkpoint import _jsonable


def test_numpy_scalars_in_a_list_reach_the_record():
    out = _jsonable({"k": [np.float64(1.5), np.int64(2)]})
    assert json.loads(json.dumps(out)) == {"k": [1.5, 2]}


@pytest.mark.parametrize("value,expected", [
    (np.int64(2), 2),
    (np.int32(2), 2),
    (np.float32(1.5), 1.5),
    (np.bool_(True), True),
])
def test_every_numpy_scalar_type_survives_a_list(value, expected):
    out = _jsonable({"k": [value]})
    assert out["k"] == [expected]
    assert json.loads(json.dumps(out))["k"] == [expected]


def test_a_mixed_list_keeps_its_length_and_order():
    """The positional facet: dropping only the ints re-indexes the list."""
    out = _jsonable({"k": [np.float64(1.0), np.int64(2), np.float64(3.0),
                           np.bool_(False)]})
    assert out["k"] == [1.0, 2, 3.0, False]


def test_an_all_numpy_integer_list_is_not_recorded_as_empty():
    out = _jsonable({"n_stars": list(np.array([7, 8, 9]))})
    assert out["n_stars"] == [7, 8, 9]


def test_skycoord_and_ndarray_members_are_still_stripped():
    """Unchanged: measurement inputs stay out of the record."""
    out = _jsonable({"coords": SkyCoord([1.0], [2.0], unit="deg"),
                     "dra": np.array([1.0, 2.0]),
                     "off_mas": np.float64(4.0)})
    assert "coords" not in out
    assert "dra" not in out
    assert out["off_mas"] == 4.0


def test_ndarray_elements_inside_a_list_are_still_stripped():
    out = _jsonable({"k": [np.array([1.0, 2.0]), np.int64(3)]})
    assert out["k"] == [3]


def test_dict_elements_inside_a_list_are_still_recursed():
    out = _jsonable({"cells": [{"n": np.int64(4), "coords": SkyCoord(
        [1.0], [2.0], unit="deg")}]})
    assert out["cells"] == [{"n": 4}]
