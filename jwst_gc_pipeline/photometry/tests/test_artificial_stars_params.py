"""Guard: artificial-star detection parameters must match production
(review #130 item 1 -- single-sourcing + drift pins)."""
import inspect
import re

from jwst_gc_pipeline.photometry import artificial_stars as a
from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS


def test_snr_and_m2_shape_single_sourced():
    assert a.M1_SNR == MANUAL_DEFAULTS['local_snr_threshold']
    assert a.M2_SNR == MANUAL_DEFAULTS['manual_iter2_local_snr']
    assert a.M2_ROUNDLO == MANUAL_DEFAULTS['manual_resid_roundlo']
    assert a.M2_ROUNDHI == MANUAL_DEFAULTS['manual_resid_roundhi']
    assert a.M2_SHARPLO == MANUAL_DEFAULTS['manual_resid_sharplo']
    assert a.M2_SHARPHI == MANUAL_DEFAULTS['manual_resid_sharphi']


def test_m1_sharp_literals_pinned_to_cataloging():
    # the m1 shape bounds are inline literals in cataloging.py; this pin
    # breaks if production changes them, so the completeness tool cannot
    # silently diverge from the real selection function
    src = inspect.getsource(cataloging)
    m = re.search(r'sharplo=([\d.]+),\s*sharphi=([\d.]+)', src)
    assert m, 'm1 sharp literals not found in cataloging.py -- update the pin'
    assert float(m.group(1)) == a.M1_SHARPLO
    assert float(m.group(2)) == a.M1_SHARPHI


def test_draw_positions_min_separation():
    import numpy as np
    rng = np.random.default_rng(1)
    valid = np.ones((512, 512), bool)
    xs, ys = a.draw_positions(rng, 512, 512, 300, valid, min_sep=10.0)
    assert len(xs) > 200
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.column_stack([xs, ys])).query(
        np.column_stack([xs, ys]), k=2)
    assert d[:, 1].min() >= 10.0
