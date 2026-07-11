"""Regression: an over-fluxed phantom fit (model peak >> local data peak) that
gouges a NEGATIVE crater in the residual must be DROPPED, not kept.

The brick F182M m6 clump shipped a flux=25434 phantom on ~5-count sky: model
peak 3352, a -2419 sigma residual crater, yet qfit=0.001 (the inflated model
LOWERS qfit, so the normal qfit<=0.2 gate admits it).  The per-frame
model_overshoot flag was cleared by the refit and dropped by the merge, so
neither the fit-time refit nor the merged-catalog vetting caught it.  The
field-agnostic FINAL overshoot drop in ``_manual_phot_pass`` (via
``_filter_or_flag_model_overshoot(action='drop')`` at
``manual_overshoot_drop_ratio``) is the net that removes it.  A real star, whose
model peak ~ its data peak (ratio ~1), must survive.
"""
import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.cataloging import _filter_or_flag_model_overshoot
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS


class _FakePhot:
    """Minimal stand-in for the PSFPhotometry object the guard mutates."""
    def __init__(self, results):
        self.results = results


# The drop pass calls the guard with flag_nonpositive_data=True (both
# instruments), so the test must match that call to exercise the shipped path.
_NONPOS = True


def _scene():
    """A real star + two phantoms on a flat background.

    - real star (10,10): data bump ~500, model peak ~500 -> ratio ~1 (KEEP)
    - phantom A (28,28): data slightly above bkg (the brick circ1 geometry,
      dpk>0), model peak 3352 -> huge ratio (DROP)
    - phantom B (28,10): data AT background (dpk<=0 blind spot), model peak
      900 -> caught only via flag_nonpositive_data (DROP)
    """
    ny = nx = 40
    data = np.full((ny, nx), 3.0, dtype=float)   # local_bkg
    modsky = np.zeros((ny, nx), dtype=float)

    data[10, 10] = 505.0
    modsky[10, 10] = 500.0

    data[28, 28] = 7.0        # dpk = 7-3 = 4 > 0, like circ1
    modsky[28, 28] = 3352.0

    data[28, 10] = 3.0        # dpk = 0 exactly (blind spot)
    modsky[28, 10] = 900.0

    res = Table()
    res['x_fit'] = [10.0, 28.0, 10.0]
    res['y_fit'] = [10.0, 28.0, 28.0]
    res['local_bkg'] = [3.0, 3.0, 3.0]
    res['flux_fit'] = [500.0, 25434.0, 6800.0]
    return _FakePhot(res), modsky, data


def test_default_drop_ratio_present_and_sane():
    r = MANUAL_DEFAULTS['manual_overshoot_drop_ratio']
    assert r > 1.5, "drop ratio must be clearly above the ~1 of a real star"


def test_phantoms_dropped_real_star_kept():
    phot, modsky, data = _scene()
    drop_ratio = MANUAL_DEFAULTS['manual_overshoot_drop_ratio']
    over = _filter_or_flag_model_overshoot(
        phot, modsky, data, ratio=drop_ratio, action='drop', label='test',
        flag_nonpositive_data=_NONPOS)
    # both phantoms removed (dpk>0 AND dpk<=0 blind spot); real star remains
    assert len(phot.results) == 1, "both phantoms must drop, real star kept"
    assert np.isclose(float(phot.results['flux_fit'][0]), 500.0)
    # mask flags exactly the two phantoms (rows 1 and 2)
    assert over.tolist() == [False, True, True]


def test_real_star_never_dropped_at_default_ratio():
    """A well-fit star whose model matches the data (ratio ~1) is never a crater."""
    phot, modsky, data = _scene()
    # keep only the real star
    phot.results = phot.results[:1]
    over = _filter_or_flag_model_overshoot(
        phot, modsky, data, ratio=MANUAL_DEFAULTS['manual_overshoot_drop_ratio'],
        action='drop', label='test', flag_nonpositive_data=_NONPOS)
    assert len(phot.results) == 1
    assert not bool(np.any(over))
