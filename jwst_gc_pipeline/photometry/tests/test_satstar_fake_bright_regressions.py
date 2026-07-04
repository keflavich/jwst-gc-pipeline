"""Regression tests for the satstar FAKE-BRIGHT predicate, encoded from REAL
observed failures so the heuristic can't silently regress to over-tuned-for-one-
field behaviour.

Each case is a recorded (model_peak, local_peak, has_saturated_core) measured at
a user-flagged position in a real product:

  * 2526 G0 F770W cloud-c filament (2026-06-23): the field-general stress test.
    FAKE stars sit on faint, smooth, FINITE extended emission (no NaN core) yet
    were fit with model peaks 4e4-3e5 -> must be REJECTED.  Real saturated stars
    have genuine NaN cores -> must be KEPT even when their local peak is faint
    (~2000), which an absolute local-peak cut alone would wrongly kill.
  * sickle F770W (2026-06-22): fakes on brighter emission (local peak up to
    ~2300) -> must still be REJECTED; real A/B have NaN cores + bright peaks.

The key field-generality property: an ABSOLUTE local-peak threshold cannot serve
both fields (sickle fakes reach 2300 > 2526 real stars at 2065).  The NaN-core
exemption is what makes it general -- these tests lock that in.
"""
import pytest
import numpy as np
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    is_fake_bright, is_small_radius_emission_phantom, is_bright_flux_phantom)

# (label, model_peak, local_peak, has_saturated_core, expect_fake)
CASES = [
    # --- 2526 cloud-c filament FAKES: finite smooth emission, must REJECT ---
    ('2526-FAKE-1', 262085.0, 337.0, False, True),
    ('2526-FAKE-2',  39317.0, 408.0, False, True),
    # --- 2526 real SATURATED stars: genuine NaN core, must KEEP ---
    #     SAT-faint has a faint local peak (2065) -- the case an absolute cut
    #     would wrongly kill.  Test it both at its observed under-fit model and
    #     at a hypothetical CORRECT bright model (post fix #4) -- both must KEEP.
    ('2526-SAT-faint-underfit', 3622.0, 2065.0, True, False),
    ('2526-SAT-faint-fixedbright', 80000.0, 2065.0, True, False),
    ('2526-SAT-miss', 120000.0, 3859.0, True, False),
    # --- sickle FAKES on brighter emission: must still REJECT ---
    ('sickle-FAKE-hi', 626707.0, 1782.0, False, True),
    ('sickle-FAKE-lo',  12187.0, 1104.0, False, True),
    # --- sickle real A/B: NaN core + bright peak, must KEEP ---
    ('sickle-A', 166374.0, 6177.0, True, False),
    ('sickle-B',  41240.0, 10963.0, True, False),
    # --- under-subtracted real star (model below model_min): never fake ---
    ('undersub-faint', 3622.0, 1500.0, False, False),
    # --- correctly-fit bright star, no core flag but bright local peak: keep ---
    ('bright-real-nocore', 100000.0, 8000.0, False, False),
]


@pytest.mark.parametrize('label,mpk,lpk,satcore,expect', CASES,
                         ids=[c[0] for c in CASES])
def test_fake_bright_classification(label, mpk, lpk, satcore, expect):
    assert is_fake_bright(mpk, lpk, has_saturated_core=satcore) is expect, label


def test_nan_inputs_never_fake():
    assert is_fake_bright(float('nan'), 100.0) is False
    assert is_fake_bright(1e6, float('nan')) is False


def test_disabled_thresholds_never_fake():
    assert is_fake_bright(1e6, 1.0, model_min=0, localpk_max=0) is False


def test_saturated_core_always_exempt():
    # even an absurd model on a faint peak is KEPT if a genuine NaN core exists
    assert is_fake_bright(1e9, 1.0, has_saturated_core=True) is False


# --- small-radius (neighbour-robust) prominence/core phantom predicate ---
# Recorded SMALL-fixed-radius (sa=400, ~11px) prominence/core from the 2526 G0
# F770W coadd.  The sat_area-scaled radius (sa=1600) wrongly inflated FAKE-1 to
# prom 49 by reaching a neighbour; the small radius reads its true 1.6 -> phantom.
# (label, prom_small, core_small, expect_phantom)
SMALL_RADIUS_CASES = [
    ('2526-FAKE-1', 1.6, 412.0, True),    # at sa=1600 this read prom 49 -> evaded
    ('2526-FAKE-2', 3.5, 412.0, True),
    ('2526-SAT-faint', 110.1, 2065.0, False),  # real, peaked at every radius
    ('2526-SAT-miss', 42.7, 3859.0, False),
    ('huge-real-zeroed-core', float('nan'), float('nan'), False),  # NaN -> kept
    ('low-prom-ok-core', 3.0, 5000.0, True),   # low prominence alone rejects
    ('ok-prom-low-core', 50.0, 500.0, True),   # low core alone rejects
]


@pytest.mark.parametrize('label,prom,core,expect', SMALL_RADIUS_CASES,
                         ids=[c[0] for c in SMALL_RADIUS_CASES])
def test_small_radius_phantom(label, prom, core, expect):
    assert is_small_radius_emission_phantom(prom, core) is expect, label


# --- POST-FIT bright-flux phantom predicate (W51 F770W) ---------------------
# Recorded (flux_fit, flux_init, ssr_ratio) from the W51 F770W consolidated
# satstar catalog at user-flagged positions.  These sit on saturated extended
# emission where EVERY pre-fit metric ranks the phantom >= a real star (verified:
# seed prominence REAL=8.6 was the LOWEST of the three); only the post-fit flux +
# ssr/ratio signals separate them.  The flux_floor precondition is load-bearing:
# it protects the real deeply-saturated spiky star whose faint wing-seed gives a
# ratio of 61 but whose flux (2e4) is far below the 1e5 floor.
# (label, flux_fit, flux_init, ssr_ratio, expect_phantom)
BRIGHT_PHANTOM_CASES = [
    # user-flagged spurious super-bright phantoms -> REJECT
    ('W51-SPUR1-highssr',    2.84e6, 175806.0, 139.0, True),   # ssr 139
    ('W51-SPUR2-hiratio',    7.04e6,   6677.0,  10.1, True),   # flux/init 1054
    # user's real well-fit bright star -> KEEP (ssr 12, ratio 14)
    ('W51-REAL-bright',      2.89e6, 200806.0,  12.1, False),
    # real deep-sat spiky star: high ratio (61) but modest flux 2e4 < floor -> KEEP
    ('W51-real-spiky-lowflux', 2.0e4,   328.0,  62.0, False),
    # bright + clean fit (low ssr, low ratio) -> KEEP
    ('bright-clean',         5.0e6, 300000.0,  15.0, False),
    # ratio-only trigger at high flux -> REJECT
    ('bright-hiratio-lowssr', 3.0e6,  20000.0,   8.0, True),   # ratio 150
]


@pytest.mark.parametrize('label,flux,finit,ssr,expect', BRIGHT_PHANTOM_CASES,
                         ids=[c[0] for c in BRIGHT_PHANTOM_CASES])
def test_bright_flux_phantom(label, flux, finit, ssr, expect):
    assert is_bright_flux_phantom(flux, finit, ssr) is expect, label


def test_bright_phantom_disabled_by_zero_floor():
    # flux_floor<=0 disables the gate entirely (default OFF -> no regression)
    assert is_bright_flux_phantom(7e6, 100.0, 999.0, flux_floor=0.0) is False


def test_bright_phantom_nan_inputs_never_phantom():
    assert is_bright_flux_phantom(float('nan'), 100.0, 999.0) is False
    # non-finite flux_init just disables the ratio arm; ssr arm still valid
    assert is_bright_flux_phantom(5e6, float('nan'), 999.0) is True
    assert is_bright_flux_phantom(5e6, float('nan'), 5.0) is False
