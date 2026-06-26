"""Regression tests for the saturated-star accept gate.

Guards the 2026-06-18 fix: the NIRCam ``ssr_ratio < 1`` gate was silently
deleting REAL must-detect bright saturated stars (high snr, good qfit) because
STPSF's wing-amplitude mismatch pushes ssr_ratio > 1 even for excellent fits.
These stars then lingered full-strength in the residual and were missing from
the model.  ``accept_satstar_fit`` now subordinates the ssr gate to confidence:
a high-snr, good-qfit fit is accepted regardless of ssr_ratio; the ssr gate
only filters low-confidence fits.

See sickle F480M brightstar_regression stars (snr 58-329, qfit 0.27-2.26,
ssr 1.0-2.6) that frame 00008 rejected 4/11 -> 10/11 after the fix.
"""
import pytest

from jwst_gc_pipeline.reduction.saturated_star_finding import accept_satstar_fit

# NIRCam keep thresholds (from get_saturated_stars, non-MIRI branch)
NRC = dict(is_miri=False, qfit_max_keep=5.0, sidelobe_min_keep=-10.0,
           ssr_ratio_max_keep=1.0, snr_min_keep=3.0)
MIRI = dict(is_miri=True, qfit_max_keep=15.0, sidelobe_min_keep=-40.0,
            ssr_ratio_max_keep=2.0, snr_min_keep=2.0)


def _fit(**over):
    base = dict(result_is_none=False, fluxerr=1.0, snr=100.0, flux=1e5,
                qfit=0.5, sidelobe_resid_sigma=2.0, ssr_ratio=0.3)
    base.update(over)
    return base


# ---- the bug: real bright stars rejected purely on ssr_ratio ----------------
# Verbatim metrics of the 4 must-detect stars frame 00008 wrongly rejected.
@pytest.mark.parametrize("snr,qfit,ssr", [
    (329.0, 0.27, 2.02),   # src6
    (58.0, 0.36, 2.58),    # src9
    (135.0, 1.12, 1.29),   # src5
    (262.0, 2.26, 1.05),   # src10
])
def test_bright_star_with_bad_ssr_is_accepted(snr, qfit, ssr):
    assert accept_satstar_fit(**_fit(snr=snr, qfit=qfit, ssr_ratio=ssr), **NRC) is True


# ---- the safeguard ssr still provides for LOW-confidence fits ----------------
def test_lowconf_fit_high_ssr_still_rejected():
    # marginal snr (below the trust threshold) + bad ssr -> ssr gate applies
    assert accept_satstar_fit(**_fit(snr=5.0, qfit=4.0, ssr_ratio=3.0), **NRC) is False


def test_lowconf_fit_good_ssr_accepted():
    assert accept_satstar_fit(**_fit(snr=5.0, qfit=4.0, ssr_ratio=0.5), **NRC) is True


# ---- the original failure mode (white-point/no source) is still rejected -----
def test_whitepoint_failure_rejected_by_qfit():
    # 0310g_00002 catastrophic case: snr~2, qfit=27 -> snr and qfit gates reject
    assert accept_satstar_fit(**_fit(snr=2.0, qfit=27.0, ssr_ratio=5.0), **NRC) is False


# ---- basic gates unchanged ---------------------------------------------------
def test_negative_flux_rejected():
    assert accept_satstar_fit(**_fit(flux=-1.0), **NRC) is False


def test_low_snr_rejected():
    assert accept_satstar_fit(**_fit(snr=1.0), **NRC) is False


def test_high_qfit_rejected_even_if_high_snr():
    # qfit above the cap is rejected regardless of snr (not a trustworthy fit)
    assert accept_satstar_fit(**_fit(snr=500.0, qfit=9.0, ssr_ratio=0.1), **NRC) is False


def test_strong_negative_sidelobe_rejected():
    assert accept_satstar_fit(**_fit(sidelobe_resid_sigma=-30.0), **NRC) is False


def test_nan_ssr_accepted_when_otherwise_good():
    assert accept_satstar_fit(**_fit(ssr_ratio=float('nan')), **NRC) is True


def test_result_none_rejected():
    assert accept_satstar_fit(**_fit(result_is_none=True), **NRC) is False


# ---- MIRI branch: ssr dropped entirely --------------------------------------
def test_miri_ignores_ssr():
    # huge ssr, but finite positive qfit below cap + good snr -> accepted
    assert accept_satstar_fit(**_fit(snr=1525.0, qfit=1.30, ssr_ratio=45.0), **MIRI) is True


def test_miri_requires_positive_qfit():
    assert accept_satstar_fit(**_fit(qfit=-1.0), **MIRI) is False
