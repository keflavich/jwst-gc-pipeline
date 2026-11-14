"""Noise-only calibration helpers for the LOO chi statistic (issue #1008, item 2).

``loo_crop.py`` with env ``NOISEONLY=<seed>`` replaces the data of every exposure by
the fitted model plus Gaussian noise drawn from the cutout variance, then repeats the
whole cold-start fit and the leave-one-out scoring on the simulated data.  If the
chi = (d-pred)/sqrt(1/w+varQ) statistic is calibrated, sigma(chi) reads 1 in every
radial bin.  numpy only, so the helpers can be tested without finufft.
"""
import numpy as np

# radial bins of the real-data line printed by loo_crop.py, plus the 300-500 px bin of
# the README table (it needs a crop half-width of at least 300 px)
BINS = [(30, 50), (50, 80), (80, 120), (120, 200), (200, 300)]
BINS_NOISEONLY = BINS+[(300, 500)]


def draw_noise_only(model, d, w, rng):
    """Simulated data: ``model`` plus a Gaussian draw of variance 1/w where w > 0.

    ``d`` is returned unchanged where w <= 0 (those pixels carry no weight in any fit
    or statistic).  The variance is the cutout variance (Poisson + read noise,
    cutout.py), evaluated at the observed rather than the simulated signal.
    """
    model = np.asarray(model, float); d = np.asarray(d, float); w = np.asarray(w, float)
    good = w > 0
    noise = np.zeros_like(model)
    noise[good] = rng.normal(0.0, 1.0, int(good.sum()))/np.sqrt(w[good])
    return np.where(good, model+noise, d)


def sigma_chi(chi):
    """Robust sigma of chi: uncentred 1.4826*median|chi| and centred 1.4826*median|chi-median|,
    plus mean(chi^2) and the pixel count (NaN statistics for an empty selection)."""
    chi = np.asarray(chi, float)
    if chi.size == 0:
        return dict(n=0, sig=np.nan, sig_centred=np.nan, chi2=np.nan)
    return dict(n=int(chi.size),
                sig=float(1.4826*np.median(np.abs(chi))),
                sig_centred=float(1.4826*np.median(np.abs(chi-np.median(chi)))),
                chi2=float(np.mean(chi**2)))


def radial_report(chi, R, ok, bins=BINS_NOISEONLY):
    """sigma_chi per radial bin ``lo <= R < hi`` over the pixels selected by ``ok``."""
    return {f'{lo}-{hi}': sigma_chi(chi[ok & (R >= lo) & (R < hi)]) for lo, hi in bins}
