"""The vetting S/N floor uses the uncertainty of the merged flux.

A merged catalog's flux is the mean over nmatch per-frame fits, while its
flux_err is the weighted mean of the per-frame errors (one frame's
uncertainty).  flux_err_prop = 1/sqrt(sum 1/sigma_i^2) is the uncertainty of
the merged flux, ~flux_err/sqrt(nmatch).  With ``snr_floor_propagated`` the
floor is applied to flux / flux_err_prop.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS


def _mk(flux_err_prop, n=3):
    """star_like (flags=1, qfit 0.5) sources with per-frame S/N 4."""
    t = Table({
        'id': np.arange(n),
        'skycoord': SkyCoord((266.5 + np.arange(n) * 0.001) * u.deg,
                             np.full(n, -28.8) * u.deg),
        'qfit': np.full(n, 0.5),
        'flux': np.full(n, 40.0),
        'flux_err': np.full(n, 10.0),
        'flags': np.ones(n, int),
        'local_bkg': np.zeros(n),
    })
    if flux_err_prop is not None:
        t['flux_err_prop'] = np.asarray(flux_err_prop, float)
    return t


def _kept(t, on):
    out = _filter_extended_emission(t, local_snr_min=5.0, snr_floor_propagated=on,
                                    sky_clean_keep=False)
    return set(np.asarray(out['id']).tolist())


def test_floor_on_merged_flux_uncertainty():
    # merged-flux S/N: 16 (16 frames), 8, 4
    t = _mk([2.5, 5.0, 10.0])
    assert _kept(t, False) == set()          # per-frame S/N 4 < 5 for all
    assert _kept(t, True) == {0, 1}


def test_missing_or_bad_flux_err_prop_keeps_per_frame_floor():
    t = _mk([np.nan, 0.0, 2.5])
    assert _kept(t, True) == {2}
    t = _mk(None)
    t['flux_err'] = [10.0, 5.0, 10.0]        # per-frame S/N 4, 8, 4
    assert _kept(t, True) == {1}


def test_qfit_confident_exemption_unchanged():
    t = _mk([10.0, 10.0, 10.0])
    t['qfit'] = [0.1, 0.5, 0.5]
    assert _kept(t, True) == {0}


def test_pipeline_default_on():
    assert MANUAL_DEFAULTS['manual_ext_snr_floor_propagated'] is True
