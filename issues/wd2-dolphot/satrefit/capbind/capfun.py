"""Array re-implementation of the pipeline recovered-core cap (recovered_core_peak + recovered_cap_flux) on stored region pixels.
Verified against the pipeline functions in stage1 output (cap_H_pipe)."""
import numpy as np


def cap_arrays(cut, psf, ur, pkidx, ppk, excl=None, excl_gate=True, min_psf_frac=0.005, max_lost=0.2, min_px=3, return_bind=False):
    """cut, psf, ur: arrays over region pixels (raster order).  excl: extra unrecoverable mask (rule i).
    excl_gate: excluded pixels also count in the lost fraction (variant gm); False: only inside the flux bound (variant go).
    Returns cap (NaN = skipped) [, binding pixel index into the region arrays or -1]."""
    nreg = len(cut)
    un_gate = ur | excl if (excl is not None and excl_gate) else ur
    un_cap = ur | excl if excl is not None else ur
    nan = (np.nan, -1) if return_bind else np.nan
    if nreg == 0:
        return nan
    lost = un_gate.sum() / nreg
    rec_g = ~un_gate & np.isfinite(cut) & (cut > 0)
    if lost >= max_lost or rec_g.sum() < min_px:
        return nan
    measured = ~un_cap & np.isfinite(cut)
    rec = measured & (cut > 0)
    if not rec.any() or not ppk > 0:
        return nan
    psf_meas = np.where(measured & np.isfinite(psf), psf, -np.inf)
    ip = int(np.argmax(psf_meas))
    pk_meas = pkidx >= 0 and bool(measured[pkidx])
    if pk_meas or not cut[ip] > 0:
        cc = np.where(rec, cut, -np.inf)
        b = int(np.argmax(cc))
        res = cut[b] / ppk
        return (float(res), b) if return_bind else float(res)
    pp = psf[ip]
    if not (np.isfinite(pp) and pp > 0) or pp / ppk < min_psf_frac:
        return nan
    res = cut[ip] / pp
    return (float(res), ip) if return_bind else float(res)
