"""Physically-motivated saturation forward model for injecting saturated stars.

Simulates the up-the-ramp behaviour of an H2RG pixel scene so that a known-flux
injected star acquires REALISTIC saturation artefacts, then ramp-fits it exactly
as calwebb_detector1 does -> cal-like SCI (DN/s) + DQ + VAR_POISSON. Feeding the
result to ``get_saturated_stars`` and comparing recovered vs injected flux gives
the bias-vs-saturation-regime curve.

Three regimes are modelled separately (see docs/reports/SATSTAR_INJECTION_MODEL.md):

  (a) per-pixel nonlinearity  -- CRDS ``linearity`` polynomial, applied FORWARD
      (linear e- -> measured DN). Isolated-pixel effect.
  (b) hard saturation         -- CRDS ``saturation`` full-well (DN); group-0
      saturation -> DO_NOT_USE + VAR_POISSON NaN (truly lost).
  (c) charge migration        -- JWST ``charge_migration``: charge spills from a
      pixel above ``signal_threshold`` (~25000 DN) into its LESS-FULL neighbours
      (brighter-fatter limit; Plazas+2018). Excess in neighbours; the flagging is
      only applied for NGROUPS>=3 (so NGROUPS<=2 keeps the excess -- the gc2211
      vs brick #210 dichotomy). Coupling ``f_bf`` is the single free parameter,
      calibrated against the #210 fit-flux-vs-footprint curves.

All detector numbers come from the CRDS reference files the pipeline itself uses;
nothing is hardcoded except the documented charge_migration threshold default.
"""
import os

import numpy as np
from astropy.io import fits
from jwst.datamodels import dqflags

P = dqflags.pixel
CRDS = os.environ.get('CRDS_PATH', '/orange/adamginsburg/jwst/crds')
_REFDIR = f'{CRDS}/references/jwst/nircam'
# charge_migration signal_threshold: pipeline docs say g2g diffs drop past
# ~25000 ADU; the step default is in DN. Overridable per run.
DEFAULT_MIGRATION_THRESH_DN = 25000.0


def _latest(pattern):
    import glob
    fs = sorted(glob.glob(f'{_REFDIR}/{pattern}'))
    if not fs:
        raise FileNotFoundError(f'no CRDS ref matching {pattern} under {_REFDIR}')
    return fs[-1]


def load_detector_refs(detector, box=None):
    """Return dict of gain (e-/DN), fullwell (DN), linearity coeffs for ``detector``.

    ``box`` = (y0,y1,x0,x1) to slice to a cutout (the injected-star region); the
    refs are 2048x2048 so slicing keeps memory small. Picks the CRDS ref whose
    DETECTOR matches; falls back to the highest-numbered file.
    """
    det = detector.upper()

    def _pick(pattern, ext):
        import glob
        for f in sorted(glob.glob(f'{_REFDIR}/{pattern}'), reverse=True):
            with fits.open(f) as h:
                if h[0].header.get('DETECTOR', '').upper() == det:
                    return f
        return _latest(pattern)

    sl = (slice(box[0], box[1]), slice(box[2], box[3])) if box else (slice(None), slice(None))
    with fits.open(_pick('*gain*', 'SCI')) as h:
        gain = np.asarray(h['SCI'].data[sl], float)
    with fits.open(_pick('*satur*', 'SCI')) as h:
        fullwell = np.asarray(h['SCI'].data[sl], float)   # DN
    with fits.open(_pick('*linear*', 'COEFFS')) as h:
        coeffs = np.asarray(h['COEFFS'].data[(slice(None),) + sl], float)  # (ncoef,ny,nx)
    gain = np.where(np.isfinite(gain) & (gain > 0), gain, np.nanmedian(gain[gain > 0]))
    fullwell = np.where(np.isfinite(fullwell) & (fullwell > 0), fullwell, 65535.0)
    return dict(gain=gain, fullwell=fullwell, coeffs=coeffs, detector=det)


def _poly_correct(dn, coeffs):
    """CRDS linearity CORRECTION: measured DN -> linear DN (sum_k c_k dn^k)."""
    out = np.zeros_like(dn)
    for k in range(coeffs.shape[0]):
        out += coeffs[k] * dn ** k
    return out


def _poly_deriv(dn, coeffs):
    out = np.zeros_like(dn)
    for k in range(1, coeffs.shape[0]):
        out += k * coeffs[k] * dn ** (k - 1)
    return out


def nonlinearize(linear_dn, coeffs, iters=12):
    """FORWARD nonlinearity: linear DN -> measured DN, by Newton-inverting the
    CRDS correction polynomial (monotonic up to saturation)."""
    dn = np.array(linear_dn, float)          # start from identity guess
    for _ in range(iters):
        f = _poly_correct(dn, coeffs) - linear_dn
        d = _poly_deriv(dn, coeffs)
        d = np.where(np.abs(d) < 1e-6, 1e-6, d)
        dn = dn - f / d
        dn = np.clip(dn, 0, None)
    return dn


def _redistribute(Q, f_bf, thresh_e):
    """Charge migration / brighter-fatter: move charge from each pixel that is
    ABOVE ``thresh_e`` toward its 4 LESS-FULL neighbours, conserving total charge.

    For each direction the outgoing flow from a pixel is
        flow = min( f_bf * (Q-thresh)_+ * (Q-Q_nbr)_+/(thresh+Q) , (Q-thresh)_+/4 )
    The (Q-Q_nbr)_+ weighting encodes "charge attracted to less-full neighbours"
    (Plazas+2018) and grows as the core saturates. Each pixel LOSES the sum of its
    four outgoing flows and GAINS its neighbours' flows directed at it.
    """
    over = np.clip(Q - thresh_e, 0.0, None)
    if not np.any(over):
        return Q
    leaving = np.zeros_like(Q)
    deposit = np.zeros_like(Q)
    for ax, sh in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        Qn = np.roll(Q, sh, axis=ax)                 # neighbour value at each cell
        w = np.clip(Q - Qn, 0.0, None)               # only spill to less-full nbr
        flow = f_bf * over * w / (thresh_e + Q)
        edge = [slice(None), slice(None)]
        edge[ax] = (0 if sh == 1 else -1)            # kill wrap-around at the border
        flow[tuple(edge)] = 0.0
        flow = np.minimum(flow, over / 4.0)          # can't move more than its overflow share
        leaving += flow
        deposit += np.roll(flow, sh, axis=ax)        # cell at +sh receives this flow
    return Q - leaving + deposit


def simulate_cal(rate_e_s, refs, ngroups, tgroup, f_bf=0.0,
                 migration_thresh_dn=DEFAULT_MIGRATION_THRESH_DN,
                 apply_charge_migration_flag=None, readnoise_dn=0.0, rng=None):
    """Simulate a cal-level SCI/DQ/VAR_POISSON for a scene of true count-rates.

    Parameters
    ----------
    rate_e_s : 2D array, TRUE linear count rate (e-/s) of the scene (star+bkg).
    refs : from load_detector_refs (gain, fullwell DN, linearity coeffs), same shape.
    ngroups, tgroup : ramp geometry (header NGROUPS, TGROUP seconds).
    f_bf : charge-migration coupling (0 = off). Calibrated against #210.
    apply_charge_migration_flag : default None -> True iff ngroups>=3 (pipeline rule).

    Returns dict: sci (DN/s), dq (uint32), var_poisson (DN/s^2 with NaN where lost),
    sat_group (int, group index of first saturation or -1), n_good.
    """
    gain, fullwell, coeffs = refs['gain'], refs['fullwell'], refs['coeffs']
    if apply_charge_migration_flag is None:
        apply_charge_migration_flag = ngroups >= 3
    thresh_e = migration_thresh_dn * gain
    ny, nx = rate_e_s.shape
    dn_cube = np.zeros((ngroups, ny, nx), float)
    sat_cube = np.zeros((ngroups, ny, nx), bool)
    cl_cube = np.zeros((ngroups, ny, nx), bool)   # charge-loss (excess) groups
    Q = np.zeros((ny, nx), float)                  # accumulated linear charge (e-)
    for g in range(ngroups):
        Q = Q + rate_e_s * tgroup
        if f_bf > 0:
            Q = _redistribute(Q, f_bf, thresh_e)
        lin_dn = Q / gain
        meas_dn = nonlinearize(lin_dn, coeffs)
        sat = meas_dn >= fullwell
        meas_dn = np.minimum(meas_dn, fullwell)
        # charge-migration flag: group past the (DN) threshold
        cl = meas_dn >= migration_thresh_dn
        if rng is not None and readnoise_dn > 0:
            meas_dn = meas_dn + rng.normal(0, readnoise_dn, meas_dn.shape)
        dn_cube[g] = meas_dn
        sat_cube[g] = sat
        cl_cube[g] = cl
    return _ramp_fit(dn_cube, sat_cube, cl_cube, tgroup, gain,
                     apply_charge_migration_flag)


def _ramp_fit(dn_cube, sat_cube, cl_cube, tgroup, gain, apply_cm_flag):
    """Slope of the good groups (not saturated; not charge-loss if flagging on).
    Mirrors calwebb_detector1: DO_NOT_USE + VAR_POISSON NaN where group0 lost."""
    ng, ny, nx = dn_cube.shape
    good = ~sat_cube
    if apply_cm_flag:
        good = good & ~cl_cube
    g = good.astype(float)
    ngood = good.sum(axis=0)
    sat_group = np.where(sat_cube.any(axis=0), sat_cube.argmax(axis=0), -1)
    g0lost = sat_cube[0] | (cl_cube[0] & apply_cm_flag)

    # vectorised least-squares slope over each pixel's good groups (t = g*tgroup)
    t = (np.arange(ng) * tgroup)[:, None, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        n = np.where(ngood > 0, ngood, np.nan)
        tmean = (g * t).sum(0) / n
        ymean = (g * dn_cube).sum(0) / n
        num = (g * (t - tmean) * (dn_cube - ymean)).sum(0)
        den = (g * (t - tmean) ** 2).sum(0)
        slope = num / den                                    # DN/s, n>=2
    sci = np.where(ngood >= 2, slope, np.nan)
    # single-good-group fallback: value/tgroup
    one = ngood == 1
    if np.any(one):
        first_good = good.argmax(0)
        yv = np.take_along_axis(dn_cube, first_good[None], 0)[0]
        sci = np.where(one, yv / tgroup, sci)
    var = np.where(ngood >= 1, np.clip(sci, 0, None) / np.maximum(gain, 1e-3)
                   / np.maximum(ngood * tgroup, tgroup), np.nan)
    dq = np.zeros((ny, nx), np.uint32)
    dq[sat_cube.any(axis=0)] |= P['SATURATED']
    if apply_cm_flag:
        dq[cl_cube.any(axis=0)] |= P['CHARGELOSS'] if 'CHARGELOSS' in P else 0
    lost = g0lost | (ngood < 1)
    dq[lost] |= P['DO_NOT_USE'] | P['SATURATED']
    var[lost] = np.nan
    return dict(sci=sci, dq=dq, var_poisson=var, sat_group=sat_group, n_good=ngood)
