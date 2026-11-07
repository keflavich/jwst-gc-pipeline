"""Reference data and the frozen pane calibration (recipe R1).

    Q = (rate * k / FLAT - b_g) / a_g        [MJy/sr, the exposure's own cal level]
    k = median(cal / rate) over the imager pixels of the same exposure

Dropped pane pixels (NaN), as STScI cal/resample drop only DO_NOT_USE: rate DQ DO_NOT_USE, non-finite rate, and the
static pane bad pixels (BADPIX == 1, the pane analogue of the CRDS bad-pixel mask).  Kept but flagged: rate DQ
SATURATED pixels with a finite slope (from their unsaturated groups) and near-saturated ramps (rate * NGROUPS *
TGROUP >= 40000 DN).  The calibration-side estimators (overlap QA, anchoring, dataset gain) exclude those (EST_DROP),
as R1 was fitted without them.  Everything is computed on the crop of the detector frame that holds the three panes
(rows 10-694, columns 0-243; see data/F770W/qpm_R1_F770W.fits).
"""
from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

DATA = Path(__file__).resolve().parent / "data"
SHAPE = (1024, 1032)
LYOT = (slice(745, 1024), slice(0, 300))      # Lyot pane box (science in cal; excluded from k and offsets)
X_LO, X_HI = 5, 1026                          # reference pixels and the NaN column 4 are dropped
FLUENCE_DN = 40000.0
CAL_DROP = 1 | 512                            # DO_NOT_USE | NON_SCIENCE: what resample drops
VALID_RANGE = (30.0, 450.0)                   # validated cal-level brightness [MJy/sr]
CONTROL_DX = 591                              # control region = pane footprint + 591 columns (imager)

# per-exposure pane DQ bits (the DQ extension of <stem>_qpm.fits)
DQ_DO_NOT_USE = 1      # SCI is NaN (bad pixel, rate DO_NOT_USE / non-finite, or outside the panes)
DQ_SATURATED = 2       # rate DQ SATURATED, finite slope: kept (slope from the unsaturated groups), unvalidated
DQ_FLUENCE = 4         # rate * NGROUPS * TGROUP >= 40000 DN (near-saturated ramp): kept, unvalidated
DQ_BADPIX = 8          # static bad pixel (BADPIX == 1)
DQ_EDGE = 16           # within the de-smearing edge margin of a pane edge or a large NaN structure: RL deficits
DQ_ABOVE = 32          # cal-level value (SCI + sky offset) above the validated range (> 450 MJy/sr)
DQ_BELOW = 64          # cal-level value (SCI + sky offset) below the validated range (< 30 MJy/sr)
DQ_RATE_OTHER = 128    # rate DQ DO_NOT_USE or non-finite rate
EST_DROP = DQ_SATURATED | DQ_FLUENCE    # kept in the image, excluded from every calibration-side estimator


@lru_cache(maxsize=None)
def refdata(filt: str = "F770W", kernels: str = "production", calibration_dir=None) -> SimpleNamespace:
    """Read a checksummed calibration bundle; cached per process and bundle path."""
    if filt != "F770W":
        raise ValueError("4QPM calibration supports F770W only")
    d = Path(calibration_dir).resolve() if calibration_dir else DATA / filt
    required = {"qpm_R1_F770W.fits", "law_R1_F770W.json", "kernels_F770W.npz", "geometry_F770W.json"}
    sums = dict(line.split()[::-1] for line in (d / "SHA256SUMS").read_text().splitlines() if line.strip())
    if set(sums) != required:
        raise ValueError(f"Incomplete calibration checksums: {d}")
    for filename, checksum in sums.items():
        if hashlib.sha256((d / filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Calibration checksum mismatch: {d / filename}")
    with fits.open(d / "qpm_R1_F770W.fits") as h:
        x0, y0 = h[0].header["CROPX0"], h[0].header["CROPY0"]
        flat = h["FLAT"].data.astype(np.float64)
        ferr = h["FLAT_ERR"].data.astype(np.float64)
        badpix = h["BADPIX"].data.copy()
        region = h["REGION"].data.copy()
    ny, nx = region.shape
    law = json.load(open(d / "law_R1_F770W.json"))
    a = np.array([np.nan] + [law["panes"][g]["a"] for g in "123"])
    b = np.array([np.nan] + [law["panes"][g]["b"] for g in "123"])
    z = np.load(d / "kernels_F770W.npz")
    names = {"production": "K{}_production", "refit10678": "K{}"}
    if kernels not in names:
        raise ValueError(f"kernels must be one of {list(names)}")
    geo = json.load(open(d / "geometry_F770W.json"))
    pane = region > 0
    if set(np.unique(region)) != {0, 1, 2, 3} or region.shape != flat.shape or flat.shape != ferr.shape:
        raise ValueError("Invalid calibration image dimensions or pane labels")
    if not (np.isfinite(flat[pane]).all() and (flat[pane] > 0).all()
            and np.isfinite(ferr[pane]).all() and (ferr[pane] >= 0).all()
            and np.isfinite(a[1:]).all() and (a[1:] > 0).all() and np.isfinite(b[1:]).all()):
        raise ValueError("Invalid flat, flat uncertainty, or response law")
    for g in (1, 2, 3):
        for key in (f"K{g}", f"K{g}_production"):
            k = z[key]
            if k.shape != (5, 21) or not np.isfinite(k).all() or (k < 0).any() or not np.isclose(k.sum(), 1):
                raise ValueError(f"Invalid non-negative unit-sum kernel: {key}")
        for axis in ("cx", "cy"):
            if np.shape(geo["panes"][str(g)][axis]) != (3,) or not np.isfinite(geo["panes"][str(g)][axis]).all():
                raise ValueError(f"Invalid pane {g} geometry")
    return SimpleNamespace(
        directory=str(d), checksum=hashlib.sha256((d / "SHA256SUMS").read_bytes()).hexdigest(),
        filt=filt, x0=x0, y0=y0, crop=(slice(y0, y0 + ny), slice(x0, x0 + nx)),
        flat=flat, flat_err=ferr, badpix=badpix, region=region, law=law, a=a, b=b,
        kernel_id=kernels,
        kernels={g: z[names[kernels].format(g)] for g in (1, 2, 3)},
        mk_kernels={g: z[f"K{g}_production"] for g in (1, 2, 3)},   # the R1 law and anchoring used these
        models={g: (np.array(geo["panes"][str(g)]["cx"]), np.array(geo["panes"][str(g)]["cy"]))
                for g in (1, 2, 3)},
    )


def full_region(ref: SimpleNamespace) -> np.ndarray:
    """Full-frame region map: 1-3 panes, 4 Lyot box, 0 imager (as fullframe_preview._init)."""
    reg = np.zeros(SHAPE, np.uint8)
    reg[LYOT] = 4
    reg[ref.crop] = np.where(ref.region > 0, ref.region, reg[ref.crop])
    return reg


def read_exposure(sci: str, rate: str, ref: SimpleNamespace) -> SimpleNamespace:
    """Science frame (cal or crf) + its rate file; k and the exposure metadata."""
    import stdatamodels.jwst.datamodels as dm

    with fits.open(sci) as h:
        ph = h[0].header
        meta = dict(filter=ph.get("FILTER"), subarray=ph.get("SUBARRAY", "FULL"),
                    s_bkdsub=str(ph.get("S_BKDSUB", "") or ""))
    m = dm.open(sci)
    C = m.data.astype(np.float32)
    cdq = np.asarray(m.dq)
    C[(cdq & CAL_DROP) > 0] = np.nan
    wcs = m.meta.wcs
    m.close()
    with fits.open(rate) as h:
        R = h["SCI"].data.astype(np.float64)
        rdq = h["DQ"].data
        err = h["ERR"].data.astype(np.float64)
        vrn = h["VAR_RNOISE"].data.astype(np.float64) if "VAR_RNOISE" in h else None
        hd = h[0].header
        meta.update(ngroups=int(hd["NGROUPS"]), tgroup=float(hd["TGROUP"]), nints=int(hd.get("NINTS", 1)),
                    readpatt=str(hd.get("READPATT", "")),
                    mjd=float(hd.get("EXPMID", hd.get("EXPSTART", np.nan))))
    reg = full_region(ref)
    Rk = np.where((rdq & 1) > 0, np.nan, R)
    imager = (reg == 0) & np.isfinite(C) & (Rk > 0)
    meta["k_npix"] = int(imager.sum())
    k = float(np.median((C / Rk)[imager])) if imager.any() else np.nan
    cols = np.zeros(SHAPE, bool)
    cols[:, X_LO:X_HI + 1] = True
    im = (reg == 0) & cols
    meta["imager_satfrac"] = float(np.mean((rdq[reg == 0] & 2) > 0))
    meta["outlier_frac"] = float(np.mean((cdq[im] & 16) > 0))
    S = np.where((reg == 0) & cols, C, np.nan).astype(np.float32)   # imager only (no panes, no Lyot)
    return SimpleNamespace(sci=str(sci), rate=str(rate), C=C, S=S, R=R, rdq=rdq, err=err, vrn=vrn,
                           k=k, wcs=wcs, meta=meta)


def calibrate(R, rdq, err, vrn, k, ngroups, tgroup, ref):
    """Calibrated pane image on the crop frame.

    R, rdq, err, vrn: full-frame or crop-frame rate SCI, DQ, ERR, VAR_RNOISE (vrn may be None).
    Returns Q (float32), VAR (total variance), VAR_RN (read-noise variance; drizzle weights), DQ (uint16).
    Variances: rate ERR^2 and VAR_RNOISE scaled by (k / (a FLAT))^2, plus the flat error (Y / (a FLAT) * FLAT_ERR /
    FLAT)^2 in VAR.  The (a, b) law uncertainty is systematic and is not in VAR.
    """
    if R.shape == SHAPE:
        R, rdq, err = R[ref.crop], rdq[ref.crop], err[ref.crop]
        vrn = None if vrn is None else vrn[ref.crop]
    reg = ref.region
    pane = reg > 0
    a, b = ref.a[reg], ref.b[reg]
    F = ref.flat
    sat = (rdq & 2) > 0
    flu = R * ngroups * tgroup >= FLUENCE_DN
    bp = ref.badpix == 1
    other = ((rdq & 1) > 0) | ~np.isfinite(R)
    good = pane & ~(bp | other)                # sat / flu pixels are kept (flagged; EST_DROP for the estimators)
    with np.errstate(all="ignore"):
        Y = R * k
        Q = np.where(good, (Y / F - b) / a, np.nan).astype(np.float32)
        g2 = (k / (a * F)) ** 2
        var = np.where(good, g2 * err ** 2 + (Y / (a * F) * ref.flat_err / F) ** 2, np.nan)
        var_rn = np.where(good, g2 * (vrn if vrn is not None else err ** 2), np.nan)
    dq = np.zeros(reg.shape, np.uint16)
    dq[sat] |= DQ_SATURATED
    dq[flu] |= DQ_FLUENCE
    dq[bp] |= DQ_BADPIX
    dq[other] |= DQ_RATE_OTHER
    dq[~good] |= DQ_DO_NOT_USE
    return Q, var.astype(np.float32), var_rn.astype(np.float32), dq
