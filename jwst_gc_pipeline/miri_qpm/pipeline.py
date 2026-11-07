"""Shared pipeline: pass A (per exposure: calibrate, optional registration, sky offset, overlap and control blocks,
pane stars), decisions (registration fallback, optional dataset gain and anchoring), pass B (per exposure: apply the
offsets, de-smear, trim, write <stem>_qpm.fits, drizzle onto the output grid), mosaic, QA.

Modes '10678' and 'external' differ only in their defaults and allowed options (resolve_settings); the workflow
(workflow.py) runs 'external'.  One run can take several input groups (`inputs`: name, sci, rate_dir, register, ...)
onto one reference and grid.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from . import __version__, calib, geometry, mosaic, offsets, psf, qa, register

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
DEFAULTS = dict(
    sci=None, rate_dir=None, ref=None, out_dir=None, name="qpm", calibration_dir=None,
    register=None, anchor="off", dataset_gain="off",
    kernels="production", rl_iter=psf.RL_ITER, edge_trim=psf.EDGE_TRIM, edge_mode="flag", science_mask=None, inputs=None,
    grid="full", n_proc=4, limit=0, write_exposures=True, quicklook=True, gc_like="auto",
    search_px=list(register.SEARCH_PX), min_match=register.MIN_MATCH, min_match_frac=register.MIN_MATCH_FRAC,
    max_rms=register.MAX_RMS, allow_bkdsub=False, fill_bright_holes=True,
)
VALID_NG = (5, 8)
SATFRAC_WARN = 0.05
OUTLIER_WARN = 0.2
MIN_K_PIX = 10000
GC_BOX = (1.5, 0.75)    # |l|, |b| [deg]: 'GC-like' box (10678 spans l -0.07..0.84, b -0.11..0.62; GC tests inside)
EXPOSURE_KEYS = ("register", "gc_like", "allow_bkdsub", "search_px", "min_match", "min_match_frac", "max_rms", "limit")
GROUP_KEYS = ("name", "sci", "rate_dir") + EXPOSURE_KEYS      # allowed in an [[inputs]] table


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------- inputs
def find_inputs(cfg):
    """One input group (sci, rate_dir, limit) -> rows (stem, sci, rate), notes."""
    pats = cfg["sci"] if isinstance(cfg["sci"], (list, tuple)) else [cfg["sci"]]
    rows, notes, seen = [], [], set()
    for f in sorted({f for p in pats for f in glob.glob(str(p))}):
        m = re.match(r"(jw\d+_[0-9a-z]+_\d+_mirimage)", Path(f).name)
        if not m or m.group(1) in seen:
            continue
        stem = m.group(1)
        rate = Path(cfg["rate_dir"]) / f"{stem}_rate.fits"
        h = fits.getheader(f)
        if h.get("FILTER") != "F770W" or h.get("SUBARRAY", "FULL") != "FULL":
            notes.append(f"{stem}: {h.get('FILTER')} {h.get('SUBARRAY')} skipped (F770W FULL only)")
            continue
        if not rate.exists():
            notes.append(f"{stem}: no rate file in {cfg['rate_dir']}, skipped")
            continue
        seen.add(stem)
        rows.append((stem, str(f), str(rate)))
    if cfg["limit"]:
        rows = rows[:cfg["limit"]]
    return rows, notes


def merge_groups(groups):
    """All input groups -> rows (stem, sci, rate, group index), notes.  A stem found in several groups is kept in the
    first group only (no exposure is processed twice)."""
    rows, notes, seen = [], [], set()
    for g, gi in enumerate(groups):
        grows, gnotes = find_inputs(gi)
        notes += gnotes
        for stem, sci, rate in grows:
            if stem in seen:
                notes.append(f"{stem}: also in input group {gi['name']!r}; kept in the first group only")
                continue
            seen.add(stem)
            rows.append((stem, sci, rate, g))
    return rows, notes


def fallback_similarity(r, ok, gof):
    """Median similarity of the registered frames `ok` of r's own input group, from r's dither set when it has
    registered members, else from the whole group; None when the group has no registered frame."""
    same = [q for q in ok if gof[q["i"]] == gof[r["i"]]]
    grp = [q["p"] for q in same if q["stem"][:20] == r["stem"][:20]] or [q["p"] for q in same]
    return np.median(grp, 0) if grp else None


# --------------------------------------------------------------- worker state
_W = {}


def _quiet():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", module="astropy.wcs")


def _init(cfg):
    _quiet()                       # worker processes only
    try:
        _W["cfg"] = cfg
        _W["ref"] = calib.refdata("F770W", cfg["kernels"], cfg.get("calibration_dir"))
        img, _, hdr = open_reference(cfg)
        _W["img"], _W["rw"] = img, WCS(hdr)
        _W["XY"] = geometry.pane_positions(_W["ref"])
    except Exception as e:         # raising here would make the Pool respawn workers forever
        _W["init_error"] = repr(e)


def open_reference(cfg):
    """(image, ERR or None, header) of the reference, memory-mapped, read through the optional science mask."""
    rf = fits.open(cfg["ref"], memmap=True)
    rh = mosaic.ref_hdu(rf)
    img = rh.data
    err = rf["ERR"].data if "ERR" in rf and rf["ERR"].data is not None and rf["ERR"].data.shape == img.shape else None
    if cfg.get("science_mask"):
        m = mosaic.ref_hdu(fits.open(cfg["science_mask"], memmap=True)).data
        img = mosaic.MaskedRef(img, m)
        err = None if err is None else mosaic.MaskedRef(err, m)
    return img, err, rh.header


def _guard(fn, job, i, stem, stage):
    """Run fn(job); an exception becomes a skip of this exposure instead of aborting the run."""
    try:
        if "init_error" in _W:
            raise RuntimeError("worker start-up failed: " + _W["init_error"])
        return fn(job)
    except Exception as e:
        return dict(i=i, stem=stem, skip=f"error in {stage}: {e!r}")


def skip_reason(meta, k, cfg):
    if meta["s_bkdsub"].upper() == "COMPLETE" and not cfg["allow_bkdsub"]:
        return "S_BKDSUB = COMPLETE: background-subtracted cal is not proportional to rate (k invalid)"
    if not np.isfinite(k) or meta["k_npix"] < MIN_K_PIX:
        return f"k from only {meta['k_npix']} imager pixels"
    return None


def _bilinear(cut, px, py):
    fin = np.isfinite(cut)
    v = ndimage.map_coordinates(np.where(fin, cut, 0), [py, px], order=1, cval=0.0)
    w = ndimage.map_coordinates(fin.astype(np.float32), [py, px], order=1, cval=0.0)
    return np.where(w > 0.999, v, np.nan)


def _cutout(wcs, rw, img, p, c, margin=40):
    gy, gx = np.mgrid[0:1024:32, 0:1032:32].astype(float)
    px, py = geometry.det_to_ref(wcs, rw, gx.ravel(), gy.ravel(), p, c)
    x0, x1 = int(max(0, px.min() - margin)), int(min(img.shape[1], px.max() + margin))
    y0, y1 = int(max(0, py.min() - margin)), int(min(img.shape[0], py.max() + margin))
    centre = (float(np.mean(px)), float(np.mean(py)))
    if x1 - x0 < 50 or y1 - y0 < 50:
        return None, 0, 0, centre
    return np.array(img[y0:y1, x0:x1], dtype=np.float32), x0, y0, centre


def _exposure_warnings(meta, wcs, cfg):
    w = []
    selected = calib.refdata("F770W", cfg.get("kernels", "production"), cfg.get("calibration_dir"))
    if selected.law.get("status") == "candidate":
        w.append("candidate calibration: scientific validation pending")
    if not VALID_NG[0] <= meta["ngroups"] <= VALID_NG[1]:
        w.append(f"NGROUPS outside the validated {VALID_NG[0]}-{VALID_NG[1]} | NGROUPS {meta['ngroups']}")
    dt = max(selected.law["validity"]["training_mjd"][0] - meta["mjd"],
             meta["mjd"] - selected.law["validity"]["training_mjd"][1], 0.0) / 365.25
    meta["epoch_offset_yr"] = dt        # informational: no time drift was detected robustly, so no epoch warning
    if meta["imager_satfrac"] > SATFRAC_WARN:
        w.append(f"imager saturated fraction > {SATFRAC_WARN} | {meta['imager_satfrac']:.3f}")
    if meta["outlier_frac"] > OUTLIER_WARN:
        w.append(f"OUTLIER-flagged imager fraction > {OUTLIER_WARN}: heavy outlier flagging can bias "
                 f"k = median(cal/rate) | {meta['outlier_frac']:.2f}")
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra, dec = wcs(516.0, 512.0, with_bounding_box=False)
    gal = SkyCoord(ra * u.deg, dec * u.deg).galactic
    l, b = ((gal.l.deg + 180) % 360) - 180, gal.b.deg
    meta["glon"], meta["glat"] = float(l), float(b)
    gc = (abs(l) <= GC_BOX[0] and abs(b) <= GC_BOX[1]) if cfg["gc_like"] == "auto" else bool(cfg["gc_like"])
    meta["gc_like"] = gc
    if not gc:
        w.append("non-GC field: the calibration was validated on GC fields only (W51: +3 to +6 MJy/sr at the "
                 f"faint end without anchoring) | l {l:+.2f} b {b:+.2f}")
    return w


# --------------------------------------------------------------- pass A
def analyse(job):
    return _guard(_analyse, job, job[0], job[1], "pass A")


def _analyse(job):
    i, stem, sci, rate, g, p_fixed = job
    cfg = {**_W["cfg"], **_W["cfg"]["inputs"][g]}          # the exposure's group settings
    ref, img, rw = _W["ref"], _W["img"], _W["rw"]
    res = dict(i=i, stem=stem, sci=sci, rate=rate, g=g, skip=None, warnings=[], registered=False, fallback=False,
               nmatch=0, nstars=0, rms=np.nan, p=np.array([1.0, 0, 0, 0]), c=np.zeros(2), off=np.nan, n_off=0,
               Qb=np.zeros(0), MKb=np.zeros(0), gb=np.zeros(0, np.int8), Cb=np.zeros(0), Mb=np.zeros(0),
               cgb=np.zeros(0, np.int8), cands=[], corners=np.zeros((0, 2)), pane_ref_cover=0.0)
    exp = calib.read_exposure(sci, rate, ref)
    meta = exp.meta
    res["k"], res["meta"] = exp.k, meta
    res["skip"] = skip_reason(meta, exp.k, cfg)
    if res["skip"]:
        return res
    res["warnings"] = _exposure_warnings(meta, exp.wcs, cfg)
    Q, var, var_rn, dq = calib.calibrate(exp.R, exp.rdq, exp.err, exp.vrn, exp.k,
                                         meta["ngroups"], meta["tgroup"], ref)
    Q = np.where((dq & calib.EST_DROP) > 0, np.nan, Q)    # every pass-A estimator: no saturated / >= 40 kDN pixels
    p, c = res["p"], res["c"]
    cut, cx0, cy0, centre = _cutout(exp.wcs, rw, img, p, c)
    res["c"] = c = np.array(centre)
    if cut is not None:
        rx, ry = register.ref_stars(cut, cx0, cy0)
        if p_fixed is not None:
            res.update(p=np.asarray(p_fixed), fallback=True)
        elif cfg["register"]:
            r = register.register_frame(exp.S, exp.wcs, rw, rx, ry, c, cfg["search_px"], cfg["min_match"],
                                        cfg["min_match_frac"], cfg["max_rms"])
            res.update(p=r["p"], registered=r["ok"], nmatch=r["nmatch"], rms=r["rms"], nstars=r["nstars"])
        p = res["p"]
        if not np.array_equal(p, [1.0, 0, 0, 0]):        # re-cut around the registered position
            cut, cx0, cy0, _ = _cutout(exp.wcs, rw, img, p, c)

    def to_ref(X, Y):
        return geometry.det_to_ref(exp.wcs, rw, X, Y, p, c)

    corners = np.array([[calib.X_LO, 0], [calib.X_HI, 0], [calib.X_HI, 1023], [calib.X_LO, 1023]], float)
    res["corners"] = np.c_[to_ref(corners[:, 0], corners[:, 1])]
    if cut is None:
        res["warnings"].append("no overlap with the reference: no sky offset")
        return res
    res["off"], res["n_off"] = offsets.sky_offset(exp.S, to_ref, cut, cx0, cy0)
    if not np.isfinite(res["off"]):
        res["warnings"].append("no science sky offset (< 1000 imager pixels overlap the reference)")

    # pane overlap blocks (pre-RL Q vs MK, both at the exposure's cal level)
    X, Y = _W["XY"]
    reg = ref.region
    pane = reg > 0
    M = np.full(reg.shape, np.nan)
    px, py = to_ref(X[pane], Y[pane])
    M[pane] = _bilinear(cut, px - cx0, py - cy0)
    res["pane_ref_cover"] = float(np.mean(np.isfinite(M[pane & np.isfinite(Q)]))) if np.isfinite(Q).any() else 0.0
    MK = offsets.convolve_mk(M, reg, ref.mk_kernels)
    # cal level, as the R1 law: Q against MK + the exposure's sky offset (the 30-450 MJy/sr range is on that level)
    res["Qb"], res["MKb"], res["gb"] = offsets.blocks(Q, MK + res["off"], reg, ref.y0, ref.x0)

    # control: imager pixels at the pane footprint + 591 columns, identical estimator, no kernel
    cs = (ref.crop[0], slice(ref.crop[1].start + calib.CONTROL_DX, ref.crop[1].stop + calib.CONTROL_DX))
    Cc, Rc, dqc = exp.C[cs], exp.R[cs], exp.rdq[cs]
    okc = (np.isfinite(Cc) & ((dqc & 2) == 0) & (Rc * meta["ngroups"] * meta["tgroup"] < calib.FLUENCE_DN)
           & pane & (ref.badpix != 1))
    Mc = np.full(reg.shape, np.nan)
    yy, xx = np.nonzero(pane)
    px, py = to_ref(xx + ref.x0 + calib.CONTROL_DX + 0.0, yy + ref.y0 + 0.0)
    Mc[pane] = _bilinear(cut, px - cx0, py - cy0)
    res["Cb"], res["Mb"], res["cgb"] = offsets.blocks(Cc, Mc + res["off"], reg, ref.y0, ref.x0, okc)

    # pane stars for the geometry check (reported only): a full RL of the cal-level Q, the same input as pass B
    Qrl = psf.deconvolve(Q, reg, ref.kernels, cfg["rl_iter"])
    res["cands"] = register.pane_candidates(Qrl, ref, exp.wcs, rw, p, c, rx, ry)
    return res


# --------------------------------------------------------------- pass B
def _common_header(h, cfg, ref):
    """Recipe, point-source bias, validity range and de-smearing keywords of both the exposure and mosaic headers."""
    h["QPMVER"] = (__version__, "miri_qpm version")
    h["QPMRECIP"] = (ref.law["recipe"], "4QPM calibration recipe")
    h["QPMCSUM"] = (ref.checksum, "SHA256 of calibration manifest")
    h["QPMCALST"] = (ref.law.get("status", "supplied"), "calibration validation status")
    PSB = ref.law["point_source_bias"]
    for g in (1, 2, 3):
        h[f"QPMA{g}"] = (float(ref.a[g]), f"pane {g} law a")
        h[f"QPMB{g}"] = (float(ref.b[g]), f"pane {g} law b [MJy/sr-equivalent]")
    h["QPMPSB"] = (PSB["s_common"], "expected point-source bias, NOT applied")
    h["QPMPSBE"] = (PSB["s_common_sys"], "systematic uncertainty of QPMPSB")
    for g in (1, 2, 3):
        h[f"QPMPSB{g}"] = (PSB["s_pane"][str(g)], f"pane {g} point-source bias, NOT applied")
    h["QPMVMIN"] = (calib.VALID_RANGE[0], "reference range for cal-level flags [MJy/sr]")
    h["QPMVMAX"] = (calib.VALID_RANGE[1], "reference range for cal-level flags [MJy/sr]")
    h["QPMKERN"] = (ref.kernel_id, "Richardson-Lucy kernel set")
    h["QPMRLIT"] = (int(cfg["rl_iter"]), "Richardson-Lucy iterations")
    h["QPMTRIM"] = (int(cfg["edge_trim"]), "edge margin [px]")
    h["QPMTRIMM"] = (cfg["edge_mode"], "edge margin: flag (kept, DQ 16) or drop (NaN)")
    return h


def _exp_header(r, corr, cfg, ref):
    h = _common_header(fits.Header(), cfg, ref)
    h["QPMFORM"] = ("Q=(rate*k/FLAT-b)/a", "per pane; RL, then minus QPMSKY (+QPMAOFg)")
    h["QPMK"] = (float(r["k"]), "median(cal/rate), imager [MJy/sr/(DN/s)]")
    h["QPMSKY"] = (float(r["off_used"]), "science sky offset subtracted [MJy/sr]")
    h["QPMSKYOK"] = (bool(np.isfinite(r["off"])), "sky offset measured against the reference")
    h["QPMANCM"] = ("L3g" if corr["offset"] else "none", "pane anchoring")
    for g in (1, 2, 3):
        h[f"QPMAOF{g}"] = (float(corr["offset"].get(g, 0.0)), f"pane {g} anchoring offset subtracted [MJy/sr]")
        h[f"QPMGAIN{g}"] = (float(corr["gain"][g]), f"pane {g} dataset gain divided out")
    for g in (1, 2, 3):
        h[f"QPMRLV{g}"] = (ref.law.get("rl_variance_factors", {}).get(str(g), psf.RL_VAR_FACTOR[g]), f"pane {g} RL variance factor applied to ERR^2")
    h["QPMFLUDN"] = (calib.FLUENCE_DN, "near-saturation cut rate*NGROUPS*TGROUP [DN]")
    h["QPMREG"] = (bool(r["registered"]), "registered to the reference (star voting)")
    h["QPMREGFB"] = (bool(r["fallback"]), "registration from its dither set (fallback)")
    h["QPMTX"], h["QPMTY"] = (float(r["p"][2]), "registration shift x [ref px]"), (float(r["p"][3]), "shift y")
    h["QPMRS"], h["QPMRROT"] = (float(r["p"][0]), "registration s cos(theta)"), (float(r["p"][1]), "s sin(theta)")
    h["QPMRCX"], h["QPMRCY"] = (float(r["c"][0]), "registration centre x [ref px]"), (float(r["c"][1]), "centre y")
    h["CROPX0"] = (ref.x0, "0-based detector column of pixel 0")
    h["CROPY0"] = (ref.y0, "0-based detector row of pixel 0")
    h["QPMSCI"] = (Path(r["sci"]).name, "science input")
    h["QPMRATE"] = (Path(r["rate"]).name, "rate input")
    h["QPMGROUP"] = (cfg["inputs"][r["g"]]["name"][:68], "input group")
    h["QPMNWARN"] = (len(r["warnings"]), "validity warnings (COMMENT cards)")
    for w in r["warnings"]:
        h["COMMENT"] = "WARNING: " + w.replace(" | ", ": ")
    h["COMMENT"] = "4QPM panes only, detector frame crop. A pane pixel (x, y) sees the sky the cal WCS assigns"
    h["COMMENT"] = "to (x + dx, y + dy): dx, dy = static pane geometry (miri_qpm data/geometry_F770W.json)."
    h["COMMENT"] = "Point sources read QPMPSB of their true flux relative to extended emission: NOT corrected."
    return h


def desmear_and_subtract(Q, var, var_rn, reg, kernels, n_iter, gain, level, variance_factors=None):
    """Per pane g: divide by gain[g], de-smear, then subtract level[g] (sky offset + anchoring offset).

    RL is multiplicative and floored, so its result depends on the additive level: it must see the cal-level image
    (as validated in the PSF study), never one with the sky removed.  A gain commutes with RL."""
    Q, var, var_rn = Q.copy(), var.copy(), var_rn.copy()
    for g in (1, 2, 3):
        m = reg == g
        Q[m] /= gain[g]
        var[m] /= gain[g] ** 2
        var_rn[m] /= gain[g] ** 2
    Q = psf.deconvolve(Q, reg, kernels, n_iter)
    for g in (1, 2, 3):
        Q[reg == g] -= level[g]
    return Q, psf.scale_variance(var, reg, variance_factors), psf.scale_variance(var_rn, reg, variance_factors)


def finalize_exposure(job):
    r = job[0]
    out = _guard(_finalize_exposure, job, r["i"], r["stem"], "pass B")
    return out if isinstance(out, tuple) else (r["i"], None, None, None, out["skip"])


def _finalize_exposure(job):
    r, corr, grid, out_dir = job
    cfg, ref, rw = _W["cfg"], _W["ref"], _W["rw"]
    exp = calib.read_exposure(r["sci"], r["rate"], ref)
    meta = exp.meta
    Q, var, var_rn, dq = calib.calibrate(exp.R, exp.rdq, exp.err, exp.vrn, exp.k,
                                         meta["ngroups"], meta["tgroup"], ref)
    reg = ref.region
    level = {g: r["off_used"] + corr["offset"].get(g, 0.0) for g in (1, 2, 3)}
    Q, var, var_rn = desmear_and_subtract(Q, var, var_rn, reg, ref.kernels, cfg["rl_iter"], corr["gain"], level,
                                              ref.law.get("rl_variance_factors"))
    # QA excludes saturated / >= 40 kDN pixels and their chessboard margin.
    est = (dq & calib.EST_DROP) > 0
    est |= psf.edge_margin(np.where(est, np.nan, Q), reg, cfg["edge_trim"], round_=False)
    edge = psf.edge_margin(Q, reg, cfg["edge_trim"])
    dq[edge] |= calib.DQ_EDGE
    if cfg["edge_mode"] == "drop":
        Q[edge] = np.nan
    fin = np.isfinite(Q)
    cal = Q + r["off_used"]            # cal level (anchoring is a correction to it, so it is not added back)
    above, below = fin & (cal > calib.VALID_RANGE[1]), fin & (cal < calib.VALID_RANGE[0])
    dq[above] |= calib.DQ_ABOVE
    dq[below] |= calib.DQ_BELOW
    dq[~fin] |= calib.DQ_DO_NOT_USE
    var[~fin] = np.nan
    satf = fin & ((dq & calib.EST_DROP) > 0)
    stats = dict(n_pane_valid={g: int((fin & (reg == g)).sum()) for g in (1, 2, 3)},
                 n_edge={g: int((edge & (reg == g)).sum()) for g in (1, 2, 3)},
                 n_satflu={g: int((satf & (reg == g)).sum()) for g in (1, 2, 3)},
                 n_above={g: int((above & (reg == g)).sum()) for g in (1, 2, 3)})
    if cfg["write_exposures"]:
        h = _exp_header(r, corr, cfg, ref)
        ed = Path(out_dir) / "exposures"
        ed.mkdir(parents=True, exist_ok=True)
        fits.HDUList([fits.PrimaryHDU(header=h),
                      fits.ImageHDU(Q.astype(np.float32), name="SCI", header=fits.Header([("BUNIT", "MJy/sr")])),
                      fits.ImageHDU(np.sqrt(var).astype(np.float32), name="ERR",
                                    header=fits.Header([("BUNIT", "MJy/sr")])),
                      fits.ImageHDU(dq, name="DQ")]).writeto(ed / f"{r['stem']}_qpm.fits", overwrite=True)
    # end-to-end QA: final pane pixels (cal level) against the unconvolved reference (+ sky offset)
    X, Y = _W["XY"]
    post = None
    cut, cx0, cy0, _ = _cutout(exp.wcs, rw, _W["img"], r["p"], r["c"])
    if cut is not None and np.isfinite(r["off"]):
        pane = reg > 0
        M = np.full(reg.shape, np.nan)
        px, py = geometry.det_to_ref(exp.wcs, rw, X[pane], Y[pane], r["p"], r["c"])
        M[pane] = _bilinear(cut, px - cx0, py - cy0)
        post = offsets.blocks(cal, M + r["off"], reg, ref.y0, ref.x0, valid=~est)
    # drizzle onto the local cutout of the output grid
    gx0, gy0, shape = grid
    px, py = geometry.det_to_ref(exp.wcs, rw, X.ravel(), Y.ravel(), r["p"], r["c"])
    pixmap = np.dstack([px.reshape(X.shape) - gx0, py.reshape(X.shape) - gy0])
    w = np.where(fin & (var_rn > 0), 1.0 / var_rn, 0.0)
    flagimgs = {mosaic.FLAG_EDGE: edge & fin,           # empty when edge_mode = 'drop'
                mosaic.FLAG_ABOVE: above, mosaic.FLAG_BELOW: below,
                mosaic.FLAG_SAT: fin & ((dq & calib.DQ_SATURATED) > 0),
                mosaic.FLAG_FLUENCE: fin & ((dq & calib.DQ_FLUENCE) > 0)}
    cut = mosaic.drizzle_local(Q, var, flagimgs, w, pixmap, shape)
    return r["i"], cut, stats, post, None


# --------------------------------------------------------------- main
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def resolve_settings(cfg, mode):
    """DEFAULTS < cfg, validated.  The rate directory and the reference are opened here, in the parent: a worker
    that fails at start-up would otherwise make the Pool respawn workers forever."""
    cfg = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    if mode not in ("10678", "external"):
        raise SystemExit("mode must be '10678' or 'external'")
    for k in ("ref", "out_dir"):
        if not cfg.get(k):
            raise SystemExit(f"missing required setting '{k}'")
    # input groups: [[inputs]] tables, or one group from the top-level sci / rate_dir
    if cfg["inputs"]:
        if cfg.get("sci") or cfg.get("rate_dir"):
            raise SystemExit("give either [[inputs]] tables or top-level sci / rate_dir, not both")
        raw = cfg["inputs"]
    else:
        raw = [dict(name=cfg["name"], sci=cfg["sci"], rate_dir=cfg["rate_dir"])]
    groups = []
    for gi in raw:
        bad = sorted(set(gi) - set(GROUP_KEYS))
        if bad:
            raise SystemExit(f"unknown keys in input group {gi.get('name')!r}: {bad} (allowed: {list(GROUP_KEYS)})")
        g = {**{k: cfg[k] for k in EXPOSURE_KEYS}, **gi}
        for k in ("name", "sci", "rate_dir"):
            if not g.get(k):
                raise SystemExit(f"missing required setting '{k}'" + (f" in input group {gi.get('name')!r}"
                                                                      if cfg["inputs"] else ""))
        if g["register"] is None:
            g["register"] = mode == "external"
        if isinstance(g["gc_like"], str) and g["gc_like"] != "auto":
            g["gc_like"] = g["gc_like"].lower() in ("1", "true", "yes", "on")
        if not Path(g["rate_dir"]).is_dir():
            raise SystemExit(f"rate_dir is not a directory: {g['rate_dir']}")
        groups.append(g)
    if len({g["name"] for g in groups}) != len(groups):
        raise SystemExit("input group names must be unique")
    if len(groups) > 1 and cfg["dataset_gain"] != "off":
        raise SystemExit("the dataset gain is one gain per dataset: not available with several input groups")
    cfg["inputs"] = groups
    cfg["register"] = any(g["register"] for g in groups)
    if cfg["anchor"] not in ("off", "on"):
        raise SystemExit("anchor must be 'off' or 'on'")
    if cfg["dataset_gain"] not in ("off", "auto"):
        raise SystemExit("dataset_gain must be 'off' or 'auto'")
    if mode == "10678" and cfg["dataset_gain"] != "off":
        raise SystemExit("the dataset gain is not available in the 10678 pipeline (R1 is trained on 10678)")
    if cfg["edge_mode"] not in ("drop", "flag"):
        raise SystemExit("edge_mode must be 'drop' or 'flag'")
    if cfg["kernels"] not in ("production", "refit10678"):
        raise SystemExit("kernels must be 'production' or 'refit10678'")
    try:
        with fits.open(cfg["ref"], memmap=True) as h:
            hd = mosaic.ref_hdu(h)
            rw = WCS(hd.header)
            shape = hd.shape
    except Exception as e:
        raise SystemExit(f"cannot read the reference mosaic {cfg['ref']}: {e!r}")
    if len(shape) != 2 or not rw.has_celestial:
        raise SystemExit(f"the reference mosaic {cfg['ref']} has no 2-D celestial image")
    if rw.has_distortion:
        raise SystemExit(f"the reference mosaic {cfg['ref']} has SIP / lookup-table distortion: use a TAN mosaic")
    if cfg["science_mask"]:
        try:
            with fits.open(cfg["science_mask"], memmap=True) as h:
                mshape = mosaic.ref_hdu(h).shape
        except Exception as e:
            raise SystemExit(f"cannot read the science mask {cfg['science_mask']}: {e!r}")
        if tuple(mshape) != tuple(shape):
            raise SystemExit(f"the science mask {cfg['science_mask']} is {mshape}, the reference {shape}: "
                             "it must be on the reference grid")
    ref = calib.refdata("F770W", cfg["kernels"], cfg.get("calibration_dir"))
    if "rl_iterations" in ref.law and cfg["rl_iter"] != ref.law["rl_iterations"]:
        raise ValueError("RL iterations must match the selected calibration noise model")
    if cfg["n_proc"] < 1 or cfg["rl_iter"] < 0 or cfg["grid"] not in ("full", "footprint"):
        raise ValueError("Invalid process count, RL iterations, or output grid")
    return cfg


def run(cfg, mode):
    """Run the pipeline (mode '10678' or 'external'); returns the report dict."""
    added = [v for v in THREAD_VARS if v not in os.environ]
    os.environ.update(dict.fromkeys(added, "1"))      # one BLAS thread in the spawned workers (they inherit it)
    try:
        with warnings.catch_warnings():
            _quiet()
            return _run(cfg, mode)
    finally:
        for v in added:
            os.environ.pop(v, None)


def _run(cfg, mode):
    t0 = time.time()
    cfg = resolve_settings(cfg, mode)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    groups = cfg["inputs"]
    rows, notes = merge_groups(groups)
    if not rows:
        raise SystemExit("no F770W FULL science frames with matching rate files")
    gof = [row[3] for row in rows]          # input group of exposure i
    if mode == "10678" and len(groups) == 1:
        notes += [f"{s}: not a GO 10678 exposure" for s, *_ in rows if not s.startswith("jw10678")]
    counts = ", ".join(f"{gi['name']}: {gof.count(g)}" for g, gi in enumerate(groups))
    log(f"miri-qpm {mode}: {len(rows)} exposures in {len(groups)} input group(s) ({counts}), "
        f"reference {cfg['ref']}, science_mask={cfg['science_mask']}, anchor={cfg['anchor']}, "
        f"dataset_gain={cfg['dataset_gain']}, kernels={cfg['kernels']}")
    ctx = get_context("spawn")
    with ctx.Pool(min(cfg["n_proc"], len(rows)), initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(analyse, [(i, *row, None) for i, row in enumerate(rows)], chunksize=1)
        res.sort(key=lambda r: r["i"])
        skipped = [r for r in res if r["skip"]]
        res = [r for r in res if not r["skip"]]
        for r in skipped:
            log(f"SKIP {r['stem']}: {r['skip']}")
        regd = [r for r in res if groups[gof[r["i"]]]["register"]]
        if regd:
            ok = [r for r in regd if r["registered"]]
            jobs = []
            for r in regd:
                if r["registered"]:
                    continue
                # shortcut: the median similarity of other frames (about their own centres) of the same input
                # group, as transfer_06
                pf = fallback_similarity(r, ok, gof)
                if pf is not None:
                    jobs.append((r["i"], r["stem"], r["sci"], r["rate"], gof[r["i"]], pf))
                else:
                    r["warnings"].append("registration failed: cal WCS used")
            redo = {q["i"]: q for q in pool.map(analyse, jobs, chunksize=1)} if jobs else {}
            for q in redo.values():
                if q["skip"]:
                    log(f"SKIP {q['stem']}: {q['skip']}")
                else:
                    q["warnings"].append("registration failed: dither-set median used")
            res = [redo.get(r["i"], r) for r in res]
            skipped += [r for r in res if r["skip"]]
            res = [r for r in res if not r["skip"]]
            log(f"registered {len(ok)}/{len(regd)}; fallback {len(jobs)}")
        if not res:
            raise SystemExit("every exposure was skipped")
        for r in res:
            r["off_used"] = r["off"] if np.isfinite(r["off"]) else 0.0
        offs = np.array([r["off"] for r in res])
        log(f"sky offsets {np.nanmin(offs):+.2f}..{np.nanmax(offs):+.2f} MJy/sr "
            f"({int(np.isfinite(offs).sum())}/{len(res)} measured)")

        # optional overlap terms
        gain, grep = {p: 1.0 for p in (1, 2, 3)}, dict(adopted=False, reason="off")
        grp, unit = offsets.groups_of([r["stem"] for r in res])
        if cfg["dataset_gain"] == "auto":
            bg = np.concatenate([np.full(len(r["Qb"]), grp[j]) for j, r in enumerate(res)])
            gain, grep = offsets.dataset_gain(np.concatenate([r["Qb"] for r in res]),
                                              np.concatenate([r["MKb"] for r in res]),
                                              np.concatenate([r["gb"] for r in res]).astype(int), bg)
        log(f"dataset gain: {grep.get('reason')} {gain}")
        corr = []
        for r in res:
            anc = offsets.anchor_offsets(r["Qb"], r["MKb"], r["gb"], gain)
            r["anchor"] = anc
            off = {p: o for p, (n, o) in anc.items() if cfg["anchor"] == "on" and np.isfinite(o)}
            corr.append(dict(gain=gain, offset=off))

        # output grid
        ref_img, ref_err, ref_hdr = open_reference(cfg)
        pts = np.vstack([r["corners"] for r in res])
        whdr, shape, gx0, gy0 = mosaic.output_grid(ref_hdr, ref_img.shape, pts, cfg["grid"])
        gbytes = shape[0] * shape[1] * 14 / 1e9
        log(f"output grid {shape[1]} x {shape[0]} ({cfg['grid']}; reference shifted by {-gx0}, {-gy0}); "
            f"{gbytes:.2f} GB on disk")
        st = os.statvfs(out_dir)
        if st.f_bavail * st.f_frsize / 1e9 - gbytes < 8:
            raise SystemExit("not enough free disk space for the output mosaic (keeps >= 8 GB free)")
        out = out_dir / f"{name}_qpm_i2d.fits"
        part = out.with_name(out.name + ".partial")       # renamed to `out` only once finalized
        phdr = _mosaic_header(cfg, mode, res, corr, gain, grep)
        mm = mosaic.create(part, phdr, whdr, shape, comments=_ext_comments())
        expflags = []
        for r, c in zip(res, corr):
            f = mosaic.FLAG_LOCAL if (c["offset"] or any(v != 1.0 for v in c["gain"].values())) else 0
            f |= mosaic.FLAG_WARN if r["warnings"] else 0
            f |= mosaic.FLAG_NOSKY if not np.isfinite(r["off"]) else 0
            expflags.append(f)
        stats = {}
        jobs = [(r, c, (gx0, gy0, shape), str(out_dir)) for r, c in zip(res, corr)]
        byi = {r["i"]: j for j, r in enumerate(res)}
        post, failed = {}, {}
        for n, (i, cut, s, pb, err) in enumerate(pool.imap_unordered(finalize_exposure, jobs, chunksize=1)):
            if err:
                failed[i] = err
                log(f"SKIP {res[byi[i]]['stem']}: {err}")
                continue
            stats[i] = s
            if pb is not None:
                post[i] = pb
            if cut is not None:
                mosaic.accumulate(mm, cut, expflags[byi[i]])
            if n % 25 == 0 or n == len(jobs) - 1:
                log(f"pass B {n + 1}/{len(jobs)}")
    if failed:
        for r in res:
            if r["i"] in failed:
                r["skip"] = failed[r["i"]]
                skipped.append(r)
        keep = [j for j, r in enumerate(res) if r["i"] not in failed]
        res, corr = [res[j] for j in keep], [corr[j] for j in keep]
    mosaic.finalize(mm, ref_img, ref_err, gx0, gy0, fill_bright_holes=bool(cfg["fill_bright_holes"]))
    step = max(1, int(np.ceil(np.sqrt(shape[0] * shape[1] / 4e7))))
    pix = qa.pixel_stats(mm, step)
    ident = qa.science_identity(mm, ref_img, gx0, gy0)
    log(f"science layer bit-identical to the reference: {ident['bit_identical']} "
        f"({ident['n_reference_finite']} px; {ident['n_not_identical']} differ)")
    if cfg["quicklook"]:
        qa.quicklook(mm, out_dir / f"{name}_qpm_quicklook.png", title=f"{name} ({mode})")
    del mm
    if failed:      # the header was written before pass B; same card, so it is rewritten in place
        fits.setval(part, "QPMNEXP", value=len(res))
    os.replace(part, out)

    # tables and report
    ref = calib.refdata("F770W", cfg["kernels"], cfg.get("calibration_dir"))
    erows = []
    for r, c in zip(res, corr):
        m = r["meta"]
        for p in (1, 2, 3):
            n, o = r["anchor"][p]
            erows.append((r["stem"], groups[gof[r["i"]]]["name"], p, m["ngroups"], m["nints"], m["mjd"], round(m["epoch_offset_yr"], 3),
                          m["glon"], m["glat"], m["imager_satfrac"], m["outlier_frac"], m["s_bkdsub"], r["k"],
                          r["off"], r["n_off"], r["registered"], r["fallback"], r["nmatch"], r["rms"],
                          r["p"][2], r["p"][3], r["pane_ref_cover"], n, o, p in c["offset"], c["gain"][p],
                          stats[r["i"]]["n_pane_valid"][p], stats[r["i"]]["n_edge"][p], stats[r["i"]]["n_satflu"][p],
                          stats[r["i"]]["n_above"][p],
                          qa.range_bias(r["Qb"], r["MKb"], r["gb"], p),
                          qa.range_bias(*post[r["i"]], p) if r["i"] in post else np.nan,
                          "; ".join(w.replace(" | ", ": ") for w in r["warnings"])))
    et = Table(rows=erows, names=("stem", "group", "pane", "ngroups", "nints", "mjd", "epoch_offset_yr", "glon", "glat",
                                  "imager_satfrac", "outlier_frac", "s_bkdsub", "k", "sky_offset", "sky_offset_npix",
                                  "registered", "registration_fallback", "nmatch", "reg_rms_px", "reg_tx_px",
                                  "reg_ty_px", "pane_ref_cover", "n_anchor_blocks", "anchor_offset", "anchored",
                                  "dataset_gain", "n_pane_valid", "n_edge_margin", "n_sat_fluence", "n_above_450",
                                  "bias_30_450_pre_rl", "bias_30_450_post_rl", "warnings"))
    et.meta["comments"] = ["miri_qpm per exposure-pane table; anchor_offset is listed even when not applied "
                           "(anchored = False); sky_offset NaN = not measured (0 used); n_pane_valid = finite "
                           "pane pixels written; n_edge_margin / n_sat_fluence / n_above_450 = of those, in the RL "
                           "edge margin / rate SATURATED or >= 40000 DN (kept, flagged) / cal level (SCI + sky offset) "
                           "> 450 MJy/sr; group = input group",
                           "bias_30_450_pre_rl / _post_rl = this exposure-pane's median overlap bias (pane / reference "
                           "- 1, blocks with reference 30-450 MJy/sr, NaN if < 8 blocks) before de-smearing and local "
                           "terms / of the final pixels; an outlier flags a misplaced or bad exposure"]
    et.write(out_dir / f"{name}_qpm_exposures.ecsv", overwrite=True)
    def _wcount(rr):
        wc = {}
        for r in rr:
            for w in r["warnings"]:
                key = w.split(" | ")[0]
                wc[key] = wc.get(key, 0) + 1
        return wc

    wcount = _wcount(res)
    per_group = {}
    for g, gi in enumerate(groups):
        sel = [j for j, r in enumerate(res) if gof[r["i"]] == g]
        rr, cc = [res[j] for j in sel], [corr[j] for j in sel]
        go = np.array([r["off"] for r in rr], float)
        per_group[gi["name"]] = dict(
            settings={k: gi[k] for k in GROUP_KEYS}, n_exposures=len(rr),
            skipped=[r["stem"] for r in skipped if gof[r["i"]] == g],
            registration=dict(enabled=gi["register"], registered=int(sum(r["registered"] for r in rr)),
                              fallback=int(sum(r["fallback"] for r in rr)),
                              rms_px_median=float(np.nanmedian([r["rms"] for r in rr])) if gi["register"] and
                              np.isfinite([r["rms"] for r in rr]).any() else None),
            sky_offset=dict(min=float(np.nanmin(go)) if np.isfinite(go).any() else None,
                            max=float(np.nanmax(go)) if np.isfinite(go).any() else None,
                            measured=int(np.isfinite(go).sum())),
            overlap=qa.overlap_summary(rr, cc, [post[r["i"]] for r in rr if r["i"] in post]) if rr else None,
            warnings=_wcount(rr))
    rep = dict(
        miri_qpm_version=__version__, mode=mode, settings=cfg, n_exposures=len(res),
        skipped={r["stem"]: r["skip"] for r in skipped}, input_notes=notes,
        recipe=dict(id=ref.law["recipe"], checksum=ref.checksum, directory=ref.directory, status=ref.law.get("status", "supplied"), formula=ref.law["formula"], a=ref.a[1:].tolist(), b=ref.b[1:].tolist(),
                    a_sd=[ref.law["panes"][g]["a_sd"] for g in "123"],
                    b_sd=[ref.law["panes"][g]["b_sd"] for g in "123"], kernels=ref.kernel_id),
        point_source_bias_not_applied=ref.law["point_source_bias"],
        registration=dict(enabled=cfg["register"], registered=int(sum(r["registered"] for r in res)),
                          fallback=int(sum(r["fallback"] for r in res))),
        sky_offset=dict(min=float(np.nanmin(offs)), max=float(np.nanmax(offs)),
                        measured=int(np.isfinite(offs).sum())),
        anchoring=dict(mode=cfg["anchor"],
                       applied_fraction={p: float(np.mean([p in c["offset"] for c in corr])) for p in (1, 2, 3)},
                       available_fraction={p: float(np.mean([np.isfinite(r["anchor"][p][1]) for r in res]))
                                           for p in (1, 2, 3)}),
        dataset_gain=dict(grep, gain_applied=gain, group_unit=unit, n_groups=int(len(set(grp)))),
        overlap=qa.overlap_summary(res, corr, list(post.values())), groups=per_group,
        geometry_check=qa.geometry_check(res, ref.models, ref.region, ref.x0, ref.y0),
        pixels=pix, science_identity=ident, output_grid=dict(shape=shape, x0=gx0, y0=gy0, mode=cfg["grid"]),
        warnings=wcount, runtime_s=round(time.time() - t0, 1),
        files=dict(mosaic=out.name, exposures_table=f"{name}_qpm_exposures.ecsv",
                   exposure_files="exposures/<stem>_qpm.fits" if cfg["write_exposures"] else None,
                   quicklook=f"{name}_qpm_quicklook.png" if cfg["quicklook"] else None),
    )
    json.dump(_jsonable(rep), open(out_dir / f"{name}_qpm_report.json", "w"), indent=1)
    ov, po = rep["overlap"]["range_30_450"], rep["overlap"].get("post_rl_30_450", {})
    log("overlap bias 30-450 MJy/sr (pre-RL pane / control / final pane): " +
        "; ".join(f"p{p} {_pct(ov[p]['bias'])} / {_pct(ov[p]['control'])} / "
                  f"{_pct(po.get(p, {}).get('bias'))} (n={ov[p]['n_blocks']})" for p in ov))
    if len(groups) > 1:
        for gname, gr in per_group.items():
            if gr["overlap"]:
                o = gr["overlap"]["range_30_450"]
                log(f"  group {gname} ({gr['n_exposures']} exp): pre-RL 30-450 " +
                    " ".join(f"p{p} {_pct(o[p]['bias'])}" for p in o))
    for w, n in wcount.items():
        log(f"WARNING x{n}: {w}")
    log(f"wrote {out} ({time.time() - t0:.0f} s)")
    return rep


def _pct(v):
    return "n/a" if v is None else f"{100 * v:+.2f}%"


def _ext_comments():
    return {
        "SCI": ["Reference science values copied unchanged where finite (SOURCE 1); elsewhere the drizzled,",
                "de-smeared 4QPM pane data (SOURCE 2-3).  Pane point sources read ~0.942 of their flux: NOT",
                "corrected (QPMPSB).  Reference holes (saturated cores) are filled with pane data (SOURCE 3).",
                "Pane values outside 30-450 MJy/sr are unvalidated (FLAGS 1/8).  Above 450 MJy/sr the panes",
                "read ~1 % (GO 10678) to 2-4 % (external GC) below the reference; above ~620 MJy/sr there is no",
                "unsaturated imager reference, so filled saturated cores (SOURCE 3, up to ~4000 MJy/sr) are untested.",
                "Pane pixels are blank only where the rate has no usable slope (DO_NOT_USE), as in STScI products;",
                "rate-SATURATED and >= 40000 DN pixels and the 3-px RL edge margin are kept and flagged (64/128/2)."],
        "ERR": ["SOURCE 1: the reference ERR if it has one, else NaN.  SOURCE 2-3: drizzled pane variance",
                "(rate ERR, flat error, RL noise factor), combined with the drizzle weights; it ignores the",
                "correlated noise of RL (anti-correlated along x) and drizzle, and the calibration systematics."],
        "WHT": ["Pane drizzle weight: sum of (1 / read-noise variance) x overlap area; 0 = no pane value in SCI.",
                "Science pixels keep their weight in the reference product."],
        "SOURCE": ["0 none, 1 science (reference), 2 pane only, 3 pane inside the science footprint (a hole",
                   "of the reference), 4 hole with pane data > 450 MJy/sr: left NaN as in the reference"],
        "FLAGS": ["pane pixels (SOURCE 2-4) only: 1 > 450 MJy/sr (unvalidated); 2 RL edge margin (2-8 % deficits",
                  "possible within 3 px of pane edges / blank cores); 4 local terms (anchoring / dataset gain) applied;",
                  "8 < 30 MJy/sr (unvalidated); 16 contributing exposure has a validity warning (NGROUPS, saturation,",
                  "outliers, non-GC, registration; not epoch); 32 contributing exposure has no science sky offset;",
                  "64 rate SATURATED (slope from the unsaturated groups); 128 >= 40000 DN ramp (64/128 unvalidated)"],
    }


def _mosaic_header(cfg, mode, res, corr, gain, grep):
    ref = calib.refdata("F770W", cfg["kernels"], cfg.get("calibration_dir"))
    h = fits.Header([("FILTER", "F770W"), ("BUNIT", "MJy/sr")])
    _common_header(h, cfg, ref)
    h["QPMMODE"] = (mode, "miri-qpm pipeline")
    h["QPMREFIM"] = (Path(cfg["ref"]).name[:68], "reference mosaic (science layer)")
    h["QPMNEXP"] = (len(res), "exposures contributing panes")
    h["QPMANCM"] = ("L3g" if cfg["anchor"] == "on" else "none", "per exposure-pane anchoring")
    for g in (1, 2, 3):
        h[f"QPMANC{g}"] = (round(float(np.mean([g in c["offset"] for c in corr])), 4),
                           f"pane {g}: fraction of exposures anchored")
        h[f"QPMGAIN{g}"] = (float(gain[g]), f"pane {g} dataset gain applied")
    h["QPMGMOD"] = ("Lg" if grep.get("adopted") else "none", "dataset gain")
    h["QPMREG"] = (bool(cfg["register"]), "frames registered by star voting (any group)")
    h["QPMSMASK"] = (Path(cfg["science_mask"]).name[:68] if cfg["science_mask"] else "none",
                     "science mask on the reference grid")
    h["QPMNGRP"] = (len(cfg["inputs"]), "input groups")
    for g, gi in enumerate(cfg["inputs"][:99]):
        h[f"QPMGRP{g}"] = (gi["name"][:50], f"input group {g}, register {bool(gi['register'])}")
    h["QPMFILLB"] = (bool(cfg["fill_bright_holes"]), "fill holes w/ pane >450 MJy/sr (F: SOURCE 4)")
    nw = sum(bool(r["warnings"]) for r in res)
    h["QPMNWARN"] = (nw, "exposures with validity warnings (see report)")
    h["COMMENT"] = "F770W mosaic: the reference science layer, unchanged, plus the MIRI 4QPM coronagraph panes"
    h["COMMENT"] = "(frozen GO 10678 calibration R1, de-smeared) where the reference has no data. See the"
    h["COMMENT"] = "_qpm_report.json and docs/MIRI_QPM.md for accuracy, validity and limitations."
    return h
