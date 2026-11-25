"""Fit a reviewable F770W calibration from local exposures and reference images.

The estimators are pane-normalised alternating least squares for the flat,
relative clipped affine response fits on 8x8 blocks, non-negative PSF matching
on isolated-star stacks, and offset-voting affine pane geometry. Training and
validation fields are explicit. Calibration candidates never replace a bundle.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.optimize import nnls
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from . import calib, geometry, mosaic, offsets, pipeline, psf, register
from .workflow import sha256, write_json


class InsufficientCalibrationData(ValueError):
    """The supplied data cannot constrain every requested calibration component."""


def robust_line(x, y, relative=True, clip=3.5):
    """Fit y = a*x + b, with relative residuals and six MAD-clipping passes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    for _ in range(6):
        if good.sum() < 10 or np.ptp(x[good]) <= 1e-6 * np.median(x[good]):
            raise InsufficientCalibrationData("Response fit needs at least 10 blocks spanning brightness")
        weight = 1 / x[good] if relative else np.ones(good.sum())
        A = np.c_[x[good], np.ones(good.sum())] * weight[:, None]
        a, b = np.linalg.lstsq(A, y[good] * weight, rcond=None)[0]
        residual = (y - a*x - b) / (x if relative else 1)
        center = np.median(residual[good])
        scatter = 1.4826 * np.median(np.abs(residual[good] - center))
        keep = good & (np.abs(residual - center) <= clip * max(scatter, 1e-12))
        if np.array_equal(keep, good):
            return float(a), float(b), good
        good = keep
    return float(a), float(b), good


def fit_flat(samples, region, min_n=3, n_iter=4):
    """ALS: Y_ij = F_i (a_j MK_ij + b_j), one affine per exposure and pane."""
    pix, exp, Y, M = (np.asarray(samples[k]) for k in ("pix", "exp", "Y", "M"))
    reg = region.ravel()
    size = reg.size
    pane = reg[pix]

    def ratio(values, predictions, use):
        count = np.bincount(pix[use], minlength=size)
        den = np.bincount(pix[use], weights=predictions[use], minlength=size)
        num = np.bincount(pix[use], weights=values[use], minlength=size)
        flat = np.full(size, np.nan)
        ok = (count >= min_n) & (den > 0)
        flat[ok] = num[ok] / den[ok]
        for g in (1, 2, 3):
            m = (reg == g) & np.isfinite(flat) & (flat > 0)
            if m.sum() < 100:
                raise InsufficientCalibrationData(f"Pane {g}: insufficient flat overlap")
            flat[reg == g] /= np.median(flat[m])
        return flat, count

    use = np.isfinite(Y) & np.isfinite(M) & (M > 0)
    F, count = ratio(Y, M, use)
    key = exp.astype(np.int64) * 4 + pane
    order = np.argsort(key, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(key[order])) + 1)
    for _ in range(n_iter):
        Q = Y / F[pix]
        pred = np.full(len(Y), np.nan)
        for idx in groups:
            good = np.isfinite(Q[idx]) & np.isfinite(M[idx])
            if good.sum() < 200:
                continue
            try:
                a, b, _ = robust_line(M[idx], Q[idx])
            except InsufficientCalibrationData:
                continue
            pred[idx] = a * M[idx] + b
        residual = Y / (F[pix] * pred) - 1
        use = np.isfinite(residual) & (pred > 0)
        for g in (1, 2, 3):
            m = use & (pane == g)
            if not m.any():
                raise InsufficientCalibrationData(f"Pane {g}: no constrained exposure response")
            center = np.median(residual[m])
            sigma = 1.4826 * np.median(np.abs(residual[m] - center))
            use[m] &= np.abs(residual[m] - center) <= 5 * max(sigma, 1e-12)
        Fnew, count = ratio(Y, pred, use)
        change = np.nanmax(np.abs(Fnew / F - 1))
        F = Fnew
        if change < 1e-4:
            break
    # The pane median supplies pixels with insufficient overlap; NSAMP records them.
    for g in (1, 2, 3):
        m = reg == g
        F[m & ~np.isfinite(F)] = np.nanmedian(F[m])
    return F.reshape(region.shape), count.reshape(region.shape)


def block_samples(samples, flat, ref):
    pix, exp = samples["pix"], samples["exp"]
    yy, xx = np.unravel_index(pix, ref.region.shape)
    g = ref.region.ravel()[pix]
    key = (((exp.astype(np.int64)*4 + g) * 130 + (yy+ref.y0)//8)*130 + (xx+ref.x0)//8)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    first = np.unique(inv, return_index=True)[1]
    Y = samples["Y"] / flat.ravel()[pix]
    M = samples["M"]
    good = cnt >= 32
    return dict(Y=(np.bincount(inv, weights=Y)/cnt)[good],
                M=(np.bincount(inv, weights=M)/cnt)[good], g=g[first][good],
                field=samples["field"][first][good], exp=exp[first][good])


def fit_law(blocks, nboot=400, seed=5):
    rng = np.random.default_rng(seed)
    result = {}
    for g in (1, 2, 3):
        take = blocks["g"] == g
        X, Y, field = (blocks[k][take] for k in ("M", "Y", "field"))
        a, b, good = robust_line(X, Y)
        if a <= 0:
            raise ValueError(f"Pane {g}: non-positive response")
        fields = np.unique(field)
        if len(fields) < 2:
            raise InsufficientCalibrationData(f"Pane {g}: response needs overlap in two training fields")
        indices = {f: np.flatnonzero(field == f) for f in fields}
        boot = []
        for _ in range(nboot):
            idx = np.concatenate([indices[f] for f in rng.choice(fields, len(fields))])
            aa, bb, _ = robust_line(X[idx], Y[idx])
            boot.append((aa, bb))
        boot = np.asarray(boot)
        result[str(g)] = dict(a=a, b=b, a_sd=float(boot[:, 0].std()), b_sd=float(boot[:, 1].std()),
                             a_ci68=np.percentile(boot[:, 0], [16, 84]).tolist(),
                             b_ci68=np.percentile(boot[:, 1], [16, 84]).tolist(),
                             n_blocks=int(len(X)), n_used=int(good.sum()), b_over_a=b/a,
                             mk_p5_p95=np.percentile(X, [5, 95]).tolist())
    return result


def fit_kernel(imager, pane):
    cols = []
    big = np.pad(imager, 10)
    for dy in range(-2, 3):
        for dx in range(-10, 11):
            cols.append(np.roll(np.roll(big, dy, 0), dx, 1)[10:-10, 10:-10].ravel())
    kernel, _ = nnls(np.asarray(cols).T, pane.ravel())
    if not np.isfinite(kernel).all() or kernel.sum() <= 0:
        raise InsufficientCalibrationData("PSF matching did not yield a positive kernel")
    return (kernel / kernel.sum()).reshape(5, 21)


def star_stamps(exp, ref):
    """Isolated unsaturated stellar stamps, centred on their measured peaks."""
    Q, _, _, dq = calib.calibrate(exp.R, exp.rdq, exp.err, exp.vrn, exp.k,
                                  exp.meta["ngroups"], exp.meta["tgroup"], ref)
    reg = calib.full_region(ref)
    im = exp.S.copy()
    fluence = exp.R * exp.meta["ngroups"] * exp.meta["tgroup"]
    im[((exp.rdq & 3) != 0) | (fluence >= 25000)] = np.nan
    Q[((dq & (calib.EST_DROP | calib.DQ_DO_NOT_USE)) != 0) | (fluence[ref.crop] >= 25000)] = np.nan
    all_stamps = {}
    for g in (0, 1, 2, 3):
        img = im if g == 0 else np.where(ref.region == g, Q, np.nan)
        region = (reg == 0) if g == 0 else (ref.region == g)
        filled = ndimage.binary_fill_holes(region)
        x, y = register.peaks(img, nmax=400, nsig=5)
        stack = []
        if len(x) < 2:
            all_stamps[g] = stack
            continue
        tree = cKDTree(np.c_[x, y])
        hp = img - ndimage.uniform_filter(np.where(np.isfinite(img), img, np.nanmedian(img)), 15)
        v = hp[np.rint(y).astype(int), np.rint(x).astype(int)]
        yy, xx = np.mgrid[-12:13, -12:13]
        annulus = (np.hypot(xx, yy) > 8) & (np.hypot(xx, yy) <= 12)
        for n, (xf, yf) in enumerate(zip(x[:60], y[:60])):
            xi, yi = round(float(xf)), round(float(yf))
            if xi < 15 or yi < 15 or xi >= img.shape[1]-15 or yi >= img.shape[0]-15:
                continue
            near = [j for j in tree.query_ball_point([xf, yf], 7) if j != n]
            if near and np.any(v[near] > 0.1*v[n]):
                continue
            box = (slice(yi-15, yi+16), slice(xi-15, xi+16))
            if not filled[box].all():
                continue
            stamp = img[box]
            bad = ~np.isfinite(stamp)
            if bad.mean() > 0.05 or bad[11:20, 11:20].sum() > 2:
                continue
            stamp = psf.fill_nan(stamp, ~bad, 3)
            stamp = ndimage.shift(stamp, (yi-yf, xi-xf), order=3, mode="nearest")[3:-3, 3:-3]
            stamp -= np.median(stamp[annulus])
            if stamp[12, 12] <= 0 or stamp[12, 12] < 0.9*stamp.max() or stamp.sum() <= 0:
                continue
            stack.append((stamp / stamp.sum()).astype(np.float32))
        all_stamps[g] = stack
    return all_stamps


def _field_state(field, ref, stack):
    handle = stack.enter_context(fits.open(field["reference"], memmap=True))
    hdu = mosaic.ref_hdu(handle)
    rw = WCS(hdu.header)
    if not rw.has_celestial or rw.has_distortion or len(hdu.shape) != 2:
        raise ValueError(f"{field['name']}: reference requires a plain celestial TAN mosaic")
    if not field.get("starless"):
        raise ValueError(f"{field['name']}: refitting requires starless on the reference grid for diffuse sample selection")
    star_hdu = mosaic.ref_hdu(stack.enter_context(fits.open(field["starless"], memmap=True)))
    sw = WCS(star_hdu.header)
    yy, xx = np.mgrid[0:hdu.shape[0]:complex(0, 5), 0:hdu.shape[1]:complex(0, 5)]
    px, py = sw.world_to_pixel_values(*rw.pixel_to_world_values(xx, yy))
    if star_hdu.shape != hdu.shape or not np.allclose(np.array([px, py]), [xx, yy], atol=1e-4, rtol=0):
        raise ValueError(f"{field['name']}: starless must match the reference WCS and shape")
    return hdu.data, star_hdu.data, rw


def exposure_context(row, ref, img, rw, settings):
    exp = calib.read_exposure(row["cal"], row["rate"], ref)
    reason = pipeline.skip_reason(exp.meta, exp.k, {"allow_bkdsub": False})
    if reason:
        raise ValueError(f"{row['stem']}: {reason}")
    p, c = np.array([1., 0, 0, 0]), np.zeros(2)
    cut, x0, y0, center = pipeline._cutout(exp.wcs, rw, img, p, c, margin=340)
    if cut is None:
        raise InsufficientCalibrationData(f"{row['stem']}: no reference overlap")
    c = np.array(center)
    rx, ry = register.ref_stars(cut, x0, y0)
    if settings.get("register", True):
        registration = register.register_frame(exp.S, exp.wcs, rw, rx, ry, c)
        if not registration["ok"]:
            raise InsufficientCalibrationData(f"{row['stem']}: registration failed; excluded calibration needs review")
        p = registration["p"]
    transform = lambda x, y: geometry.det_to_ref(exp.wcs, rw, x, y, p, c)
    off, n = offsets.sky_offset(exp.S, transform, cut, x0, y0)
    if not np.isfinite(off):
        raise InsufficientCalibrationData(f"{row['stem']}: cannot measure imager sky offset")
    return exp, p, c, cut, x0, y0, rx, ry, off


def _sample(image, px, py):
    """Read only the required reference window, preserving unsupported pixels."""
    result = np.full(px.shape, np.nan)
    good = np.isfinite(px) & np.isfinite(py)
    if not good.any():
        return result
    x0, x1 = max(0, int(np.floor(px[good].min()))-2), min(image.shape[1], int(np.ceil(px[good].max()))+3)
    y0, y1 = max(0, int(np.floor(py[good].min()))-2), min(image.shape[0], int(np.ceil(py[good].max()))+3)
    if x1 > x0 and y1 > y0:
        result[good] = pipeline._bilinear(np.asarray(image[y0:y1, x0:x1], float), px[good]-x0, py[good]-y0)
    return result


def sample_exposure(exp, ref, img, starless, rw, p, c, off, control=False):
    if control:
        Y, X = np.indices(ref.region.shape, dtype=float)
        X += ref.x0 + calib.CONTROL_DX
        Y += ref.y0
        crop = (ref.crop[0], slice(ref.x0+calib.CONTROL_DX, ref.x0+calib.CONTROL_DX+ref.region.shape[1]))
    else:
        X, Y = geometry.pane_positions(ref)
        crop = ref.crop
    px, py = geometry.det_to_ref(exp.wcs, rw, X, Y, p, c)
    M = _sample(img, px, py)
    stars = M - _sample(starless, px, py)
    M += off
    M[ref.region == 0] = np.nan
    MK = M if control else offsets.convolve_mk(M, ref.region, ref.kernels)
    gy, gx = np.gradient(M)
    grad = np.hypot(gx, gy)
    rate, dq = exp.R[crop], exp.rdq[crop]
    valid = ((ref.region > 0) & (ref.badpix != 1) & ((dq & 3) == 0)
             & np.isfinite(rate) & np.isfinite(MK) & (MK > 0)
             & np.isfinite(stars) & np.isfinite(grad)
             & (stars <= np.maximum(1, 0.01*MK)) & (0.15*grad <= 0.01*MK)
             & (rate * exp.meta["ngroups"] * exp.meta["tgroup"] < calib.FLUENCE_DN))
    values = exp.C[crop] if control else rate*exp.k
    valid &= np.isfinite(values)
    pix = np.flatnonzero(valid).astype(np.int32)
    return {"pix": pix, "Y": values.ravel()[pix].astype(np.float32),
            "M": MK.ravel()[pix].astype(np.float32)}


def _collect(fields, ref, cfg, directory, do_stamps=False, do_samples=False):
    directory.mkdir(parents=True, exist_ok=True)
    stamps = {g: [] for g in range(4)}
    sample_files, candidates, metadata = [], [], []
    ei = 0
    for fi, field in enumerate(fields):
        settings = {**cfg.get("qpm", {}), **field.get("qpm", {})}
        with ExitStack() as stack:
            img, starless, rw = _field_state(field, ref, stack)
            for row in field["exposures"]:
                print(f"refit {directory.name}: {field['name']} {row['stem']}", flush=True)
                exp, p, c, cut, x0, y0, rx, ry, off = exposure_context(row, ref, img, rw, settings)
                if do_stamps:
                    for g, values in star_stamps(exp, ref).items():
                        stamps[g].extend(values)
                if do_samples:
                    sample = sample_exposure(exp, ref, img, starless, rw, p, c, off)
                    n = len(sample["pix"])
                    sample.update(exp=np.full(n, ei, np.int32), field=np.full(n, fi, np.int32))
                    f = directory / f"{row['stem']}.npz"
                    np.savez(f, **sample)
                    sample_files.append(f)
                    control = sample_exposure(exp, ref, img, starless, rw, p, c, off, control=True)
                    control.update(exp=np.full(len(control["pix"]), ei, np.int32),
                                   field=np.full(len(control["pix"]), fi, np.int32))
                    np.savez(directory / f"{row['stem']}_control.npz", **control)
                else:
                    Q, _, _, dq = calib.calibrate(exp.R, exp.rdq, exp.err, exp.vrn, exp.k,
                                                 exp.meta["ngroups"], exp.meta["tgroup"], ref)
                    Q[(dq & calib.EST_DROP) != 0] = np.nan
                    Q = psf.deconvolve(Q, ref.region, ref.kernels)
                    candidates.extend((ei, *r) for r in register.pane_candidates(Q, ref, exp.wcs, rw, p, c, rx, ry))
                metadata.append({"field": field["name"], "stem": row["stem"], "visit": row["stem"][:13],
                                 "mjd": exp.meta["mjd"], "ngroups": exp.meta["ngroups"], "k": exp.k,
                                 "sky_offset": off, "registration": p.tolist()})
                ei += 1
    return stamps, np.asarray(candidates).reshape(-1, 6), sample_files, metadata


def _load_samples(files):
    parts = {k: [] for k in ("pix", "Y", "M", "exp", "field")}
    for filename in files:
        with np.load(filename) as z:
            for key in parts:
                parts[key].append(z[key])
    return {k: np.concatenate(v) for k, v in parts.items()}


def _subset(samples, mask):
    return {k: v[mask] for k, v in samples.items()}


def fit_psfs(stamps, min_stars=30):
    stacks, counts = {}, {}
    for g in range(4):
        counts[str(g)] = len(stamps[g])
        if len(stamps[g]) < min_stars:
            raise InsufficientCalibrationData(f"Region {g}: need {min_stars} isolated star stamps, got {len(stamps[g])}")
        stacks[g] = np.median(stamps[g], axis=0)
        stacks[g] /= stacks[g].sum()
    kernels = {g: fit_kernel(stacks[0], stacks[g]) for g in (1, 2, 3)}
    return kernels, stacks, counts


def _geometry(candidates):
    models, stats = register.fit_regions(candidates)
    for g in (1, 2, 3):
        if g not in models or stats[g]["n"] < 30 or stats[g]["frames"] < 3:
            raise InsufficientCalibrationData(f"Pane {g}: geometry requires 30 matches across at least three exposures")
        use = candidates[:, 1] == g
        x, y = candidates[use, 2:4].T
        A = np.c_[np.ones(len(x)), (x-500)/500, (y-500)/500]
        if np.linalg.matrix_rank(A) < 3 or np.linalg.cond(A) > 1e4:
            raise InsufficientCalibrationData(f"Pane {g}: stars do not constrain an affine geometry")
    return models, stats


def noise_factors(kernels, n_iter=30):
    rng = np.random.default_rng(771)
    noise = rng.normal(0, 0.1, (256, 256))
    d = 100 + noise
    result = {}
    for g in (1, 2, 3):
        out = psf.richardson_lucy(np.pad(d, psf.PAD, mode="edge"), kernels[g], n_iter)
        out = out[psf.PAD:-psf.PAD, psf.PAD:-psf.PAD]
        inner = (slice(32, -32), slice(32, -32))
        result[str(g)] = float(np.var(out[inner])/np.var(noise[inner]))
    return result


def validation(samples, ref):
    blocks = block_samples(samples, ref.flat, ref)
    report = {}
    for field in np.unique(blocks["field"]):
        per_pane = {}
        for g in (1, 2, 3):
            select = (blocks["field"] == field) & (blocks["g"] == g)
            X = blocks["M"][select]
            Q = (blocks["Y"][select] - ref.b[g])/ref.a[g]
            bins = []
            for lo, hi in ((30, 100), (100, 300), (300, 450)):
                m = (X >= lo) & (X < hi)
                residual = Q[m]/X[m]-1
                bins.append({"range": [lo, hi], "n": int(m.sum()),
                             "bias": float(np.median(residual)) if len(residual) else None})
            per_pane[str(g)] = bins
        report[str(int(field))] = per_pane
    return report


def refit(prepared):
    cfg = prepared["config"]
    options = cfg.get("calibration", {})
    allowed = {"output_dir", "seed_dir", "iterations", "bootstrap", "min_stars"}
    if set(options) - allowed:
        raise ValueError(f"Unknown calibration settings: {sorted(set(options)-allowed)}")
    if not options.get("output_dir"):
        raise ValueError("[calibration] requires output_dir for a new candidate bundle")
    output = Path(options["output_dir"])
    if output.exists():
        raise FileExistsError(f"Calibration output must be a new directory: {output}")
    train = [f for f in prepared["fields"] if f.get("train", True)]
    heldout = [f for f in prepared["fields"] if not f.get("train", True)]
    if len(train) < 2 or not heldout:
        raise InsufficientCalibrationData("Refit requires at least two training fields and one field with train=false")
    train_stems = {r["stem"] for f in train for r in f["exposures"]}
    if train_stems & {r["stem"] for f in heldout for r in f["exposures"]}:
        raise ValueError("Training and validation fields share exposures")
    for f in train + heldout:
        if not f.get("starless"):
            raise ValueError(f"{f['name']}: supply a starless reference for diffuse calibration")
    n_iter, nboot = int(options.get("iterations", 2)), int(options.get("bootstrap", 400))
    if n_iter < 1 or nboot < 20:
        raise ValueError("Use at least one calibration iteration and 20 bootstrap draws")
    seed = calib.refdata(calibration_dir=options.get("seed_dir"))
    ref = deepcopy(seed)
    working = output.with_name(output.name + ".partial")
    if working.exists():
        raise FileExistsError(f"Calibration working directory exists: {working}; choose a new output_dir")
    working.mkdir(parents=True)
    trace = []
    for cycle in range(n_iter):
        folder = working / f"fit_{cycle+1}"
        stamps, _, _, _ = _collect(train, ref, cfg, folder / "stars", do_stamps=True)
        ref.kernels, stacks, counts = fit_psfs(stamps, int(options.get("min_stars", 30)))
        ref.mk_kernels = ref.kernels
        _, candidates, _, _ = _collect(train, ref, cfg, folder / "geometry")
        ref.models, geometry_stats = _geometry(candidates)
        np.save(folder / "geometry_candidates.npy", candidates)
        _, _, files, metadata = _collect(train, ref, cfg, folder / "samples", do_samples=True)
        samples = _load_samples(files)
        ref.flat, nsamp = fit_flat(samples, ref.region)
        blocks = block_samples(samples, ref.flat, ref)
        law = fit_law(blocks, nboot)
        ref.a = np.array([np.nan] + [law[str(g)]["a"] for g in (1, 2, 3)])
        ref.b = np.array([np.nan] + [law[str(g)]["b"] for g in (1, 2, 3)])
        trace.append({"cycle": cycle+1, "stars": counts, "geometry": geometry_stats,
                      "law": law, "samples": len(samples["pix"])})
        np.savez(folder / "psfs.npz", **{f"P{g}": stacks[g] for g in stacks},
                 **{f"K{g}": ref.kernels[g] for g in ref.kernels})
    visits = sorted({m["visit"] for m in metadata})
    if len(visits) < 4:
        raise InsufficientCalibrationData("Flat uncertainty requires at least four visits")
    parity = {v: i % 2 for i, v in enumerate(visits)}
    ep = np.array([parity[m["visit"]] for m in metadata])
    halves = [fit_flat(_subset(samples, ep[samples["exp"]] == half), ref.region)[0] for half in (0, 1)]
    ref.flat_err = 0.5*np.abs(halves[0]-halves[1])
    _, validation_candidates, _, _ = _collect(heldout, ref, cfg, working / "validation_geometry")
    validation_stamps, _, _, _ = _collect(heldout, ref, cfg, working / "validation_stars", do_stamps=True)
    validation_psfs = {}
    for g in (1, 2, 3):
        n = len(validation_stamps[g])
        n_imager = len(validation_stamps[0])
        if n >= 10 and n_imager >= 10:
            imager_stack = np.median(validation_stamps[0], axis=0)
            pane_stack = np.median(validation_stamps[g], axis=0)
            imager_stack /= imager_stack.sum()
            pane_stack /= pane_stack.sum()
            prediction = fftconvolve(imager_stack, ref.kernels[g], mode="same")
            validation_psfs[str(g)] = {"n_stars": n, "n_imager_stars": n_imager,
                                        "residual_l1": float(np.abs(pane_stack-prediction).sum())}
        else:
            validation_psfs[str(g)] = {"n_stars": n, "n_imager_stars": n_imager,
                                        "residual_l1": None, "reason": "fewer than 10 stamps"}
    _, _, validation_files, validation_meta = _collect(heldout, ref, cfg, working / "validation_samples", do_samples=True)
    scores = validation(_load_samples(validation_files), ref)
    control_ref = deepcopy(ref)
    control_ref.flat = np.ones_like(ref.flat)
    control_ref.a, control_ref.b = np.ones(4), np.zeros(4)
    controls = validation(_load_samples(sorted((working / "validation_samples").glob("*_control.npz"))), control_ref)
    noise = noise_factors(ref.kernels)
    diagnostics = {"status": "candidate", "seed_checksum": seed.checksum,
                   "training_fields": [f["name"] for f in train], "validation_fields": [f["name"] for f in heldout],
                   "fits": trace, "validation_diffuse": scores, "validation_imager_control": controls,
                   "validation_psf": validation_psfs, "rl_variance_factors": noise,
                   "flat_fraction_with_at_least_three_samples": {
                       str(g): float(np.mean(nsamp[ref.region == g] >= 3)) for g in (1, 2, 3)},
                   "training_exposures": metadata, "validation_exposures": validation_meta,
                   "validation_scope": "Held-out fields; conditional on supplied pane mask and seed calibration. Scientific review required."}
    np.save(working / "validation_geometry_candidates.npy", validation_candidates)
    # Geometry residuals are scored against the fitted model, without refitting on validation.
    geometry_report = {}
    for g in (1, 2, 3):
        rows = validation_candidates[validation_candidates[:, 1] == g]
        if len(rows):
            dx, dy = geometry.region_shift(ref.models, g, rows[:, 2], rows[:, 3])
            residual = np.hypot(rows[:, 4]-dx, rows[:, 5]-dy)
            keys = np.unique(rows[:, [0, 2, 3]], axis=0, return_inverse=True)[1]
            order = np.lexsort((residual, keys))
            first = np.r_[True, np.diff(keys[order]) != 0]
            r = residual[order[first]]
            geometry_report[str(g)] = {"n_stars": len(r), "within_0p7_px": int((r < 0.7).sum()),
                                        "median_candidate_distance_px": float(np.median(r))}
        else:
            geometry_report[str(g)] = {"n_stars": 0}
    diagnostics["validation_geometry"] = geometry_report
    bundle = working / "bundle"
    bundle.mkdir()
    with fits.open(Path(seed.directory) / "qpm_R1_F770W.fits") as h:
        h["FLAT"].data = ref.flat.astype(np.float32)
        h["FLAT_ERR"].data = ref.flat_err.astype(np.float32)
        h[0].header["RECIPE"] = "R1-refit"
        h[0].header["CALSTAT"] = "candidate"
        h.append(fits.ImageHDU(nsamp.astype(np.int32), name="NSAMP"))
        h.writeto(bundle / "qpm_R1_F770W.fits")
    doc = deepcopy(ref.law)
    doc.update(recipe="R1-refit", status="candidate", panes=law,
               training={"fields": [f["name"] for f in train], "n_exposures": len(metadata),
                         "n_samples": len(samples["pix"]), "seed_checksum": seed.checksum,
                         "uncertainty": f"{nboot}-draw field bootstrap of law, flat held fixed; visit-half flat difference"},
               rl_variance_factors=noise, rl_iterations=30,
               point_source_bias_status="supplied diagnostic; not revalidated for this candidate")
    doc["validity"]["training_mjd"] = [min(m["mjd"] for m in metadata), max(m["mjd"] for m in metadata)]
    from astropy.time import Time
    doc["validity"]["training_dates"] = [Time(t, format="mjd").isot[:10] for t in doc["validity"]["training_mjd"]]
    doc["program_trained_on"] = sorted({m["stem"][2:7] for m in metadata})
    doc["flat_err_median"] = [float(np.median(ref.flat_err[ref.region == g])) for g in (1, 2, 3)]
    doc["validity"]["status"] = "candidate; supplied brightness/ramp limits retained for flags"
    write_json(bundle / "law_R1_F770W.json", doc)
    geo = {"formula": "dx=cx[0]+cx[1]*(x-500)/500+cx[2]*(y-500)/500; dy likewise with cy",
           "panes": {str(g): {"cx": ref.models[g][0].tolist(), "cy": ref.models[g][1].tolist()} for g in (1, 2, 3)}}
    write_json(bundle / "geometry_F770W.json", geo)
    np.savez(bundle / "kernels_F770W.npz", **{f"K{g}": ref.kernels[g] for g in (1, 2, 3)},
             **{f"K{g}_production": ref.kernels[g] for g in (1, 2, 3)})
    names = sorted([p.name for p in bundle.iterdir()])
    (bundle / "SHA256SUMS").write_text("".join(f"{sha256(bundle/name)}  {name}\n" for name in names))
    write_json(working / "calibration_report.json", pipeline._jsonable(diagnostics))
    calib.refdata(calibration_dir=str(bundle))
    working.rename(output)
    return output / "bundle"
