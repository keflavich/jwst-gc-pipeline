"""Merged mosaic: the reference science layer, unchanged, with the pane data drizzled only where it has no data.

Output grid = the reference mosaic's pixel grid (same WCS, CRPIX shifted), either extended to cover every frame's
footprint ('full') or cut to the union footprint of the processed frames ('footprint').  The output FITS is created
on disk first and filled through memory maps, so the full 10678 grid (~31k x 25k px) never sits in memory:
per exposure, the panes are drizzled onto a local cutout and accumulated (SCI holds sum(w x), ERR holds
sum(w^2 var), WHT holds sum(w) until finalize()).

Extensions: SCI, ERR, WHT, SOURCE, FLAGS.
  SOURCE: 0 no data, 1 science (reference value copied unchanged), 2 pane only (outside the science footprint),
          3 pane inside the science footprint (a hole of the reference),
          4 (only with fill_bright_holes=False) a hole with pane data above 450 MJy/sr, left NaN.
  A hole is a connected NaN region of the reference enclosed by science pixels: typically a saturated star core.
  The pane has a lower throughput and does not saturate there.  By default every hole is filled (3); pane values
  above 450 MJy/sr there are unvalidated (FLAG_ABOVE).  fill_bright_holes=False leaves such holes NaN (4).
  FLAGS (pane pixels, SOURCE 2-4 only): see FLAG_* below.  WHT is 0 at SOURCE 4 (no pane value in SCI).
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

FLAG_ABOVE = 1       # a contributing pane pixel is > 450 MJy/sr at its exposure's cal level: unvalidated
FLAG_EDGE = 2        # a contributing pane pixel lies in the de-smearing edge margin (RL deficits of 2-8 % possible)
FLAG_LOCAL = 4       # a contributing exposure-pane has a local term applied (anchoring and/or dataset gain)
FLAG_BELOW = 8       # a contributing pane pixel is < 30 MJy/sr at its exposure's cal level: unvalidated
FLAG_WARN = 16       # a contributing exposure carries a validity warning (NGROUPS, saturation, non-GC, ...)
FLAG_NOSKY = 32      # a contributing exposure had no science sky offset (its pane level is not tied to the reference)
FLAG_SAT = 64        # a contributing pane pixel is rate SATURATED (slope from its unsaturated groups): unvalidated
FLAG_FLUENCE = 128   # a contributing pane pixel has rate * NGROUPS * TGROUP >= 40000 DN (near saturation): unvalidated
TILE, MARGIN = 2048, 256


def ref_hdu(hdul):
    """The reference image HDU: 'SCI' if present, else the first 2-D image."""
    if "SCI" in hdul:
        return hdul["SCI"]
    for h in hdul:
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return h
    raise ValueError("no 2-D image in the reference file")


class MaskedRef:
    """A reference image (or its ERR) read through a science mask: pixels where the mask is 0 / False read NaN, i.e.
    they are not science anywhere (science layer, sky offsets, overlap QA, holes).  Supports 2-D slicing and .shape
    only; slices come back as native float32, so the full image never sits in memory."""

    def __init__(self, img, mask):
        self.img, self.mask, self.shape, self.ndim = img, mask, img.shape, 2

    def __getitem__(self, sl):
        return np.where(np.asarray(self.mask[sl]) > 0, self.img[sl], np.nan).astype(np.float32)


def output_grid(ref_header, ref_shape, pts, mode="full", margin=5):
    """(header, shape, x0, y0): output pixel (x, y) = reference pixel (x + x0, y + y0)."""
    ny, nx = ref_shape
    lo = np.floor(np.nanmin(pts, 0)).astype(int) - margin
    hi = np.ceil(np.nanmax(pts, 0)).astype(int) + margin
    if mode == "full":
        x0, y0 = min(0, lo[0]), min(0, lo[1])
        x1, y1 = max(nx, hi[0] + 1), max(ny, hi[1] + 1)
    elif mode == "footprint":
        x0, y0, x1, y1 = lo[0], lo[1], hi[0] + 1, hi[1] + 1
    else:
        raise ValueError("grid must be 'full' or 'footprint'")
    w = WCS(ref_header)
    if w.has_distortion:
        raise ValueError("reference WCS has SIP / lookup-table distortion, which to_header() drops: use a TAN mosaic")
    hdr = w.to_header()
    hdr["CRPIX1"] -= x0
    hdr["CRPIX2"] -= y0
    hdr["REFX0"] = (int(x0), "output x = reference x - REFX0")
    hdr["REFY0"] = (int(y0), "output y = reference y - REFY0")
    return hdr, (int(y1 - y0), int(x1 - x0)), int(x0), int(y0)


EXTS = (("SCI", ">f4", -32), ("ERR", ">f4", -32), ("WHT", ">f4", -32), ("SOURCE", "u1", 8), ("FLAGS", "u1", 8))


def create(path, phdr, wcs_hdr, shape, comments=None):
    """Write the FITS skeleton (zero data, sparse on disk) and return {ext: np.memmap}."""
    offs = {}
    comments = comments or {}
    with open(path, "wb") as f:
        f.write(fits.PrimaryHDU(header=phdr).header.tostring().encode("ascii"))
        for name, dt, bitpix in EXTS:
            h = fits.Header([("XTENSION", "IMAGE"), ("BITPIX", bitpix), ("NAXIS", 2), ("NAXIS1", shape[1]),
                             ("NAXIS2", shape[0]), ("PCOUNT", 0), ("GCOUNT", 1), ("EXTNAME", name)])
            h.update(wcs_hdr)
            if name == "SCI" or name == "ERR":
                h["BUNIT"] = "MJy/sr"
            for c in comments.get(name, []):
                h["COMMENT"] = c
            f.write(h.tostring().encode("ascii"))
            offs[name] = (f.tell(), dt)
            n = shape[0] * shape[1] * np.dtype(dt).itemsize
            f.seek(n + (-n % 2880), 1)
        f.truncate()
    return {k: np.memmap(path, dtype=dt, mode="r+", offset=o, shape=shape) for k, (o, dt) in offs.items()}


def drizzle_local(Q, var, flagimgs, w, pixmap, shape):
    """Drizzle one exposure's panes onto the local cutout of the output grid they cover.
    flagimgs: {FLAG bit: bool image} carried along (an output pixel gets the bit if any contributing input pixel has
    it).  Returns (y0, x0, img, wht, var_img, {bit: img}) or None.  var is propagated with squared weights."""
    from drizzle.resample import Drizzle

    ok = w > 0
    if not ok.any():
        return None
    px, py = pixmap[..., 0], pixmap[..., 1]
    x0, x1 = max(int(np.floor(px[ok].min())) - 2, 0), min(int(np.ceil(px[ok].max())) + 3, shape[1])
    y0, y1 = max(int(np.floor(py[ok].min())) - 2, 0), min(int(np.ceil(py[ok].max())) + 3, shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    bits = sorted(flagimgs)
    d = Drizzle(kernel="square", out_shape=(y1 - y0, x1 - x0), disable_ctx=True)
    pm = np.dstack([px - x0, py - y0]).astype(np.float64)
    d.add_image(np.where(ok, Q, 0).astype(np.float32), exptime=1.0, pixmap=pm,
                weight_map=np.where(ok, w, 0).astype(np.float32),
                data2=[np.where(ok, var, 0).astype(np.float32)] +
                      [np.where(ok & flagimgs[b], 1, 0).astype(np.float32) for b in bits],
                in_units="cps")
    return y0, x0, d.out_img, d.out_wht, d.out_img2[0], dict(zip(bits, d.out_img2[1:]))


def accumulate(mm, cut, expflags):
    y0, x0, img, wht, var, fimg = cut
    sl = (slice(y0, y0 + img.shape[0]), slice(x0, x0 + img.shape[1]))
    m = wht > 0
    mm["SCI"][sl] += np.where(m, img * wht, 0)
    mm["ERR"][sl] += np.where(m, wht * wht * var, 0)
    mm["WHT"][sl] += wht
    f = np.where(m, expflags, 0)
    for bit, im in fimg.items():
        f |= np.where(m & (im > 0), bit, 0)
    mm["FLAGS"][sl] |= f.astype(np.uint8)


def finalize(mm, ref_img, ref_err, x0, y0, fill_bright_holes=True):
    """Tile by tile: SCI = reference where finite (bit-identical), else the pane mean; SOURCE; ERR; FLAGS.
    Every hole is filled (SOURCE 3) unless fill_bright_holes=False (then > 450 MJy/sr holes -> SOURCE 4).
    shortcut: SOURCE 3/4 (hole of the reference) is decided inside a tile + 256-px margin; a reference hole larger
    than that which reaches the margin is labelled 2.  Enlarge MARGIN if such holes matter.
    The hole test reads FLAGS over the margin: finalized tiles keep the FLAGS of pane pixels, so it is consistent."""
    ny, nx = mm["SCI"].shape
    rny, rnx = ref_img.shape
    for ty in range(0, ny, TILE):
        for tx in range(0, nx, TILE):
            # extended tile (for the hole test) and its core inside it
            ey0, ey1 = max(ty - MARGIN, 0), min(ty + TILE + MARGIN, ny)
            ex0, ex1 = max(tx - MARGIN, 0), min(tx + TILE + MARGIN, nx)
            r = np.full((ey1 - ey0, ex1 - ex0), np.nan, np.float32)
            e = np.full(r.shape, np.nan, np.float32)
            ry0, ry1 = max(ey0 + y0, 0), min(ey1 + y0, rny)
            rx0, rx1 = max(ex0 + x0, 0), min(ex1 + x0, rnx)
            if ry1 > ry0 and rx1 > rx0:
                dst = (slice(ry0 - y0 - ey0, ry1 - y0 - ey0), slice(rx0 - x0 - ex0, rx1 - x0 - ex0))
                r[dst] = ref_img[ry0:ry1, rx0:rx1]
                if ref_err is not None:
                    e[dst] = ref_err[ry0:ry1, rx0:rx1]
            sci = np.isfinite(r)
            core = (slice(ty - ey0, min(ty + TILE, ny) - ey0), slice(tx - ex0, min(tx + TILE, nx) - ex0))
            osl = (slice(ty, min(ty + TILE, ny)), slice(tx, min(tx + TILE, nx)))
            wht = np.array(mm["WHT"][osl])
            num = np.array(mm["SCI"][osl])
            v2 = np.array(mm["ERR"][osl])
            pane = (wht > 0) & ~sci[core]
            src = np.where(sci[core], 1, 0).astype(np.uint8)
            src[pane] = 2
            if pane.any():
                lab, n = ndimage.label(~sci)
                edge_lab = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
                hole = (lab > 0) & ~np.isin(lab, edge_lab)
                src[pane & hole[core]] = 3
                if not fill_bright_holes:
                    above = (np.asarray(mm["FLAGS"][ey0:ey1, ex0:ex1]) & FLAG_ABOVE) > 0
                    bright = np.isin(lab, np.unique(lab[hole & above]))     # whole holes touched by > 450
                    src[pane & bright[core]] = 4
            use = pane & (src != 4)
            out = np.where(sci[core], r[core], np.nan).astype(np.float32)
            err = np.where(sci[core], e[core], np.nan).astype(np.float32)
            with np.errstate(all="ignore"):
                out[use] = num[use] / wht[use]
                err[use] = np.sqrt(v2[use]) / wht[use]
            flags = np.where(pane, np.array(mm["FLAGS"][osl]), 0).astype(np.uint8)
            mm["WHT"][osl] = np.where(src == 4, 0, wht)
            mm["SCI"][osl] = out
            mm["ERR"][osl] = err
            mm["SOURCE"][osl] = src
            mm["FLAGS"][osl] = flags
    for m in mm.values():
        m.flush()
