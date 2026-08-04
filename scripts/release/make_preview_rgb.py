#!/usr/bin/env python
"""
Generate a web-preview RGB image for a staged release field from its science
mosaics.  Default channels are the short-wavelength trio R=F212N, G=F187N,
B=F182M, which share a common pixel grid (no reprojection needed); pass
``--filters`` to override (2 or 3 filters; with 2, green is synthesized as the
mean of the other two).  ``--reproject`` reprojects all channels onto the first
channel's WCS (needed when mixing pixel scales, e.g. SW + LW).  ``--observation``
selects one pointing of a multi-pointing field (images/<obs>/<filter>/).

A field staged as several mosaics per filter (module-split fields such as
arches/quintuplet, which ship ``-nrca_i2d`` + ``-nrcb_i2d`` instead of a single
``-merged_i2d``) is coadded onto one common grid automatically; that grid's
longest axis is capped by ``--max-axis``.

Writes into ``<release>/<version>/<field>/preview/``:
    <field>[_<obs>]_rgb_<R>_<G>_<B>.png   full resolution
    <field>[_<obs>]_rgb_<R>_<G>_<B>.jpg   web-downsampled (<= --max-width px)

Stretch: per-channel percentile clip + AsinhStretch.  Output is oriented for
display (origin flipped to image convention).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.visualization import AsinhStretch
from astropy.wcs import WCS
from PIL import Image

from stage_release import field_release_dir


def science_paths(field_dir, filt, observation):
    """Every science mosaic staged for one filter, sorted.

    Most fields stage a single full-field ``-merged_i2d``; module-split fields
    (arches/quintuplet: ``-nrca_i2d`` + ``-nrcb_i2d``) stage one mosaic per
    module, and those are coadded onto a common grid by ``load_science``."""
    # observation doubles as a subdir selector ("o023", or "MIRI")
    sub = field_dir / "images"
    if observation:
        sub = sub / observation
    matches = (glob.glob(str(sub / filt / "*-merged_i2d.fits"))
               or glob.glob(str(sub / filt / "*_i2d.fits")))
    if not matches:
        raise FileNotFoundError(f"no science mosaic for {filt} "
                                f"(sub={observation}) in {field_dir}")
    return sorted(matches)


def science_path(field_dir, filt, observation):
    return science_paths(field_dir, filt, observation)[0]


def _mosaic_header(paths, max_axis):
    """Common celestial grid covering every input, capped at ``max_axis`` px."""
    from reproject.mosaicking import find_optimal_celestial_wcs
    headers = [fits.getheader(p, "SCI") for p in paths]
    wcs_list = [((h["NAXIS2"], h["NAXIS1"]), WCS(h, relax=True)) for h in headers]
    wcs_out, shape_out = find_optimal_celestial_wcs(wcs_list)
    # cap the output size: a preview never needs the native grid of a
    # multi-module mosaic (which can be tens of thousands of px across)
    scale = max(shape_out) / max_axis
    if scale > 1:
        wcs_out, shape_out = find_optimal_celestial_wcs(
            wcs_list, resolution=abs(wcs_out.wcs.cdelt[0]) * scale * u.deg)
    header = wcs_out.to_header()
    header["NAXIS"] = 2
    header["NAXIS1"], header["NAXIS2"] = shape_out[1], shape_out[0]
    return header


def load_science(field_dir, filt, observation=None, ref_header=None,
                 max_axis=8000):
    paths = science_paths(field_dir, filt, observation)
    if len(paths) == 1 and ref_header is None:
        path = paths[0]
        return (fits.getdata(path, "SCI").astype("float32"),
                fits.getheader(path, "SCI"))
    if ref_header is None:
        ref_header = _mosaic_header(paths, max_axis)
    ref_wcs = WCS(ref_header, relax=True)
    shape_out = (ref_header["NAXIS2"], ref_header["NAXIS1"])
    if len(paths) == 1:
        # reproject onto the reference WCS
        from reproject import reproject_interp
        out, _ = reproject_interp(
            (fits.getdata(paths[0], "SCI").astype("float32"),
             WCS(fits.getheader(paths[0], "SCI"), relax=True)),
            ref_wcs, shape_out=shape_out)
        return out.astype("float32"), ref_header
    # several mosaics for one filter (per-module fields) -> coadd them
    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
    out, _ = reproject_and_coadd(
        [(fits.getdata(p, "SCI").astype("float32"),
          WCS(fits.getheader(p, "SCI"), relax=True)) for p in paths],
        ref_wcs, shape_out=shape_out, reproject_function=reproject_interp,
        combine_function="mean", match_background=False)
    return out.astype("float32"), ref_header


def stretch(channel, low_pct, high_pct, asinh_a):
    finite = channel[np.isfinite(channel) & (channel != 0)]
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    norm = np.clip((channel - lo) / (hi - lo), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    return AsinhStretch(a=asinh_a)(norm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", default="cloudc")
    parser.add_argument("--version", default="v1.0-2026.06")
    parser.add_argument("--release-root",
                        default="/orange/adamginsburg/jwst/releases")
    parser.add_argument("--filters", nargs="+", metavar="FILT",
                        default=["F212N", "F187N", "F182M"],
                        help="2 or 3 filters (R [G] B); 2 -> green = mean")
    parser.add_argument("--observation", default=None,
                        help="pointing of a multi-pointing field (e.g. o023)")
    parser.add_argument("--reproject", action="store_true",
                        help="reproject channels onto the first channel's grid")
    parser.add_argument("--low-percentile", type=float, default=40.0,
                        help="black point percentile")
    parser.add_argument("--high-percentile", type=float, default=99.8,
                        help="white point percentile")
    parser.add_argument("--asinh-a", type=float, default=0.03)
    parser.add_argument("--max-width", type=int, default=3000,
                        help="web JPEG max width in px")
    parser.add_argument("--max-height", type=int, default=3000,
                        help="web JPEG max height in px")
    parser.add_argument("--max-axis", type=int, default=8000,
                        help="cap (px) on the longest axis of the common grid "
                             "built when a filter has several mosaics to coadd")
    args = parser.parse_args(argv)

    if len(args.filters) not in (2, 3):
        parser.error("--filters takes 2 or 3 filter names")

    field_dir = field_release_dir(args.field, args.version, args.release_root)

    # A filter staged as several mosaics (one per module) has to be coadded onto
    # a common grid, and every other channel must then land on that same grid --
    # so it implies --reproject even when the pixel scales already match.
    multi = any(len(science_paths(field_dir, f, args.observation)) > 1
                for f in args.filters)

    # load channels; with --reproject, all are resampled onto the first's WCS
    ref_header = None
    loaded = []
    for f in args.filters:
        data, hdr = load_science(field_dir, f, args.observation, ref_header,
                                 max_axis=args.max_axis)
        loaded.append(data)
        if (args.reproject or multi) and ref_header is None:
            ref_header = hdr

    stretched = [stretch(c, args.low_percentile, args.high_percentile,
                         args.asinh_a) for c in loaded]
    if len(stretched) == 2:
        r_name, b_name = args.filters
        g_name = "mean"
        red, blue = stretched
        # crop to common shape (≤1 px differences when not reprojected)
        h = min(red.shape[0], blue.shape[0])
        w = min(red.shape[1], blue.shape[1])
        red, blue = red[:h, :w], blue[:h, :w]
        rgb = np.dstack([red, (red + blue) / 2, blue])
    else:
        r_name, g_name, b_name = args.filters
        h = min(c.shape[0] for c in stretched)
        w = min(c.shape[1] for c in stretched)
        rgb = np.dstack([c[:h, :w] for c in stretched])

    rgb8 = (rgb * 255).clip(0, 255).astype("uint8")
    # FITS origin is bottom-left; flip to image (top-left) convention
    rgb8 = np.flipud(rgb8)

    out_dir = field_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_tag = f"_{args.observation}" if args.observation else ""
    stem = f"{args.field}{obs_tag}_rgb_{r_name}_{g_name}_{b_name}".lower()

    img = Image.fromarray(rgb8, mode="RGB")
    png_path = out_dir / f"{stem}.png"
    img.save(png_path)
    print(f"wrote {png_path}  ({img.width}x{img.height})")

    # bound BOTH axes: a portrait field (arches is ~1:2.2 on sky) is under the
    # width cap while still being 5000 px tall, i.e. a multi-MB "web" JPEG.
    scale = min(1.0, args.max_width / img.width, args.max_height / img.height)
    if scale < 1.0:
        web = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))), Image.LANCZOS)
    else:
        web = img
    jpg_path = out_dir / f"{stem}.jpg"
    web.save(jpg_path, format="JPEG", quality=90, progressive=True)
    print(f"wrote {jpg_path}  ({web.width}x{web.height})")


if __name__ == "__main__":
    main()
