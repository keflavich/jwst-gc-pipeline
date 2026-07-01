#!/usr/bin/env python
"""
Generate a web-preview RGB image for a staged release field from its science
mosaics.  Default channels are the short-wavelength trio R=F212N, G=F187N,
B=F182M, which share a common pixel grid (no reprojection needed); pass
``--filters`` to override (2 or 3 filters; with 2, green is synthesized as the
mean of the other two).  ``--reproject`` reprojects all channels onto the first
channel's WCS (needed when mixing pixel scales, e.g. SW + LW).  ``--observation``
selects one pointing of a multi-pointing field (images/<obs>/<filter>/).

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
from astropy.io import fits
from astropy.visualization import AsinhStretch
from astropy.wcs import WCS
from PIL import Image

from stage_release import field_release_dir


def science_path(field_dir, filt, observation):
    # observation doubles as a subdir selector ("o023", or "MIRI")
    sub = field_dir / "images"
    if observation:
        sub = sub / observation
    matches = (glob.glob(str(sub / filt / "*-merged_i2d.fits"))
               or glob.glob(str(sub / filt / "*_i2d.fits")))
    if not matches:
        raise FileNotFoundError(f"no science mosaic for {filt} "
                                f"(sub={observation}) in {field_dir}")
    return matches[0]


def load_science(field_dir, filt, observation=None, ref_header=None):
    path = science_path(field_dir, filt, observation)
    data = fits.getdata(path, "SCI").astype("float32")
    if ref_header is None:
        return data, fits.getheader(path, "SCI")
    # reproject onto the reference WCS
    from reproject import reproject_interp
    out, _ = reproject_interp((data, WCS(fits.getheader(path, "SCI"))),
                              WCS(ref_header),
                              shape_out=(ref_header["NAXIS2"], ref_header["NAXIS1"]))
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
    args = parser.parse_args(argv)

    if len(args.filters) not in (2, 3):
        parser.error("--filters takes 2 or 3 filter names")

    field_dir = field_release_dir(args.field, args.version, args.release_root)

    # load channels; with --reproject, all are resampled onto the first's WCS
    ref_header = None
    loaded = []
    for f in args.filters:
        data, hdr = load_science(field_dir, f, args.observation, ref_header)
        loaded.append(data)
        if args.reproject and ref_header is None:
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

    if img.width > args.max_width:
        scale = args.max_width / img.width
        web = img.resize((args.max_width, int(img.height * scale)),
                         Image.LANCZOS)
    else:
        web = img
    jpg_path = out_dir / f"{stem}.jpg"
    web.save(jpg_path, format="JPEG", quality=90, progressive=True)
    print(f"wrote {jpg_path}  ({web.width}x{web.height})")


if __name__ == "__main__":
    main()
