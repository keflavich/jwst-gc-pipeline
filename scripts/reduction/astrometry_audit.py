"""Per-field astrometry audit for JWST-GC mosaics.

Three checks, run per field on the released ``i2d`` mosaics:

1. **Inter-module** (NRCA vs NRCB) — the proper-motion killer.  Reference-FREE: match
   detections in the per-module ``-nrca``/``-nrcb`` (or ``-nrcalong``/``-nrcblong``)
   mosaics to each other in their overlap and take the median offset.  A non-zero
   inter-module shift injects spurious PM (~offset / baseline).

2. **Per-filter internal consistency** — reference-FREE: bulk offset of each filter's
   mosaic against an anchor filter (the one with the most detections), via the crowding
   robust pair-histogram cross-correlation (NOT nearest-neighbour, which collapses to ~0
   in dense fields and hides a real bulk offset).

3. **Absolute frame** (optional) — bulk offset of each mosaic vs a reference catalogue
   (VIRAC2 or Gaia, PM-propagated to the observation epoch), same xcorr method.

Bulk offsets always use the pair-histogram peak (``xcorr``) so a >search-radius shift is
recovered; inter-module uses a tight direct match because after alignment the residual is
small.  Emits a JSON summary and a printed table; flags anything above the thresholds.

Usage:
    python -m data_qa.astrometry_audit --field sgrb2 [--refcat PATH] [--out result.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS

BASE = "/orange/adamginsburg/jwst"
# flag thresholds (mas)
THRESH = dict(intermodule=15.0, perfilter=30.0, absolute=75.0, crossobs=30.0)
XMAXSEP = 2.5 * u.arcsec   # pair-histogram search radius (recovers offsets up to this)
XBIN = 0.05               # arcsec histogram bin
MIN_PEAK_RATIO = 4.0      # peak/background below this -> bulk tie ambiguous


# --------------------------------------------------------------------------- detection
def detect(path, thr=50.0, fwhm=2.5):
    """High-SNR detections on a drizzled mosaic -> (SkyCoord, instrumental mag)."""
    from photutils.detection import DAOStarFinder
    with fits.open(path) as hdul:
        sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
        w = WCS(sci.header)
        data = sci.data.astype("float32")
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    tbl = DAOStarFinder(fwhm=fwhm, threshold=thr * std)(data - med)
    if tbl is None or len(tbl) == 0:
        return None, None
    sc = SkyCoord(w.pixel_to_world(tbl["xcentroid"], tbl["ycentroid"]))
    mag = -2.5 * np.log10(np.clip(np.asarray(tbl["flux"], float), 1e-9, None))
    return sc, mag


# --------------------------------------------------------------------------- offsets
def xcorr(a: SkyCoord, b: SkyCoord, maxsep=XMAXSEP, binarc=XBIN):
    """Bulk on-sky offset (mas) to move ``a`` onto ``b`` via the peak of the 2-D
    histogram of all pair separations within ``maxsep``.  Crowding-proof.  Returns dict
    with dra/ddec/off (mas), npairs, peak_ratio, or None.

    Thin wrapper over the sanctioned ``photometry.astrometry_offsets.measure_offset``
    (single source of truth for offset-histogram stacking) -- so the audit inherits
    the window-SWEEP: if the initial ``maxsep`` window shows no coherent tie, it
    escalates the window (3->10->30->60") and reports the real peak. A fixed 2.5"
    window is exactly what let brick-1182 v001's ~20" offset read as ~0."""
    from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
    r = measure_offset(a, b, maxsep=maxsep, bin_arcsec=binarc, min_pairs=30)
    if r is None:
        return None
    # preserve this module's return contract (peak_ratio == contrast)
    return dict(dra=r["dra"], ddec=r["ddec"], off=r["off"],
                npairs=r["npairs"], peak_ratio=r["contrast"])


def direct_intermodule(sc_a: SkyCoord, sc_b: SkyCoord, radius=0.1 * u.arcsec):
    """Median offset between the SAME stars seen in module A and module B (their overlap).
    Reference-free; a non-zero value is a real inter-module frame offset."""
    ia, ib, sep, _ = search_around_sky(sc_a, sc_b, radius)
    if len(ia) < 8:
        return None
    dra = (sc_a[ia].ra - sc_b[ib].ra).to(u.arcsec).value * np.cos(np.radians(sc_a[ia].dec.value)) * 1000
    ddec = (sc_a[ia].dec - sc_b[ib].dec).to(u.arcsec).value * 1000
    return dict(dra=float(np.median(dra)), ddec=float(np.median(ddec)),
                off=float(np.hypot(np.median(dra), np.median(ddec))),
                nshared=int(len(ia)), scatter=float(mad_std(ddec)))


# --------------------------------------------------------------------------- reference
def load_reference(path, epoch):
    """Load a VIRAC2 or Gaia reference catalogue and PM-propagate to ``epoch`` (jyear).
    Auto-detects columns.  Returns (SkyCoord, mag) or (None, None)."""
    if not path or not os.path.exists(path):
        return None, None
    t = Table.read(path)
    cols = {c.lower(): c for c in t.colnames}

    def col(*names):
        for n in names:
            if n.lower() in cols:
                return np.asarray(t[cols[n.lower()]], float)
        return None

    ra = col("RAJ2000", "ra", "RA")
    dec = col("DEJ2000", "dec", "DEC")
    if ra is None or dec is None:
        return None, None
    pmra = col("pmRA", "pmra")
    pmdec = col("pmDE", "pmdec")
    mag = col("Ksmag", "phot_g_mean_mag", "Gmag", "mag")
    ref_ep = 2014.0 if ("RAJ2000".lower() in cols) else 2016.0   # VIRAC2 vs Gaia
    dt = epoch - ref_ep
    if pmra is not None and pmdec is not None:
        pmra = np.nan_to_num(pmra); pmdec = np.nan_to_num(pmdec)
        ra = ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec))
        dec = dec + pmdec * dt / 3.6e6
    good = np.isfinite(ra) & np.isfinite(dec)
    sc = SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
    return sc, (mag[good] if mag is not None else None)


# --------------------------------------------------------------------------- field
FILT_RE = re.compile(r"clear-(f\d{3,4}[wnm])-(merged|nrca|nrcb|nrcalong|nrcblong)_i2d", re.I)


def find_mosaics(field):
    """Return {filter: {module: path}} for the field's science i2d mosaics."""
    out = {}
    for p in glob.glob(f"{BASE}/{field}/*/pipeline/*_i2d.fits"):
        low = os.path.basename(p).lower()
        if "resbgsub" in low or "_model_" in low or "_residual_" in low:
            continue
        m = FILT_RE.search(low)
        if not m:
            continue
        filt, mod = m.group(1).upper(), m.group(2).lower()
        out.setdefault(filt, {})[mod] = p
    return out


def epoch_of(path):
    """jyear from a mosaic header DATE-OBS."""
    from astropy.time import Time
    for ext in (0, ("SCI", 1), 1):
        try:
            d = fits.getheader(path, ext=ext).get("DATE-OBS")
        except (KeyError, IndexError, OSError):
            d = None
        if d:
            return float(Time(d).jyear)
    return None


def channel(filt):
    """SW (<=2.12um) vs LW channel from a filter name like 'F212N'."""
    return "SW" if int(filt[1:4]) <= 212 else "LW"


def _merged_mosaics(spec):
    """{filter: path} of a field's (or 'field:obs') merged science i2d mosaics.

    ``spec`` is a field name (``arches``) or ``field:obs`` (``gc2211:046``).  The
    optional obs token restricts to that observation's mosaics -- needed for
    multi-obs fields (gc2211 has obs 023/028/046/049/050 in one field dir).
    """
    field, _, obs = spec.partition(":")
    obs = obs.lstrip("o")
    out = {}
    for pat in (f"{BASE}/{field}/*/pipeline/*_i2d.fits",
                f"{BASE}/{field}/images-merged/*_i2d.fits"):
        for p in glob.glob(pat):
            low = os.path.basename(p).lower()
            if any(s in low for s in ("resbgsub", "_model_", "_residual_", "starless")):
                continue
            m = FILT_RE.search(low)
            if not m or m.group(2) != "merged":
                continue
            if obs and f"o{obs}" not in low.replace("-", "").replace("_", ""):
                continue
            out.setdefault(m.group(1).upper(), p)   # first hit per filter
    return out


def cross_obs_consistency(base_spec, peer_specs, thr=50.0):
    """Flag relative astrometric offsets between overlapping observations.

    The per-field audit ties each field to VIRAC independently; two fields can
    each pass (<absolute threshold) yet disagree with EACH OTHER by more than
    that -- a PM-killer for any cross-epoch/overlap science.  (This is exactly
    how gc2211 o046 slipped through: never tied, ~108 mas off arches, but each
    field's own vs-VIRAC audit was clean.)

    For each channel (SW, LW) we cross-correlate ``base_spec``'s densest merged
    mosaic against each peer's same-channel merged mosaic.  Cross-FILTER within a
    channel is fine (same stars; the offset-histogram is positional), so
    arches F212N vs gc2211 F200W is a valid SW pair.  Flags off > crossobs
    threshold with a confident peak.
    """
    base = _merged_mosaics(base_spec)
    if not base:
        return dict(base=base_spec, error="no merged mosaics")
    base_det = {f: detect(p)[0] for f, p in base.items()}
    res = dict(base=base_spec, peers={}, flags=[])
    for peer in peer_specs:
        pm = _merged_mosaics(peer)
        pairs = {}
        for chan in ("SW", "LW"):
            bf = [f for f in base if channel(f) == chan and base_det.get(f) is not None]
            pf = [f for f in pm if channel(f) == chan]
            if not bf or not pf:
                continue
            # densest available filter in each field's channel
            bfilt = max(bf, key=lambda f: len(base_det[f]))
            pfilt = pf[0]
            pdet = detect(pm[pfilt])[0]
            if pdet is None:
                continue
            r = xcorr(base_det[bfilt], pdet)
            if not r:
                continue
            r.update(base_filter=bfilt, peer_filter=pfilt)
            pairs[chan] = r
            if r["off"] > THRESH["crossobs"] and r["peak_ratio"] >= MIN_PEAK_RATIO:
                res["flags"].append(
                    f"crossobs {base_spec}({bfilt}) vs {peer}({pfilt}) {chan}: "
                    f"{r['off']:.0f} mas (>{THRESH['crossobs']:.0f})")
        res["peers"][peer] = pairs
    return res


def print_cross_report(r):
    print(f"\n===== CROSS-OBS  base={r['base']} =====")
    if r.get("error"):
        print("  ERROR:", r["error"]); return
    for peer, pairs in r["peers"].items():
        if not pairs:
            print(f"  vs {peer}: no overlapping channel/mosaics"); continue
        for chan, d in pairs.items():
            flag = "  *** RELATIVE-FRAME OFFSET ***" if (
                d["off"] > THRESH["crossobs"] and d["peak_ratio"] >= MIN_PEAK_RATIO) else ""
            print(f"  vs {peer} {chan} ({d['base_filter']} vs {d['peer_filter']}): "
                  f"({d['dra']:+.0f},{d['ddec']:+.0f}) |{d['off']:.0f} mas| "
                  f"peak/bg={d['peak_ratio']:.0f} n={d['npairs']}{flag}")
    print("  FLAGS:", "; ".join(r["flags"]) if r["flags"] else "none")


def audit_field(field, refcat=None, thr=50.0):
    mosaics = find_mosaics(field)
    if not mosaics:
        return dict(field=field, error="no mosaics found")
    # module key for A / B given the two conventions
    modA = lambda mods: mods.get("nrca") or mods.get("nrcalong")
    modB = lambda mods: mods.get("nrcb") or mods.get("nrcblong")

    epoch = next((epoch_of(next(iter(m.values()))) for m in mosaics.values()), None)
    ref_sc, _ = load_reference(refcat, epoch) if (refcat and epoch) else (None, None)

    # detect on every mosaic once (cache)
    det = {}
    for filt, mods in mosaics.items():
        for mod, path in mods.items():
            sc, _ = detect(path, thr=thr)
            det[(filt, mod)] = sc
            print(f"  detect {field} {filt} {mod}: "
                  f"{0 if sc is None else len(sc)} src", flush=True)

    res = dict(field=field, epoch=epoch, filters=sorted(mosaics), intermodule={},
               perfilter={}, absolute={}, flags=[])

    # 1) inter-module (reference-free)
    for filt, mods in sorted(mosaics.items()):
        a, b = det.get((filt, "nrca")) or det.get((filt, "nrcalong")), \
               det.get((filt, "nrcb")) or det.get((filt, "nrcblong"))
        if a is None or b is None:
            continue
        r = direct_intermodule(a, b)
        if r:
            res["intermodule"][filt] = r
            if r["off"] > THRESH["intermodule"]:
                res["flags"].append(f"intermodule {filt}: {r['off']:.0f} mas "
                                    f"(>{THRESH['intermodule']:.0f})")

    # 2) per-filter internal consistency -- WITHIN channel only.  SW and LW have
    # independent distortion solutions and detect different stellar populations, so a
    # SW-vs-LW catalog cross-match is unreliable (few true pairs -> a chance histogram
    # peak can dominate: this produced spurious ~420 mas "offsets" that the absolute tie
    # -- uniform ~18 mas across all filters -- disproves).  Compare each filter to the
    # densest filter of its OWN channel; require many pairs before flagging.  Cross-
    # channel agreement is assessed by the absolute check instead.
    # (channel() is defined at module scope.)

    merged = {f: det.get((f, "merged")) for f in mosaics if det.get((f, "merged")) is not None}
    res["anchor"] = {}
    if merged:
        for chan in ("SW", "LW"):
            cf = {f: sc for f, sc in merged.items() if channel(f) == chan}
            if len(cf) < 2:
                continue
            anchor = max(cf, key=lambda f: len(cf[f]))
            res["anchor"][chan] = anchor
            for filt, sc in sorted(cf.items()):
                if filt == anchor:
                    continue
                r = xcorr(sc, cf[anchor])
                if r:
                    r["anchor"] = anchor
                    res["perfilter"][filt] = r
                    # npairs>=5000 was the original confidence guard, but it
                    # SUPPRESSED real offsets in sparse fields: ngc6334 F115W/
                    # F162M/F182M sit 60-66 mas off their channel anchor with
                    # peak/bg 123-225 at only ~500-1200 pairs -- an unambiguous
                    # histogram peak.  A strong peak ratio at moderate pair
                    # counts is just as conclusive as a weak one at 5000.
                    if (r["off"] > THRESH["perfilter"] and r["peak_ratio"] >= MIN_PEAK_RATIO
                            and (r["npairs"] >= 5000
                                 or (r["npairs"] >= 300 and r["peak_ratio"] >= 20))):
                        res["flags"].append(f"perfilter {filt} vs {anchor}: {r['off']:.0f} mas")
        # 3) absolute vs reference
        if ref_sc is not None:
            for filt, sc in sorted(merged.items()):
                r = xcorr(sc, ref_sc)
                if r:
                    res["absolute"][filt] = r
                    if r["off"] > THRESH["absolute"] and r["peak_ratio"] >= MIN_PEAK_RATIO:
                        res["flags"].append(f"absolute {filt}: {r['off']:.0f} mas vs ref")
    return res


def print_report(r):
    print(f"\n===== {r['field']}  (epoch {r.get('epoch')}) =====")
    if r.get("error"):
        print("  ERROR:", r["error"]); return
    print(f"  filters: {', '.join(r['filters'])}   anchors={r.get('anchor', {})}")
    if r["intermodule"]:
        print("  INTER-MODULE (nrca-nrcb, reference-free):")
        for f, d in r["intermodule"].items():
            flag = "  *** PM-KILLER ***" if d["off"] > THRESH["intermodule"] else ""
            print(f"    {f:6s} ({d['dra']:+.1f},{d['ddec']:+.1f}) |{d['off']:.1f} mas| "
                  f"n={d['nshared']}{flag}")
    if r["perfilter"]:
        print("  PER-FILTER (within-channel, internal):")
        for f, d in r["perfilter"].items():
            print(f"    {f:6s} vs {d.get('anchor','?')} |{d['off']:.1f} mas| "
                  f"peak/bg={d['peak_ratio']:.0f} n={d['npairs']}")
    if r["absolute"]:
        print("  ABSOLUTE vs reference:")
        for f, d in r["absolute"].items():
            print(f"    {f:6s} |{d['off']:.1f} mas| peak/bg={d['peak_ratio']:.0f}")
    print("  FLAGS:", "; ".join(r["flags"]) if r["flags"] else "none")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", required=True)
    ap.add_argument("--refcat", default=None, help="VIRAC2/Gaia reference catalogue path")
    ap.add_argument("--thr", type=float, default=50.0, help="detection threshold (sigma)")
    ap.add_argument("--out", default=None, help="write JSON result here")
    ap.add_argument("--cross-check", nargs="+", default=None, metavar="FIELD[:obs]",
                    help="also check --field's frame against these overlapping "
                         "observations (field name or field:obs, e.g. gc2211:046). "
                         "Flags relative offsets that per-field vs-VIRAC audits miss.")
    args = ap.parse_args(argv)
    r = audit_field(args.field, refcat=args.refcat, thr=args.thr)
    print_report(r)
    cross = None
    if args.cross_check:
        cross = cross_obs_consistency(args.field, args.cross_check, thr=args.thr)
        print_cross_report(cross)
        r["flags"] = list(r.get("flags", [])) + cross.get("flags", [])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(field_audit=r, cross_obs=cross), fh, indent=2)
        print(f"\nwrote {args.out}")
    # non-zero exit if anything flagged (field or cross-obs) -> CI/campaign gate
    return 1 if r.get("flags") else 0


if __name__ == "__main__":
    sys.exit(main())
