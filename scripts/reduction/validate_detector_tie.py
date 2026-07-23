#!/usr/bin/env python
"""Before/after validation of the m2 per-detector tie on cached m1 catalogs.

Measures, on a set of per-frame m1 catalogs of ONE (field, filter, visit):

  1. the all-detector visit consensus (``build_visit_consensus`` -- the real
     m2 machinery);
  2. the per-detector ties (``detector_tie.measure_visit_detector_ties``);
  3. the cross-module (A/B seam) detector-pair same-star offsets, per
     overlapping frame pair, aggregated per detector pair -- BEFORE and AFTER
     applying the measured ties in-memory (no files are modified);
  4. the per-frame same-star tie to the VIRAC2 reference, pair-weighted over
     frames -- BEFORE and AFTER (the tie is internal: the absolute frame must
     not move beyond the mean applied shift).

Acceptance (DETECTOR_TIE_DESIGN.md section 5): the seam terms collapse toward
the ~1 mas same-detector control floor and the VIRAC2 bulk is unchanged
within error.

Inputs are either the GDC-experiment cache format (``*_corr.fits`` with
``ra_crds``/``dec_crds`` columns and DETECTOR/EXPOSURE meta) via
``--corrdir``, or real pipeline per-frame catalogs
(``*_m1_daophot_basic.fits``, ``skycoord`` column + reliable-star cut) via
``--m1-glob``.

Every offset here is measured with the sanctioned machinery only:
``measure_offset`` (swept histogram) + ``local_residual_map`` (same-star,
precondition-gated).  No NN-median anywhere (CLAUDE.md rule #1).

Example (GDC cache):
    python scripts/reduction/validate_detector_tie.py \
        --corrdir .../gdc_experiment/corrected/brick_f212n --variant crds \
        --refcat /orange/adamginsburg/jwst/brick/catalogs/gaia_virac2_refcat_epoch2022.70.fits \
        --out /tmp/dettie_brick_f212n.json
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jwst_gc_pipeline.photometry.astrometry_offsets import (  # noqa: E402
    local_residual_map, measure_offset)
from jwst_gc_pipeline.photometry.detector_tie import (  # noqa: E402
    _samestar_refine, group_frames_by_detector, measure_visit_detector_ties)
from jwst_gc_pipeline.photometry.interframe_overlap import (  # noqa: E402
    _footprint_intersection, _in_bounds)
from jwst_gc_pipeline.photometry.visit_consensus import (  # noqa: E402
    build_visit_consensus, catalog_coords, load_reference_catalog,
    select_reliable_stars)

MAX_CONTROL_PAIRS = 60


def load_frames(args):
    """dict label('nrca1_e01') -> SkyCoord of reliable stars + meta."""
    frames = {}
    if args.corrdir:
        for path in sorted(glob.glob(os.path.join(args.corrdir, "*_corr.fits"))):
            t = Table.read(path)
            det = str(t.meta["DETECTOR"]).lower()
            exp = int(t.meta["EXPOSURE"])
            frames[f"{det}_e{exp:02d}"] = SkyCoord(
                ra=np.asarray(t[f"ra_{args.variant}"]) * u.deg,
                dec=np.asarray(t[f"dec_{args.variant}"]) * u.deg, frame="icrs")
    else:
        for path in sorted(glob.glob(args.m1_glob)):
            t = Table.read(path)
            keep = select_reliable_stars(t)
            det = str(t.meta.get("MODULE", t.meta.get("DETECTOR", "?"))).lower()
            exp = int(str(t.meta.get("EXPOSURE", 0))[-5:])
            frames[f"{det}_e{exp:02d}"] = catalog_coords(t)[keep]
    if not frames:
        raise SystemExit("no frames found")
    return frames


def as_consensus_tables(frames):
    """visit_consensus-format tables (already-cut stars: no further cuts)."""
    tables = []
    for label, sc in frames.items():
        det, e = label.rsplit("_e", 1)
        t = Table()
        t["skycoord"] = sc
        t.meta.update(VISIT="001", EXPOSURE=f"{int(e):05d}", MODULE=det,
                      FILTER="X", RAOFFSET=0.0, DEOFFSET=0.0)
        tables.append(t)
    return tables


def shift_coords(sc, dra_mas, ddec_mas):
    cosd = np.cos(np.radians(sc.dec.deg))
    return SkyCoord(ra=(sc.ra.deg + dra_mas / 3.6e6 / cosd) * u.deg,
                    dec=(sc.dec.deg + ddec_mas / 3.6e6) * u.deg, frame="icrs")


def samestar_pair(a, b, context):
    """Same-star offset (b - a, mas) of one overlapping frame pair; None when
    unmeasurable.  Sanctioned: swept histogram -> giant-cell refinement."""
    m = measure_offset(a, b, min_pairs=25, sweep=True, context=context)
    if m is None or not m["ok"] or m["swept"] or m["off"] > 90.0:
        return None
    lrm = local_residual_map(a, b, m, cell_arcsec=1e9,
                             match_radius=0.3 * u.arcsec, min_stars=10,
                             context=context + " giant")
    if not lrm["cells"]:
        return None
    c = max(lrm["cells"], key=lambda cc: cc["n"])
    return dict(dra=float(m["dra"] + c["dra_mas"]),
                ddec=float(m["ddec"] + c["ddec_mas"]), n=int(c["n"]),
                sem=float(np.hypot(c["dra_sem"], c["ddec_sem"])))


def classify_pairs(frames):
    """Overlapping frame pairs: AB (cross-module, the headline), xdet
    (same-module cross-detector) and same (same-detector control, sampled)."""
    labels = sorted(frames)
    pairs = {"AB": [], "xdet": [], "same": []}
    for la, lb in itertools.combinations(labels, 2):
        bounds, n_a, n_b = _footprint_intersection(frames[la], frames[lb])
        if bounds is None or min(n_a, n_b) < 25:
            continue
        deta, detb = la.rsplit("_e", 1)[0], lb.rsplit("_e", 1)[0]
        kind = ("same" if deta == detb
                else "AB" if deta[:4] != detb[:4] else "xdet")
        pairs[kind].append((la, lb, bounds))
    rng = np.random.default_rng(1182)
    for kind in ("xdet", "same"):
        if len(pairs[kind]) > MAX_CONTROL_PAIRS:
            idx = rng.choice(len(pairs[kind]), MAX_CONTROL_PAIRS, replace=False)
            pairs[kind] = [pairs[kind][i] for i in sorted(idx)]
    return pairs


def measure_seams(frames, pairs, tag):
    """Per detector-pair aggregation of the same-star pair offsets."""
    per_pair = {}
    raw = []
    t0 = time.time()
    todo = [(k, p) for k in ("AB", "xdet", "same") for p in pairs[k]]
    for i, (kind, (la, lb, bounds)) in enumerate(todo):
        a_in = frames[la][_in_bounds(frames[la], bounds)]
        b_in = frames[lb][_in_bounds(frames[lb], bounds)]
        rec = samestar_pair(a_in, b_in, f"{tag} {la}|{lb}")
        if rec is None:
            continue
        deta, detb = la.rsplit("_e", 1)[0], lb.rsplit("_e", 1)[0]
        key = tuple(sorted([deta, detb]))
        s = 1 if (deta, detb) == key else -1
        per_pair.setdefault((kind,) + key, []).append(
            (s * rec["dra"], s * rec["ddec"], rec["n"]))
        raw.append(dict(kind=kind, a=la, b=lb, **rec))
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i + 1}/{len(todo)} pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    agg = {}
    for (kind, da, db), vals in sorted(per_pair.items()):
        v = np.array(vals)
        offs = np.hypot(v[:, 0], v[:, 1])
        mad_sem = 1.4826 * np.median(np.abs(offs - np.median(offs)))
        mad_sem /= max(np.sqrt(len(v)), 1.0)
        agg[f"{kind}:{da}-{db}"] = dict(
            n_meas=len(v), dra_med=float(np.median(v[:, 0])),
            ddec_med=float(np.median(v[:, 1])),
            off_med=float(np.hypot(np.median(v[:, 0]), np.median(v[:, 1]))),
            sem=float(mad_sem))
    return agg, raw


def measure_virac_bulk(frames, refcat, tag):
    """Pair-weighted mean over frames of the per-frame same-star VIRAC2 tie
    (per frame -- pooling exposures duplicates stars and breaks matched-pair
    uniqueness)."""
    if refcat is None:
        return None
    ref_all = refcat["all"]
    vals = []
    for label, sc in sorted(frames.items()):
        m = measure_offset(sc, ref_all, sweep=True, context=f"{tag} {label} vs ref")
        if m is None or not m.get("ok") or m.get("swept"):
            continue
        refined = _samestar_refine(sc, ref_all, m, context=f"{tag} {label} ss")
        if refined is not None:
            dra, ddec, _s1, _s2, n = refined
            vals.append((dra, ddec, n))
    if not vals:
        return None
    v = np.array(vals)
    w = v[:, 2] / v[:, 2].sum()
    return dict(dra=float(np.sum(w * v[:, 0])), ddec=float(np.sum(w * v[:, 1])),
                n_frames=len(v), n_pairs=int(v[:, 2].sum()),
                frame_scatter_mas=float(np.std(np.hypot(v[:, 0], v[:, 1]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corrdir", help="GDC-cache dir of *_corr.fits")
    ap.add_argument("--variant", default="crds", help="corrdir coord variant")
    ap.add_argument("--m1-glob", help="glob of *_m1_daophot_basic.fits")
    ap.add_argument("--refcat", help="gaia_virac2 seed refcat path")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--label", default="", help="tag for logs")
    args = ap.parse_args()
    if not args.corrdir and not args.m1_glob:
        ap.error("need --corrdir or --m1-glob")

    t0 = time.time()
    frames = load_frames(args)
    print(f"[{args.label}] {len(frames)} frames", flush=True)
    refcat = load_reference_catalog(args.refcat) if args.refcat else None

    # 1. all-detector visit consensus (the real m2 machinery)
    cons = build_visit_consensus(as_consensus_tables(frames),
                                 context=f"{args.label} consensus")
    print(f"[{args.label}] consensus: {len(cons['coords'])} stars, "
          f"{cons['n_components']} component(s) "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 2. per-detector ties (reuse the consensus-build per-frame histograms)
    hist_by_label = {}
    for e in cons["exposures"]:
        det = str(e["key"][2]).lower()
        hist_by_label[f"{det}_e{int(e['key'][1]):02d}"] = e["vs_consensus"]
    frames_by_det = group_frames_by_detector(
        (("001", int(label.rsplit("_e", 1)[1]),
          label.rsplit("_e", 1)[0], "X"),
         sc, hist_by_label.get(label))
        for label, sc in frames.items())
    ties = measure_visit_detector_ties(frames_by_det, cons["coords"],
                                       refcat=refcat, context=args.label)
    for det, rec in sorted(ties.items()):
        print(f"[{args.label}] tie {det}: "
              f"({rec['dra_mas']:+.2f},{rec['ddec_mas']:+.2f}) mas "
              f"sem={rec['sem_mas']:.2f} n={rec['n_pairs']} "
              f"apply={rec['apply']} "
              f"{'' if rec['apply'] else rec['refuse_reason']}", flush=True)

    # 3. seams before/after
    pairs = classify_pairs(frames)
    print(f"[{args.label}] pairs: AB={len(pairs['AB'])} "
          f"xdet={len(pairs['xdet'])} same={len(pairs['same'])}", flush=True)
    seam_before, _ = measure_seams(frames, pairs, f"{args.label} before")
    shifted = {}
    for label, sc in frames.items():
        det = label.rsplit("_e", 1)[0]
        rec = ties.get(det)
        if rec is not None and rec.get("apply"):
            shifted[label] = shift_coords(sc, rec["dra_mas"], rec["ddec_mas"])
        else:
            shifted[label] = sc
    seam_after, _ = measure_seams(shifted, pairs, f"{args.label} after")

    # 4. VIRAC2 bulk before/after
    virac_before = measure_virac_bulk(frames, refcat, f"{args.label} before")
    virac_after = measure_virac_bulk(shifted, refcat, f"{args.label} after")

    out = dict(
        label=args.label, n_frames=len(frames),
        consensus_stars=int(len(cons["coords"])),
        ties={d: {k: v for k, v in r.items() if k != "frame_ties"}
              for d, r in ties.items()},
        seam_before=seam_before, seam_after=seam_after,
        virac_before=virac_before, virac_after=virac_after,
        runtime_s=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print(f"\n[{args.label}] seam table (median same-star offset, mas):")
    print(f"{'pair':32s} {'n':>4s} {'before dRA/dDec':>18s} "
          f"{'after dRA/dDec':>18s}")
    for key in sorted(set(seam_before) | set(seam_after)):
        b = seam_before.get(key)
        a = seam_after.get(key)
        bs = (f"{b['dra_med']:+6.2f}/{b['ddec_med']:+6.2f}" if b else "n/a")
        as_ = (f"{a['dra_med']:+6.2f}/{a['ddec_med']:+6.2f}" if a else "n/a")
        print(f"{key:32s} {b['n_meas'] if b else 0:>4d} {bs:>18s} {as_:>18s}")
    if virac_before and virac_after:
        print(f"[{args.label}] VIRAC2 bulk: before "
              f"({virac_before['dra']:+.2f},{virac_before['ddec']:+.2f}) -> "
              f"after ({virac_after['dra']:+.2f},{virac_after['ddec']:+.2f}) "
              f"mas ({virac_before['n_pairs']} pairs)")
    print(f"[{args.label}] wrote {args.out} ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
