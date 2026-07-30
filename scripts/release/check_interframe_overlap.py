#!/usr/bin/env python
"""BLOCKING gate: reference-free inter-frame overlap registration.

The brick-1182 F200W seam failure (2026-07-12) proved that our existing gates are
STRUCTURALLY BLIND to a per-visit residual:

- ``registration_failsafes.py`` matches the mosaic against its OWN merged catalog.
  Both are built from the same ``_crf`` frames, so if visit-001 carries a ~90 mas
  residual, the catalog inherits it too -> mosaic and catalog AGREE (offset ~0) ->
  PASS, even though both are wrong. A self-referential truth cannot see a shared
  error. It also searches only +-2.5" (no sweep) so a gross overlap offset reads as
  "no overlap", not FAIL.
- Bulk / coarse-grid vs-reference checks average the two visits together in the
  overlap and read ~0.

The ONLY check that sees it is REFERENCE-FREE and PAIRWISE: detect on each
per-exposure ``_crf`` frame (each on its own corrected GWCS), pool by
exposure-group (visit x module), and histogram-stack every OVERLAPPING pair's
MUTUAL offset. Overlapping same-instrument groups must co-register to < 30 mas.
Non-zero exit on FAIL so it gates a release/staging chain.

Usage::

    python check_interframe_overlap.py --field brick --filter F200W
    python check_interframe_overlap.py --field brick --scan           # all filters
    # scope to specific released observations (default: auto-derived from the
    # field's merged mosaics, so stray crf from other programs are excluded):
    python check_interframe_overlap.py --field brick --scan --observations 02221-001,01182-004
    # optional external absolute cross-check (fine grid vs VIRAC2/Gaia):
    python check_interframe_overlap.py --field brick --filter F200W --refcat <path>
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.detection import find_peaks

from jwst_gc_pipeline.photometry.interframe_overlap import (
    overlap_offset_grid, pairwise_overlap_offsets, DEFAULT_OVERLAP_TOL_MAS)
from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset_grid

BASE = os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst")
TOL_MAS = float(os.environ.get("OVERLAP_TOL_MAS", DEFAULT_OVERLAP_TOL_MAS))
# external-reference fine-grid gate
GRID_N = int(os.environ.get("OVERLAP_GRID_N", 16))
GRID_MAX_OFF_MAS = float(os.environ.get("OVERLAP_GRID_MAX_OFF_MAS", 80.0))


def _detect(path, nsigma=8.0, box=5):
    with fits.open(path, memmap=True) as h:
        d = np.asarray(h["SCI"].data)
        w = WCS(h["SCI"].header)
    fin = np.isfinite(d)
    if fin.sum() < 2000:
        return None
    _, med, std = sigma_clipped_stats(d[fin], sigma=3.0, maxiters=3)
    tb = find_peaks(np.where(fin, d, 0.0), threshold=med + nsigma * std, box_size=box)
    if tb is None or len(tb) < 20:
        return None
    sky = w.pixel_to_world(np.array(tb["x_peak"]), np.array(tb["y_peak"]))
    return SkyCoord(sky.ra, sky.dec)


# Precise per-exposure crf filename parser.  JWST products are named
#   jw<PPPPP><OOO><VVV>_<GGSAA>_<EEEEE>_<detector>[_<lineage>...]_o<OOO>_crf.fits
# (PPPPP=proposal, OOO=observation, VVV=visit; the trailing _oOOO_ repeats the
# observation).  We anchor on that exact structure rather than a permissive
# `jw*_*_nrc*_o*` glob: a wildcard `o*` matches EVERY observation, so a shared
# target directory (the brick dir also holds 2221 o002 = cloudc crf) leaks stray
# frames from other observations/programs into the verdict.  The regex both
# validates a name and yields (proposal, observation, visit, module) exactly.
_CRF_RE = re.compile(
    r"^jw(?P<prop>\d{5})(?P<obs>\d{3})(?P<visit>\d{3})_\d+_\d+_"
    r"(?P<det>mirimage|nrc[ab](?:long|[1-4])?)"
    r"(?P<lineage>(?:_[a-z0-9]+)*?)_o(?P<obs2>\d{3})_crf\.fits$")

#: Lineage tokens of the RETIRED post-resample realignment path
#: (``realign_to_vvv`` / ``sync_gwcs_to_fits_wcs``, removed from the code
#: 2026-07-11 -- see ASTROMETRY_WCS_CORRECTION_FLOW.md).  Its OUTPUTS are older
#: than that removal (the cloudc set is dated 2023-08-01), and still on disk, and
#: still match ``jw*_crf.fits``.
#:
#: They must never be pooled into a registration verdict: in those files the FITS
#: header and the ASDF GWCS disagree with EACH OTHER by arcseconds, because that
#: path realigned one representation and not the other.  Measured 2026-07-29 on
#: cloudc/F405N, all 32 ``*_realigned_to_vvv_*_crf.fits``:
#:
#:     visit 001 (16 frames):  7.972 - 8.067 arcsec FITS-vs-GWCS
#:     visit 002 (16 frames):  4.091 - 4.170 arcsec
#:     live *_destreak_o002_crf (32):  6.681 - 7.972 mas, median 7.327
#:
#: The ~3.9 arcsec VISIT-TO-VISIT differential matters more here than the
#: absolute size, because this gate groups by (obs+visit, module) -- a
#: visit-dependent shift is precisely what it exists to catch.  Those files also
#: lack the ``-SIP`` CTYPE suffix, so astropy warns when reading them: a second,
#: independent reason no reader gets the intended answer out of them.
#:
#: SCOPE, stated plainly: this blocklist removes 32 files in ONE directory
#: (cloudc/F405N is the only directory archive-wide with retired-path crf).  It
#: does NOT solve lineage staleness in general -- 46 of 127 pipeline directories
#: still admit more than one lineage copy of the same exposure (mostly
#: ``['', '_destreak']`` or ``['_align', '_destreak']``), and in cloudc/F405N the
#: ``bare`` copy sits 8.5 arcsec from the live ``_destreak`` one by same-pixel sky
#: position.  Selecting a single family per exposure is the general fix; see the
#: follow-up issue.  Do not read this blocklist as more than it is.
_RETIRED_LINEAGE_TOKENS = ("realigned", "refcat", "vvv")


def _retired_lineage(name):
    """True if ``name`` is a retired-path crf (the reason _parse_crf rejected it)."""
    m = _CRF_RE.match(os.path.basename(name))
    if m is None:
        return False
    return bool(set(m.group("lineage").split("_")) & set(_RETIRED_LINEAGE_TOKENS))


def _parse_crf(name):
    """Parse a per-exposure crf basename into its identifying fields, or None
    if it is not a well-formed crf name.  ``module`` collapses the detector to
    the physical NIRCam module (nrca/nrcb) or ``mirimage``; SW and LW share a
    module label but never share a filter, so the per-filter check keeps them
    apart.  ``obs_key`` = ``"<proposal>-<observation>"`` is the release-scoping
    key (proposal-aware, so 1182-o004 and 2221-o001 never collide)."""
    m = _CRF_RE.match(os.path.basename(name))
    if m is None:
        return None
    # the leading obs and the trailing _oOOO_ must agree; a name whose tokens
    # disagree (a product-named crf copied into a per-exposure name) is a parse
    # failure, not a coin-flip on whichever token the code happens to read
    if m.group("obs2") != m.group("obs"):
        return None
    # Retired-path products (see _RETIRED_LINEAGE_TOKENS): their FITS header and
    # their GWCS disagree by arcseconds, so pooling them makes the verdict a
    # coin-flip on which representation the reader used.
    lineage = set(m.group("lineage").split("_")) - {""}
    if lineage & set(_RETIRED_LINEAGE_TOKENS):
        return None
    det = m.group("det")
    module = "mirimage" if det == "mirimage" else det[:4]
    return dict(prop=m.group("prop"), obs=m.group("obs"), visit=m.group("visit"),
                det=det, module=module,
                obs_key=f"{m.group('prop')}-{m.group('obs')}")


def _group_key(crf_path):
    """Group per-exposure crf by (obs+visit, module).  Filename like
    jw01182004001_04101_00001_nrca3_destreak_o004_crf.fits -> '004001:nrca'."""
    p = _parse_crf(crf_path)
    if p is None:
        return "det?:det?"
    return f"{p['obs']}{p['visit']}:{p['module']}"


# regex for a merged science mosaic of EITHER instrument (nircam merged_i2d /
# miri ..._data_i2d), incl. the combined multi-observation form
# jwPPPPP-oOOO-MMM (an association of obs OOO and MMM -> both are released).
_MOSAIC_OBS_RE = re.compile(
    r"^jw(?P<prop>\d{5})-o(?P<obs>\d{3})(?:-(?P<obs2>\d{3}))?_t001_"
    r"(?:nircam|miri)_")


def _mosaic_obs_keys(name):
    """The ``"<proposal>-<observation>"`` key(s) a merged-mosaic basename
    encodes (a combined ``-oOOO-MMM`` mosaic encodes both)."""
    m = _MOSAIC_OBS_RE.match(os.path.basename(name))
    if m is None:
        return set()
    keys = {f"{m.group('prop')}-{m.group('obs')}"}
    if m.group("obs2"):
        keys.add(f"{m.group('prop')}-{m.group('obs2')}")
    return keys


# explicit sentinel: "scoping disabled -- accept every well-formed crf".
# Distinct from ``None`` (derive the scope) so a derivation that finds NOTHING
# is never silently indistinguishable from a deliberate opt-out.
NO_SCOPE = object()


def _field_observations(field, filt):
    """The ``"<proposal>-<observation>"`` keys RELEASED in ONE filter directory,
    from the merged science mosaics on disk there.  Derivation is PER FILTER
    DIRECTORY (not per field): a ``{field}/{filt}/pipeline`` dir holds one
    instrument's products, so a MIRI filter derives MIRI observations and a
    NIRCam filter derives NIRCam ones -- and a stray obs sharing a proposal-obs
    key across instruments (brick MIRI F2550W and the cloudc NIRCam crf are both
    2221 o002) can never leak between them.  ``images-merged/`` is included for
    the per-observation release layout (gc2211's o050 lives there).

    Matches ONLY the canonical science-mosaic form
    ``..._t001_<instr>_clear-<filt>-...`` -- a shared dir also holds
    NON-canonical stray products (the cloudc combined
    ``jw02221-o002_t001_nircam_f405n-f444w_i2d.fits`` has no ``clear`` and is not
    a release mosaic); matching those would re-derive the stray's observation."""
    obs = set()
    fl = filt.lower()
    for pat in (f"{BASE}/{field}/{filt}/pipeline/jw*-o*_t001_nircam_clear-{fl}-*_i2d.fits",
                f"{BASE}/{field}/{filt}/pipeline/jw*-o*_t001_miri_clear-{fl}-*_i2d.fits",
                f"{BASE}/{field}/images-merged/jw*-o*_t001_nircam_clear-{fl}-*_i2d.fits",
                f"{BASE}/{field}/images-merged/jw*-o*_t001_miri_clear-{fl}-*_i2d.fits"):
        for p in glob.glob(pat):
            obs |= _mosaic_obs_keys(p)
    return obs


def build_groups(field, filt, observations=None):
    """Detect on each per-exposure crf and pool by (obs+visit, module).

    Scoping (excludes stray crf from other programs/observations in a shared
    target dir -- the brick dir also holds 2221 o002 = cloudc crf):

    - ``observations=None`` (default): scope to the RELEASED observations derived
      from THIS filter directory's mosaics (``_field_observations``);
    - an explicit iterable of ``"<proposal>-<observation>"`` keys: RESTRICT the
      derived scope to their intersection (restrict-only -- a broad field-level
      set, e.g. one that also lists a MIRI observation, can never RE-ADMIT a
      stray that the per-directory derivation already excluded);
    - ``NO_SCOPE``: disable scoping, accept every well-formed crf.

    A derivation that yields nothing (no mosaic on disk) is announced loudly and
    leaves scoping OFF for that filter -- on a shared target dir that is exactly
    when a stray would slip in, so it must not pass silently.
    """
    if observations is NO_SCOPE:
        scope = None
    else:
        derived = _field_observations(field, filt)
        if observations is None:
            scope = derived or None
            if scope is None:
                print(f"  WARNING: {field}/{filt}: could not derive released "
                      f"observations (no mosaic on disk) -- scoping DISABLED; a "
                      f"stray crf from another observation could pollute this "
                      f"verdict", flush=True)
        else:
            passed = set(observations)
            # restrict-only: intersect with the per-directory derivation when it
            # exists; fall back to the passed set only when nothing was derived
            scope = (derived & passed) if derived else passed
    # Enumerate broadly, then let the precise _parse_crf regex decide -- a name
    # that is not a well-formed crf, or belongs to an out-of-scope observation,
    # is dropped.  MIRI crf carry no _destreak token; the parser covers them
    # (excluding MIRI once silently PASSED the F2550W doubled-star saga).
    frames = []
    n_retired = 0
    for fn in sorted(glob.glob(f"{BASE}/{field}/{filt}/pipeline/jw*_crf.fits")):
        p = _parse_crf(fn)
        if p is None:
            # _parse_crf returns None both for "not a well-formed crf name" and
            # for "deliberately excluded retired-path product".  Count the second
            # so a silently smaller frame set stays distinguishable from an empty
            # directory: check_filter fails closed on nframes == 0, but a PARTIAL
            # exclusion (a visit losing all its frames) would pass with fewer
            # groups and no trace.
            if _retired_lineage(fn):
                n_retired += 1
            continue
        if scope is not None and p["obs_key"] not in scope:
            continue
        frames.append(fn)
    if n_retired:
        print(f"  {field}/{filt}: excluded {n_retired} retired-path crf "
              f"(realign_to_vvv lineage; FITS header and GWCS disagree by "
              f"arcseconds in those files)", flush=True)
    groups = {}
    ndet = {}
    for fn in frames:
        s = _detect(fn)
        if s is None:
            continue
        k = _group_key(fn)
        groups.setdefault(k, []).append(s)
        ndet[k] = ndet.get(k, 0) + len(s)
    pooled = {k: SkyCoord(np.concatenate([c.ra.deg for c in v]) * u.deg,
                          np.concatenate([c.dec.deg for c in v]) * u.deg)
              for k, v in groups.items()}
    return pooled, ndet, len(frames)


def _refcat(path):
    t = Table.read(path)
    cols = {c.lower(): c for c in t.colnames}
    ra, dec = t[cols["ra"]], t[cols["dec"]]
    rc = SkyCoord(ra.data * u.deg if ra.unit is None else ra,
                  dec.data * u.deg if dec.unit is None else dec)
    src = cols.get("source")
    gaia = None
    if src is not None:
        gm = np.array([s in (b"GaiaDR3", "GaiaDR3") for s in t[src]])
        gaia = rc[gm] if gm.any() else None
    return rc, gaia


def check_filter(field, filt, refcat=None, verbose=True, observations=None):
    pooled, ndet, nframes = build_groups(field, filt, observations=observations)
    # FAIL-CLOSED on "found nothing": a gate that goes green because its glob
    # matched zero files (renamed products, naming drift) is the silent
    # false-agreement class this repo bans.  Distinguish it from a genuine
    # single-group field (frames found, nothing to pairwise-check).
    if nframes == 0:
        if verbose:
            print(f"  {field} {filt}: NO crf frames matched -- cannot verify "
                  f"inter-frame registration (glob mismatch / missing products?)",
                  flush=True)
        return dict(field=field, filt=filt, PASS=False, could_not_verify=True,
                    note="no crf frames matched")
    if not pooled:
        if verbose:
            print(f"  {field} {filt}: {nframes} crf but detection produced NO "
                  f"usable groups -- cannot verify", flush=True)
        return dict(field=field, filt=filt, PASS=False, could_not_verify=True,
                    note="no detections from any crf")
    if len(pooled) < 2:
        if verbose:
            print(f"  {field} {filt}: single exposure-group from {nframes} crf "
                  f"-- nothing to overlap-check", flush=True)
        return dict(field=field, filt=filt, PASS=True,
                    note=f"single exposure-group ({nframes} crf)")

    # PER-TILE (local) reference-free check: a per-visit residual is spatially
    # varying, so a field-pooled single offset can average below tol while a thin
    # seam is ~90 mas off. Grid it.
    # TWO-LAYER frame-vs-frame check, both reference-free:
    #   fine  : per-tile grid on mutual-coverage cells (owns the 30-100 mas
    #           seam regime; blind BY CONSTRUCTION to offsets > its cell
    #           margin, which empty the mutual-coverage cells);
    #   gross : field-pooled SWEPT histogram over the intersection populations
    #           (owns the >margin regime -- the brick-1182 v001 ~20" class --
    #           a gross rigid offset still overlaps in the strip, so the swept
    #           peak recovers it with no reference catalog).
    # A pair verifies only if a layer POSITIVELY measured it; a pair NEITHER
    # layer can measure is could-not-verify -> exit 2 (fail closed: "a
    # failsafe that cannot run is not a passing failsafe").
    res = overlap_offset_grid(pooled, tol_mas=TOL_MAS, nx=GRID_N, ny=GRID_N,
                              maxsep=3.0 * u.arcsec)
    pw = {(r["a"], r["b"]): r for r in pairwise_overlap_offsets(
        pooled, tol_mas=TOL_MAS, maxsep=3.0 * u.arcsec)}
    GROSS_MAS = 3.0 * 3000.0  # beyond the grid's ~3*maxsep cell margin
    bad, unverifiable, overlapped = [], [], []
    for r in res:
        if not r["overlap"]:
            continue
        overlapped.append(r)
        p = pw.get((r["a"], r["b"])) or pw.get((r["b"], r["a"])) or {}
        r["pairwise"] = {k: p.get(k) for k in
                         ("off_mas", "dra_mas", "ddec_mas", "contrast",
                          "n_peak", "measurable", "swept", "ok")}
        if r.get("could_not_verify"):
            # fine layer blind -> the gross layer must decide
            if p.get("measurable"):
                if not p["ok"]:
                    r["fail_reason"] = (f"gross pairwise offset "
                                        f"{p['off_mas']:.0f} mas (swept="
                                        f"{p.get('swept')})")
                    bad.append(r)
            else:
                unverifiable.append(r)
        else:
            if not r["ok"]:
                r["fail_reason"] = (r.get("fail_reason")
                                    or "per-tile misregistration")
                bad.append(r)
            elif (p.get("measurable") and p.get("off_mas") is not None
                    and p["off_mas"] > GROSS_MAS):
                # tiles measured fine locally but the pooled swept peak sits
                # beyond the grid's sight -- gross regime, fail
                r["fail_reason"] = f"gross pairwise offset {p['off_mas']:.0f} mas"
                bad.append(r)
    if verbose:
        print(f"  {field} {filt}: {nframes} crf -> {len(pooled)} groups, "
              f"{len(overlapped)} overlapping pairs, {len(bad)} FAIL, "
              f"{len(unverifiable)} could-not-verify (tol {TOL_MAS:.0f} mas, "
              f"{GRID_N}x{GRID_N} tiles + pooled swept)", flush=True)
        for r in sorted(overlapped, key=lambda r: -(r["worst_off_mas"] or 0))[:8]:
            tag = ("FAIL" if r in bad
                   else ("could-not-verify" if r in unverifiable else "ok"))
            _w = r["worst_off_mas"]
            _p = r.get("pairwise", {})
            print(f"      {tag}: {r['a']} | {r['b']}  worst tile off="
                  f"{'n/a' if _w is None else f'{_w:.0f} mas'} "
                  f"({r['n_ok']}/{r['n_total']} tiles ok, "
                  f"{r.get('n_no_coverage', 0)} no-mutual-coverage cells; "
                  f"pooled off={'n/a' if _p.get('off_mas') is None else f'{_p['off_mas']:.0f} mas'}"
                  f"{' MEASURABLE' if _p.get('measurable') else ''}"
                  f"{'; ' + r['fail_reason'] if r.get('fail_reason') else ''})",
                  flush=True)
        for r in unverifiable:
            print(f"      COULD NOT VERIFY: {r['a']} | {r['b']} -- footprints "
                  f"intersect but neither the per-tile grid (no mutual-coverage "
                  f"cells) nor the pooled swept histogram (no measurable peak) "
                  f"could measure the pair.  Fail-closed: requires the external "
                  f"reference map (--refcat) or a fixed reduction to stage.",
                  flush=True)

    ext_fail = False
    ext_ran = False
    if refcat:
        rc, gaia = _refcat(refcat)
        allsrc = SkyCoord(np.concatenate([p.ra.deg for p in pooled.values()]) * u.deg,
                          np.concatenate([p.dec.deg for p in pooled.values()]) * u.deg)
        # GC reference-frame policy (gc-gaia-frame-not-catalog): VIRAC2 is the GC
        # reference catalog and the ONLY absolute frame that GATES here. Gaia is
        # the FRAME but far too sparse (~1.8k over the Brick vs ~113k VIRAC2) to
        # per-tile a dense field -- on a fine grid its cells are star-starved and
        # false-fail, so it is measured as a DIAGNOSTIC only and never sets
        # ext_fail. A real absolute-frame problem shows up against dense VIRAC2.
        for rn, rr, gates in (("VIRAC2", rc, True), ("Gaia", gaia, False)):
            if rr is None:
                continue
            g = measure_offset_grid(allsrc, rr, nx=GRID_N, ny=GRID_N,
                                    maxsep=3.0 * u.arcsec, max_off_mas=GRID_MAX_OFF_MAS,
                                    context=f"{field}/{filt}/{rn}")
            if gates:
                ext_ran = True
            if verbose:
                tag = "" if gates else " [diagnostic, non-gating: Gaia too sparse]"
                print(f"      fine grid {GRID_N}x{GRID_N} vs {rn}: clean={g['clean']} "
                      f"worst_off={g['worst_off_mas']:.0f} mas "
                      f"(max {GRID_MAX_OFF_MAS:.0f}) n_ok={g['n_ok']}/{g['n_total']}"
                      f"{tag}", flush=True)
            if gates and not g["clean"]:
                ext_fail = True

    # FAIL-CLOSED: a pair NEITHER frame-vs-frame layer could measure only
    # passes when the external reference map ran AND is clean; otherwise the
    # filter is could-not-verify (exit 2 -- refused by stage_release).
    could_not_verify = bool(unverifiable) and not (ext_ran and not ext_fail)
    if unverifiable and ext_ran and not ext_fail and verbose:
        print(f"      could-not-verify pair(s) accepted via the CLEAN external "
              f"reference map", flush=True)
    return dict(field=field, filt=filt,
                PASS=bool(not bad and not ext_fail and not could_not_verify),
                could_not_verify=could_not_verify,
                n_fail=len(bad), pairs=res)


def field_filters(field):
    fs = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                 for p in glob.glob(f"{BASE}/{field}/*/pipeline/")})
    return [f for f in fs if f.upper().startswith("F")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--scan", action="store_true", help="every filter of the field")
    ap.add_argument("--refcat", default=None,
                    help="optional external refcat for the fine-grid absolute cross-check")
    ap.add_argument("--observations", default=None,
                    help="csv of <proposal>-<observation> keys (e.g. "
                         "02221-001,01182-004) to scope the check to; stray crf "
                         "from other programs/observations in a shared target dir "
                         "are excluded.  Default: the field's released "
                         "observations (derived from the merged mosaics on disk).")
    args = ap.parse_args(argv)

    filts = field_filters(args.field) if args.scan else [args.filter]
    if not filts or filts == [None]:
        print("ERROR: give --filter or --scan", file=sys.stderr)
        return 2
    any_fail = False
    any_noverify = False
    for f in filts:
        r = check_filter(args.field, f, refcat=args.refcat,
                         observations=(set(args.observations.split(","))
                                       if args.observations else None))
        if r.get("could_not_verify"):
            any_noverify = True
        elif not r.get("PASS"):
            any_fail = True
    if any_fail:
        print(f"\nOVERLAP GATE: FAIL for {args.field} -- inter-frame misregistration "
              f"(> {TOL_MAS:.0f} mas). Do NOT stage; re-examine per-visit alignment.",
              flush=True)
        return 1
    if any_noverify:
        # exit 2 = could-not-verify: distinct from a measured FAIL, but still
        # refused by stage_release (its rc != 0 branch) -- fail closed, never
        # green-because-the-glob-matched-nothing.
        print(f"\nOVERLAP GATE: COULD NOT VERIFY {args.field} -- at least one filter "
              f"had no matchable crf frames / no detections. Fix the products or the "
              f"glob; a gate that finds nothing is not a passing gate.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
