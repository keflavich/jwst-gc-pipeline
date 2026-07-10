#!/usr/bin/env python
"""
Curate and stage final JWST-GC pipeline products into a fixed-in-time release tree
for distribution via the ``JWST root`` Globus guest collection.

The pipeline working directories contain many intermediate products (per-exposure,
per-module, and per-merge-iteration files).  This script discovers the *canonical*
deliverables for a field -- the plain science mosaics, the highest-available merge
iteration of the residual/model images, and the final merged photometry catalogs --
and stages them into

    <release_root>/<version>/<field>/
        images/<FILT>/   science i2d + residual/model i2d (highest iteration)
        catalogs/        field-wide merged catalog (full + quality-cut) + seed
                         + per-filter vetted catalogs
        README.md
        MANIFEST.json    machine-readable list of every staged file w/ provenance
        CHECKSUMS.sha256

Default mode is a dry run that only prints the manifest.  Use ``--stage`` to build
the tree (symlinks by default; ``--copy`` for a frozen, source-independent release),
``--set-acl`` to grant public read on the Globus collection, and ``--print-urls`` to
emit the HTTPS download URLs.

Globus collection (the jwst endpoint):
    name            JWST root
    collection id   d9873d5e-0fbd-4980-aedf-4ca56f65a045  (guest, POSIX, Public)
    root maps to    /orange/adamginsburg/jwst/
    HTTPS base      https://g-92a536.55ba.08cc.data.globus.org
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- Globus collection constants ---------------------------------------------
GLOBUS_COLLECTION_ID = "d9873d5e-0fbd-4980-aedf-4ca56f65a045"
GLOBUS_COLLECTION_ROOT = Path("/orange/adamginsburg/jwst")
GLOBUS_HTTPS_BASE = "https://g-92a536.55ba.08cc.data.globus.org"
# modern (v3) globus CLI, already logged in as adamginsburg@ufl.edu
GLOBUS_CLI = "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/globus"

# --- per-field configuration -------------------------------------------------
# Add fields here as their pipelines complete.  proposal_prefix is the leading
# token of the merged-mosaic filenames (e.g. jw05365-o001_t001_nircam_...).
FIELDS = {
    # proposal_prefix may be a single string or a list (fields whose filters
    # span multiple observations, e.g. brick = JWST 1182 wide + 2221 med/narrow).
    "sgrb2": {
        "data_dir": Path("/orange/adamginsburg/jwst/sgrb2"),
        "proposal_prefix": "jw05365-o001_t001_nircam_clear",
        # MIRI science mosaics (explicit; not auto-discovered). Use the full
        # combined (o002-998) mosaics, not the per-observation i2d.
        "miri": [
            {"filter": "F770W",
             "src": "/orange/adamginsburg/jwst/sgrb2/F770W/pipeline/jw05365-o002-998_t001_miri_clear-f770w-mirimage_data_i2d.fits"},
            {"filter": "F1280W",
             "src": "/orange/adamginsburg/jwst/sgrb2/F1280W/pipeline/jw05365-o002-998_t001_miri_clear-f1280w-mirimage_data_i2d.fits"},
            {"filter": "F2550W",
             "src": "/orange/adamginsburg/jwst/sgrb2/F2550W/pipeline/jw05365-o002-998_t001_miri_clear-f2550w-mirimage_data_i2d.fits"},
        ],
    },
    "cloudc": {
        "data_dir": Path("/orange/adamginsburg/jwst/cloudc"),
        "proposal_prefix": "jw02221-o002_t001_nircam_clear",
    },
    "sgrc": {
        "data_dir": Path("/orange/adamginsburg/jwst/sgrc"),
        "proposal_prefix": "jw04147-o012_t001_nircam_clear",
    },
    "brick": {
        "data_dir": Path("/orange/adamginsburg/jwst/brick"),
        "proposal_prefix": ["jw01182-o004_t001_nircam_clear",
                            "jw02221-o001_t001_nircam_clear"],
        "miri": [
            {"filter": "F2550W",
             "src": "/orange/adamginsburg/jwst/brick/images/jw02221-o002_t001_miri_f2550w_i2d.fits"},
        ],
    },
    # gc2211: multi-pointing / multi-epoch (JWST 2211). Each observation is a
    # distinct pointing (o023 & o049 are repeat epochs of one position; o028/
    # o046/o050 are separate positions). Only o023 & o050 have m7 images; the
    # others (o028/o046/o049) are still mid-pipeline and are held for a later
    # release. Images are laid out per observation under images/<obs>/.
    "gc2211": {
        "data_dir": Path("/orange/adamginsburg/jwst/gc2211"),
        "proposal_prefix": "jw02211",
        "observations": ["o023", "o050"],
    },
    # sickle: NIRCam science mosaics + MIRI; catalogs still in progress so they are
    # NOT shipped yet (skip_catalogs). NIRCam is single-module (nrcb only), so the
    # mosaics are listed explicitly (no_auto_images): F210M has a canonical -merged_,
    # the rest are -nrcb_. No NIRCam residual/model (cataloging ongoing).
    "sickle": {
        "data_dir": Path("/orange/adamginsburg/jwst/sickle"),
        "proposal_prefix": "jw03958-o007_t001_nircam_clear",
        "no_auto_images": True,
        "skip_catalogs": True,
        "nircam": [
            {"filter": "F187N",
             "src": "/orange/adamginsburg/jwst/sickle/F187N/pipeline/jw03958-o007_t001_nircam_clear-f187n-nrcb_i2d.fits"},
            {"filter": "F210M",
             "src": "/orange/adamginsburg/jwst/sickle/F210M/pipeline/jw03958-o007_t001_nircam_clear-f210m-merged_i2d.fits"},
            {"filter": "F335M",
             "src": "/orange/adamginsburg/jwst/sickle/F335M/pipeline/jw03958-o007_t001_nircam_clear-f335m-nrcb_i2d.fits"},
            {"filter": "F470N",
             "src": "/orange/adamginsburg/jwst/sickle/F470N/pipeline/jw03958-o007_t001_nircam_clear-f470n-nrcb_i2d.fits"},
            {"filter": "F480M",
             "src": "/orange/adamginsburg/jwst/sickle/F480M/pipeline/jw03958-o007_t001_nircam_clear-f480m-nrcb_i2d.fits"},
        ],
        "miri": [
            {"filter": "F770W",
             "src": "/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958-o001-002_t001_miri_clear-f770w-mirimage_data_i2d.fits"},
            {"filter": "F1130W",
             "src": "/orange/adamginsburg/jwst/sickle/F1130W/pipeline/jw03958-o001-002_t001_miri_clear-f1130w-mirimage_data_i2d.fits"},
            {"filter": "F1500W",
             "src": "/orange/adamginsburg/jwst/sickle/F1500W/pipeline/jw03958-o001-002_t001_miri_clear-f1500w-mirimage_data_i2d.fits"},
        ],
    },
    # --- Galactic Plane fields (grouped under <version>/galactic_plane/) -------
    # These are NOT Galactic Center fields; they live in a separate group folder
    # both on disk and on the webpage. Standard single-pointing pipeline layout.
    # w51: SF complex (JWST 6151). Field-wide m7 merged catalog ready.
    "w51": {
        "data_dir": Path("/orange/adamginsburg/jwst/w51"),
        "proposal_prefix": "jw06151-o001_t001_nircam_clear",
        "group": "galactic_plane",
    },
    # wd1: Westerlund 1 (JWST 1905). Per-filter m7 ready, but the field-wide
    # merged catalog has not been built yet -> ships images + per-filter vetted
    # only until the merge step runs.
    "wd1": {
        "data_dir": Path("/orange/adamginsburg/jwst/wd1"),
        "proposal_prefix": "jw01905-o001_t001_nircam_clear",
        "group": "galactic_plane",
    },
    # wd2: Westerlund 2 (JWST 3523). Per-filter m7 ready (17 filters), field-wide
    # merged catalog not yet built -> images + per-filter vetted only for now.
    "wd2": {
        "data_dir": Path("/orange/adamginsburg/jwst/wd2"),
        "proposal_prefix": "jw03523-o005_t001_nircam_clear",
        "group": "galactic_plane",
    },
}


def field_release_dir(field, version, release_root):
    """Release directory for a field: ``<release_root>/<version>/[<group>/]<field>``.
    Fields with a ``group`` in their config are nested under that group folder
    (e.g. galactic_plane) to keep them separate from the Galactic Center fields."""
    base = Path(release_root) / version
    group = FIELDS.get(field, {}).get("group")
    if group:
        base = base / group
    return base / field

# Filter subdirectories live directly under the field directory.
FILTER_DIR_RE = re.compile(r"^F\d{3,4}[WMN]$")


def iteration_rank(token):
    """Rank a merge-iteration token so the highest/best sorts largest.

    Tokens seen, in increasing order of quality:
        m2 < m3 < m4 < resbgsub_m5 < resbgsub_m6 < resbgsub_m7

    Ranking = 10*N + (1 if resbgsub else 0), so resbgsub_m5 (51) > m4 (40).
    Returns None if the token is not a recognized iteration.
    """
    match = re.fullmatch(r"(resbgsub_)?m(\d+)", token)
    if match is None:
        return None
    resbgsub, number = match.group(1), int(match.group(2))
    return number * 10 + (1 if resbgsub else 0)


# image filename: <prefix>-<filt>-merged_<iter>_daophot_basic_mergedcat_<kind>_i2d.fits
IMAGE_RE = re.compile(
    r"-(?P<filt>f\d{3,4}[wmn])-merged_(?P<iter>(?:resbgsub_)?m\d+)"
    r"_daophot_basic_mergedcat_(?P<kind>residual|model)_i2d\.fits$"
)


def _collect_images(pipeline, prefixes, filt, observation=None):
    """Science + highest-iteration residual/model for one filter under the
    given prefix(es).  ``observation`` tags multi-pointing items."""
    items = []
    science = None
    for prefix in prefixes:
        cand = pipeline / f"{prefix}-{filt}-merged_i2d.fits"
        if cand.is_file():
            science = cand
            break
    if science is not None:
        items.append({
            "category": "image", "kind": "science", "filter": filt.upper(),
            "iteration": None, "observation": observation,
            "instrument": "NIRCam", "src": str(science),
        })

    best = {"residual": None, "model": None}  # kind -> (rank, path, iter)
    for prefix in prefixes:
        for path in pipeline.glob(f"{prefix}-{filt}-merged_*_i2d.fits"):
            name = path.name
            if "smoothed_bg" in name:
                continue
            match = IMAGE_RE.search(name)
            if match is None:
                continue
            rank = iteration_rank(match.group("iter"))
            if rank is None:
                continue
            kind = match.group("kind")
            current = best[kind]
            if current is None or rank > current[0]:
                best[kind] = (rank, path, match.group("iter"))
    for kind in ("residual", "model"):
        if best[kind] is not None:
            _, path, iteration = best[kind]
            items.append({
                "category": "image", "kind": kind, "filter": filt.upper(),
                "iteration": iteration, "observation": observation,
                "instrument": "NIRCam", "src": str(path),
            })
    return items


def discover_miri(field_cfg):
    """MIRI science mosaics are listed explicitly per field (they vary in
    location/naming and quality, so they are curated by hand, not auto-found)."""
    items = []
    for entry in field_cfg.get("miri", []):
        src = Path(entry["src"])
        if not src.is_file():
            continue
        items.append({
            "category": "image", "kind": "science",
            "filter": entry["filter"].upper(), "iteration": None,
            "observation": entry.get("observation"), "instrument": "MIRI",
            "src": str(src),
        })
    return items


def discover_nircam(field_cfg):
    """Explicitly-listed NIRCam science mosaics (``nircam`` config key, same shape
    as ``miri``).  Use when the auto-discovered ``<prefix>-<filt>-merged_i2d.fits``
    naming does not apply -- e.g. single-module (nrcb-only) fields whose mosaic is
    ``...-<filt>-nrcb_i2d.fits``.  Routed to images/<FILTER>/ like any NIRCam image."""
    items = []
    for entry in field_cfg.get("nircam", []):
        src = Path(entry["src"])
        if not src.is_file():
            continue
        items.append({
            "category": "image", "kind": "science",
            "filter": entry["filter"].upper(), "iteration": None,
            "observation": entry.get("observation"), "instrument": "NIRCam",
            "src": str(src),
        })
    return items


def discover_images(field_cfg):
    """Return image deliverable dicts for a field.

    Single-pointing fields: per filter, the plain science mosaic plus the
    highest-iteration residual/model (full-field ``-merged_`` only; per-module
    and ``smoothed_bg`` variants excluded; ``proposal_prefix`` may be a list to
    span observations, e.g. brick).

    Multi-pointing fields (``observations`` in config): the same, but per
    (observation, filter), with each observation's prefix ``<prefix>-<obs>...``;
    items are tagged with their observation and laid out under images/<obs>/.
    """
    data_dir = field_cfg["data_dir"]
    base_prefix = field_cfg["proposal_prefix"]
    observations = field_cfg.get("observations")

    filter_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and FILTER_DIR_RE.match(d.name)
    )

    items = []
    for fdir in filter_dirs:
        filt = fdir.name.lower()
        pipeline = fdir / "pipeline"
        if not pipeline.is_dir():
            continue
        if observations:
            for obs in observations:
                prefixes = [f"{base_prefix}-{obs}_t001_nircam_clear"]
                items += _collect_images(pipeline, prefixes, filt, observation=obs)
        else:
            prefixes = base_prefix if isinstance(base_prefix, list) else [base_prefix]
            items += _collect_images(pipeline, prefixes, filt)
    return items


CAT_BASE = "basic_merged_indivexp_photometry_tables_merged"
# combined (all-pointings) merged table; the (?!_o\d) guard keeps per-pointing
# "..._m7_o023.fits" variants OUT of the combined match.
COMBINED_RE = re.compile(
    rf"^{re.escape(CAT_BASE)}_(?P<iter>(?:resbgsub_)?m\d+)"
    rf"(?P<qc>_qualcuts_oksep2221)?\.(?P<ext>fits|ecsv)$"
)
# per-pointing merged table: "..._m7_o023.fits", "..._m7_o023_qualcuts...fits"
PERPOINT_RE = re.compile(
    rf"^{re.escape(CAT_BASE)}_(?P<iter>(?:resbgsub_)?m\d+)_(?P<obs>o\d+)"
    rf"(?P<qc>_qualcuts_oksep2221)?\.(?P<ext>fits|ecsv)$"
)
# per-filter vetted, optionally per-pointing (excludes *_vetted_carta.fits)
VETTED_RE = re.compile(
    r"^(?P<filt>f\d{3,4}[wmn])_merged_indivexp_merged_"
    r"(?P<iter>(?:resbgsub_)?m\d+)_dao_basic(?:_(?P<obs>o\d+))?_vetted\.fits$"
)
# quality floor: do not ship per-filter vetted catalogs below this iteration
# (anything earlier is still mid-pipeline / draft). resbgsub_m5 == rank 51.
MIN_VETTED_RANK = 51


def _emit_table_group(items, entry, observation):
    for key, kind in (("full_fits", "catalog_full"),
                      ("full_ecsv", "catalog_full"),
                      ("qualcuts", "catalog_qualcut")):
        if key in entry:
            items.append({
                "category": "catalog", "kind": kind, "filter": None,
                "iteration": entry["iter"], "observation": observation,
                "src": str(entry[key]),
            })


def discover_catalogs(field_cfg, field):
    """Return catalog deliverable dicts: the combined merged table (full +
    quality-cut), the seed catalog, per-filter vetted catalogs, and -- for
    multi-pointing fields -- the per-pointing merged tables and vetted catalogs.
    Highest merge iteration is selected in each group."""
    cat_dir = field_cfg["data_dir"] / "catalogs"
    observations = field_cfg.get("observations")
    items = []
    if not cat_dir.is_dir():
        return items

    # combined (all-pointings) merged table -- highest iteration
    combined = {}  # rank -> {iter, full_fits, full_ecsv, qualcuts}
    for path in cat_dir.glob(f"{CAT_BASE}_*"):
        m = COMBINED_RE.match(path.name)
        if m is None:
            continue
        rank = iteration_rank(m.group("iter"))
        if rank is None:
            continue
        entry = combined.setdefault(rank, {"iter": m.group("iter")})
        slot = "qualcuts" if m.group("qc") else f"full_{m.group('ext')}"
        entry[slot] = path
    if combined:
        _emit_table_group(items, combined[max(combined)], None)

    # per-pointing merged tables (multi-pointing fields) -- highest iter per obs
    if observations:
        per_obs = {}  # obs -> {rank -> entry}
        for path in cat_dir.glob(f"{CAT_BASE}_*"):
            m = PERPOINT_RE.match(path.name)
            if m is None or m.group("obs") not in observations:
                continue
            rank = iteration_rank(m.group("iter"))
            if rank is None:
                continue
            entry = per_obs.setdefault(m.group("obs"), {}).setdefault(
                rank, {"iter": m.group("iter")})
            slot = "qualcuts" if m.group("qc") else f"full_{m.group('ext')}"
            entry[slot] = path
        for obs in sorted(per_obs):
            ranks = per_obs[obs]
            _emit_table_group(items, ranks[max(ranks)], obs)

    # seed catalog
    seed = cat_dir / f"seed_union_iter3_{field}.fits"
    if seed.is_file():
        items.append({
            "category": "catalog", "kind": "seed", "filter": None,
            "iteration": "iter3", "observation": None, "src": str(seed),
        })

    # per-filter vetted catalogs -- highest iteration per (filter, observation)
    best_pf = {}  # (filt, obs) -> (rank, path, iter)
    for path in cat_dir.glob("*_dao_basic*_vetted.fits"):
        m = VETTED_RE.match(path.name)
        if m is None:
            continue
        obs = m.group("obs")
        if obs is not None and (not observations or obs not in observations):
            continue
        rank = iteration_rank(m.group("iter"))
        if rank is None or rank < MIN_VETTED_RANK:
            continue
        key = (m.group("filt"), obs)
        current = best_pf.get(key)
        if current is None or rank > current[0]:
            best_pf[key] = (rank, path, m.group("iter"))
    for (filt, obs) in sorted(best_pf, key=lambda k: (k[0], k[1] or "")):
        rank, path, iteration = best_pf[(filt, obs)]
        items.append({
            "category": "catalog", "kind": "catalog_per_filter_vetted",
            "filter": filt.upper(), "iteration": iteration, "observation": obs,
            "src": str(path),
        })

    return items


def assign_dest(item, field):
    """Compute the destination path of an item relative to the field release dir."""
    src_name = Path(item["src"]).name
    if item["category"] == "image":
        if item.get("instrument") == "MIRI":
            return Path("images") / "MIRI" / item["filter"] / src_name
        obs = item.get("observation")
        if obs:
            return Path("images") / obs / item["filter"] / src_name
        return Path("images") / item["filter"] / src_name
    # catalogs stay flat; per-pointing filenames already carry the _oNNN tag
    return Path("catalogs") / src_name


def build_manifest(field, version):
    field_cfg = FIELDS[field]
    items = []
    # auto-discovered per-filter NIRCam mosaics (skip with no_auto_images, e.g.
    # fields whose NIRCam mosaics are listed explicitly via the `nircam` key)
    if not field_cfg.get("miri_only") and not field_cfg.get("no_auto_images"):
        items += discover_images(field_cfg)
    items += discover_nircam(field_cfg)   # explicit NIRCam list (if any)
    items += discover_miri(field_cfg)
    # catalogs: skip while cataloging is still in progress (skip_catalogs)
    if not field_cfg.get("miri_only") and not field_cfg.get("skip_catalogs"):
        items += discover_catalogs(field_cfg, field)
    for item in items:
        src = Path(item["src"])
        item["dest"] = str(assign_dest(item, field))
        item["size_bytes"] = src.stat().st_size if src.is_file() else None
        # per-file version: defaults to the field release version so every file carries
        # an explicit version on the download page. A file bumped independently (e.g. a
        # re-tied mosaic staged into an otherwise-older release) can override this.
        item.setdefault("version", version)
    return items


def sha256sum(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def human_size(num_bytes):
    if num_bytes is None:
        return "?"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024


def print_manifest(items):
    print(f"\n{'CATEGORY':<9} {'KIND':<26} {'FILT':<6} {'ITER':<14} {'SIZE':>8}  SRC")
    print("-" * 110)
    for it in items:
        print(f"{it['category']:<9} {it['kind']:<26} "
              f"{(it['filter'] or ''):<6} {(it['iteration'] or ''):<14} "
              f"{human_size(it['size_bytes']):>8}  {it['src']}")
    total = sum(it["size_bytes"] or 0 for it in items)
    print("-" * 110)
    print(f"{len(items)} files, total {human_size(total)}\n")


def stage(items, field, version, release_root, mode, do_checksum):
    field_dir = field_release_dir(field, version, release_root)
    field_dir.mkdir(parents=True, exist_ok=True)

    # reuse checksums from a prior manifest for files that are unchanged, so
    # re-staging (e.g. to add one MIRI mosaic) doesn't re-hash tens of GB.
    prior = {}  # dest -> (size, sha256)
    manifest_path = field_dir / "MANIFEST.json"
    if manifest_path.is_file():
        for f in json.loads(manifest_path.read_text()).get("files", []):
            if "sha256" in f and f.get("size_bytes") is not None:
                prior[f["dest"]] = (f["size_bytes"], f["sha256"])

    checksum_lines = []
    for it in items:
        src = Path(it["src"]).resolve()
        dest = field_dir / it["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        src_size = src.stat().st_size
        unchanged = (mode == "copy" and dest.is_file()
                     and not dest.is_symlink()
                     and dest.stat().st_size == src_size)
        if not unchanged:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            if mode == "copy":
                shutil.copy2(src, dest)
            else:
                dest.symlink_to(src)

        if do_checksum:
            cached = prior.get(it["dest"])
            if unchanged and cached and cached[0] == src_size:
                digest = cached[1]  # reuse; content unchanged
            else:
                digest = sha256sum(src)
            it["sha256"] = digest
            checksum_lines.append(f"{digest}  {it['dest']}")

    # release-relative globus path and URL for each item
    for it in items:
        rel_to_collection = (field_dir / it["dest"]).relative_to(GLOBUS_COLLECTION_ROOT)
        it["globus_path"] = "/" + str(rel_to_collection)
        it["url"] = GLOBUS_HTTPS_BASE + it["globus_path"]

    # write MANIFEST.json
    manifest = {
        "field": field,
        "version": version,
        "group": FIELDS.get(field, {}).get("group"),
        # globus-collection-relative path of this field's release dir
        # (includes the group folder when set), e.g.
        # /releases/v1.0-2026.06/galactic_plane/w51
        "release_path": "/" + str(field_dir.relative_to(GLOBUS_COLLECTION_ROOT)),
        "built": datetime.datetime.now().astimezone().isoformat(),
        "mode": mode,
        "globus_collection_id": GLOBUS_COLLECTION_ID,
        "globus_https_base": GLOBUS_HTTPS_BASE,
        "files": items,
    }
    (field_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    if do_checksum:
        (field_dir / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n")

    write_readme(field_dir, field, version, items, mode)

    # world-readable
    subprocess.run(["chmod", "-R", "a+rX", str(field_dir)], check=True)
    return field_dir


def write_readme(field_dir, field, version, items, mode):
    images = [it for it in items if it["category"] == "image"]
    catalogs = [it for it in items if it["category"] == "catalog"]
    lines = [
        f"# JWST Galactic Center survey -- {field} -- release {version}",
        "",
        "Final reduced products from the JWST-GC photometry pipeline.",
        f"Staged {datetime.datetime.now().astimezone().isoformat()} (mode: {mode}).",
        "",
        "Files are distributed via the `JWST root` Globus guest collection;",
        "direct-download URLs are listed in `MANIFEST.json` (requires a free Globus login).",
        "",
        "## Images (`images/<FILTER>/`)",
        "",
        "- `*-merged_i2d.fits`        : science mosaic (drizzled)",
        "- `*_residual_i2d.fits`      : PSF-photometry residual (highest merge iteration)",
        "- `*_model_i2d.fits`         : PSF model image (highest merge iteration)",
        "",
        f"{len(images)} image files across "
        f"{len({it['filter'] for it in images})} filters.",
        "",
        "## Catalogs (`catalogs/`)",
        "",
        "- `basic_merged_indivexp_photometry_tables_merged_*` : final merged photometry",
        "  table (`.fits` + `.ecsv`); `_qualcuts_oksep2221` is the quality-filtered subset.",
        "- `*_dao_basic_vetted.fits` : per-filter vetted catalogs.",
        "- `seed_union_iter3_*.fits` : seed source list.",
        "",
        "## Integrity",
        "",
        "`CHECKSUMS.sha256` lists SHA-256 for every file. `MANIFEST.json` records",
        "provenance (original pipeline path, merge iteration, size, checksum, URL).",
        "",
    ]
    (field_dir / "README.md").write_text("\n".join(lines))


def set_acl(field, version, release_root):
    """Grant all-authenticated-users (free Globus login required) read on the
    field's release path."""
    field_path = field_release_dir(field, version, release_root)
    rel = "/" + str(field_path.relative_to(GLOBUS_COLLECTION_ROOT)) + "/"
    cmd = [
        GLOBUS_CLI, "endpoint", "permission", "create",
        f"{GLOBUS_COLLECTION_ID}:{rel}",
        "--permissions", "r",
        "--all-authenticated",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", default="sgrb2", choices=sorted(FIELDS))
    parser.add_argument("--version", default="v1.0-2026.06")
    parser.add_argument("--release-root",
                        default="/orange/adamginsburg/jwst/releases")
    parser.add_argument("--stage", action="store_true",
                        help="build the release tree (default: dry-run manifest only)")
    parser.add_argument("--copy", action="store_true",
                        help="copy files instead of symlinking (frozen release)")
    parser.add_argument("--no-checksum", action="store_true",
                        help="skip SHA-256 computation (faster dry staging)")
    parser.add_argument("--set-acl", action="store_true",
                        help="grant all-authenticated-users read on the release path")
    parser.add_argument("--print-urls", action="store_true",
                        help="print HTTPS download URLs (requires --stage)")
    parser.add_argument("--allow-registration-fail", action="store_true",
                        help="stage even if the local-registration failsafe FAILs "
                             "(a band is locally misregistered). DANGEROUS -- only for "
                             "deliberate overrides; ALSO requires ALLOW_REGISTRATION_FAIL=1 "
                             "in the environment. The default refuses to stage.")
    args = parser.parse_args(argv)

    items = build_manifest(args.field, args.version)
    if not items:
        print(f"No deliverables discovered for field '{args.field}'.", file=sys.stderr)
        return 1
    print_manifest(items)

    if not args.stage:
        print("Dry run. Re-run with --stage to build the release tree.")
        return 0

    # ---- LOCAL-REGISTRATION GATE ----------------------------------------------------
    # A field-average astrometry check passes over a LOCALIZED several-arcsec seam
    # misregistration in one band (brick 1182 F115W, 2026-07: 1.8" visit-seam junk,
    # bulk ~0). Before staging, run the spatially-resolved cross-band + own-catalog
    # failsafe over every band; REFUSE to stage if any band FAILs. This makes that
    # corruption unable to reach a release by construction.
    # The override is deliberately hard to reach: --allow-registration-fail ALONE is
    # not enough, it also requires ALLOW_REGISTRATION_FAIL=1 in the environment. This
    # stops an agent from flipping a red gate green with a single flag (the exact
    # failure mode that keeps letting 4" astrometry into releases).
    override = args.allow_registration_fail and os.environ.get("ALLOW_REGISTRATION_FAIL") == "1"
    if args.allow_registration_fail and not override:
        print("\nREFUSING TO STAGE: --allow-registration-fail also requires "
              "ALLOW_REGISTRATION_FAIL=1 in the environment. This override bypasses "
              "the astrometry failsafe -- only set it with a written justification.",
              file=sys.stderr)
        return 2
    if not override:
        gate = Path(__file__).with_name("registration_failsafes.py")
        rc = subprocess.run([sys.executable, str(gate), "--field", args.field, "--scan"]).returncode
        if rc == 1:
            print(f"\nREFUSING TO STAGE '{args.field}': local-registration failsafe FAILED "
                  f"-- a band's mosaic is locally misregistered vs the other bands / its own "
                  f"catalog (see the scan output above). Fix the reduction, or override with "
                  f"--allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1 (dangerous).",
                  file=sys.stderr)
            return 2
        if rc != 0:
            # Fail CLOSED: a failsafe that cannot run is NOT a passing failsafe.
            print(f"\nREFUSING TO STAGE '{args.field}': registration failsafe could not run "
                  f"(rc={rc}); cannot confirm astrometry. Fix the failsafe, or override with "
                  f"--allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1 (dangerous).",
                  file=sys.stderr)
            return 2

    mode = "copy" if args.copy else "symlink"
    field_dir = stage(items, args.field, args.version, args.release_root,
                      mode, not args.no_checksum)
    print(f"Staged {len(items)} files into {field_dir} (mode: {mode}).")

    if args.set_acl:
        set_acl(args.field, args.version, args.release_root)

    if args.print_urls:
        print("\nDownload URLs:")
        for it in items:
            print(it["url"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
