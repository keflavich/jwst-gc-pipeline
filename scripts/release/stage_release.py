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

# `release_freshness` is a sibling MODULE, not a package member, so a bare
# import only resolves when this file is run as a script from a cwd that
# happens to contain it.  Loading `stage_release` any other way -- which is
# what the release-gate tests do, via importlib against an absolute path --
# raised ModuleNotFoundError at import time and took the whole COLLECTION
# down: pytest then ran nothing at all without --continue-on-collection-errors,
# so two gate test files silently stopped protecting anything.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_freshness            # noqa: E402  (needs the path above)
import exposure_bundle              # noqa: E402  (same reason)
import astrometry_provenance        # noqa: E402  (same reason)

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
    # cloudc: NIRCam from JWST 2221 o002, and MIRI from TWO SEPARATE PROGRAMS
    # imaging NON-OVERLAPPING pointings of the same cloud -- 2221 o001 (F2550W)
    # and 2526 o021 (F770W).  Both are cloudc and both belong in its release;
    # neither is discoverable from `proposal_prefix`, which names one NIRCam
    # program.
    #
    # The MIRI observation numbering is INVERTED with respect to NIRCam within
    # 2221, and that trap is why these are spelled out rather than derived:
    #
    #     cloudc  NIRCam 2221-o002    cloudc  MIRI 2221-o001  (F2550W)
    #     brick   NIRCam 2221-o001    brick   MIRI 2221-o002  (F2550W)
    #
    # Verified by field centre rather than by name, since the names invite
    # exactly the wrong pairing:
    #
    #     cloudc NIRCam F405N   266.5872 -28.5902
    #     cloudc MIRI  F2550W   266.5695 -28.5918   <- 2221-o001, ~1' away: cloudc
    #     cloudc MIRI  F770W    266.5823 -28.6271   <- 2526-o021, ~2' S: cloudc,
    #                                                  and disjoint from F2550W
    #     brick  NIRCam F405N   266.5356 -28.7128
    #     brick  MIRI  F2550W   266.5372 -28.7066   <- 2221-o002, brick's own
    #
    # F770W is the weakest of the three on distance alone (3.7' to cloudc vs
    # 5.7' to brick, against 3.9x and 23x for the other two).  What settles it
    # is PROGRAM MEMBERSHIP: 2526 appears nowhere in brick's release, and the
    # frames are filed under cloudc/F770W/.  TARGPROP cannot separate the 2221
    # pair at all -- both read BRICK-IKP2016-G0.253+0.015 -- which is why the
    # centres are quoted rather than the header target name.
    "cloudc": {
        "data_dir": Path("/orange/adamginsburg/jwst/cloudc"),
        "proposal_prefix": "jw02221-o002_t001_nircam_clear",
        "miri": [
            {"filter": "F770W",
             "src": "/orange/adamginsburg/jwst/cloudc/F770W/pipeline/"
                    "jw02526-o021_t001_miri_clear-f770w-mirimage_data_i2d.fits"},
            {"filter": "F2550W",
             "src": "/orange/adamginsburg/jwst/cloudc/F2550W/pipeline/"
                    "jw02221-o001_t001_miri_clear-f2550w-mirimage_data_i2d.fits"},
        ],
    },
    "sgrc": {
        "data_dir": Path("/orange/adamginsburg/jwst/sgrc"),
        "proposal_prefix": "jw04147-o012_t001_nircam_clear",
    },
    # arches / quintuplet: the two GC starburst clusters (JWST 2045, o001 & o003).
    # The reduction drizzles these per module, so there is no `-merged_i2d`; the
    # canonical science products are the NRCA and NRCB mosaics (both current: the
    # last m2 offsets correction was 2026-07-31, these were drizzled 2026-08-01).
    # Cataloging is still running, so image-only. `make_preview_rgb.py` coadds the
    # two modules onto a common grid for the preview.
    "arches": {
        "data_dir": Path("/orange/adamginsburg/jwst/arches"),
        "proposal_prefix": "jw02045-o001_t001_nircam_clear",
        "no_auto_images": True, "skip_catalogs": True,
        "nircam": [
            {"filter": "F212N", "src": "/orange/adamginsburg/jwst/arches/F212N/pipeline/jw02045-o001_t001_nircam_clear-f212n-nrca_i2d.fits"},
            {"filter": "F212N", "src": "/orange/adamginsburg/jwst/arches/F212N/pipeline/jw02045-o001_t001_nircam_clear-f212n-nrcb_i2d.fits"},
            {"filter": "F323N", "src": "/orange/adamginsburg/jwst/arches/F323N/pipeline/jw02045-o001_t001_nircam_clear-f323n-nrca_i2d.fits"},
            {"filter": "F323N", "src": "/orange/adamginsburg/jwst/arches/F323N/pipeline/jw02045-o001_t001_nircam_clear-f323n-nrcb_i2d.fits"},
        ],
    },
    "quintuplet": {
        "data_dir": Path("/orange/adamginsburg/jwst/quintuplet"),
        "proposal_prefix": "jw02045-o003_t001_nircam_clear",
        "no_auto_images": True, "skip_catalogs": True,
        "nircam": [
            {"filter": "F212N", "src": "/orange/adamginsburg/jwst/quintuplet/F212N/pipeline/jw02045-o003_t001_nircam_clear-f212n-nrca_i2d.fits"},
            {"filter": "F212N", "src": "/orange/adamginsburg/jwst/quintuplet/F212N/pipeline/jw02045-o003_t001_nircam_clear-f212n-nrcb_i2d.fits"},
            {"filter": "F323N", "src": "/orange/adamginsburg/jwst/quintuplet/F323N/pipeline/jw02045-o003_t001_nircam_clear-f323n-nrca_i2d.fits"},
            {"filter": "F323N", "src": "/orange/adamginsburg/jwst/quintuplet/F323N/pipeline/jw02045-o003_t001_nircam_clear-f323n-nrcb_i2d.fits"},
        ],
    },
    # sgra: Sgr A* (JWST 1939). Image-only: F212N + F405N mosaics are current;
    # F115W's mosaics were stale-tagged (*_im0_badastrom) by the m2 astrometry
    # checkpoint on 2026-07-28 when it corrected the offsets table, so that band
    # is held until it is re-drizzled. Catalogs are not certified yet.
    # `skip_catalogs` is what ENFORCES that: without it, a re-stage that forgets
    # `--images-only` would publish six catalogs including the F115W one, built
    # against the offsets that same checkpoint corrected by 32-55 mas -- a
    # catalog with no image beside it, on a superseded solution.
    "sgra": {
        "data_dir": Path("/orange/adamginsburg/jwst/sgra"),
        "proposal_prefix": "jw01939-o001_t001_nircam_clear",
        "skip_catalogs": True,
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
    # o046/o050 are separate positions). Images are laid out per observation
    # under images/<obs>/.
    #
    # o028/o046/o049 were held as "still mid-pipeline"; they were re-drizzled
    # 2026-08-08 and measured against VIRAC2 with the swept offset-histogram
    # estimator (3" window, unswept, window_edge_fraction 0.01-0.03, so genuine
    # ties rather than window-edge aliases):
    #
    #     o023  F200W  55 mas  contrast  66     o046  F200W  34 mas  contrast 148
    #     o023  F277W  75 mas  contrast  65     o046  F277W  35 mas  contrast  91
    #     o028  F150W  59 mas  contrast  67     o049  F200W  41 mas  contrast 106
    #     o028  F277W  52 mas  contrast 158     o049  F277W  49 mas  contrast  77
    #
    # That is the re-reduction landing: o023 measured ~3.25" off before it.
    # o028 lies outside the field's main reference catalogue and needs the
    # per-pointing `gaia_virac2_refcat_epoch2023.71_o028.fits`.
    #
    # o050 is DELIBERATELY absent: it has no current mosaic at all, every
    # product of it is quarantined, and it was the worst-aligned pointing
    # (~5.6"). Re-add it here once it has been re-reduced and tied.
    "gc2211": {
        "data_dir": Path("/orange/adamginsburg/jwst/gc2211"),
        "proposal_prefix": "jw02211",
        "observations": ["o023", "o028", "o046", "o049"],
    },
    # sickle: NIRCam science mosaics + MIRI; catalogs still in progress so they are
    # NOT shipped yet (skip_catalogs). NIRCam is single-module (nrcb only), so the
    # mosaics are listed explicitly (no_auto_images) and ALL FIVE are `-nrcb_`.
    # F210M used to point at a `-merged_i2d.fits`, which on a single-module field
    # cannot be a merge -- it was a leftover from an April 2026 generation. The m2
    # checkpoint quarantined it (`..._im0_badastrom.fits`, 2026-08-05T03:29:26Z), and
    # v1.1 had already shipped it: v1.1's F210M is DATE 2026-04-19 / jwst_1535.pmap
    # while its other four bands are 2026-06-27 / jwst_1537.pmap -- a different
    # reduction generation AND a different CRDS context inside one release.
    # No NIRCam residual/model (cataloging ongoing).
    "sickle": {
        "data_dir": Path("/orange/adamginsburg/jwst/sickle"),
        "proposal_prefix": "jw03958-o007_t001_nircam_clear",
        "no_auto_images": True,
        "skip_catalogs": True,
        "nircam": [
            {"filter": "F187N",
             "src": "/orange/adamginsburg/jwst/sickle/F187N/pipeline/jw03958-o007_t001_nircam_clear-f187n-nrcb_i2d.fits"},
            {"filter": "F210M",
             "src": "/orange/adamginsburg/jwst/sickle/F210M/pipeline/jw03958-o007_t001_nircam_clear-f210m-nrcb_i2d.fits"},
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
    # ngc6334: PRIVATE (shared only with H. Bouy, not public). Two programs
    # (jw06778 + jw07213); F200W exists in both, we ship the 06778 version. Merged
    # NIRCam mosaics listed explicitly (two prefixes -> no_auto_images); image-only.
    "ngc6334": {
        "data_dir": Path("/orange/adamginsburg/jwst/ngc6334"),
        "no_auto_images": True,
        "skip_catalogs": True,
        "nircam": [
            {"filter": "F090W", "src": "/orange/adamginsburg/jwst/ngc6334/F090W/pipeline/jw06778-o001_t001_nircam_clear-f090w-merged_i2d.fits"},
            {"filter": "F115W", "src": "/orange/adamginsburg/jwst/ngc6334/F115W/pipeline/jw07213-o001_t001_nircam_clear-f115w-merged_i2d.fits"},
            {"filter": "F162M", "src": "/orange/adamginsburg/jwst/ngc6334/F162M/pipeline/jw07213-o001_t001_nircam_clear-f162m-merged_i2d.fits"},
            {"filter": "F182M", "src": "/orange/adamginsburg/jwst/ngc6334/F182M/pipeline/jw07213-o001_t001_nircam_clear-f182m-merged_i2d.fits"},
            {"filter": "F187N", "src": "/orange/adamginsburg/jwst/ngc6334/F187N/pipeline/jw06778-o001_t001_nircam_clear-f187n-merged_i2d.fits"},
            {"filter": "F200W", "src": "/orange/adamginsburg/jwst/ngc6334/F200W/pipeline/jw06778-o001_t001_nircam_clear-f200w-merged_i2d.fits"},
            {"filter": "F277W", "src": "/orange/adamginsburg/jwst/ngc6334/F277W/pipeline/jw06778-o001_t001_nircam_clear-f277w-merged_i2d.fits"},
            {"filter": "F335M", "src": "/orange/adamginsburg/jwst/ngc6334/F335M/pipeline/jw06778-o001_t001_nircam_clear-f335m-merged_i2d.fits"},
            {"filter": "F356W", "src": "/orange/adamginsburg/jwst/ngc6334/F356W/pipeline/jw07213-o001_t001_nircam_clear-f356w-merged_i2d.fits"},
            {"filter": "F405N", "src": "/orange/adamginsburg/jwst/ngc6334/F405N/pipeline/jw07213-o001_t001_nircam_clear-f405n-merged_i2d.fits"},
            {"filter": "F444W", "src": "/orange/adamginsburg/jwst/ngc6334/F444W/pipeline/jw07213-o001_t001_nircam_clear-f444w-merged_i2d.fits"},
            # F470N held out only to avoid the gate override: it is VERIFIED internally
            # consistent (0.9 mas vs the pooled LW bands, all tiles OK, contrast 357); the
            # per-cell gate false-fails it at its fine 20x20 grid because F470N is sparse
            # (886 det). Re-add once the gate no longer false-fails sparse narrow bands.
            # {"filter": "F470N", "src": "/orange/adamginsburg/jwst/ngc6334/F470N/pipeline/jw06778-o001_t001_nircam_clear-f470n-merged_i2d.fits"},
        ],
    },
    # --- Globular clusters (Anderson programs), grouped under <version>/globular_clusters/ ---
    # Public image-only; W2 filters (F150W2/F322W2) and few bands. m4 has two pointings.
    "m4": {
        "data_dir": Path("/orange/adamginsburg/jwst/m4"),
        "no_auto_images": True, "skip_catalogs": True, "group": "globular_clusters",
        "nircam": [
            {"filter": "F150W2", "observation": "o002", "src": "/orange/adamginsburg/jwst/m4/F150W2/pipeline/jw01979-o002_t001_nircam_clear-f150w2-merged_i2d.fits"},
            {"filter": "F322W2", "observation": "o002", "src": "/orange/adamginsburg/jwst/m4/F322W2/pipeline/jw01979-o002_t001_nircam_clear-f322w2-merged_i2d.fits"},
            {"filter": "F150W2", "observation": "o003", "src": "/orange/adamginsburg/jwst/m4/F150W2/pipeline/jw01979-o003_t001_nircam_clear-f150w2-merged_i2d.fits"},
            {"filter": "F322W2", "observation": "o003", "src": "/orange/adamginsburg/jwst/m4/F322W2/pipeline/jw01979-o003_t001_nircam_clear-f322w2-merged_i2d.fits"},
        ],
    },
    "m92": {
        "data_dir": Path("/orange/adamginsburg/jwst/m92"),
        "no_auto_images": True, "skip_catalogs": True, "group": "globular_clusters",
        "nircam": [
            {"filter": "F090W", "src": "/orange/adamginsburg/jwst/m92/F090W/pipeline/jw01334-o001_t001_nircam_clear-f090w-merged_i2d.fits"},
            {"filter": "F150W", "src": "/orange/adamginsburg/jwst/m92/F150W/pipeline/jw01334-o001_t001_nircam_clear-f150w-merged_i2d.fits"},
            {"filter": "F277W", "src": "/orange/adamginsburg/jwst/m92/F277W/pipeline/jw01334-o001_t001_nircam_clear-f277w-merged_i2d.fits"},
            {"filter": "F444W", "src": "/orange/adamginsburg/jwst/m92/F444W/pipeline/jw01334-o001_t001_nircam_clear-f444w-merged_i2d.fits"},
        ],
    },
    "ngc6397": {
        "data_dir": Path("/orange/adamginsburg/jwst/ngc6397"),
        "no_auto_images": True, "skip_catalogs": True, "group": "globular_clusters",
        "nircam": [
            {"filter": "F150W2", "src": "/orange/adamginsburg/jwst/ngc6397/F150W2/pipeline/jw01979-o001_t001_nircam_clear-f150w2-merged_i2d.fits"},
            {"filter": "F322W2", "src": "/orange/adamginsburg/jwst/ngc6397/F322W2/pipeline/jw01979-o001_t001_nircam_clear-f322w2-merged_i2d.fits"},
        ],
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


def _badastrom_sibling(src):
    """The quarantined twin of a vanished product, when one exists.

    The m2 astrometry checkpoint does not delete a mosaic built on superseded
    offsets -- it renames it to ``<name>_im0_badastrom.fits`` and writes a
    ``.why.json`` beside it.  So when an explicitly-listed src has gone missing,
    that sibling is the record of what happened to it, and naming it in the
    refusal is what turns "file not found" into "the checkpoint took it".
    """
    name = src.name
    suffix = "_i2d.fits"
    if name.endswith(suffix):
        sibling = src.with_name(name[:-len(suffix)] + "_i2d_im0_badastrom.fits")
    else:
        sibling = src.with_name(src.stem + "_im0_badastrom" + src.suffix)
    return sibling if sibling.is_file() else None


def _quarantine_note(sibling):
    """``(date; reason)`` from the checkpoint's ``.why.json``, or ``''``."""
    why = Path(str(sibling) + ".why.json")
    if not why.is_file():
        return ""
    try:
        record = json.loads(why.read_text())
    except (OSError, ValueError):
        return ""
    bits = [record.get("date"), record.get("reason")]
    bits = [b for b in bits if b]
    return " (" + "; ".join(bits) + ")" if bits else ""


def _missing_listed_src(instrument, entry, src):
    """One line describing an explicitly-listed source file that is not there."""
    label = f"{instrument} {entry['filter'].upper()}"
    obs = entry.get("observation")
    if obs:
        label += f"/{obs}"
    detail = f"{label}: listed src does not exist -- {src}"
    sibling = _badastrom_sibling(src)
    if sibling is not None:
        detail += (f"\n      the astrometry checkpoint quarantined it to "
                   f"{sibling.name}{_quarantine_note(sibling)}")
    return detail


def discover_miri(field_cfg, missing=None):
    """MIRI science mosaics are listed explicitly per field (they vary in
    location/naming and quality, so they are curated by hand, not auto-found).

    An entry whose src is absent is recorded in ``missing`` (when a list is
    passed) rather than being silently dropped -- see ``discover_nircam``.
    """
    items = []
    for entry in field_cfg.get("miri", []):
        src = Path(entry["src"])
        if not src.is_file():
            if missing is not None:
                missing.append(_missing_listed_src("MIRI", entry, src))
            continue
        items.append({
            "category": "image", "kind": "science",
            "filter": entry["filter"].upper(), "iteration": None,
            "observation": entry.get("observation"), "instrument": "MIRI",
            "src": str(src),
        })
    return items


def discover_nircam(field_cfg, missing=None):
    """Explicitly-listed NIRCam science mosaics (``nircam`` config key, same shape
    as ``miri``).  Use when the auto-discovered ``<prefix>-<filt>-merged_i2d.fits``
    naming does not apply -- e.g. single-module (nrcb-only) fields whose mosaic is
    ``...-<filt>-nrcb_i2d.fits``.  Routed to images/<FILTER>/ like any NIRCam image.

    An entry whose src is absent is recorded in ``missing`` (when a list is passed)
    rather than being silently dropped.  Skipping it quietly is what let sickle's
    F210M vanish from the release set: the m2 checkpoint had quarantined the listed
    ``-merged_i2d.fits`` to ``..._im0_badastrom.fits``, so a re-stage would have
    shipped four bands where the config asks for five, with nothing printed.  These
    entries are curated by hand -- an absent one means the config is stale, which is
    a fact about the release, not a file to skip.
    """
    items = []
    for entry in field_cfg.get("nircam", []):
        src = Path(entry["src"])
        if not src.is_file():
            if missing is not None:
                missing.append(_missing_listed_src("NIRCam", entry, src))
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
# The oksep quality-cut suffix carries the target's OWN proposal token(s):
# merge_catalogs._qualcuts_oksep_suffix() builds it from the field's registered
# proposals, so wd1 and w51 write their own program numbers and only the fields
# that really are program 2221 (brick, cloudc) write that one.  Matching a
# single program's token here silently skipped every other field's
# quality-filtered table: the loops below `continue` on a non-match, so wd1's
# and w51's tables sat on disk and never reached a release.
QUALCUTS_RE = r"_qualcuts_oksep[0-9A-Za-z-]+"


def field_qualcuts_suffix(field):
    """The quality-cut suffix this field's own catalogs should carry, or None.

    Imported lazily: stage_release runs as a script from scripts/release, and
    a missing/!importable pipeline package must not stop a release from being
    staged -- the caller falls back to alphabetical order.
    """
    try:
        from jwst_gc_pipeline.photometry.merge_catalogs import (
            _qualcuts_oksep_suffix)
    except ImportError:
        return None
    return _qualcuts_oksep_suffix(field)


def _qualcuts_sort_key(field):
    """Order a catalog directory so the LAST quality-cut table wins on merit.

    The loops below assign ``entry["qualcuts"] = path``, so with an unsorted
    glob the winner was whatever the directory listing happened to yield last.
    Eleven fields carry a mislabelled ``_qualcuts_oksep2221`` table written  # noqa: qualcuts-token
    before the suffix was per-field, and w51 holds that one NEXT TO its correct
    ``_qualcuts_oksep6151`` at the same iteration -- so which table reached  # noqa: qualcuts-token
    the release was decided by inode order.  Sorting puts the field's own token
    last, and is otherwise alphabetical so a rerun stages the same file twice.
    """
    own = field_qualcuts_suffix(field)

    def key(path):
        return (own is not None and own in path.name, path.name)

    return key
# combined (all-pointings) merged table; the (?!_o\d) guard keeps per-pointing
# "..._m7_o023.fits" variants OUT of the combined match.
COMBINED_RE = re.compile(
    rf"^{re.escape(CAT_BASE)}_(?P<iter>(?:resbgsub_)?m\d+)"
    rf"(?P<qc>{QUALCUTS_RE})?\.(?P<ext>fits|ecsv)$"
)
# per-pointing merged table: "..._m7_o023.fits", "..._m7_o023_qualcuts...fits"
PERPOINT_RE = re.compile(
    rf"^{re.escape(CAT_BASE)}_(?P<iter>(?:resbgsub_)?m\d+)_(?P<obs>o\d+)"
    rf"(?P<qc>{QUALCUTS_RE})?\.(?P<ext>fits|ecsv)$"
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
    for path in sorted(cat_dir.glob(f"{CAT_BASE}_*"), key=_qualcuts_sort_key(field)):
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
        for path in sorted(cat_dir.glob(f"{CAT_BASE}_*"),
                           key=_qualcuts_sort_key(field)):
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


SAME_RUN_TOL_MAS = 30.0   # a shipped image and its catalog from the SAME run agree within this


def _detect_i2d(path, thr=50.0):
    """Bright-source SkyCoords from a science mosaic (for the same-run tie check)."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats
    from astropy.coordinates import SkyCoord
    from photutils.detection import DAOStarFinder
    with fits.open(path) as h:
        sci = h["SCI"] if "SCI" in h else h[1]
        w = WCS(sci.header)
        d = sci.data.astype("float32")
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=2.5, threshold=thr * std)(d - med)
    if t is None or len(t) == 0:
        return None
    return SkyCoord(w.pixel_to_world(t["xcentroid"], t["ycentroid"]))


def check_image_catalog_match(items, tol_mas=SAME_RUN_TOL_MAS):
    """SAME-RUN gate. Every shipped science image must agree astrometrically with the
    shipped per-filter catalog of the same (filter, observation) to < ``tol_mas``.

    A mismatch means the image and catalog were produced by DIFFERENT pipeline /
    cataloging runs (different astrometric solutions) and must NOT be released together
    -- they will disagree by construction and look like an astrometry bug (e.g. brick
    2221 F182M: 07-08 catalog vs 07-11 image, ~10-15 mas apart). Uses the sanctioned
    offset-histogram (NO NN-median). Returns a list of ((filter, obs), off_mas) failures.
    """
    import numpy as np
    import astropy.units as u
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
    imgs = {(it["filter"], it.get("observation")): it for it in items
            if it["category"] == "image" and it.get("kind") == "science" and it.get("filter")}
    cats = {(it["filter"], it.get("observation")): it for it in items
            if it.get("kind") == "catalog_per_filter_vetted" and it.get("filter")}
    fails = []
    for key in sorted(set(imgs) & set(cats), key=lambda k: (k[0], k[1] or "")):
        det = _detect_i2d(imgs[key]["src"])
        if det is None:
            continue
        t = Table.read(cats[key]["src"])
        if "skycoord" not in t.colnames:
            continue
        csc = SkyCoord(t["skycoord"])
        csc = csc[np.isfinite(csc.ra.deg)]
        r = measure_offset(det, csc, maxsep=3.0 * u.arcsec, sweep=False)
        off = None if r is None else r["off"]
        ok = off is not None and off <= tol_mas
        tag = "ok" if ok else "MISMATCH -> different runs"
        print(f"  same-run {key[0]} {key[1] or ''}: image<->catalog "
              + ("no tie" if off is None else f"{off:.1f} mas") + f"  {tag}", flush=True)
        if not ok:
            fails.append((key, off))
    return fails


# Absolute-frame gate: a shipped catalog MUST be on the Gaia(DR3)=VIRAC2 frame, not a
# deprecated crowdsource/VVV/2MASS frame (which is ~20-90 mas off Gaia and silently
# propagated into the NIRSpec 6927 MSA plan). We enforce it astrometrically: the catalog
# bulk offset vs the field's Gaia-tied refcat must be < FRAME_TOL_MAS. Per-field refcat =
# the Gaia-tied seed used by the reduction (fields.yaml `reference_catalog:`).
FRAME_TOL_MAS = 15.0
FRAME_REFCAT = {
    # field: the Gaia-tied refcat the reduction was (re)anchored to. Extend as confirmed.
    "brick": "/orange/adamginsburg/jwst/brick/catalogs/gaia_virac2_refcat_epoch2022.70.fits",
    # arches: same construction as brick's (GaiaDR3 3656 + VIRAC2 174193, identical
    # columns).  Confirmed by measuring the staged mosaics against it with the swept
    # offset-histogram estimator: F212N nrca/nrcb 9.5/8.4 mas, F323N nrca/nrcb
    # 10.6/9.3 mas, every one un-swept at 3" with window_edge_fraction 0.00.
    #
    # This one matters for the OVERLAP gate rather than the frame gate: arches ships
    # no catalogs, so `check_catalog_on_frame` has nothing to test, but arches is
    # module-split with `geometry: disjoint` -- precisely the thin/sparse inter-module
    # overlap where the reference-free frame-vs-frame histogram is unreliable and the
    # same-star residual map vs VIRAC2 is the authoritative arbiter.  Staging it
    # printed "no Gaia refcat mapped for 'arches' in FRAME_REFCAT" while the refcat
    # sat on disk beside the ones already in use.
    "arches": "/orange/adamginsburg/jwst/arches/catalogs/gaia_virac2_refcat_epoch2023.64.fits",
    # quintuplet: arches's twin -- same program (2045), same module-split
    # `geometry: disjoint`, same two filters, same refcat construction (177791
    # rows).  Measured the same way, on the mosaics `build_manifest` selects:
    # F212N nrca/nrcb 6.4/6.3 mas, F323N nrca/nrcb 6.6/7.3 mas; contrast
    # 157-266, none swept at 3", window_edge_fraction 0.002.  A tighter tie than
    # arches's own (8.5-10.4 mas), so mapping arches while leaving its twin
    # unmapped would have been an accident of which field was looked at first.
    "quintuplet": "/orange/adamginsburg/jwst/quintuplet/catalogs/"
                  "gaia_virac2_refcat_epoch2024.62.fits",
}

# NOT mapped, deliberately, though a `gaia_virac2_refcat_*.fits` exists for each:
#
# * gc2211 -- its pointings are DISJOINT and o028 lies outside the field-wide
#   refcat's footprint (measuring against it reads `ref_in_fov=0` and no tie at any
#   window; it has its own `..._o028.fits`).  A single field-level entry would hand
#   the arbiter a reference that cannot see one of the pointings, which is worse
#   than no arbiter: the gate would fail-closed on good data.  Mapping gc2211 needs
#   per-observation refcats, which this dict cannot express.
# * cloudef -- its o002 mosaics are ~185 mas off this refcat in all four bands
#   (o005 F480M is 5.4 mas), so the field is not tied to it yet.  Mapping it now
#   would assert a frame the data does not sit on.
# * sgra -- absent for the OPPOSITE reason to gc2211's, and the distinction is
#   the point: its reference is fine, and the arbiter is simply never reached.
#   The overlap gate finds NO OVERLAPPING PAIRS in any band, so there is no tie
#   to break:
#
#       sgra F115W: 96 crf -> 2 groups, 0 overlapping pairs, 0 FAIL
#       sgra F212N: 96 crf -> 2 groups, 0 overlapping pairs, 0 FAIL
#       sgra F405N: 24 crf -> 2 groups, 0 overlapping pairs, 0 FAIL
#
#   `overlap_arbiter_refcat('sgra')` returning None is therefore correct rather
#   than a gap.  Recorded because sgra has the DENSEST refcat of the four
#   (191,462 rows, against brick's 115,032 and arches/quintuplet's ~177,800)
#   sitting unused, so "why is sgra not mapped?" is the obvious question and its
#   answer is not gc2211's.

# ---------------------------------------------------------------------------
# The star list the OVERLAP gate uses to arbitrate a pair it cannot measure
# frame-against-frame.  A separate registry from FRAME_REFCAT above, because the
# two jobs have opposite requirements.
#
# FRAME_REFCAT feeds a BLOCKING absolute-frame check: "is this catalogue on the
# right sky?"  That needs a dense catalogue -- a sparse one gives a noisy bulk
# tie and would refuse good data.
#
# This registry feeds a TIE-BREAK: two exposures overlap on a sliver too thin
# to compare directly, so both are compared against a common list of stars
# instead.  Sparse is fine for that, and far better than nothing -- with no
# list the pair is simply unmeasurable and the field cannot stage.  w51's Gaia
# list is ~9,500 rows against the Galactic Centre fields' ~100,000, which is
# why it belongs here and NOT in FRAME_REFCAT.
OVERLAP_ARBITER_REFCAT = {
    "w51": "/orange/adamginsburg/jwst/w51/catalogs/gaia_refcat.fits",
}


def overlap_arbiter_refcat(field):
    """The star list to arbitrate an unmeasurable overlap pair, or ``None``.

    Prefers the field's own arbiter entry; falls back to its absolute-frame
    catalogue, which is denser and works for this too.
    """
    for source in (OVERLAP_ARBITER_REFCAT, FRAME_REFCAT):
        path = source.get(field)
        if path and os.path.exists(path):
            return path
    return None



def _frame_bulk_offset(sc, ref, detect_sc=None):
    """The catalog's bulk offset vs the reference, by the SANCTIONED method
    (CLAUDE.md): histogram-stack + SWEEP to DETECT the tie (density-immune,
    catches a gross >window shift like brick-1182 v001 ~700 mas), then refine
    the PRECISE bulk same-star via ``local_residual_map`` (a single giant cell)
    -- which itself REFUSES unless the verified global tie is already small, so
    pairs are unambiguous.  This is NOT an ad-hoc dense NN-median.

    ``detect_sc`` supplies a DIFFERENT source list for the detection step only;
    the refinement, and therefore the gated number, still comes from ``sc``.
    The two steps want opposite populations and the caller cannot serve both
    with one list:

    * detection needs the sources the reference actually contains.  The GC
      refcat is Gaia+VIRAC2, i.e. BRIGHT stars, and in a NIRCam short-wavelength
      band most of those are saturated -- 3775 of brick F182M's 6322 matched
      pairs.  Drop them and the histogram loses the majority of its TRUE pairs
      while keeping every wrong one, so the peak walks off onto a spurious lag:
      F182M read (+288, -466) mas at contrast 16 without them and (-4.9, +3.9)
      mas at contrast 25 with them.  Same catalog, same reference, 1.4% of the
      rows removed.
    * refinement wants clean centroids.  A saturated star's centroid carries a
      flux-dependent bias with nothing to do with the frame (worst in the narrow
      Pa-alpha F187N, where it read a false 68 mas OFF-FRAME).

    Detecting with the bright stars and refining without them gives both: brick
    F182M then reads 1.10 mas over 4474 unsaturated same-star pairs.

    Returns ``(off_mas_or_None, source)``.  ``source`` is ``"same-star"`` for a
    refined fine tie, ``"histogram"`` when the tie is large/unverifiable (the
    genuinely-off-frame case -- the raw sweep value, which is meant to be large),
    or ``"no-tie"``.  The histogram peak is knowingly biased several mas against
    a DENSE reference (histogram-vs-samestar-offset-bias), so we never gate on it
    for a fine tie -- only same-star.  A large sweep value is used as-is because
    for a gross shift the overstatement is immaterial (it fails the gate anyway)
    and the refinement legitimately refuses."""
    import numpy as np
    import astropy.units as u
    from jwst_gc_pipeline.photometry.astrometry_offsets import (
        measure_offset, local_residual_map, GlobalTieNotVerifiedError)
    r = measure_offset(sc if detect_sc is None else detect_sc, ref,
                       maxsep=3.0 * u.arcsec, sweep=True)
    if r is None:
        return None, "no-tie"
    off, source = r["off"], "histogram"
    if r.get("ok") and not r.get("swept"):
        try:
            # cell_arcsec=1e9 = ONE cell spanning the whole footprint: we want the
            # single field-wide same-star bulk here, not a per-cell distortion map,
            # so the giant cell pools every matched pair into one robust residual.
            lrm = local_residual_map(sc, ref, r, cell_arcsec=1e9,
                                     match_radius=0.3 * u.arcsec, min_stars=200)
            cells = lrm.get("cells") or []
            if cells:
                c = max(cells, key=lambda cc: cc["n"])
                sdra = r["dra"] + c["dra_mas"]
                sddec = r["ddec"] + c["ddec_mas"]
                off, source = float(np.hypot(sdra, sddec)), "same-star"
        except GlobalTieNotVerifiedError:
            pass   # tie too large to refine -> keep the (large) histogram value
    return off, source


def _obs_keys_from_name(name):
    """``"<proposal>-<observation>"`` key(s) from a product/prefix basename
    ``jwPPPPP-oOOO[-MMM]...`` (a combined ``-oOOO-MMM`` encodes both)."""
    m = re.match(r"^jw(?P<prop>\d{5})-o(?P<obs>\d{3})(?:-(?P<obs2>\d{3}))?", name)
    if m is None:
        return set()
    keys = {f"{m.group('prop')}-{m.group('obs')}"}
    if m.group("obs2"):
        keys.add(f"{m.group('prop')}-{m.group('obs2')}")
    return keys


def _instrument_of(item):
    """``"miri"`` / ``"nircam"`` for a manifest item, from the single source of
    truth (`photometry.naming.MIRI_FILTERS`).  Items with no filter (a merged
    catalog, the offsets table) belong to the NIRCam side of a release: every
    field's NIRCam bands are what those products are built from."""
    from jwst_gc_pipeline.photometry.naming import MIRI_FILTERS
    return "miri" if str(item.get("filter") or "").lower() in MIRI_FILTERS \
        else "nircam"


def gate_by_instrument(field, items, run_gate):
    """Run the inter-frame overlap gate PER INSTRUMENT.

    ``(kept_items, withheld, refusal)``.  ``run_gate(instrument)`` returns the
    gate's exit code for that instrument's bands.

    NIRCam and MIRI are independent observations of the same sky -- different
    detectors, different exposures, usually a different program entirely
    (cloudc's MIRI is 2221-o001 and 2526-o021 against its NIRCam 2221-o002).  A
    MIRI verdict therefore says NOTHING about the NIRCam mosaics, and refusing
    the whole field on one withholds good data for a reason that does not apply
    to it.  cloudc was refused exactly that way on 2026-08-19, over a MIRI band
    its release did not even ship.

    So a failing MIRI gate WITHHOLDS the MIRI products and the rest of the
    release goes out, with the reason recorded for the manifest.  Nothing is
    loosened: a NIRCam failure still refuses the field, because the NIRCam
    mosaics are every field's main deliverable and there is nothing left to ship
    without them.
    """
    withheld = {}
    for instrument in sorted({_instrument_of(it) for it in items}):
        rc = run_gate(instrument)
        if rc == 0:
            continue
        why = ("FAILED -- two overlapping visits/detectors are misregistered vs "
               "EACH OTHER (>30 mas), so the drizzle overlap has doubled/smeared "
               "stars even though each frame is fine vs a reference"
               if rc == 1 else
               f"could not run (rc={rc}); frame-vs-frame registration is "
               f"unconfirmed")
        if instrument == "nircam":
            return items, {}, (
                f"REFUSING TO STAGE '{field}': NIRCam inter-frame OVERLAP gate "
                f"{why}. Re-examine per-visit alignment. Override with "
                f"--allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1 "
                f"(dangerous).")
        withheld[instrument] = why
        print(f"\nWITHHOLDING {instrument.upper()} from '{field}': its "
              f"inter-frame overlap gate {why}.\n  Those products are NOT "
              f"staged; everything else ships. Recorded in MANIFEST.json as "
              f"`withheld_instruments`, so the release cannot read as complete.",
              flush=True)
    if not withheld:
        return items, {}, None
    kept = [it for it in items if _instrument_of(it) not in withheld]
    print(f"  dropped {len(items) - len(kept)} {'/'.join(sorted(withheld))} "
          f"item(s) from the manifest", flush=True)
    if not kept:
        return kept, withheld, (
            f"REFUSING TO STAGE '{field}': withholding "
            f"{'/'.join(sorted(withheld))} leaves nothing to ship.")
    return kept, withheld, None


def _release_observations(field_cfg):
    """Every ``"<proposal>-<observation>"`` key this field's release covers,
    across ALL instruments -- so the reference-free overlap gate's scope is not
    silently NIRCam-only.  Sources:

    - ``proposal_prefix`` entries carrying ``-oNNN`` (brick = 1182 o004 +
      2221 o001);
    - a bare ``jwPPPPP`` proposal_prefix combined with the ``observations`` list
      (gc2211 = ``jw02211`` + ``["o023","o028","o046","o049"]`` -> 02211-023,
      02211-028, 02211-046, 02211-049);
    - explicit per-instrument mosaic lists (``miri`` / ``nircam`` config keys),
      whose ``src`` basenames carry the MIRI observations that ``proposal_prefix``
      (a NIRCam prefix) omits -- e.g. sgrb2 MIRI o002/o998, sickle MIRI o001/o002.

    The gate INTERSECTS this per filter directory, so listing a MIRI observation
    here can never re-admit a stray NIRCam crf that shares its proposal-obs key
    (brick MIRI F2550W and the cloudc NIRCam strays are both 2221 o002)."""
    prefixes = field_cfg.get("proposal_prefix", [])
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    obs = set()
    bare_props = []
    for pref in prefixes:
        keys = _obs_keys_from_name(pref)
        if keys:
            obs |= keys
        else:
            mb = re.match(r"^jw(\d{5})$", pref)
            if mb:
                bare_props.append(mb.group(1))
    for o in field_cfg.get("observations", []):
        mo = re.match(r"^o?(\d{3})$", str(o))
        if mo:
            obs |= {f"{bp}-{mo.group(1)}" for bp in bare_props}
    for key in ("miri", "nircam"):
        for entry in field_cfg.get(key, []):
            obs |= _obs_keys_from_name(os.path.basename(str(entry.get("src", ""))))
    return obs


def check_catalog_on_frame(items, field, tol_mas=FRAME_TOL_MAS):
    """Every shipped per-filter catalog must lie on the field's Gaia-tied reference frame.
    Measures the catalog's bulk offset vs the Gaia refcat by the sanctioned same-star
    method (``_frame_bulk_offset``); a bulk > ``tol_mas`` means the catalog is on a WRONG
    frame (crowdsource/VVV/2MASS) and must not ship. Returns list of ((filter,obs), off_mas)
    failures, or [] if no refcat is mapped for the field (can't enforce -> caller warns).

    Saturated / replaced-saturated sources are EXCLUDED from the REFINEMENT, which is
    the number this gates on: their centroids carry a strong flux-dependent bias (worst
    in the narrow Pa-alpha F187N, where the brightest quartile pulls the raw bulk by
    tens of mas) that has nothing to do with the frame.  Including them made F187N read
    68 mas (a false OFF-FRAME) while the clean same-star tie is ~1 mas.

    They are KEPT for the DETECTION step, because the reference is a bright-star catalog
    and in a short-wavelength NIRCam band most of the stars it shares with the release
    are saturated.  Excluding them from detection too made brick F182M read 547 mas --
    a false OFF-FRAME that refused the whole field's catalogs -- while the same rows
    tie at 1.10 mas once the peak is found with the bright stars present.  See
    ``_frame_bulk_offset``."""
    refpath = FRAME_REFCAT.get(field)
    if not refpath or not os.path.exists(refpath):
        return None
    import numpy as np
    import astropy.units as u
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    rt = Table.read(refpath)
    rcol = "skycoord" if "skycoord" in rt.colnames else None
    ref = SkyCoord(rt[rcol]) if rcol else SkyCoord(rt["ra"] * u.deg, rt["dec"] * u.deg)
    ref = ref[np.isfinite(ref.ra.deg)]
    fails = []
    cats = [it for it in items if it.get("kind") == "catalog_per_filter_vetted" and it.get("filter")]
    for it in cats:
        t = Table.read(it["src"])
        col = next((c for c in ("skycoord", "skycoord_ref") if c in t.colnames), None)
        if col is None:
            continue
        finite = np.isfinite(SkyCoord(t[col]).ra.deg)
        sat = np.zeros(len(t), dtype=bool)
        for satcol in ("is_saturated", "replaced_saturated"):
            if satcol in t.colnames:
                sat |= np.asarray(t[satcol], dtype=bool)
        sc_all = SkyCoord(t[col])[finite]
        sc = SkyCoord(t[col])[finite & ~sat]
        # Detect with every source (the bright ones carry the true pairs), refine
        # on the unsaturated ones (clean centroids).  `detect_sc=sc_all` is a
        # no-op when nothing is flagged.
        off, source = _frame_bulk_offset(sc, ref, detect_sc=sc_all)
        ok = off is not None and off <= tol_mas
        # Report the saturated-INCLUDED offset alongside the (gating) clean one --
        # print only, never gates.  Saturated-star centroids carry a strong
        # flux-dependent bias (worst in F187N) and are excluded from the gate, but
        # that bias is itself a finding; keeping the with-saturated number in the
        # staging log means a future regression in saturated-star astrometry shows
        # up in the same place, instead of the gate silently absorbing it.
        off_sat, _ = _frame_bulk_offset(sc_all, ref) if sat.any() else (None, "")
        satnote = "" if not sat.any() else (
            f"  [with-saturated: {'no tie' if off_sat is None else f'{off_sat:.1f} mas'}"
            f", {int(sat.sum())} sat]")
        print(f"  frame {it['filter']} {it.get('observation') or ''}: bulk vs Gaia-refcat "
              + ("no tie" if off is None else f"{off:.1f} mas ({source})")
              + f"  {'ok' if ok else 'OFF-FRAME'}" + satnote, flush=True)
        if not ok:
            fails.append(((it["filter"], it.get("observation")), off))
    return fails


# Photometric-continuity gate: the released merged catalog must be photometrically
# CONTINUOUS across the saturation boundary, and its near-degenerate colors
# (F405N-F410M, F182M-F187N) must be magnitude-flat. A jump/drift >= this floor
# means saturated-star or sub-floor-strip photometry is on a different flux scale
# than normal photometry (the CMD breaks at the boundary) -- the exact defect class
# of the 2026-07 satstar campaign. Certifiers: photometry/saturation_continuity.py.
CONTINUITY_TOL_MAG = 0.10
# DIRECTIONAL (band_sat, band_ref): band_sat = the earlier-saturating band whose
# replaced_saturated photometry is under test; band_ref = the reference band the
# color is binned in. saturation_continuity is NOT order-symmetric -- a reversed
# pair returns a plausible-but-different number (a reversed 0.042 once reached the
# checklist against a true 0.170), so it is called with band_sat=/band_ref= below.
CONTINUITY_PAIRS = [("f182m", "f187n"), ("f410m", "f405n")]

# A NIRCam-LONG observation read out with NGROUPS<=2 (BRIGHT2) cannot recover the
# deepest saturated cores -- they are railed at group 0, so there is no unsaturated
# sample to reconstruct their flux from. The recovered-satstar F410M-F405N color
# then carries a residual bias in the saturation-onset bins that no reduction
# change closes: an observation-design floor, not a pipeline defect, and the
# boundary metric cannot exclude the satstar rows (measuring the satstar<->normal
# transition IS its purpose), so science_only does not help here the way it does
# for the flatness gate.
#
# This exemption is scoped FOUR ways and FAILS CLOSED on the unknown -- it is not
# a whole-pair whitelist:
#   1. only the (band_sat, band_ref) pairs listed below (F410M-F405N);
#   2. only the C1-boundary-jump metric kind (the railed-core mechanism); a
#      C2-locus-offset -- a whole-locus color shift, a different defect -- blocks;
#   3. only when the field's band_sat readout is NGROUPS <= max_ngroups, taken as
#      the DEEPEST readout across the shipped science mosaics AT GATE TIME (w51
#      NGROUPS=5 and sgrb2 NGROUPS=3-4 are NOT exempt);
#   4. only when the jump is below max_jump_mag -- a GROSS break is not the deep-core
#      floor and blocks (exercised by the synthetic gross-break/regression tests; no
#      CURRENT field reaches the gate with both a measurable gross F410M-F405N jump
#      and a shippable 2-group mosaic, so this condition has no real-data example yet).
# If the mosaic/NGROUPS cannot be read, the pair is NOT exempt (blocks).
#
# max_jump_mag is a HARD-CODED constant, deliberately NOT env-overridable: it moves
# a blocking threshold, and a single env knob would be a one-factor waiver of a
# release gate (the repo's other overrides are two-factor: --allow-registration-fail
# AND ALLOW_REGISTRATION_FAIL=1). 0.25 sits just above the measured 0.170 floor, so
# a regression that materially WORSENS the boundary jump (e.g. ~0.30) still blocks
# rather than hiding under a wide ceiling.
#
# Measured (2026-08), and WHY each field lands where it does -- note on today's
# measurable data the ceiling+NGROUPS do the separating; the C1 guard and the
# ceiling cover cases the current fields do not contain:
#   brick  0.170  NGROUPS=2  C1   -> WARN (worst bin n_sat=38 at F405N 12-13 onset)
#   w51    0.577  NGROUPS=5  C1   -> FAIL (not the railed regime; condition 3)
#   cloudc 2.841  NGROUPS=None    -> FAIL (its F410M science mosaic is quarantined
#          _i2d_im0_badastrom, so no readout is shippable -> fail-closed, condition 3;
#          the 2.841 jump is ALSO above the ceiling, but the gate stops at NGROUPS).
# So the ceiling (condition 4) is not yet demonstrated on real data -- do not cite
# cloudc as the ceiling case; it is the fail-closed-on-missing-mosaic case.
#
# PROVISIONAL: saturated-star recovery photometry improvement is under active
# investigation; remove this entry once the recovered deep-core flux scale is fixed.
CONTINUITY_BOUNDARY_KNOWN_LIMITS = {
    ("f410m", "f405n"): dict(
        max_ngroups=2,
        max_jump_mag=0.25,
        kind="C1-boundary-jump",
        reason="NGROUPS<=2 (BRIGHT2) railed deep-core satstar color floor "
               "(provisional; recovery photometry under investigation)"),
}


def _band_ngroups(items, band):
    """The DEEPEST (max) NGROUPS across ``band``'s shipped science i2d mosaics.

    Returns ``max(NGROUPS)`` over every readable F<band> science mosaic, or None
    when none is shipped / readable -- the caller treats None as "cannot verify
    the readout" and does NOT grant a known-limit exemption (fail closed).  Taking
    the MAX, not the first, is deliberate: a field that ships several mosaics for
    a filter (per-module nrca/nrcb, per-pointing) must be judged by its DEEPEST
    readout, so a shallow-listed 2-group mosaic cannot grant an exemption while a
    deeper NGROUPS>=3 readout for the same filter is also present."""
    from astropy.io import fits
    found = []
    for it in items:
        if not (it.get("category") == "image" and it.get("kind") == "science"
                and str(it.get("filter", "")).upper() == band.upper()
                and str(it.get("src", "")).endswith(".fits")):
            continue
        try:
            ng = fits.getheader(it["src"]).get("NGROUPS")
        except OSError:
            # a shipped band mosaic we cannot even open -> cannot certify the
            # readout -> POISON: fail closed rather than judge off its siblings.
            print(f"  _band_ngroups: {band} science mosaic unreadable "
                  f"({it['src']}) -- cannot verify readout", flush=True)
            return None
        if ng is None:
            continue          # no NGROUPS key: not a readout claim, skip
        try:
            found.append(int(ng))
        except (TypeError, ValueError):
            # a shipped mosaic whose NGROUPS is not int-parseable -> POISON.
            # The contract is "judged by the deepest readout"; a candidate whose
            # readout cannot be parsed must DEFEAT the judgement, not be dropped
            # from it (else a shallow sibling could grant the exemption).
            print(f"  _band_ngroups: {band} science mosaic has unparseable "
                  f"NGROUPS={ng!r} ({it['src']}) -- cannot verify readout",
                  flush=True)
            return None
    return max(found) if found else None


# Degenerate-pair flatness is certified on the SCIENCE subset (science_only=True):
# the rows a user analyses after cutting every saturation flag.  The recovered /
# deep-core satstar rows stay in the released table under is_saturated /
# replaced_saturated for anyone who wants them, but they are NOT required to be
# color-flat -- some carry a recovered-satstar color bias the flags exist to
# signal.  (The saturation-BOUNDARY continuity check above -- CONTINUITY_PAIRS --
# is a different, complementary gate: it measures the satstar<->normal transition
# and therefore includes the satstar rows by construction; science_only does not
# apply to it.)
#
# DEGENERATE_FLATNESS_MIN_N: the flatness metric is the worst per-bin median
# color deviation; the brightest science bins at the saturation onset are sparse
# (Brick 2026-08 F405N-F410M: n=33 at F410M=12.75 drives the raw metric to 0.386)
# and a handful of stars there should not decide a release.  Certify at min_n=200
# so a bin must hold >=200 stars to count: on Brick m8 the worst qualifying bin is
# then F410M=14.25 (n=1418, dev 0.083) and both pairs hard-block <0.10 (F405N-F410M
# 0.083, F182M-F187N 0.049), while the 2026-07-11 suppression-strip guard fixture
# -- whose strip bins hold ~800 stars each -- still fails at 0.352.  This replaces
# an earlier per-pair exemption list: no pair is whitelisted, both stay blocking,
# and a genuine well-populated suppression strip in EITHER pair is still refused.
DEGENERATE_FLATNESS_MIN_N = 200


def check_photometric_continuity(items, tol=CONTINUITY_TOL_MAG,
                                 flatness_min_n=DEGENERATE_FLATNESS_MIN_N):
    """Certify the shipped combined merged table: saturation-boundary continuity
    over CONTINUITY_PAIRS + degenerate-pair color flatness (DEGENERATE_PAIRS),
    both against ``tol``.  Only pairs whose mag_vega_ columns exist in the table
    are tested.  Returns a list of failure strings ([] = pass), or None when no
    combined merged table is shipped (caller warns; cannot enforce).

    Flatness is certified on the SCIENCE subset (all saturation flags cut) at
    ``flatness_min_n``.  If that subset is too small to measure (nan) but the
    flag-inclusive metric IS measurable and over tol, the pair still fails -- a
    gate must never print "ok" for a population it declined to measure."""
    import numpy as np
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.saturation_continuity import (
        DEGENERATE_PAIRS, degenerate_pair_flatness, saturation_continuity)
    cat_items = [it for it in items
                 if it.get("kind") == "catalog_full" and it["src"].endswith(".fits")]
    if not cat_items:
        return None
    fails = []
    for it in cat_items:
        src = it["src"]
        cat = Table.read(src)
        have = {c[len("mag_vega_"):] for c in cat.colnames
                if c.startswith("mag_vega_")}
        name = Path(src).name
        for a, b in CONTINUITY_PAIRS:
            if a not in have or b not in have:
                continue
            r = saturation_continuity(cat, band_sat=a, band_ref=b)
            over = np.isfinite(r["metric"]) and r["metric"] >= tol
            exempt, note = False, ""
            if over:
                lim = CONTINUITY_BOUNDARY_KNOWN_LIMITS.get((a, b))
                if lim is not None:
                    ng = _band_ngroups(items, a)
                    if (r["kind"] == lim["kind"] and ng is not None
                            and 1 <= ng <= lim["max_ngroups"]
                            and r["metric"] < lim["max_jump_mag"]):
                        exempt = True
                        # Record the waiver on the catalog ITEM so it persists into
                        # MANIFEST.json (items -> manifest["files"]): a shipped
                        # catalog must carry a machine-readable record that a
                        # boundary gate was waived, not just a stdout line.
                        it.setdefault("continuity_waivers", []).append(dict(
                            pair=f"{a}-{b}", metric=round(float(r["metric"]), 4),
                            kind=r["kind"], ngroups=ng,
                            ceiling_mag=lim["max_jump_mag"], reason=lim["reason"]))
                        note = (f"  known limit (kind={r['kind']}, NGROUPS={ng}<="
                                f"{lim['max_ngroups']}, jump {r['metric']:.3f}<"
                                f"{lim['max_jump_mag']} mag): {lim['reason']} "
                                f"[recorded in MANIFEST]")
                    else:
                        note = (f"  NOT exempt (needs kind={lim['kind']}, NGROUPS<="
                                f"{lim['max_ngroups']}, jump<{lim['max_jump_mag']} mag; "
                                f"got kind={r['kind']}, NGROUPS={ng}, "
                                f"jump {r['metric']:.3f})")
            status = "ok" if not over else ("WARN(known-limit)" if exempt else "FAIL")
            print(f"  continuity {a}-{b} [{name}]: "
                  + (f"{r['metric']:.3f} mag ({r['kind']})" if np.isfinite(r["metric"])
                     else "n/a") + f"  {status}" + (note if over else ""), flush=True)
            if over and not exempt:
                fails.append(f"{a}-{b} continuity {r['metric']:.3f} mag [{name}]")
        for a, b in DEGENERATE_PAIRS:
            if a not in have or b not in have:
                continue
            # Blocking metric: the shipped science subset (all saturation flags
            # cut), at flatness_min_n so a sparse saturation-onset bin cannot
            # decide the release.
            r = degenerate_pair_flatness(cat, a, b, science_only=True,
                                         min_n=flatness_min_n)
            # Informational: full-inclusive drift (recovered/deep-core rows kept),
            # so a regressing satstar flux scale is still visible in the log. This
            # is a DIAGNOSTIC, not a blocker, and a satstar flux-scale offset lives
            # in exactly the sparse bins flatness_min_n suppresses -- so it is
            # measured at the default min_n, NOT flatness_min_n, or it would hide
            # the very thing it exists to show.
            r_full = degenerate_pair_flatness(cat, a, b, include_flags=True)
            sci_finite = np.isfinite(r["metric"])
            full_finite = np.isfinite(r_full["metric"])
            over = sci_finite and r["metric"] >= tol
            # Fail-open guard: never pass a population we declined to measure. If
            # the science subset is unmeasurable (too few rows at flatness_min_n),
            # the pair is NOT-CERTIFIED unless the flag-inclusive metric is itself
            # measurable AND clean (a larger, flatter population vouches for it).
            # Both-unmeasurable therefore blocks -- a small catalog with a real
            # strip no longer slips through the raised min_n.
            not_certified = (not sci_finite) and (
                (not full_finite) or (r_full["metric"] >= tol))
            ok = not (over or not_certified)
            wb = r["worst_bin"]
            sci = (f"{r['metric']:.3f} @{b}={wb['magB_lo']:.2f}(n={wb['n']})"
                   if sci_finite and wb else ("n/a" if not sci_finite else f"{r['metric']:.3f}"))
            full = (f"{r_full['metric']:.3f}" if full_finite else "n/a")
            status = "ok" if ok else ("NOT-CERTIFIED" if not_certified else "FAIL")
            print(f"  degenerate-pair {a}-{b} [{name}]: science drift {sci} mag "
                  f"vs plateau {r['plateau']:+.3f}  (full-inclusive {full} mag @min_n=default)  "
                  f"{status}", flush=True)
            if over:
                fails.append(f"{a}-{b} degenerate-pair science drift "
                             f"{r['metric']:.3f} mag [{name}]")
            elif not_certified:
                _why = (f"flag-inclusive drift {r_full['metric']:.3f} mag"
                        if full_finite else "flag-inclusive also unmeasurable")
                fails.append(f"{a}-{b} degenerate-pair science subset unmeasurable "
                             f"(<10*{flatness_min_n} rows); {_why} [{name}]")
    return fails


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
    if item["category"] == exposure_bundle.EXPOSURE_CATEGORY:
        # Mirrors the images/ layout exactly, so "the frames behind
        # images/<obs>/<FILT>" is `exposures/<obs>/<FILT>` with no lookup --
        # which is also what makes one Globus folder link per group a usable
        # bulk download.
        parts = [Path("exposures")]
        if item.get("instrument") == "MIRI":
            parts.append(Path("MIRI"))
        if item.get("observation"):
            parts.append(Path(item["observation"]))
        parts.append(Path(item["filter"] or "unknown"))
        return Path(*parts) / src_name
    # catalogs stay flat; per-pointing filenames already carry the _oNNN tag
    return Path("catalogs") / src_name


def build_manifest(field, version, images_only=False, missing=None,
                   exposures=True, exposure_problems=None):
    """Deliverable dicts for a field.

    ``missing`` -- pass a list to collect one description per explicitly-listed
    (``nircam`` / ``miri``) src that is not on disk.  The caller is expected to
    refuse to stage when it comes back non-empty; the auto-discovered products
    are not affected (nothing lists them, so nothing can go absent from a list).

    ``exposures`` -- also offer the detector-frame frames each science mosaic was
    drizzled from (``exposure_bundle``).  These are provenance-derived from the
    mosaics' own associations, so they are added AFTER the mosaic set is final
    and they can never change which mosaics ship.  ``exposure_problems``
    collects one line per mosaic whose input list could not be established;
    unlike ``missing`` this is a report, not a refusal -- a mosaic is a
    deliverable, its input list is a convenience, and withholding a certified
    mosaic because its association went missing would be the wrong trade.
    """
    field_cfg = FIELDS[field]
    items = []
    # auto-discovered per-filter NIRCam mosaics (skip with no_auto_images, e.g.
    # fields whose NIRCam mosaics are listed explicitly via the `nircam` key)
    if not field_cfg.get("miri_only") and not field_cfg.get("no_auto_images"):
        items += discover_images(field_cfg)
    items += discover_nircam(field_cfg, missing=missing)   # explicit NIRCam list (if any)
    items += discover_miri(field_cfg, missing=missing)
    # catalogs: skip while cataloging is still in progress (skip_catalogs), or for an
    # explicit image-only release (--images-only): ship mosaics without catalogs.
    if not field_cfg.get("miri_only") and not field_cfg.get("skip_catalogs") and not images_only:
        items += discover_catalogs(field_cfg, field)
    if images_only:
        # science mosaics only: drop the catalog-derived residual/model i2d, which encode
        # the (uncertified) catalog fit, and any catalog products.
        items = [it for it in items if it.get("kind") == "science"]
    if exposures:
        # After `images_only` has already settled which mosaics ship, so the
        # frames offered are the frames behind the mosaics on the page and
        # nothing else. Detector frames are not catalog-derived, so an
        # image-only release keeps them.
        items += exposure_bundle.discover_exposures(
            [it for it in items
             if it.get("category") == "image" and it.get("kind") == "science"],
            search_root=field_cfg["data_dir"], problems=exposure_problems)
    # The pointing-correction table: kilobytes, COPIED and checksummed like a
    # mosaic. The frames are symlinks whose headers a re-reduction rewrites, so
    # this table -- not the FITS -- is what makes this version's astrometry
    # reconstructible afterwards.
    table = astrometry_provenance.stage_item(field, field_cfg, version)
    if table is not None:
        items.append(table)
    for item in items:
        src = Path(item["src"])
        # `stage_item` already knows where the table goes; everything else is
        # placed by category.
        item.setdefault("dest", str(assign_dest(item, field)))
        item["size_bytes"] = src.stat().st_size if src.is_file() else None
        # per-file version: defaults to the field release version so every file carries
        # an explicit version on the download page. A file bumped independently (e.g. a
        # re-tied mosaic staged into an otherwise-older release) can override this.
        item.setdefault("version", version)
    # needs every `dest` assigned, so it runs after the loop above
    exposure_bundle.link_parents(items)
    return items


# ---- generation-span check ---------------------------------------------------------
# A field's bands are drizzled as one batch, so the science mosaics of one reduction
# generation carry DATE headers hours apart and a single CRDS_CTX.  Measured over the
# 19 field/instrument sets in FIELDS on 2026-08-05, every single-generation set spans
# <= 0.6 d (wd2 0.53 d over 17 filters is the widest), while a genuinely mixed set is
# an order of magnitude wider: v1.1's shipped sickle set spans 69 d (F210M 2026-04-19
# / jwst_1535.pmap vs the other four 2026-06-27 / jwst_1537.pmap), ngc6334 24.6 d,
# wd1 17.8 d.  7 d sits ~10x above the widest single batch and ~2.5x below the
# narrowest set this leg is meant to separate (wd1), so it does not flag a re-drizzle
# that ran over a weekend.  The CRDS leg is what catches a mixed set the date leg
# cannot: w51's 11 bands span only 3.4 d but three CRDS contexts.  Reported per
# instrument -- NIRCam and MIRI are reduced in separate batches, so a NIRCam-vs-MIRI
# date gap is expected and is not a complaint.
GENERATION_SPAN_DAYS = 7.0


def _image_provenance(path):
    """``(DATE, CRDS_CTX)`` from a mosaic's primary header; ``(None, None)`` if the
    file cannot be opened or carries neither keyword."""
    from astropy.io import fits
    try:
        with fits.open(path) as hdul:
            header = hdul[0].header
            return header.get("DATE"), header.get("CRDS_CTX")
    except (OSError, ValueError):
        return None, None


def _parse_header_date(value):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def check_generation_span(items, max_span_days=GENERATION_SPAN_DAYS, verbose=True):
    """Report staged science images that did not come from one reduction generation.

    Returns a list of complaint strings (empty == the set is one generation).  Two
    independent legs, because they catch different mistakes: a DATE span wider than
    ``max_span_days`` (one band left behind by a re-drizzle) and more than one
    CRDS_CTX (bands calibrated against different reference files, which a date span
    alone can miss when the re-reduction happened to be quick).
    """
    by_instrument = {}
    for it in items:
        if it.get("category") != "image" or it.get("kind") != "science":
            continue
        date, ctx = _image_provenance(it["src"])
        key = it.get("instrument") or "unknown"
        by_instrument.setdefault(key, []).append(
            (it.get("filter") or "?", it.get("observation"), date, ctx, it["src"]))

    complaints = []
    for instrument in sorted(by_instrument):
        members = by_instrument[instrument]
        stamped = [(_parse_header_date(d), f, o, c)
                   for f, o, d, c, _s in members]
        stamped = [row for row in stamped if row[0] is not None]
        contexts = sorted({c for _f, _o, _d, c, _s in members if c})
        span_days = None
        if stamped:
            span_days = (max(r[0] for r in stamped)
                         - min(r[0] for r in stamped)).total_seconds() / 86400.0
        if verbose:
            span_txt = "n/a" if span_days is None else f"{span_days:.2f} d"
            print(f"  {instrument}: {len(members)} science image(s), DATE span "
                  f"{span_txt}, CRDS_CTX {', '.join(contexts) or 'n/a'}")
        if span_days is not None and span_days > max_span_days:
            oldest = min(stamped, key=lambda r: r[0])
            newest = max(stamped, key=lambda r: r[0])
            complaints.append(
                f"{instrument} DATE span {span_days:.1f} d > {max_span_days:.0f} d "
                f"({oldest[1]}{('/' + oldest[2]) if oldest[2] else ''} "
                f"{oldest[0].isoformat()} .. "
                f"{newest[1]}{('/' + newest[2]) if newest[2] else ''} "
                f"{newest[0].isoformat()})")
        if len(contexts) > 1:
            per_ctx = {}
            for f, o, _d, c, _s in members:
                if c:
                    per_ctx.setdefault(c, []).append(f"{f}{('/' + o) if o else ''}")
            detail = "; ".join(f"{c}: {', '.join(sorted(v))}"
                               for c, v in sorted(per_ctx.items()))
            complaints.append(f"{instrument} spans {len(contexts)} CRDS contexts "
                              f"({detail})")
    return complaints


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
    """One line per deliverable; exposures GROUPED.

    A field ships tens of mosaics and catalogs and can ship several thousand
    detector-frame exposures (wd1: 696).  Printing those one per line buries the
    deliverable list the operator is actually reading before a stage, so they
    are summarized per (observation, filter) instead, with the product suffix
    shown -- which is the part worth eyeballing, since it says whether a band
    fell back off `_crf` onto `_cal`.
    """
    deliverables = [it for it in items
                    if it["category"] != exposure_bundle.EXPOSURE_CATEGORY]
    exposures = [it for it in items
                 if it["category"] == exposure_bundle.EXPOSURE_CATEGORY]
    print(f"\n{'CATEGORY':<9} {'KIND':<26} {'FILT':<6} {'ITER':<14} {'SIZE':>8}  SRC")
    print("-" * 110)
    for it in deliverables:
        print(f"{it['category']:<9} {it['kind']:<26} "
              f"{(it['filter'] or ''):<6} {(it['iteration'] or ''):<14} "
              f"{human_size(it['size_bytes']):>8}  {it['src']}")
    total = sum(it["size_bytes"] or 0 for it in deliverables)
    print("-" * 110)
    print(f"{len(deliverables)} files, total {human_size(total)}")
    if exposures:
        summary = exposure_bundle.summarize(exposures)
        exp_total = sum(it["size_bytes"] or 0 for it in exposures)
        print(f"\nDETECTOR-FRAME EXPOSURES (linked, not frozen): "
              f"{len(exposures)} frames, total {human_size(exp_total)}")
        for (obs, filt) in sorted(summary):
            count, size = summary[(obs, filt)]
            print(f"  {(obs or '-'):<8} {filt:<7} {count:>5} frames  "
                  f"{human_size(size):>9}")
        print("  products: " + ", ".join(
            f"{suffix} x{n}" for suffix, n
            in sorted(exposure_bundle.suffix_histogram(exposures).items())))
    print()


def stage(items, field, version, release_root, mode, do_checksum,
          continuity_gate=None, allow_older=False, withheld_instruments=None):
    assert_writable(version, release_root, allow_older, field)
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
        # Detector-frame exposures are ALWAYS symlinked and never checksummed,
        # even under --copy.  A field-filter is ~20 GB of frames against ~500 MB
        # of mosaics, so copying them would grow the frozen tree ~40x for data
        # that a re-reduction only re-headers; and a sha256 over a symlink whose
        # target is expected to change is not the frozen-copy integrity claim
        # that CHECKSUMS.sha256 makes.  See exposure_bundle's module docstring.
        is_exposure = it.get("category") == exposure_bundle.EXPOSURE_CATEGORY
        item_mode = "hardlink" if is_exposure else mode
        src = Path(it["src"]).resolve()
        dest = field_dir / it["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        src_stat = src.stat()
        src_size = src_stat.st_size
        # Size alone is NOT a content proxy on this archive: a re-drizzle of the
        # same field/filter writes the same-shaped mosaic, so the byte count is
        # unchanged while the pixels are not (sgrc F182M and wd1 F150W are exactly
        # that between v1.0 and v1.1 -- identical size, different content). Taking
        # size as "unchanged" keeps the OLD bytes and republishes the OLD sha256
        # under a fresh build timestamp. `copy2` preserves mtime, so requiring the
        # mtime to match too is what makes the skip mean "this is that file".
        dest_stat = dest.stat() if (dest.is_file() and not dest.is_symlink()) else None
        unchanged = (item_mode == "copy" and dest_stat is not None
                     and dest_stat.st_size == src_size
                     and abs(dest_stat.st_mtime - src_stat.st_mtime) < 1e-3)
        if not unchanged:
            if item_mode == "copy":
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                shutil.copy2(src, dest)
            else:
                it["link_mode"] = exposure_bundle.link_frame(src, dest)
                it["source_identity"] = exposure_bundle.source_identity(src)

        if do_checksum and not is_exposure:
            cached = prior.get(it["dest"])
            if unchanged and cached and cached[0] == src_size:
                digest = cached[1]  # reuse; content unchanged
            else:
                digest = sha256sum(src)
            it["sha256"] = digest
            checksum_lines.append(f"{digest}  {it['dest']}")

    exposure_bundle.report_symlink_fallbacks(
        [it for it in items
         if it.get("category") == exposure_bundle.EXPOSURE_CATEGORY])

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
        # `mode` describes the deliverables. Detector-frame exposures are
        # always links and always unchecksummed, so state that separately
        # rather than let "mode": "copy" imply a frozen copy of every entry.
        # DERIVED from the entries, never asserted: see link_mode_summary.
        "exposure_mode": exposure_bundle.link_mode_summary(
            [it for it in items
             if it.get("category") == exposure_bundle.EXPOSURE_CATEGORY]),
        # Positive outcome of the photometric-continuity gate for this staging:
        # "passed" (all clean), "waived" (a documented known limit was recorded --
        # see per-file continuity_waivers), "skipped(override)", or a
        # "not_enforced/not_applicable" reason. A manifest with no waiver is thus
        # not ambiguous. None only for a stage() call outside the gated main path.
        "continuity_gate": continuity_gate,
        # Instruments whose products this release does NOT carry because their
        # own gate refused them, `{instrument: why}`.  Present so a consumer can
        # tell "this field has no MIRI" from "this field's MIRI was withheld",
        # which the file list alone cannot say.
        "withheld_instruments": withheld_instruments or {},
        "globus_collection_id": GLOBUS_COLLECTION_ID,
        "globus_https_base": GLOBUS_HTTPS_BASE,
        "files": items,
    }
    (field_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    if do_checksum:
        (field_dir / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n")

    exposure_dests = [it["dest"] for it in items
                      if it["category"] == exposure_bundle.EXPOSURE_CATEGORY]
    if exposure_dests:
        orphans, unexpected = prune_exposure_orphans(field_dir, exposure_dests)
        if orphans:
            print(f"  removed {orphans} stale exposure link(s) this release no "
                  f"longer claims")
        for rel in unexpected:
            print(f"  WARNING: {rel} under exposures/ is a real file, not a "
                  f"link -- left in place")

    write_readme(field_dir, field, version, items, mode,
                 built_at=manifest["built"],
                 withheld_instruments=withheld_instruments)

    # world-readable
    subprocess.run(["chmod", "-R", "a+rX", str(field_dir)], check=True)
    return field_dir


def _exposures_from_disk(field, version, field_dir):
    """Exposure items enumerated from the pipeline directories, no mosaic needed.

    Detector frames are a DEPENDENCY of the mosaic -- Stage 2/3 write them first
    -- so a field can release them before anything is drizzled, and should: a
    two-filter program mid-reduction has its ``_cal``/``_destreak`` frames on
    disk and nothing about them is waiting on a drizzle.

    Behind an explicit flag rather than used as a silent fallback.  The
    association path answers "which frames went INTO this mosaic"; this one
    answers "which frames belong to this field/observation/filter", which is a
    weaker claim, and the difference is real -- on wd1 F150W the two disagree,
    because that mosaic was drizzled from ``_cal`` members while a separate
    ``_o001_crf`` family also sits in the same directory.  Choosing between them
    by accident is the guesswork ``exposures_for_mosaic`` exists to refuse.
    """
    found = exposure_bundle.enumerate_field_exposures(FIELDS[field], field)
    items = []
    for (obs, filt), paths in sorted(found.items(),
                                     key=lambda kv: (kv[0][0] or "", kv[0][1])):
        for path in paths:
            items.append({
                "category": exposure_bundle.EXPOSURE_CATEGORY,
                "kind": exposure_bundle.EXPOSURE_KIND,
                "filter": filt, "iteration": None, "observation": obs,
                "instrument": "MIRI" if "mirimage" in path.name else "NIRCam",
                "src": str(path), "version": version,
            })
    for it in items:
        it["dest"] = str(assign_dest(it, field))
        src = Path(it["src"])
        it["size_bytes"] = src.stat().st_size if src.is_file() else None
        rel = (field_dir / it["dest"]).relative_to(GLOBUS_COLLECTION_ROOT)
        it["globus_path"] = "/" + str(rel)
        it["url"] = GLOBUS_HTTPS_BASE + it["globus_path"]
    return items


def prune_exposure_orphans(field_dir, keep_dests):
    """Remove staged exposure links this release no longer claims.

    Re-staging with a CHANGED frame list -- which is exactly what correcting the
    input-list rule does -- leaves the previous links in place, so the directory
    accumulates: 240 links on disk against 120 manifest entries. That matters
    beyond tidiness, because the bulk-download button transfers the FOLDER, not
    the manifest, so an orphan is a file the release still hands out while
    claiming not to.

    Removes only links -- a symlink, or a hardlink sharing its inode with a file
    outside the release tree.  A frame with a link count of 1 is the release's
    ONLY copy of those bytes and is left alone and reported: deleting it would
    destroy data rather than unpublish it, and nothing in this design writes
    one.  (Before hardlinks this checked ``is_symlink()``, which after the
    switch would have matched nothing and silently pruned no orphan at all.)
    """
    exposures_dir = Path(field_dir) / "exposures"
    if not exposures_dir.is_dir():
        return 0, []
    keep = {str(d) for d in keep_dests}
    removed, unexpected = 0, []
    for path in sorted(exposures_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(field_dir).as_posix()
        if rel in keep:
            continue
        if path.is_symlink() or path.stat().st_nlink > 1:
            path.unlink()
            removed += 1
        else:
            unexpected.append(rel)
    for directory in sorted(exposures_dir.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return removed, unexpected


class FrozenReleaseError(RuntimeError):
    """Writing into a release directory that already exists, without saying so."""


def refuse_older_version(version, release_root, allow_older, field=None):
    """``(refuse, why)`` for writing into ``version``.

    Keyed on whether THIS FIELD'S directory already exists, not on version
    ordering.  Ordering had two holes, both real:

    * ``version < max(existing)`` leaves the NEWEST published version writable
      with no flag.  Of the four field directories written on 2026-08-17 it
      would have stopped three; ``v1.3-2026.08/arches`` was already
      ``max(existing)`` and stayed open, and still is.
    * the comparison is lexicographic, so it inverts at the first two-digit
      minor: with ``v1.9`` and ``v1.10`` on disk, ``v1.9`` (frozen) is writable
      and ``v1.10`` (newest) is refused.

    A directory holding a ``MANIFEST.json`` has been staged before, whatever its
    version string sorts as, so re-cutting it is the deliberate act the flag is
    for.  ``field=None`` falls back to the ordering check for callers that have
    no field yet.
    """
    if allow_older:
        return False, ""
    if field is not None:
        manifest = Path(release_root) / version
        cfg = FIELDS.get(field, {})
        if cfg.get("group"):
            manifest = manifest / cfg["group"]
        manifest = manifest / field / "MANIFEST.json"
        if manifest.is_file():
            return True, (f"'{version}', which already holds a staged {field} release "
                          f"({manifest}); re-cutting it rewrites a MANIFEST, "
                          f"README and CHECKSUMS whose checksums are cited")
    root = Path(release_root)
    existing = sorted(p.name for p in root.iterdir()
                      if p.is_dir() and p.name.startswith("v")) if root.is_dir() else []
    if existing and version < max(existing):
        return True, (f"'{version}', which is older than the newest release on "
                      f"disk ('{max(existing)}')")
    return False, ""


def assert_writable(version, release_root, allow_older, field=None):
    """Raise ``FrozenReleaseError`` unless this release directory may be written.

    Called by the functions that DO the writing rather than only from ``main``.
    The guard used to live in two ``main()`` branches, so any other caller --
    including this repo's own tests, which call ``stage_exposures_only``
    directly -- bypassed it entirely while the docstring claimed every path went
    through one check.
    """
    refuse, why = refuse_older_version(version, release_root, allow_older, field)
    if refuse:
        raise FrozenReleaseError(
            f"REFUSING to write into {why}. Pass --allow-older-version if "
            f"re-cutting it is intended.")


def stage_exposures_only(field, version, release_root, from_disk=False,
                         allow_older=False):
    """Add the detector-frame exposures to an ALREADY-STAGED release.

    Stages nothing but symlinks, and rewrites ``MANIFEST.json`` and ``README.md``
    around them.  ``images/``, ``catalogs/`` and ``CHECKSUMS.sha256`` are not
    touched, and the mosaic gates are deliberately not re-run -- see
    ``exposure_bundle.add_to_release`` for why that is sound and for the
    ``built`` trap it avoids.

    ``from_disk`` enumerates the frames from the pipeline directories instead of
    from the staged mosaics, which is what lets a field release them BEFORE it
    has any mosaic at all -- and it will create the release directory if there
    is none yet.

    Returns ``(field_dir, n_exposures)``, or ``(None, 0)`` when there is nothing
    to add to and no way to enumerate.
    """
    assert_writable(version, release_root, allow_older, field)
    field_dir = field_release_dir(field, version, release_root)
    have_manifest = (field_dir / "MANIFEST.json").is_file()
    if not have_manifest and not from_disk:
        print(f"\nNo staged release at {field_dir} -- --exposures-only adds "
              f"frames to an EXISTING release; stage the field first, or pass "
              f"--exposures-from-disk to release the detector frames on their "
              f"own (they do not need a mosaic).", file=sys.stderr)
        return None, 0

    problems = []
    if from_disk:
        field_dir.mkdir(parents=True, exist_ok=True)
        exposures = _exposures_from_disk(field, version, field_dir)
        manifest = (json.loads((field_dir / "MANIFEST.json").read_text())
                    if have_manifest else {
            "field": field, "version": version,
            "group": FIELDS.get(field, {}).get("group"),
            "release_path": "/" + str(field_dir.relative_to(GLOBUS_COLLECTION_ROOT)),
            # A release that has never held a mosaic has no prior staging time to
            # preserve, so stamping `built` here is correct -- unlike the
            # add-to-existing path, where moving it would re-publish quarantined
            # mosaics (see exposure_bundle.add_to_release).
            "built": datetime.datetime.now().astimezone().isoformat(),
            "mode": "symlink", "continuity_gate": "not_applicable(exposures-only)",
            "globus_collection_id": GLOBUS_COLLECTION_ID,
            "globus_https_base": GLOBUS_HTTPS_BASE, "files": []})
        # No parent mosaic to inherit withholding from. Say so rather than let
        # the absence of `parent_dest` read as "checked and fine".
        print("  enumerated from the pipeline directories: these frames are not "
              "tied to a staged mosaic, so none of them is withheld by a "
              "mosaic's astrometry verdict.")
    else:
        exposures, manifest = exposure_bundle.add_to_release(
            field_dir, lambda it: assign_dest(it, field),
            GLOBUS_COLLECTION_ROOT, GLOBUS_HTTPS_BASE,
            search_root=FIELDS[field]["data_dir"], problems=problems)
    if problems:
        print(f"\nEXPOSURE PROVENANCE: could not establish the detector-frame "
              f"input list for {len(problems)} staged mosaic(s); their frames "
              f"are NOT offered:")
        for problem in problems:
            print(f"    - {problem}")
    if not exposures:
        print(f"\nNo detector-frame exposures could be resolved for '{field}'.",
              file=sys.stderr)
        return field_dir, 0

    print_manifest(exposures)
    orphans, unexpected = prune_exposure_orphans(
        field_dir, [it["dest"] for it in exposures])
    if orphans:
        print(f"  removed {orphans} stale exposure link(s) this release no "
              f"longer claims")
    for rel in unexpected:
        print(f"  WARNING: {rel} under exposures/ is a real file, not a link -- "
              f"left in place")
    for it in exposures:
        dest = field_dir / it["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        it["link_mode"] = exposure_bundle.link_frame(Path(it["src"]).resolve(), dest)
        it["source_identity"] = exposure_bundle.source_identity(it["src"])
    exposure_bundle.report_symlink_fallbacks(exposures)

    # A frames-only release must still carry the table those frames are on --
    # that is the whole point of preserving it, and the frames are the products
    # whose own bytes are not frozen.
    table = astrometry_provenance.stage_item(field, FIELDS[field],
                                             manifest.get("version", version))
    if table is not None:
        dest = field_dir / table["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(table["src"], dest)
        table["size_bytes"] = os.path.getsize(table["src"])
        table["sha256"] = sha256sum(table["src"])
        rel = dest.relative_to(GLOBUS_COLLECTION_ROOT)
        table["globus_path"] = "/" + str(rel)
        table["url"] = GLOBUS_HTTPS_BASE + table["globus_path"]
        # ...and it goes into CHECKSUMS.sha256, which this path otherwise does
        # not touch. The no-touch promise exists so the FROZEN MOSAIC hashes are
        # never disturbed; appending a line for a newly added frozen deliverable
        # completes that record rather than disturbing it. Omitting it would
        # leave the README and the page claiming a checksum that is only in
        # MANIFEST.json -- a checksum file that does not list something the
        # release ships frozen is simply wrong.
        checksums = field_dir / "CHECKSUMS.sha256"
        existing = [ln for ln in (checksums.read_text().splitlines()
                                  if checksums.is_file() else [])
                    if ln.strip() and not ln.endswith(f"  {table['dest']}")]
        checksums.write_text("\n".join(existing
                                       + [f"{table['sha256']}  {table['dest']}"]) + "\n")
        print(f"  + offsets table {Path(table['src']).name} "
              f"({table['size_bytes']} bytes, frozen copy, checksummed)")
    kept = [f for f in manifest.get("files", [])
            if f.get("category") not in (exposure_bundle.EXPOSURE_CATEGORY,
                                         astrometry_provenance.ASTROMETRY_CATEGORY)]
    manifest["files"] = kept + exposures + ([table] if table else [])
    manifest["exposure_mode"] = exposure_bundle.link_mode_summary(exposures)
    # `built` is NOT touched -- release_freshness reads it as the staging time
    # and would re-publish this field's quarantined mosaics if it moved.  The
    # separate key records when the frames were added without claiming the
    # release itself was rebuilt.
    manifest["exposures_added"] = datetime.datetime.now().astimezone().isoformat()
    (field_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    write_readme(field_dir, field, manifest.get("version", version),
                 manifest["files"], manifest.get("mode", "copy"),
                 built_at=manifest.get("built"))
    subprocess.run(["chmod", "-R", "a+rX", str(field_dir)], check=True)
    return field_dir, len(exposures)


def latest_staged_version(field, release_root):
    """Newest version directory that actually holds a staged `field`, or ``None``.

    Ordered by the release date embedded in the version string rather than
    lexicographically, which inverts at the first two-digit minor: with
    ``v1.9-2026.09`` and ``v1.10-2026.10`` on disk, a string sort calls
    ``v1.9`` the newer one.  Read-only: nothing here decides what may be
    written, only which staged tree to audit.
    """
    root = Path(release_root)
    if not root.is_dir():
        return None
    staged = [p.name for p in root.iterdir()
              if p.is_dir() and p.name.startswith("v")
              and (field_release_dir(field, p.name, release_root)
                   / "MANIFEST.json").is_file()]
    if not staged:
        return None

    def key(name):
        version = name.split("-", 1)[0].lstrip("v")
        parts = tuple(int(p) if p.isdigit() else 0
                      for p in version.split("."))
        return (parts, name)
    return max(staged, key=key)


def check_exposures(field, version, release_root):
    """Audit a staged release's detector frames.  Returns a process exit code.

    ``diverged_frames`` existed with no caller outside the tests while the
    README told readers the release "reports any that have diverged" -- a claim
    with nothing behind it.  This is what performs it.  Divergence is expected
    over time and is not a staging failure: the release still serves real,
    readable bytes, just an older generation than the pipeline now holds, which
    is the property a symlink would have hidden by following along silently.
    """
    field_dir = field_release_dir(field, version, release_root)
    manifest_path = field_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        print(f"No staged release at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    exposures = [f for f in manifest.get("files", [])
                 if f.get("category") == exposure_bundle.EXPOSURE_CATEGORY]
    if not exposures:
        print(f"{field} {version}: release ships no detector-frame exposures.")
        return 0
    recorded = manifest.get("exposure_mode")
    derived = exposure_bundle.link_mode_summary(exposures)
    print(f"{field} {version}: {len(exposures)} detector-frame exposures, "
          f"link mode {derived}"
          + (f" (MANIFEST says {recorded!r})" if recorded != derived else ""))
    fallbacks = exposure_bundle.report_symlink_fallbacks(exposures)
    diverged = exposure_bundle.diverged_frames(manifest)
    if diverged:
        print(f"  {len(diverged)} frame(s) have diverged from their source:")
        for dest, why in diverged[:20]:
            print(f"    - {dest}: {why}")
        if len(diverged) > 20:
            print(f"    ... and {len(diverged) - 20} more")
    else:
        print("  no frame has diverged from its source")
    return 1 if (diverged or recorded != derived) else 0


def _link_mode_phrase(exposures):
    """How the frames were placed, in README prose -- derived, never asserted."""
    return {"hardlink": "HARDLINKS", "symlink": "SYMLINKS",
            "mixed": "LINKS (some hard, some symbolic)"}.get(
                exposure_bundle.link_mode_summary(exposures), "LINKS")


def write_readme(field_dir, field, version, items, mode, built_at=None,
                 withheld_instruments=None):
    images = [it for it in items if it["category"] == "image"]
    catalogs = [it for it in items if it["category"] == "catalog"]
    # describe the mosaics ACTUALLY staged: a module-split field (arches,
    # quintuplet, sickle) ships `-nrca_i2d`/`-nrcb_i2d`, not `-merged_i2d`, and
    # a README promising the latter sends users looking for a file that is not
    # there.
    modules = sorted({m.group(1) for m in
                      (re.search(r"-(merged|nrca|nrcb|nrcalong|nrcblong|mirimage)"
                                 r"(?:_data)?_i2d\.fits$", os.path.basename(it["src"]))
                       for it in images) if m})
    # `mirimage` is MIRI's normal full-field product, NOT evidence of a
    # module split -- keying off "anything but merged" claimed sgrb2 (which has a
    # merged mosaic in all ten NIRCam filters, plus MIRI) had none.  Only a
    # per-module NIRCam suffix means the split.
    per_module = sorted(set(modules) & {"nrca", "nrcb", "nrcalong", "nrcblong"})
    if per_module:
        science_lines = [
            "- `*-{" + ",".join(modules) + "}_i2d.fits` : science mosaic (drizzled)",
            "",
            "  Some filters here have no single full-field NIRCam mosaic: "
            + "/".join(f"`{m}`" for m in per_module) + " are per-module",
            "  mosaics, so those filters ship one image per module."
            + ("  (`mirimage` is MIRI.)" if "mirimage" in modules else ""),
        ]
    elif "mirimage" in modules and "merged" in modules:
        science_lines = [
            "- `*-merged_i2d.fits`        : NIRCam science mosaic (drizzled)",
            "- `*-mirimage_*i2d.fits`     : MIRI science mosaic (drizzled)",
        ]
    else:
        science_lines = ["- `*-merged_i2d.fits`        : science mosaic (drizzled)"]
    kinds = {it.get("kind") for it in images}
    if "residual" in kinds:
        science_lines.append(
            "- `*_residual_i2d.fits`      : PSF-photometry residual "
            "(highest merge iteration)")
    if "model" in kinds:
        science_lines.append(
            "- `*_model_i2d.fits`         : PSF model image (highest merge iteration)")
    # Known photometric limitations the continuity gate WAIVED for this release.
    # A JSON key in MANIFEST.json is a machine surface; a downloader reads the
    # README, so a waiver must appear here in plain text too (mirrors the frame +
    # epoch declaration rule).
    _waivers = [w for it in items for w in (it.get("continuity_waivers") or [])]
    limitation_lines = []
    if _waivers:
        limitation_lines = [
            "## Known photometric limitations (READ BEFORE USING THE PHOTOMETRY)",
            "",
            "The photometric-continuity gate WAIVED the following documented limit(s)",
            "for this release (also in `MANIFEST.json`: top-level `continuity_gate: "
            "\"waived\"` and per-catalog `continuity_waivers`):",
            "",
        ]
        for w in _waivers:
            limitation_lines.append(
                f"- **{w['pair'].upper()} saturation-boundary continuity = "
                f"{w['metric']} mag** (gate floor {CONTINUITY_TOL_MAG} mag). "
                f"{w['reason']} It affects the deepest saturated stars only "
                f"(flagged `replaced_saturated`); cut those rows for unbiased "
                f"{w['pair'].upper()} colors. Provisional -- expected to improve in "
                f"a later release.")
        limitation_lines.append("")

    # A WITHHELD INSTRUMENT is the same class of fact as a waiver, and needs the
    # same treatment: `items` here is the already-filtered list, so without this
    # the README describes a field that simply HAS no MIRI.  The distinction --
    # "this field has no MIRI" vs "this field's MIRI was withheld" -- is
    # invisible in a file list, and the reader who most needs it is the one
    # reading the README rather than MANIFEST.json.
    if withheld_instruments:
        limitation_lines += [
            "## Instruments withheld from this release (READ FIRST)",
            "",
            "This release is PARTIAL.  The products below exist and were reduced, but",
            "their registration could not be confirmed, so they are NOT shipped here:",
            "",
        ]
        for instrument, why in sorted(withheld_instruments.items()):
            limitation_lines.append(
                f"- **{instrument.upper()}**: the inter-frame overlap gate "
                f"{why}. The other instrument(s) in this release are unaffected "
                f"-- they are separate observations, on separate detectors, "
                f"often from a separate program, and this verdict says nothing "
                f"about them.")
        limitation_lines += [
            "",
            "Also in `MANIFEST.json` as `withheld_instruments`.  Do not read the",
            "absence of these bands as the field not having them.",
            "",
        ]

    # Detector-frame exposures.  Stated in the README as well as MANIFEST.json
    # because the two things a downloader has to know about them -- that they
    # are symlinks into the live pipeline tree, and that they are therefore not
    # part of the frozen, checksummed release -- are the kind of thing that gets
    # discovered from a broken link months later otherwise.
    exposures = [it for it in items
                 if it["category"] == exposure_bundle.EXPOSURE_CATEGORY]
    exposure_lines = []
    if exposures:
        exp_total = sum(it["size_bytes"] or 0 for it in exposures)
        suffixes = exposure_bundle.suffix_histogram(exposures)
        exposure_lines = [
            "## Detector-frame exposures (`exposures/`)",
            "",
            f"{len(exposures)} individual exposures ({human_size(exp_total)}) -- the",
            "frames each mosaic above was drizzled from, in the ORIGINAL DETECTOR",
            "FRAME, with the full GWCS distortion chain and this pipeline's",
            "astrometric solution. Laid out to mirror `images/`, and taken from the",
            "record each mosaic carries of what went into it, so",
            "`exposures/<...>/<FILTER>/` holds exactly the frames behind",
            "`images/<...>/<FILTER>/`.",
            "",
            "Products present here: "
            + ", ".join(f"`*_{s}.fits` ({n})" for s, n in sorted(suffixes.items()))
            + ".",
            "The last detector-frame product varies by field and filter -- `_crf` is",
            "the Stage-3 (outlier/CR-flagged) frame where one was written, otherwise",
            "the `_destreak`/`_align`/`_cal` frame the mosaic was drizzled from",
            "directly.",
            "",
            f"**These are {_link_mode_phrase(exposures)} to the pipeline's own "
            f"frames, not frozen copies.**",
            "They cost no additional storage and unlike",
            "everything else in this release they are not checksummed and are not",
            "covered by `CHECKSUMS.sha256`. A re-reduction writes a NEW file rather",
            "than rewriting these bytes, so a frame here can become an older",
            "generation than the pipeline currently holds; the release records each",
            "source's identity, which",
            "`stage_release.py --check-exposures --field <field>` compares against",
            "the sources now to report the frames that have diverged. Cite the mosaics",
            "and catalogs, which are frozen; the exposures are a working convenience",
            "for re-drizzling and per-exposure work.",
            "",
        ]
        if exposure_bundle.symlink_fallbacks(exposures):
            n_sym = len(exposure_bundle.symlink_fallbacks(exposures))
            exposure_lines[-1:] = [
                f"**{n_sym} of these frames are SYMLINKS and can only be taken by",
                "Globus transfer.** This field's data are on a different filesystem",
                "from the release tree, where a hardlink is impossible. The Globus",
                "HTTPS data plane refuses a symlink pointing out of the release tree",
                "(404); the transfer API follows it. Use",
                "`globus transfer --recursive` (or the bundle buttons on the web",
                "page), not `wget` on a per-frame URL. Mosaics, catalogs and tables",
                "are real files and are unaffected.",
                "",
            ]

    # Astrometry provenance. Present for EVERY field, including -- especially --
    # the ones with nothing to report: a release that ships detector frames and
    # says nothing about what frame they are on is asserting by omission that
    # they are tied. Proposal 1939 was ~14.8" off while saying nothing.
    provenance_lines = ["## Astrometric provenance (READ BEFORE USING THE FRAMES)", ""]
    _cfg = FIELDS.get(field)
    if _cfg is None:
        # `write_readme` must stay total: an unregistered field name is a reason
        # to say the provenance is unknown, never to abort the staging that has
        # already copied the files.
        provenance_lines.append(
            f"`{field}` is not in the release registry, so no astrometric "
            f"provenance could be determined for it.")
    else:
        _record = astrometry_provenance.collect(field, _cfg)
        # The README is the FROZEN, in-tarball surface and never got the
        # webpage's treatment: it claimed a shipped frozen table for a field
        # that ships none, and an --allow-older-version re-cut wrote today's
        # ties and today's sha256 under a June `built` timestamp. Same
        # corrections, same two inputs.
        _ships_table = any(
            it.get("category") == astrometry_provenance.ASTROMETRY_CATEGORY
            for it in items)
        if _record.get("state") == "table" and not _ships_table:
            _record = dict(_record, state="table-not-shipped")
        _built = str(built_at or "")
        _on_record = _record.get("ties") or {}
        _record = dict(_record, ties={
            f: t for f, t in _on_record.items()
            if _record.get("state") != "table-not-shipped"
            and not (_built and str(t.get("date") or "") > _built)})
        provenance_lines += astrometry_provenance.summary_lines(
            _record, n_on_record=len(_on_record))
    provenance_lines.append("")

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
        *science_lines,
        "",
        f"{len(images)} image files across "
        f"{len({it['filter'] for it in images})} filters.",
        "",
    ]
    # an image-only release ships no catalogs/ directory at all
    lines += ([
        "## Catalogs (`catalogs/`)",
        "",
        "- `basic_merged_indivexp_photometry_tables_merged_*` : final merged photometry",
        "  table (`.fits` + `.ecsv`); the `_qualcuts_oksep<proposal>` variant is the",
        "  quality-filtered subset.",
        "- `*_dao_basic_vetted.fits` : per-filter vetted catalogs.",
        "- `seed_union_iter3_*.fits` : seed source list.",
        "",
    ] if catalogs else [
        "## Catalogs",
        "",
        "**This is an image-only release: no catalogs are shipped.** The mosaics",
        "are current, but the photometry catalogs for this field are not yet",
        "certified. They will follow in a later release.",
        "",
    ]) + exposure_lines + provenance_lines + limitation_lines + [
        "## Astrometric frame and epoch (READ BEFORE TARGETING)",
        "",
        "- **Reference frame:** Gaia DR3 (via the Gaia+VIRAC2 per-field reference",
        "  catalog; VIRAC2 fill is Gaia-DR3-aligned).",
        "- **Position epoch:** the OBSERVATION epoch of this field (see",
        "  `MANIFEST.json` provenance); positions are NOT propagated to any other",
        "  epoch. GC stars move ~3-8 mas/yr -- propagate per-star proper motions",
        "  before pointed follow-up (NIRSpec MSA, slit work).",
        "- Any target list or MSA plan built from these catalogs MUST record this",
        "  frame + epoch. Lesson: NIRSpec 6927 plan v11 inherited a deprecated",
        "  ~90 mas-off frame because the source list did not state its frame.",
        "",
        "## Integrity",
        "",
        "`CHECKSUMS.sha256` lists SHA-256 for every frozen deliverable"
        + (" (not the symlinked `exposures/`, see above)." if exposures else ".")
        + " `MANIFEST.json` records",
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
    # No safe default exists for anything that WRITES: v1.0-2026.06 is a FROZEN
    # release, and `stage()` unlinks-and-copies, so a forgotten `--version` used
    # to write into the oldest published tree.  Required below for every writing
    # path.  `--check-exposures` writes nothing -- it reads a staged manifest and
    # reports -- and requiring it there made the invocation this script prints in
    # every shipped README (`--check-exposures --field <field>`) exit 2 without
    # doing anything: a documented command that does not run, which is the same
    # defect as a documented check with no implementation.
    parser.add_argument("--version", default=None,
                        help="release version directory, e.g. v1.2-2026.08. "
                             "Required except with --check-exposures, which "
                             "defaults to the newest version holding the field.")
    parser.add_argument("--allow-older-version", action="store_true",
                        help="permit staging into a version older than the newest "
                             "one present under --release-root (re-cutting a frozen "
                             "release); refused by default")
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
    parser.add_argument("--no-exposures", action="store_true",
                        help="do not offer the detector-frame exposures behind each "
                             "shipped mosaic (default: offer them, symlinked)")
    parser.add_argument("--exposures-only", action="store_true",
                        help="add ONLY the detector-frame exposures to an already-"
                             "staged release: symlinks + MANIFEST/README, leaving "
                             "images/, catalogs/ and CHECKSUMS.sha256 untouched and "
                             "not re-running the mosaic gates (it cannot change which "
                             "mosaics ship). Use when a field's frames should go out "
                             "but a band has no live mosaic to re-stage.")
    parser.add_argument("--exposures-from-disk", action="store_true",
                        help="with --exposures-only: enumerate the frames from the "
                             "pipeline directories instead of from staged mosaics, "
                             "and create the release directory if there is none. "
                             "Detector frames are a DEPENDENCY of the mosaic, so a "
                             "field can release them before anything is drizzled. "
                             "Weaker provenance than the association path -- it "
                             "answers 'which frames belong to this field/obs/filter', "
                             "not 'which frames went into that mosaic' -- so it is "
                             "opt-in, never a silent fallback.")
    parser.add_argument("--check-exposures", action="store_true",
                        help="audit a STAGED release's detector frames instead of "
                             "staging: report the frames whose source has been "
                             "rewritten or removed since they were linked, and the "
                             "ones that fell back to a symlink (not HTTPS-servable). "
                             "Exits 1 when anything has diverged. This is the check "
                             "the README tells readers the release performs.")
    parser.add_argument("--images-only", action="store_true",
                        help="ship mosaics only, no catalogs (e.g. images are internally "
                             "consistent but the catalog/absolute frame is not yet certified)")
    parser.add_argument("--allow-registration-fail", action="store_true",
                        help="stage even if the local-registration failsafe FAILs "
                             "(a band is locally misregistered). DANGEROUS -- only for "
                             "deliberate overrides; ALSO requires ALLOW_REGISTRATION_FAIL=1 "
                             "in the environment. The default refuses to stage.")
    parser.add_argument("--refuse-mixed-generations", action="store_true",
                        help="turn the generation report into a refusal: exit nonzero "
                             f"when the staged science images span more than "
                             f"{GENERATION_SPAN_DAYS:.0f} days of DATE, or more than one "
                             "CRDS_CTX, within an instrument (default: report only)")
    args = parser.parse_args(argv)

    if args.check_exposures:
        version = args.version or latest_staged_version(args.field,
                                                        args.release_root)
        if version is None:
            print(f"No staged release of '{args.field}' under "
                  f"{args.release_root}", file=sys.stderr)
            return 2
        return check_exposures(args.field, version, args.release_root)
    if args.version is None:
        parser.error("--version is required for every path that writes")

    # ---- LISTED-SOURCE GATE ---------------------------------------------------------
    # `nircam`/`miri` entries are curated by hand, so an absent one means the config is
    # stale -- most often because the m2 astrometry checkpoint quarantined the product
    # to `..._im0_badastrom.fits` after correcting an offsets table. Dropping it quietly
    # (the old behaviour) ships a release that is simply short that band, with nothing
    # printed: sickle's F210M was in exactly that state on 2026-08-05. Refuse, and name
    # the quarantined sibling so the operator can see what took the file. There is
    # deliberately no override -- the fix is a one-line config edit, not a flag.
    # Handled before everything below: this path adds no mosaic and makes no
    # mosaic decision, so the gates that decide which mosaics ship have nothing
    # to rule on. Running them anyway would refuse a field whose ALREADY-STAGED
    # release is exactly what is being added to (arches: F212N has no live
    # mosaic, so the listed-source gate refuses the field while its published
    # v1.2 tree sits there gated and frozen).
    if args.exposures_only:
        # The frozen-version guard runs HERE too, not only on the full staging
        # path below. Adding an exposures/ tree and an astrometry/ table to a
        # published version is still writing into it: it rewrites MANIFEST.json,
        # README.md and CHECKSUMS.sha256 of a release whose checksums are cited.
        try:
            field_dir, n = stage_exposures_only(
                args.field, args.version, args.release_root,
                from_disk=args.exposures_from_disk,
                allow_older=args.allow_older_version)
        except FrozenReleaseError as err:
            print(f"\n{err}", file=sys.stderr)
            return 2
        if field_dir is None:
            return 2
        if not n:
            return 1
        print(f"Added {n} detector-frame exposures (linked, not checksummed) "
              f"to {field_dir}. Mosaics and catalogs unchanged.")
        return 0

    missing = []
    exposure_problems = []
    items = build_manifest(args.field, args.version, images_only=args.images_only,
                           missing=missing, exposures=not args.no_exposures,
                           exposure_problems=exposure_problems)
    if exposure_problems:
        # Reported, never a refusal -- see build_manifest.  Printed before the
        # manifest so it is not lost under a long file list.
        print(f"\nEXPOSURE PROVENANCE: could not establish the detector-frame input "
              f"list for {len(exposure_problems)} shipped mosaic(s); their frames are "
              f"NOT offered (the mosaics themselves are unaffected):")
        for problem in exposure_problems:
            print(f"    - {problem}")
    if missing:
        print(f"\nREFUSING TO STAGE '{args.field}': {len(missing)} explicitly-listed "
              f"source file(s) are not on disk. Staging would ship a release that is "
              f"silently short those bands:", file=sys.stderr)
        for entry in missing:
            print(f"    - {entry}", file=sys.stderr)
        print("  Point the field's `nircam`/`miri` entry at the current product, or "
              "re-drizzle the missing one, then re-run.", file=sys.stderr)
        return 2
    if not items:
        print(f"No deliverables discovered for field '{args.field}'.", file=sys.stderr)
        return 1

    # ---- SUPERSEDED-SOURCE GATE ----------------------------------------------------
    # A mosaic the pipeline has already quarantined as bad-astrometry must never
    # enter a release. Cloud C shipped six of them for weeks: staged 2026-07-10,
    # superseded by the 2026-07-12 astrometry fix, still on the page in August --
    # presented as evidence that the astrometry is sound.
    #
    # Deliberately called with NO recorded size, so this sees only the RENAME
    # while the page also withholds sources REBUILT since staging.  The
    # asymmetry is the point and not an oversight: there is nothing to compare
    # against at staging time -- the size is recorded BY this run, from the file
    # it is about to copy, so a rebuilt source is simply the current one and is
    # exactly what should be staged.  The size check only becomes meaningful
    # once a manifest exists to disagree with.
    #
    # Exposures are exempt, for two reasons that point the same way.  The
    # checkpoint quarantines MOSAICS, not detector frames -- there is no
    # `*_im0_badastrom.fits` twin of a `_crf` to find -- so the check has no
    # signal to give here; and each glob over a pipeline directory holding tens
    # of thousands of entries, times several thousand frames, would cost minutes
    # to establish that.  The frames are covered by the mosaic they came from:
    # `parent_dest` withholds them wherever it is withheld.
    stale = [it for it in items
             if it["category"] != exposure_bundle.EXPOSURE_CATEGORY
             and release_freshness.source_state(it["src"]) != release_freshness.LIVE]
    if stale:
        print(f"\nREFUSING TO STAGE '{args.field}': {len(stale)} product(s) have no "
              f"live source -- the pipeline has superseded or removed them since "
              f"they were produced:", file=sys.stderr)
        for it in stale[:10]:
            print(f"  {release_freshness.source_state(it['src']):11s} {it['src']}",
                  file=sys.stderr)
        if len(stale) > 10:
            print(f"  ... and {len(stale) - 10} more", file=sys.stderr)
        print("Re-run the reduction so current mosaics exist, then stage.",
              file=sys.stderr)
        return 2

    print_manifest(items)

    # ---- GENERATION-SPAN REPORT -----------------------------------------------------
    # Everything shipped for a field should come from one reduction generation. A band
    # four months older than its neighbours is obvious afterwards and invisible at the
    # time -- v1.1 shipped sickle that way -- so make it visible before the copy.
    print("GENERATION CHECK (staged science images, per instrument):")
    span_complaints = check_generation_span(items)
    if span_complaints:
        for complaint in span_complaints:
            print(f"  MIXED GENERATIONS: {complaint}")
        if args.refuse_mixed_generations:
            print(f"\nREFUSING TO STAGE '{args.field}': the staged science images are "
                  f"not from a single reduction generation (see above). Re-drizzle the "
                  f"lagging band(s) so the whole field comes from one run, or drop "
                  f"--refuse-mixed-generations to stage anyway.", file=sys.stderr)
            return 2
        print("  (report only; pass --refuse-mixed-generations to make this a refusal)")

    if not args.stage:
        print("Dry run. Re-run with --stage to build the release tree.")
        return 0

    # ---- FROZEN-VERSION GUARD -------------------------------------------------------
    # A published version is frozen: people cite its checksums.  Staging into one
    # rewrites files under a fresh `built` timestamp with no record that it
    # happened.  Refuse to target anything older than the newest version on disk
    # unless that is explicitly what is wanted.
    # `why` is what the guard actually decided; printing hardcoded ordering text
    # instead told the operator the wrong reason -- refusing v1.3 on the
    # field-manifest branch read "it is older than the newest release on disk
    # ('v1.3-2026.08')", i.e. older than itself. And `max(existing)` was left in
    # a branch that no longer implies `existing` is non-empty: the
    # field-manifest leg can refuse with no `v*` directory present at all, which
    # raises `ValueError: max() iterable argument is empty`. Both go away by
    # using the reason the guard returned.
    refuse, why = refuse_older_version(args.version, args.release_root,
                                       args.allow_older_version, args.field)
    if refuse:
        print(f"\nREFUSING TO STAGE into {why}, and a published version is frozen "
              f"-- its checksums are cited. Stage into a new version, or pass "
              f"--allow-older-version if re-cutting it is intended.",
              file=sys.stderr)
        return 2

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
    # Positive record of the continuity gate's outcome, written to MANIFEST.json so
    # a manifest with no waiver is not ambiguous (clean vs gate-skipped vs
    # overridden). Set to a definite value on every path.
    continuity_gate = "skipped(override)" if override else None
    if not override:
        gate = Path(__file__).with_name("registration_failsafes.py")
        gate_cmd = [sys.executable, str(gate), "--field", args.field, "--scan"]
        if args.images_only:   # image-only: gate on cross-band image consistency, not own-catalog
            gate_cmd.append("--images-only")
        rc = subprocess.run(gate_cmd).returncode
        if rc == 1:
            print(f"\nREFUSING TO STAGE '{args.field}': local-registration failsafe FAILED "
                  f"-- a band's mosaic is locally misregistered vs the other bands / its own "
                  f"catalog (see the scan output above). Fix the reduction, or override with "
                  f"--allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1 (dangerous).",
                  file=sys.stderr)
            return 2
        if rc == 2:
            print(f"\nREFUSING TO STAGE '{args.field}': local-registration failsafe could "
                  f"NOT VERIFY the field (see the UNVERIFIED lines above) -- e.g. a band "
                  f"with no merged mosaic in a field whose modules overlap, so that band's "
                  f"inter-module seam was never checked. An unverified band is not a passing "
                  f"band: produce the missing mosaic, or override with "
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

        # Reference-free inter-frame overlap gate (added 2026-07-12). The
        # registration failsafe above matches the mosaic vs its OWN catalog -- both
        # derive from the same _crf frames, so a per-visit residual is
        # self-referential and cancels (brick-1182 F200W seam: ~90 mas visit-001
        # residual doubled every star in the overlap, yet mosaic-vs-catalog read
        # ~0). Only a reference-free frame-vs-frame check sees it.  (Applies to
        # --images-only too: it reads the crf frames, not catalogs.)
        overlap_gate = Path(__file__).with_name("check_interframe_overlap.py")
        overlap_cmd = [sys.executable, str(overlap_gate),
                       "--field", args.field, "--scan"]
        # Scope the reference-free gate to THIS release's observations so stray
        # crf from other programs in a shared target dir (the brick dir also
        # holds 2221 o002 = cloudc frames) cannot pollute the frame-vs-frame
        # verdict.  Derived from proposal_prefix (proposal-aware); the gate also
        # self-derives from the released mosaics when this is not passed.
        rel_obs = _release_observations(FIELDS[args.field])
        if rel_obs:
            overlap_cmd += ["--observations", ",".join(sorted(rel_obs))]
        # Pass the field's Gaia/VIRAC2 refcat so the gate can resolve pairs its
        # reference-free layer cannot measure -- a sparse / thin inter-module
        # overlap (2221 nrca-long|nrcb-long) has 0 mutual-coverage tiles, so the
        # frame-vs-frame pooled histogram is unreliable there; the same-star
        # residual map vs VIRAC2 is the authoritative arbiter (fail-closed still
        # applies if the refcat is missing).
        overlap_refcat = overlap_arbiter_refcat(args.field)
        if overlap_refcat:
            overlap_cmd += ["--refcat", overlap_refcat]
        else:
            # Said BEFORE the gate runs, because the consequence is a refusal
            # further down that names the pair rather than the missing list.
            print(f"  no overlap arbiter star list for '{args.field}' "
                  f"(OVERLAP_ARBITER_REFCAT / FRAME_REFCAT) -- any pair whose "
                  f"footprints overlap too thinly to compare frame-against-frame "
                  f"will stay unmeasurable and block staging", flush=True)
        # PER INSTRUMENT: see `gate_by_instrument`.
        items, withheld, refusal = gate_by_instrument(
            args.field, items,
            lambda instrument: subprocess.run(
                overlap_cmd + ["--instrument", instrument]).returncode)
        if refusal:
            print(f"\n{refusal}", file=sys.stderr)
            return 2

        # ---- FROZEN-STAGE ASTROMETRY CHECKPOINT GATE ----------------------------------
        # The frozen stages (m3+) and the m7 cross-filter check now RECORD a
        # failure instead of raising inside the chain
        # (ASTROM_CHECKPOINT_ENFORCE=release, the default).  That is only
        # defensible because the stop moved HERE rather than disappearing: this
        # refuses any field carrying a checkpoint record with passed=false, and
        # it fails closed on records it cannot read and on a field that has none.
        ckpt_gate = Path(__file__).with_name("check_astrometry_checkpoints.py")
        ckpt_cmd = [sys.executable, str(ckpt_gate), "--field", args.field, "--scan"]
        if rel_obs:
            ckpt_cmd += ["--observations", ",".join(sorted(rel_obs))]
        # Which stages' products this release actually SHIPS, from the items'
        # own iteration tokens (`resbgsub_m7` -> m7).  A frozen-stage failure at
        # a stage that is NOT shipped, which a LATER stage of the same chain
        # measured as passing, describes an intermediate nobody downloads --
        # brick's m5/F200W, answered by m6 and m7 (issue #258).  Declaring
        # nothing supersedes nothing, so a caller that cannot say stays strict.
        shipped_stages = sorted({m.group(0) for m in
                                 (re.search(r"m\d+$", str(it.get("iteration") or ""))
                                  for it in items) if m})
        if shipped_stages:
            ckpt_cmd += ["--shipped-stages", ",".join(shipped_stages)]
        rc = subprocess.run(ckpt_cmd).returncode
        if rc == 1:
            print(f"\nREFUSING TO STAGE '{args.field}': a frozen-stage astrometry "
                  f"checkpoint FAILED (see above). The chain was allowed to finish so "
                  f"the products exist to diagnose from; the field still does not ship. "
                  f"Fix the astrometry and re-run the affected stages, or override with "
                  f"--allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1 (dangerous).",
                  file=sys.stderr)
            return 2
        if rc == 3:
            print(f"\nREFUSING TO STAGE '{args.field}': no astrometry checkpoint records "
                  f"for this field/observation set. A field that never ran the checkpoint "
                  f"is unverified, not verified. Run the cataloging chain, or override "
                  f"with --allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1.",
                  file=sys.stderr)
            return 2
        if rc != 0:
            print(f"\nREFUSING TO STAGE '{args.field}': the astrometry checkpoint gate "
                  f"could not run (rc={rc}); cannot confirm the frozen-stage astrometry.",
                  file=sys.stderr)
            return 2

        # ---- SAME-RUN GATE: image <-> catalog provenance -------------------------------
        # When a release ships BOTH images and per-filter catalogs, they MUST come from
        # the same pipeline/cataloging run. We enforce it directly: each shipped science
        # image must agree with its shipped per-filter catalog to < SAME_RUN_TOL_MAS. A
        # mismatch = different runs (different astrometric solutions) -> refuse. (Skipped
        # for --images-only, which ships no catalogs.)
        if not args.images_only:
            print("\nSAME-RUN CHECK (shipped image <-> shipped catalog):")
            fails = check_image_catalog_match(items)
            if fails:
                detail = "; ".join(f"{f}{('/' + o) if o else ''}: {v:.0f} mas"
                                   for (f, o), v in fails)
                print(f"\nREFUSING TO STAGE '{args.field}': image<->catalog SAME-RUN check "
                      f"FAILED (> {SAME_RUN_TOL_MAS:.0f} mas): {detail}. The shipped image "
                      f"and catalog are from DIFFERENT runs (different astrometric "
                      f"solutions) and must not be released together. Rebuild both from one "
                      f"run, or override with --allow-registration-fail AND "
                      f"ALLOW_REGISTRATION_FAIL=1 (dangerous).", file=sys.stderr)
                return 2

        # ---- ABSOLUTE-FRAME GATE: catalogs on the Gaia(DR3)=VIRAC2 frame ----------------
        # A catalog reduced against a deprecated crowdsource/VVV/2MASS refcat is ~20-90 mas
        # off Gaia (the frame that hit the NIRSpec 6927 MSA plan). Enforce astrometrically:
        # each shipped catalog's bulk offset vs the field's Gaia-tied refcat must be < tol.
        if not args.images_only:
            print("\nABSOLUTE-FRAME CHECK (shipped catalog <-> Gaia-tied refcat):")
            frame_fails = check_catalog_on_frame(items, args.field)
            if frame_fails is None:
                print(f"  no Gaia refcat mapped for '{args.field}' in FRAME_REFCAT -- "
                      f"cannot enforce the frame gate; add its refcat to enforce.")
            elif frame_fails:
                detail = "; ".join(f"{f}{('/' + o) if o else ''}: {v:.0f} mas"
                                   for (f, o), v in frame_fails)
                print(f"\nREFUSING TO STAGE '{args.field}': catalog(s) OFF the Gaia/VIRAC2 "
                      f"frame (> {FRAME_TOL_MAS:.0f} mas): {detail}. A ~1-pixel bulk offset "
                      f"means the catalog is on a deprecated crowdsource/VVV/2MASS frame, not "
                      f"Gaia -- it must be re-anchored + re-reduced before release. Override "
                      f"only with --allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1.",
                      file=sys.stderr)
                return 2

        # ---- PHOTOMETRIC-CONTINUITY GATE: saturated-star flux scale ---------------------
        # The astrometric gates above cannot see a catalog whose saturated-star or
        # sub-floor-strip photometry sits on a different FLUX scale (detached CMD
        # clouds, degenerate-color drift -- the 2026-07 satstar failure class). Enforce
        # the executable certifiers on the shipped combined merged table. Same
        # hard-to-reach override as the astrometry gates.
        if not args.images_only:
            print("\nPHOTOMETRIC-CONTINUITY CHECK (saturation boundary + degenerate pairs):")
            cont_fails = check_photometric_continuity(items)
            if cont_fails is None:
                print("  no combined merged table shipped -- cannot enforce the "
                      "continuity gate.")
                continuity_gate = "not_enforced(no-merged-table)"
            elif cont_fails:
                detail = "; ".join(cont_fails)
                print(f"\nREFUSING TO STAGE '{args.field}': photometric-continuity "
                      f"certification FAILED (>= {CONTINUITY_TOL_MAG:.2f} mag): {detail}. "
                      f"Saturated-star / sub-floor photometry is on a different flux scale "
                      f"than normal photometry -- the CMD breaks at the saturation "
                      f"boundary. Re-catalog with the satstar channel fixes, or override "
                      f"with --allow-registration-fail AND ALLOW_REGISTRATION_FAIL=1.",
                      file=sys.stderr)
                return 2
            else:
                continuity_gate = ("waived" if any(it.get("continuity_waivers")
                                                   for it in items) else "passed")
        else:
            continuity_gate = "not_applicable(images-only)"

    mode = "copy" if args.copy else "symlink"
    field_dir = stage(items, args.field, args.version, args.release_root,
                      mode, not args.no_checksum, continuity_gate=continuity_gate,
                      allow_older=args.allow_older_version,
                      withheld_instruments=withheld or None)
    # Broken out rather than reported as one total: "222 files (mode: copy)"
    # reads as 222 frozen copies when 216 of them are symlinks that --copy did
    # not apply to, which is the one thing about this tree an operator must not
    # misread.
    staged_exposures = [it for it in items
                        if it["category"] == exposure_bundle.EXPOSURE_CATEGORY]
    n_exposures = len(staged_exposures)
    exp_mode = exposure_bundle.link_mode_summary(staged_exposures)
    print(f"Staged {len(items) - n_exposures} deliverables into {field_dir} "
          f"(mode: {mode})"
          + (f", plus {n_exposures} detector-frame exposures "
             f"({exp_mode} links, not checksummed)." if n_exposures else "."))

    if args.set_acl:
        set_acl(args.field, args.version, args.release_root)

    if args.print_urls:
        print("\nDownload URLs:")
        for it in items:
            print(it["url"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
