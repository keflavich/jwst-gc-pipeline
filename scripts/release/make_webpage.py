#!/usr/bin/env python
"""
Generate the static distribution website for staged JWST-GC release fields.

Reads each field's ``MANIFEST.json`` (written by stage_release.py) and emits a
self-contained site:

    <out>/index.html            field grid with preview thumbnails
    <out>/<field>.html          per-field page: preview + image/catalog tables
                                with Globus download links, sizes, checksums
    <out>/assets/<field>.jpg    field preview (copied from the release preview/)

Links point at the Globus HTTPS URLs recorded in each MANIFEST.  Deploy with
``scripts/release/deploy_site.sh`` -- NOT a hand-written rsync.  ``htdocs/jwst-gc``
also holds the pipeline monitor in ``monitor/``, which is not in <out>, so an
rsync carrying --delete removes it (as happened 2026-08-06); the script protects
that tree and verifies it is still there afterwards.
"""
import argparse
import collections
import hashlib
import html
import json
import os
import re
import shutil
import urllib.parse
from pathlib import Path

import curated_images
import field_overview
import release_freshness
import make_preview_rgb
import preview_plan
from stage_release import field_release_dir

# Display label per group folder (None = Galactic Center, the default survey).
GROUP_LABEL = {
    None: "JWST Galactic Center survey",
    "galactic_plane": "JWST Galactic Plane fields",
    "globular_clusters": "JWST Globular Clusters",
}
GROUP_TITLE = {
    None: "Galactic Center",
    "galactic_plane": "Galactic Plane",
    "globular_clusters": "Globular Clusters",
}

GLOBUS_APP = "https://app.globus.org/file-manager"
GLOBUS_COLLECTION_ID = "d9873d5e-0fbd-4980-aedf-4ca56f65a045"

# Standalone token helper served from the site for scripted wget/curl downloads.
TOKEN_HELPER = '''#!/usr/bin/env python
"""Mint a Globus HTTPS bearer token for the JWST root collection so you can
download release files with wget/curl.

  pip install globus-sdk      # once
  python get_globus_token.py  # opens an ORCID/Globus login in your browser

Then:
  TOKEN=<paste the printed token>
  wget --header="Authorization: Bearer $TOKEN" -i sgrb2_files.txt
The token lasts ~48 h; re-run to get a fresh one.
"""
import globus_sdk

CLIENT_ID = "3b1925c0-a87b-452b-a492-2c9921d3bd14"   # Globus tutorial native client
COLLECTION = "d9873d5e-0fbd-4980-aedf-4ca56f65a045"  # JWST root

scope = f"https://auth.globus.org/scopes/{COLLECTION}/https"
client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
client.oauth2_start_flow(requested_scopes=scope, refresh_tokens=False)
print("\\n1. Open this URL in a browser and log in (ORCID works):\\n")
print("   " + client.oauth2_get_authorize_url() + "\\n")
code = input("2. Paste the authorization code here: ").strip()
tokens = client.oauth2_exchange_code_for_tokens(code)
token = tokens.by_resource_server[COLLECTION]["access_token"]
print("\\nBearer token (valid ~48h):\\n")
print(token)
'''

FILTER_WAVELENGTH = {  # micron, for ordering/labels
    "F115W": 1.15, "F150W": 1.50, "F162M": 1.62, "F182M": 1.82, "F187N": 1.87,
    "F200W": 2.00, "F210M": 2.10, "F212N": 2.12, "F277W": 2.77, "F300M": 3.00,
    "F323N": 3.23, "F356W": 3.56, "F360M": 3.60, "F405N": 4.05, "F410M": 4.10,
    "F444W": 4.44, "F466N": 4.66, "F470N": 4.70, "F480M": 4.80,
    # MIRI
    "F560W": 5.6, "F770W": 7.7, "F1000W": 10.0, "F1130W": 11.3, "F1280W": 12.8,
    "F1500W": 15.0, "F1800W": 18.0, "F2100W": 21.0, "F2550W": 25.5,
}

CSS = """
:root { --bg:#0d1117; --panel:#161b22; --fg:#e6edf3; --muted:#8b949e;
        --accent:#58a6ff; --border:#30363d; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); margin:0;
       font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       line-height:1.5; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
header, main, footer { max-width:1100px; margin:0 auto; padding:1.5rem; }
header { border-bottom:1px solid var(--border); }
h1 { margin:0 0 .3rem; font-size:1.7rem; }
h2 { border-bottom:1px solid var(--border); padding-bottom:.3rem; margin-top:2rem; }
.muted { color:var(--muted); font-size:.9rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
        gap:1rem; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:8px;
        overflow:hidden; }
/* fixed thumbnail aspect so the grid stays aligned whatever shape the field
   is -- `contain` letterboxes (against the black background) rather than
   cropping, because a portrait field's cluster is not necessarily centred. */
.card img { width:100%; display:block; background:#000;
            aspect-ratio:2/1; object-fit:contain; }
.card .body { padding:.8rem 1rem; }
.preview { width:100%; border:1px solid var(--border); border-radius:8px;
           margin:1rem 0; background:#000; }
.previews { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
            gap:1rem; margin:1rem 0; }
.previews figure { margin:0; }
.previews .preview { margin:0 0 .35rem; }
table { width:100%; border-collapse:collapse; font-size:.9rem; margin:.5rem 0 1.5rem; }
th, td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--border); }
th { color:var(--muted); font-weight:600; }
td.size { white-space:nowrap; color:var(--muted); }
code { background:#0b0f14; padding:.1rem .35rem; border-radius:4px; font-size:.82em; }
.tag { display:inline-block; background:#1f2937; color:var(--accent);
       border:1px solid var(--border); border-radius:999px; padding:.05rem .55rem;
       font-size:.78rem; }
.checksum { font-family:monospace; color:var(--muted); font-size:.78rem; }
.bulk { background:var(--panel); border:1px solid var(--border); border-radius:8px;
        padding:1rem 1.2rem; margin:1rem 0; }
.btn { display:inline-block; background:var(--accent); color:#0d1117;
       font-weight:600; padding:.5rem 1rem; border-radius:6px; margin:.3rem .5rem .3rem 0; }
.btn:hover { text-decoration:none; filter:brightness(1.1); }
.btn.secondary { background:#21262d; color:var(--fg); border:1px solid var(--border); }
"""

KIND_LABEL = {
    "science": "Science mosaic",
    "residual": "PSF residual",
    "model": "PSF model",
    "catalog_full": "Merged catalog (full)",
    "catalog_qualcut": "Merged catalog (quality-cut)",
    "seed": "Seed source list",
    "catalog_per_filter_vetted": "Per-filter vetted",
}


def human_size(num_bytes):
    if not num_bytes:
        return "?"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit in ("B", "KB") else f"{size:.1f}{unit}"
        size /= 1024


def page_head(title):
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{CSS}{field_overview.CSS}</style></head><body>")


def dl(item):
    return f"<a href='{html.escape(item['url'])}'>download</a>"


def _version_dropdown(field, active_version, all_versions):
    """A <select> to switch this region's page between release versions.
    Latest -> <field>.html; older -> <field>.<version>.html (static, no server)."""
    if not all_versions or len(all_versions) < 2:
        return ""
    opts = []
    latest = all_versions[0]
    for v in all_versions:
        href = f"{field}.html" if v == latest else f"{field}.{v}.html"
        sel = " selected" if v == active_version else ""
        label = html.escape(v) + (" (latest)" if v == latest else "")
        opts.append(f"<option value='{html.escape(href)}'{sel}>{label}</option>")
    return ("<label class=muted style='margin-left:1em'>version "
            "<select onchange='if(this.value)location=this.value'>"
            + "".join(opts) + "</select></label>")


def _preview_caption(stem, field):
    """'R=F444W, G=F410M, B=F405N' (and the pointing) from a preview filename.

    Filenames are ``<field>[_<obs>]_rgb_<R>_<G>_<B>``; a two-filter preview
    writes ``mean`` for green, which is what it actually is.
    """
    head, _, bands = stem.partition("_rgb_")
    parts = bands.split("_")
    obs = head[len(field) + 1:] if head.startswith(field + "_") else ""
    label = ""
    if len(parts) == 3:
        r, g, b = (p.upper() for p in parts)
        label = (f"R={r}, G=mean({r},{b}), B={b}" if g == "MEAN"
                 else f"R={r}, G={g}, B={b}")
    return (f"{obs.upper()} - " if obs else "") + (label or stem)


def _avm_read(xmp, field):
    """The ``<rdf:li>`` values of one AVM bag as floats, or ``None``."""
    block = re.search(r'<avm:' + field + r'>(.*?)</avm:' + field + r'>', xmp, re.S)
    if not block:
        return None
    items = re.findall(r'<rdf:li>(.*?)</rdf:li>', block.group(1), re.S)
    try:
        return [float(v) for v in items] or None
    except ValueError:
        return None


def _avm_bag(xmp, field, values):
    """Rewrite one ``<avm:Field>``'s ``<rdf:li>`` values in place.

    Returns ``xmp`` unchanged when the field is absent, and rewrites however
    many items the bag holds -- ``Spatial.Scale`` has two, ``Spatial.CDMatrix``
    four.  The first version hard-coded two, which silently left a CDMatrix
    untouched while the rest of the record was rescaled.
    """
    block = re.search(r'(<avm:' + field + r'>)(.*?)(</avm:' + field + r'>)',
                      xmp, re.S)
    if block is None:
        return xmp
    inner, index = block.group(2), [0]

    def _sub(match):
        i = index[0]
        index[0] += 1
        if i >= len(values):
            return match.group(0)
        return match.group(1) + f"{values[i]:.16f}" + match.group(3)

    rewritten = re.sub(r'(<rdf:li>)(.*?)(</rdf:li>)', _sub, inner, flags=re.S)
    return xmp[:block.start()] + block.group(1) + rewritten + block.group(3) \
        + xmp[block.end():]


def _avm_for_resize(xmp, orig_size, new_size):
    """The source AVM, corrected for a resize -- or ``None`` if it cannot be.

    Carrying the AVM through UNCHANGED would be worse than dropping it: these
    renders are 8-12k px wide and the web copy is 2200, so the original
    ``Spatial.Scale`` and ``Spatial.ReferencePixel`` describe a grid ~5x larger
    than the pixels they would be attached to.  A viewer that trusts AVM would
    put every source in the wrong place, on a release whose stated purpose is
    being evidence the astrometry is right.  Silence is better than a confident
    wrong answer; a corrected answer is better than both.

    ``Spatial.FITSheader`` is DROPPED rather than corrected.  It is a full FITS
    header for the original grid (CRPIX/CD of an 11796x8219 image), it is
    redundant with the AVM keywords beside it, and rewriting a FITS header with
    regexes to keep a nice-to-have is not a trade worth making.
    """
    if not xmp:
        return None
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "replace")
    if tuple(orig_size) == tuple(new_size):
        return xmp.encode("utf-8")
    # PER AXIS, from the size actually written.  One factor for both is wrong
    # whenever `round()` lands differently on the two axes, which it does for
    # every portrait render here: the limiting axis hits 2200 exactly and the
    # other is rounded, so the true x and y factors differ.  Measured on the
    # five portrait renders that exposed it, using a single factor left the
    # angular extent off by 6e-5 to 5e-4 -- tens of mas on these fields.
    sx = new_size[0] / orig_size[0]
    sy = new_size[1] / orig_size[1]
    refpix = _avm_read(xmp, r'Spatial\.ReferencePixel')
    # The pixel scale is given EITHER as Spatial.Scale (+ Spatial.Rotation) OR
    # as a Spatial.CDMatrix; both are legal AVM and this set uses both.  Only
    # Scale was handled at first, so 8 of the 32 curated renders -- every
    # gc2211 pointing, brick MIRI, sgrb2 MIRI, both sgrc -- fell to the
    # `return None` and were published with their astrometry dropped.
    pxscale = _avm_read(xmp, r'Spatial\.Scale')
    cdmatrix = _avm_read(xmp, r'Spatial\.CDMatrix')
    if not refpix or len(refpix) < 2 or not (pxscale or cdmatrix):
        return None                      # not a shape we can correct
    out = _avm_bag(xmp, r'Spatial\.ReferenceDimension',
                   [float(new_size[0]), float(new_size[1])])
    # (p - 0.5)*s + 0.5, not p*s.  AVM's ReferencePixel is 1-based like FITS
    # CRPIX, and a resize maps pixel CENTRES -- verified against PIL directly:
    # for a ramp downscaled 4x, mean|v - ((x+0.5)/s - 0.5)| = 0.003 against
    # mean|v - x/s| = 1.50.  The naive form is what pyavm's own
    # `to_wcs(target_shape=)` does (`crpix *= scale`), and it puts the reference
    # pixel up to half a source pixel out: measured across these renders, the
    # published position moved by up to 155 mas against the source AVM.
    out = _avm_bag(out, r'Spatial\.ReferencePixel',
                   [(refpix[0] - 0.5) * sx + 0.5,
                    (refpix[1] - 0.5) * sy + 0.5])
    # deg/px grows as the image shrinks; the CD matrix is the same statement in
    # matrix form, so every element scales the same way.  Rotation, projection,
    # reference value and quality are all resize-invariant and left alone.
    if pxscale:
        out = _avm_bag(out, r'Spatial\.Scale', [pxscale[0] / sx, pxscale[1] / sy])
    if cdmatrix and len(cdmatrix) == 4:
        # CD maps (dx, dy) -> (dxi, deta): the first and third elements multiply
        # dx, the second and fourth multiply dy.
        out = _avm_bag(out, r'Spatial\.CDMatrix',
                       [cdmatrix[0] / sx, cdmatrix[1] / sy,
                        cdmatrix[2] / sx, cdmatrix[3] / sy])
    out = re.sub(r'<avm:Spatial\.FITSheader>.*?</avm:Spatial\.FITSheader>', '',
                 out, flags=re.S)
    return out.encode("utf-8")


def _needs_avm_refresh(src, dest):
    """Would re-encoding add astrometry the published copy does not have?"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(src) as img:
            if not (img.info.get("XML:com.adobe.xmp") or img.info.get("xmp")):
                return False                  # nothing to carry
        with Image.open(dest) as img:
            return not (img.info.get("xmp") or img.info.get("XML:com.adobe.xmp"))
    except (OSError, ValueError):
        return True            # unreadable either side -> rebuild and find out


def web_jpeg(src, dest, max_px=2200):
    """Web-sized JPEG of a curated PNG, composited onto black (they carry alpha).

    Carries the AVM/XMP astrometry through, rescaled to the size actually
    written (see ``_avm_for_resize``).  Re-encoding used to drop it silently,
    which turned a WCS-carrying render into a flat picture the moment it was
    published -- the one property that made these images evidence rather than
    decoration.

    Skipped when the destination is newer than the source -- these renders are
    13-147 MB and rebuilding them every page build costs minutes for no change.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    if dest.is_file() and dest.stat().st_mtime >= os.path.getmtime(src):
        # ... but only if it already has everything this function now writes.
        # The mtime skip is what makes a page build affordable, and it silently
        # made the AVM work a no-op wherever a JPEG already existed: the
        # deployment target holds 31 curated JPEGs built before the AVM was
        # carried, every one newer than its source, so a rebuild there would
        # have skipped all 31 and written "avm": "absent" into every provenance
        # record while claiming the astrometry now survives.
        if not _needs_avm_refresh(src, dest):
            return dest
    img = Image.open(src)
    xmp = img.info.get("XML:com.adobe.xmp") or img.info.get("xmp")
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (0, 0, 0))
        flat.paste(img, mask=img.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")
    original_size = img.size
    scale = min(1.0, max_px / img.width, max_px / img.height)
    if scale < 1.0:
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))), Image.LANCZOS)
    corrected = _avm_for_resize(xmp, original_size, img.size)
    if corrected:
        img.save(dest, format="JPEG", quality=88, progressive=True, xmp=corrected)
    else:
        img.save(dest, format="JPEG", quality=88, progressive=True)
    return dest


def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def curated_provenance(entry, published, sidecar=None):
    """What this published picture is, and what it was made from.

    A staged mosaic reaches the page with a manifest entry, a size, a checksum
    and a source path; a curated render reached it with a filename.  This PR
    makes the curated render the field's PRIMARY image, so it was the least
    accountable thing on the page while being the most prominent -- the same
    argument the superseded-source gate makes about mosaics, pointed at the
    picture instead.

    The record is written beside the asset and collected into
    ``<field>_curated.json``.  Hashing is cached on ``(size, mtime)`` in the
    sidecar: these are 13-147 MB PNGs and there are 32 of them, so re-hashing
    every build is ~4.6 GB of reads for an answer that has not changed.
    """
    src = entry["file"]
    stat = os.stat(src)
    record = None
    if sidecar is not None and sidecar.is_file():
        try:
            cached = json.loads(sidecar.read_text())
            if (cached.get("source_bytes") == stat.st_size
                    and cached.get("source_mtime") == int(stat.st_mtime)):
                record = cached
        except (OSError, ValueError):
            record = None
    if record is None:
        record = {
            "stem": entry["stem"],
            # basename + parent only: the absolute path is a host detail and
            # this record is served publicly.
            "source": os.path.join(os.path.basename(os.path.dirname(src)),
                                   os.path.basename(src)),
            "source_name": os.path.basename(src),
            "source_bytes": stat.st_size,
            "source_mtime": int(stat.st_mtime),
            "source_sha256": _sha256(src),
        }
    record.update({
        "label": entry.get("label"),
        "pointing": entry.get("pointing"),
        "instrument": entry.get("instrument"),
        "published": os.path.basename(str(published)),
        "published_sha256": _sha256(published),
        "published_bytes": os.path.getsize(published),
        "avm": _published_avm_state(published),
    })
    if sidecar is not None:
        sidecar.write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def _published_avm_state(published):
    """``carried`` / ``absent`` -- read back off the file that was WRITTEN.

    Read back rather than inferred from what the writer intended: the whole
    defect was that re-encoding dropped the AVM while every line of code still
    said it was there.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(published) as img:
            has = bool(img.info.get("xmp") or img.info.get("XML:com.adobe.xmp"))
    except (OSError, ValueError):
        return "unknown"
    return "carried" if has else "absent"


def curated_withholding_inputs(manifest):
    """``(withheld_obs, withheld_bands, known_obs)`` for the curated rule.

    QUARANTINED only, not every superseded state.  A curated render is an
    independent product with no recorded link to a particular build of the
    mosaic, so "the source was rebuilt in place" says nothing about the picture
    -- but "the pipeline repudiated this mosaic's astrometry" condemns
    everything drawn from it, which is the gc2211 o050 case.

    The difference is most of the tree: on the REBUILT reading brick loses two
    of its three curated images to 23 rebuilt v1.0 sources and ZERO
    quarantines, and cloudc v1.1 loses its only NIRCam render to six rebuilds
    and no quarantines -- the fields whose pictures this exists to publish.
    Withholding on the quarantine still withholds gc2211 o050 and the whole of
    sgrc / sgrb2 / cloudc v1.0, which are the real repudiations.

    Split out of ``main`` for the reason the rule itself was: which STATES feed
    the rule is as much of a decision as the rule, and inline it could only be
    asserted by reading the source.
    """
    if not manifest:
        return set(), set(), set()
    reasons = release_freshness.superseded_reasons(manifest)
    quarantined = {dest for dest, why in reasons.items()
                   if why == release_freshness.QUARANTINED}
    entries = [f for f in manifest.get("files", [])
               if f.get("dest") in quarantined]
    withheld_obs = set()
    for entry in entries:
        match = re.search(r'[-_](o\d{3})[_.\-]', str(entry.get("src") or ""))
        if match:
            withheld_obs.add(match.group(1).lower())
    withheld_bands = {(e.get("filter") or "").upper() for e in entries
                      if e.get("filter")}
    # Which pointings this field actually HAS.  A curated entry's `pointing` is
    # trusted as an observation only if it looks like one or is one of these --
    # cloudc's MIRI renders carry `pointing='MIRI F770W'`, which is a label.
    known_obs = {(f.get("observation") or "").lower()
                 for f in manifest.get("files", []) if f.get("observation")}
    # No `| withheld_obs`: it can never add anything.  `withheld_obs` is built
    # by the `o\d{3}` search above, so every member already matches
    # OBS_TOKEN_RE and is recognised on shape alone.  The union looked like
    # defence in depth and was unreachable -- deleting it changed no test.
    return withheld_obs, withheld_bands, known_obs


def curated_withheld_reason(entry, withheld_obs=(), withheld_bands=(),
                            known_obs=()):
    """Why this curated render must not be published, or ``None``.

    A pure function on purpose.  This lived inline in ``main()`` and the two
    tests guarding it read the enclosing SOURCE, so `if pointing in
    withheld_obs:` could be replaced with `if False:` and the suite stayed
    green -- which republishes gc2211 o050, the render whose source was
    quarantined a day after it was made, as the field's primary image.  A
    substring assertion is not a case.

    Withhold on the OBSERVATION, not the band.  gc2211 is five observations in
    one field and all five share F200W/F277W, so a band-level test cannot
    express "o050 is repudiated, o049 is fine" -- it withholds o049, the one
    region whose astrometry is actually good (~50 mas, against o050's 5.6").

    A render with no pointing to reason about falls back to the band names in
    its label, and if THAT cannot decide either it is withheld: a curated image
    whose provenance cannot be tied to a live mosaic must not be the field's
    primary image while something on that field is repudiated.
    """
    return withheld_reason(
        pointing=entry.get("pointing"),
        bands=re.findall(r'F\d{3,4}[WMN]', str(entry.get("label") or "").upper()),
        withheld_obs=withheld_obs, withheld_bands=withheld_bands,
        known_obs=known_obs)


#: What an observation token looks like (`o023`, `o001-002`).  A `pointing`
#: that is not one of these is a LABEL, and must not be reasoned about as an
#: observation -- see `withheld_reason`.
OBS_TOKEN_RE = re.compile(r'^o\d{3}(-\d{3})?$')


def withheld_reason(pointing, bands, withheld_obs=(), withheld_bands=(),
                    known_obs=()):
    """The ONE withholding rule, for a curated render or a generated preview.

    Both ask the same question -- may this picture be published, given which
    mosaics the pipeline has repudiated -- and they had two implementations of
    it, one keyed off a registry entry and one off a preview filename.  They
    already disagreed (see ``known_obs``), which is what two copies of a rule do.

    A pointing decides ALONE, but only when it is recognisably an observation.
    That qualification is the fix for a fail-open: cloudc's curated MIRI renders
    carry ``pointing='MIRI F770W'``, which is a label, not an obs token.  It is
    never in ``withheld_obs``, so the obs branch returned "publish" and the band
    fallback never ran -- the render was published unconditionally, even with
    F770W withheld.  A pointing is therefore trusted only when it LOOKS like an
    observation (``OBS_TOKEN_RE``) or is one the manifest actually has
    (``known_obs``); anything else falls through to the bands.

    Recognising the token by shape rather than by membership alone matters: a
    live pointing is usually absent from ``withheld_obs`` by definition, so a
    membership-only test withholds every pointing whose siblings share its
    bands -- which is the exact behaviour this rule exists to avoid.
    """
    withheld_obs = {str(o).lower() for o in (withheld_obs or ())}
    withheld_bands = {str(b).upper() for b in (withheld_bands or ()) if b}
    known_obs = {str(o).lower() for o in (known_obs or ())} | withheld_obs
    pointing = str(pointing or "").lower()
    if pointing and (OBS_TOKEN_RE.match(pointing) or pointing in known_obs):
        # a picture of a live pointing stays live even when a sibling pointing
        # shares every one of its bands -- the case no band rule can express
        if pointing in withheld_obs:
            return (f"observation {pointing}'s mosaic has been superseded "
                    f"since it was rendered")
        return None
    if not (withheld_obs or withheld_bands):
        return None
    bands = {str(b).upper() for b in (bands or ()) if b}
    if not bands or (bands & withheld_bands):
        return (f"cannot tie it to a live mosaic (bands="
                f"{sorted(bands) or 'unknown'}, "
                f"withheld={sorted(withheld_bands)})")
    return None


def render_field_page(field, manifest, preview_rel, preview_channels=None,
                      all_versions=None, preview_version=None, previews=(),
                      superseded=(), reasons=None, curated=(),
                      curated_prov=(), preview_from_curated=False):
    # A staged image whose SOURCE has since been quarantined as bad-astrometry
    # must not be presented as this field's astrometry. It is withheld from the
    # page, and the withholding is stated -- the point of the release is to be
    # evidence the astrometry is right, so quietly serving a superseded mosaic
    # is worse than showing nothing. The file itself is untouched.
    superseded = set(superseded or ())
    reasons = dict(reasons or {})
    files = [f for f in manifest["files"] if f.get("dest") not in superseded]
    withheld = [f for f in manifest["files"] if f.get("dest") in superseded]
    # A preview RENDERED FROM a withheld mosaic is the thing a reader actually
    # looks at, so withholding the download row and leaving the picture up
    # publishes the bad astrometry anyway.
    #
    # Withhold on the OBSERVATION where the manifest has one.  Keying on the
    # band alone is wrong for a multi-pointing field: gc2211's five
    # observations all ship F200W/F277W, so superseding o050 dropped EVERY
    # preview -- including ones rendered entirely from o023's live mosaics --
    # while the same page still offered o023's F200W and F277W for download.
    # Both halves of the manifest entry are there; only one was being read.
    withheld_bands = {(f.get("filter") or "").upper() for f in withheld}
    withheld_obs = {(f.get("observation") or "").lower()
                    for f in withheld if f.get("observation")}
    live_obs = {(f.get("observation") or "").lower()
                for f in files if f.get("observation")}

    def _keep_preview(stem):
        # Same rule as the curated renders, through the same function -- see
        # `withheld_reason`.  Only the parsing differs: a curated entry carries
        # its pointing and bands as fields, a preview carries them in its stem.
        head, sep, bands = stem.partition("_rgb_")
        if not sep:
            # Off-convention stem: nothing can be established about what it was
            # rendered from, so there are no bands to tie it with -- which the
            # shared rule already treats as withhold-if-anything-is-withheld.
            bands = ""
        obs = head[len(field) + 1:] if head.startswith(field + "_") else ""
        return withheld_reason(
            pointing=obs, bands=[p for p in bands.split("_") if p],
            withheld_obs=withheld_obs, withheld_bands=withheld_bands,
            known_obs=live_obs) is None

    if withheld_bands or withheld_obs:
        kept = [(rel, stem) for rel, stem in previews if _keep_preview(stem)]
        if len(kept) != len(previews) and not curated:
            # `assets/<field>.jpg` is a byte COPY of previews[0], so dropping
            # that render from the gallery does not stop it being the page's
            # headline image and the index card's thumbnail.  Re-point at a
            # survivor; only an empty survivor list means no picture at all.
            #
            # Skipped when the field has a CURATED image: there `preview_rel` is
            # the curated render, which has already been through its own
            # (observation-keyed) withholding upstream, and re-pointing it at a
            # generated preview would demote the better picture.
            preview_rel = kept[0][0] if kept else None
        previews = kept
    images = [f for f in files if f["category"] == "image"]
    catalogs = [f for f in files if f["category"] == "catalog"]
    filters = sorted({f["filter"] for f in images if f["filter"]},
                     key=lambda x: FILTER_WAVELENGTH.get(x, 99))

    survey = GROUP_LABEL.get(manifest.get("group"), GROUP_LABEL[None])
    out = [page_head(f"{survey} — {field} — {manifest['version']}")]
    out.append("<header>")
    out.append(f"<h1>{html.escape(survey)} — {html.escape(field)}</h1>")
    out.append(f"<div class=muted>Release {html.escape(manifest['version'])} · "
               f"built {html.escape(manifest['built'][:10])} · "
               f"<a href='index.html'>← all fields</a>"
               f"{_version_dropdown(field, manifest['version'], all_versions)}</div>")
    out.append("</header><main>")
    if withheld:
        # Two states, two sentences.  This notice used to assert -- for every
        # withheld image -- that "the m2 astrometry checkpoint corrected the
        # offsets table and quarantined every mosaic built before it".  That is
        # true only where the pipeline actually renamed the file.  For a source
        # REBUILT in place, all that is known is one `stat` disagreeing with the
        # recorded size; a re-run, a re-chunk or a new stage all look identical,
        # and the rebuilt case is now the majority (52 of 116).  Naming a
        # quarantine that did not happen is a public-facing false statement --
        # it would have appeared on 23 of brick's 31 frozen v1.0 images.
        quarantined = [f for f in withheld
                       if reasons.get(f.get("dest")) == release_freshness.QUARANTINED]
        rebuilt = [f for f in withheld if f not in quarantined]

        def _bands(group):
            return html.escape(', '.join(sorted({f.get("filter") or "?"
                                                 for f in group})))
        lines = []
        if quarantined:
            lines.append(
                f"<b>{len(quarantined)} withheld as bad astrometry.</b> The "
                f"pipeline repudiated the mosaics these were staged from -- the "
                f"m2 astrometry checkpoint corrected the offsets table and "
                f"quarantined every mosaic built before it: "
                f"<code>{_bands(quarantined)}</code>.")
        if rebuilt:
            # Deliberately says less.  This bucket is "the bytes on disk are no
            # longer the staged ones", which also covers a reason the caller did
            # not supply, so it must not name a cause of any kind.
            lines.append(
                f"<b>{len(rebuilt)} withheld as superseded.</b> The sources "
                f"these were staged from are no longer the files they were "
                f"copied from -- they have been rebuilt or replaced since, so "
                f"these are the older bytes. Why is not recorded here, and no "
                f"claim is made about their astrometry: "
                f"<code>{_bands(rebuilt)}</code>.")
        out.append(
            "<p style='border:1px solid #b58900;padding:.6rem 1rem;border-radius:6px'>"
            + " ".join(lines)
            + " They will return when the field is re-staged from current"
              " products.</p>")
    if all_versions and manifest['version'] != all_versions[0]:
        out.append(f"<p class=muted style='border:1px solid #b58900;padding:.5em'>"
                   f"You are viewing an <b>older</b> release ({html.escape(manifest['version'])}). "
                   f"The latest is <a href='{html.escape(field)}.html'>{html.escape(all_versions[0])}</a>.</p>")

    multi = any(f.get("observation") for f in files)

    if curated:
        out.append("<p class=muted>Curated colour images -- the published "
                   "renders. Tuned stretches, channels chosen far apart, and "
                   "each program combined with itself.</p>")
        # Provenance per published render, keyed on the asset filename.  These
        # are the field's PRIMARY images and they are not staged products, so
        # without this they would be the only thing on the page with no source,
        # no checksum and no statement of what survived publication.
        prov_by_asset = {p.get("published"): p for p in (curated_prov or ())}
        out.append("<div class=previews>")
        for rel, cap in curated:
            prov = prov_by_asset.get(os.path.basename(rel))
            note = ""
            if prov:
                avm = ('carries AVM/WCS' if prov.get("avm") == "carried"
                       else 'no AVM/WCS')
                note = (f"<br><span class=checksum>"
                        f"{html.escape(str(prov.get('source_name')))} · "
                        f"sha256 {html.escape(str(prov.get('source_sha256'))[:12])} · "
                        f"{avm}</span>")
            out.append(f"<figure><img class=preview src='{html.escape(rel)}' "
                       f"loading=lazy alt='{html.escape(field)} {html.escape(cap)}'>"
                       f"<figcaption class=muted>{html.escape(cap)}{note}"
                       f"</figcaption></figure>")
        out.append("</div>")
        if curated_prov:
            out.append(f"<p class=muted>Full provenance for these renders: "
                       f"<a href='{html.escape(field)}_curated.json'>"
                       f"{html.escape(field)}_curated.json</a> -- source path and "
                       f"checksum, the checksum of the bytes served, and whether "
                       f"the AVM astrometry survived resizing. These are curated "
                       f"renders, not staged data products; the mosaics they were "
                       f"made from are in the table below.</p>")

    # `assets/<field>.jpg` is a BYTE COPY of the first curated JPEG when there
    # is one -- but `preview_channels`, `_preview_caption` and
    # `preview_version` are all computed from `previews[0]`, i.e. from a
    # generated preview that may not even be on the page.  So the hand-made
    # render was emitted a SECOND time, under a heading saying it was
    # automatically generated and a caption naming bands it does not contain:
    # cloudc showed "R=F212N, G=F187N, B=F182M" over an F466N/F405N image (and
    # those three bands were withheld), and gc2211 claimed "Rendered from the
    # v1.0-2026.06 mosaics" about a PNG from neither release.  It is already
    # shown, correctly captioned, in the curated block above; showing it again
    # under someone else's provenance is worse than not showing it.
    if preview_rel and not preview_from_curated:
        # Attribute the preview's version whenever it is not this page's. The
        # fallback exists so a re-stage does not blank the card, but "the same
        # mosaics under a new version" is not always true: cloudc, sgrc and wd1
        # all re-drizzled their preview bands between v1.0 and v1.1 (two of them
        # to a byte-different file of identical size). An older preview is still
        # a fair illustration; presenting it as this release's is not.
        stale = (preview_version and preview_version != manifest["version"])
        provenance = (f" Rendered from the {preview_version} mosaics, "
                      f"not {manifest['version']}'s." if stale else "")
        # EVERY preview, not just the first. One image cannot show a
        # multi-pointing field, and cannot carry more than three of the bands a
        # field like sgrb2 ships -- so a single preview silently hid both the
        # other pointings and most of the wavelengths.
        if previews and curated:
            out.append("<p class=muted style='margin-top:1.5rem'>Automatically "
                       "generated previews, one per pointing and enough wavelength "
                       "combinations that every released band appears in at least "
                       f"one.{html.escape(provenance)}</p>")
        if len(previews) > 1:
            out.append(f"<p class=muted>{len(previews)} colour previews: one per "
                       f"pointing, and enough wavelength combinations that every "
                       f"band below appears in at least one."
                       f"{html.escape(provenance)}</p>")
            out.append("<div class=previews>")
            for rel, stem in previews:
                cap = _preview_caption(stem, field)
                out.append(f"<figure><img class=preview src='{html.escape(rel)}' "
                           f"loading=lazy alt='{html.escape(field)} {html.escape(cap)}'>"
                           f"<figcaption class=muted>{html.escape(cap)}</figcaption>"
                           f"</figure>")
            out.append("</div>")
        else:
            # `_preview_caption` rather than `preview_channels`: the latter drops
            # the pointing, so a one-preview multi-pointing field (gc2211, m4)
            # said "RGB preview (R=F277W ...)" with no hint WHICH of its
            # pointings -- the omission this whole change exists to fix.
            cap = (_preview_caption(previews[0][1], field) if previews
                   else (f"R={preview_channels[0]}, G={preview_channels[1]}, "
                         f"B={preview_channels[2]}" if preview_channels else "Preview"))
            out.append(f"<img class=preview src='{html.escape(preview_rel)}' "
                       f"alt='{html.escape(field)} preview'>")
            out.append(f"<div class=muted>RGB preview - {html.escape(cap)}."
                       f"{html.escape(provenance)} "
                       "Full-resolution images below.</div>")

    if multi:
        out.append("<p class=muted><b>Multi-pointing / multi-epoch field.</b> "
                   "Each observation (o###) is a distinct pointing or epoch; images "
                   "are grouped by observation. The combined catalog merges all "
                   "observations (including ones whose images are not yet final and "
                   "are held for a later release); per-observation catalogs are also "
                   "provided.</p>")

    # globus-collection-relative field path (includes group folder when set)
    base = manifest.get("release_path") or f"/releases/{manifest['version']}/{field}"
    out.append("<p class=muted>Files are served from the <b>JWST&nbsp;root</b> Globus "
               "collection; downloading requires a free "
               "<a href='https://www.globus.org/'>Globus</a> login. "
               "Checksums and full provenance are in "
               f"<a href='{html.escape(manifest['globus_https_base'])}"
               f"{html.escape(base)}/MANIFEST.json'>"
               "MANIFEST.json</a>.</p>")

    # bulk download: Globus file-manager links (select-all -> transfer, no
    # tarball) + plain URL lists for scripted/wget downloads. Separate paths for
    # images, catalogs, and everything.
    coll = manifest["globus_collection_id"]
    have_images = any(f["category"] == "image" for f in files)
    have_catalogs = any(f["category"] == "catalog" for f in files)

    def app_link(subpath, label):
        url = (f"{GLOBUS_APP}?origin_id={coll}"
               f"&origin_path={urllib.parse.quote(subpath + '/')}")
        return f"<a class=btn href='{html.escape(url)}'>⬇ {label}</a>"

    buttons = []
    if have_images and have_catalogs:
        buttons.append(app_link(base, "Everything"))
    if have_images:
        buttons.append(app_link(base + "/images", "Images only"))
    if have_catalogs:
        buttons.append(app_link(base + "/catalogs", "Catalogs only"))
    txt_links = [
        f"<a href='{html.escape(field)}_files.txt'>all</a>"]
    if have_images:
        txt_links.append(f"<a href='{html.escape(field)}_images.txt'>images</a>")
    if have_catalogs:
        txt_links.append(f"<a href='{html.escape(field)}_catalogs.txt'>catalogs</a>")

    out.append(
        "<div class=bulk><b>Bulk download</b>"
        "<div class=muted style='margin:.3rem 0 .6rem'>"
        "Click a Globus button → sign in → the field's folder opens in the Globus "
        "file manager → press <b>Ctrl/Cmd-A</b> to select all, then <b>Start</b> to "
        "transfer to your own collection. Prefer scripting? Use the URL lists with "
        "<code>wget -i</code> / <code>curl</code> after authenticating.</div>"
        + "".join(buttons)
        + "<div class=muted style='margin-top:.5rem'>URL lists: "
        + " · ".join(txt_links)
        + " &nbsp;|&nbsp; <a href='download_help.html'>how to download / authenticate</a>"
        + "</div></div>")

    obs_col = "<th>Obs</th>" if multi else ""
    order = {"science": 0, "residual": 1, "model": 2}

    # per-file version: an explicit per-file version if the manifest records one,
    # else the field release version.  Lets a mixed release (some files bumped to a
    # newer version) show each file's own version.
    def file_version(f):
        return html.escape(str(f.get("version") or manifest["version"]))

    # images table grouped by (observation, filter)
    out.append("<h2>Images</h2>")
    out.append(f"<table><tr>{obs_col}<th>Filter</th><th>Type</th><th>Iteration</th>"
               "<th>Version</th><th>Size</th><th>Download</th></tr>")
    groups = {}
    for f in images:
        groups.setdefault((f.get("observation") or "", f["filter"]), []).append(f)
    for key in sorted(groups, key=lambda k: (k[0], FILTER_WAVELENGTH.get(k[1], 99))):
        obs, filt = key
        rows = sorted(groups[key], key=lambda f: order.get(f["kind"], 9))
        for i, f in enumerate(rows):
            obs_cell = (f"<td><b>{html.escape(obs)}</b></td>"
                        if multi and i == 0 else ("<td></td>" if multi else ""))
            filt_cell = (f"<b>{filt}</b> "
                         f"<span class=muted>{FILTER_WAVELENGTH.get(filt,'')}µm</span>"
                         if i == 0 else "")
            out.append(f"<tr>{obs_cell}<td>{filt_cell}</td>"
                       f"<td>{KIND_LABEL.get(f['kind'], f['kind'])}</td>"
                       f"<td><span class=tag>{html.escape(f['iteration'] or '')}</span></td>"
                       f"<td><span class=tag>{file_version(f)}</span></td>"
                       f"<td class=size>{human_size(f['size_bytes'])}</td>"
                       f"<td>{dl(f)}</td></tr>")
    out.append("</table>")

    # catalogs table
    out.append("<h2>Catalogs</h2>")
    if catalogs and not any(f["kind"] == "catalog_full" for f in catalogs):
        out.append("<p class=muted><b>Preliminary catalog release.</b> The field-wide "
                   "merged photometry table is still being built; only the per-filter "
                   "vetted catalogs are provided for now. The merged table will be added "
                   "in a later update.</p>")
    out.append(f"<table><tr><th>Catalog</th>{obs_col}<th>Filter</th><th>Iteration</th>"
               "<th>Version</th><th>Size</th><th>Download</th></tr>")
    cat_order = {"catalog_full": 0, "catalog_qualcut": 1, "seed": 2,
                 "catalog_per_filter_vetted": 3}
    for f in sorted(catalogs, key=lambda f: (f.get("observation") or "",
                                             cat_order.get(f["kind"], 9),
                                             f.get("filter") or "")):
        name = KIND_LABEL.get(f["kind"], f["kind"])
        fmt = Path(f["dest"]).suffix.lstrip(".")
        obs_cell = (f"<td>{html.escape(f.get('observation') or '—')}</td>"
                    if multi else "")
        out.append(f"<tr><td>{name} <span class=muted>({fmt})</span></td>{obs_cell}"
                   f"<td>{html.escape(f['filter'] or '—')}</td>"
                   f"<td><span class=tag>{html.escape(f['iteration'] or '')}</span></td>"
                   f"<td><span class=tag>{file_version(f)}</span></td>"
                   f"<td class=size>{human_size(f['size_bytes'])}</td>"
                   f"<td>{dl(f)}</td></tr>")
    out.append("</table>")

    out.append("</main>")
    out.append(footer())
    out.append("</body></html>")
    return "\n".join(out)


def footer():
    return ("<footer class=muted>JWST Galactic Center survey · "
            "data reduced with the "
            "<a href='https://github.com/keflavich/jwst-gc-pipeline'>jwst-gc-pipeline</a>"
            " · contact <a href='mailto:adamginsburg@ufl.edu'>adamginsburg@ufl.edu</a>"
            "</footer>")


def render_cmz_explorer(hips_url, cat_hips_url=None, moc_url=None,
                        target="Galactic Center"):
    """Standalone Aladin Lite page: CMZ color HiPS + catalog HiPS + coverage MOC.

    ``*_url`` are relative to the site root (the products live in the release
    tree).  Aladin Lite v3 loads from the CDS CDN; the HiPS/catalog/MOC are served
    from the same host as the page.
    """
    js_cat = (f"aladin.addCatalog(A.catalogHiPS('{cat_hips_url}', "
              f"{{onClick:'showTable', name:'CMZ catalog'}}));"
              if cat_hips_url else "")
    js_moc = (f"fetch('{moc_url}').then(r=>r.arrayBuffer()).then(b=>"
              f"aladin.addMOC(A.MOCFromURL('{moc_url}', "
              f"{{opacity:0.12, color:'#38bdf8', lineWidth:1}})));"
              if moc_url else "")
    out = [page_head("JWST-GC — CMZ explorer")]
    out.append("<header><h1>CMZ explorer</h1>"
               "<div class=muted>Two-color HiPS (B=F212N, R=F480M, G=0.5·(R+B)) "
               "with the source catalog and coverage overlaid · "
               "<a href='index.html'>← all fields</a></div></header><main>")
    out.append("<div id='aladin-lite-div' style='width:100%;height:75vh;"
               "border-radius:8px;overflow:hidden'></div>")
    out.append("<script src='https://aladin.cds.unistra.fr/AladinLite/api/v3/"
               "latest/aladin.js' charset='utf-8'></script>")
    out.append(
        "<script>A.init.then(() => {"
        f"const aladin = A.aladin('#aladin-lite-div', {{cooFrame:'galactic', "
        f"fov:1.0, target:'{target}', showCooGrid:false}});"
        f"aladin.setImageSurvey(aladin.createImageSurvey('cmz-color',"
        f"'CMZ two-color','{hips_url}','galactic',9,{{imgFormat:'png'}}));"
        f"{js_cat}{js_moc}"
        "});</script>")
    out.append("<p class=muted>Products: "
               f"<a href='{hips_url}'>color HiPS</a>"
               + (f" · <a href='{cat_hips_url}'>catalog HiPS</a>" if cat_hips_url else "")
               + (f" · <a href='{moc_url}'>coverage MOC</a>" if moc_url else "")
               + ".</p>")
    out.append("</main>" + footer() + "</body></html>")
    return "".join(out)


def render_help():
    out = [page_head("JWST-GC — how to download")]
    out.append("<header><h1>Downloading the data</h1>"
               "<div class=muted><a href='index.html'>← all fields</a></div>"
               "</header><main>")

    out.append("<h2>In a web browser — no token needed</h2>")
    out.append("<p>Just click any <b>download</b> link on a field page. If you are not "
               "already signed in, Globus will prompt you to log in (your "
               "<b>ORCID</b> works), then the file downloads. The bulk "
               "<b>⬇ Everything / Images / Catalogs</b> buttons open the folder in the "
               "Globus file manager — press <b>Ctrl/Cmd-A</b> to select all and "
               "<b>Start</b> a transfer to your own collection. "
               "<b>No access token is required for browser downloads.</b></p>")

    out.append("<h2>Command line with <code>globus</code> — recommended (one tool, no token)</h2>")
    out.append("<p>If your download destination is itself a Globus collection, this is "
               "the whole job: <code>globus</code> handles authentication and the "
               "transfer — <b>no <code>wget</code>, no token, no extra steps</b>.</p>")
    out.append("<pre><code>pip install globus-cli\n"
               "globus login                 # one-time; handles authentication\n"
               "globus transfer --recursive \\\n"
               f"  {GLOBUS_COLLECTION_ID}:/releases/&lt;version&gt;/&lt;field&gt;/ \\\n"
               "  &lt;YOUR_ENDPOINT_ID&gt;:/local/destination/</code></pre>"
               "<p class=muted>The destination must be a Globus collection. On an "
               "HPC/cluster you almost certainly already have one (ask your admin for "
               "its endpoint ID). On a laptop or workstation, install "
               "<a href='https://www.globus.org/globus-connect-personal'>Globus Connect "
               "Personal</a> once to make it a collection. Track progress with "
               "<code>globus task list</code>.</p>")

    out.append("<h2>No Globus collection at your destination? <code>wget</code> / <code>curl</code></h2>")
    out.append("<p>Only needed if you cannot use a Globus collection as the destination "
               "(e.g. pulling straight onto a plain web server). <code>wget</code> "
               "cannot do the interactive login itself, so you first mint a short-lived "
               "bearer token. <b>The login step still happens in your browser</b> (open "
               "a URL, sign in, paste back a code); the helper then prints the token. "
               "There is no way to copy a raw token straight from a web page.</p>")
    out.append("<pre><code>pip install globus-sdk\n"
               "python get_globus_token.py        # browser login, prints a token\n"
               "TOKEN=&lt;paste the token&gt;\n"
               "wget --header=\"Authorization: Bearer $TOKEN\" -i &lt;field&gt;_files.txt</code></pre>")
    out.append("<p>Save <a href='get_globus_token_helper.txt' download='get_globus_token.py'>"
               "get_globus_token.py</a> (or copy it below). Token is valid ~48 h; re-run "
               "for a fresh one. Each field page links <code>_files.txt</code> / "
               "<code>_images.txt</code> / <code>_catalogs.txt</code> URL lists for "
               "<code>wget -i</code>.</p>")
    out.append("<pre><code>" + html.escape(TOKEN_HELPER) + "</code></pre>")

    out.append("</main>")
    out.append(footer())
    out.append("</body></html>")
    return "\n".join(out)


def _field_cards(fields_info):
    out = ["<div class=grid>"]
    for fi in fields_info:
        thumb = (f"<img src='{html.escape(fi['preview'])}' alt='{fi['field']}'>"
                 if fi["preview"] else "")
        out.append(
            f"<a class=card href='{fi['field']}.html'>{thumb}"
            f"<div class=body><b>{html.escape(fi['field'])}</b><br>"
            f"<span class=muted>{fi['n_images']} images · {fi['n_catalogs']} catalogs · "
            f"{html.escape(fi['version'])}</span></div></a>")
    out.append("</div>")
    return out


def render_index(fields_info, overview_html=""):
    out = [page_head("JWST Galactic Center survey — data release")]
    out.append("<header><h1>JWST Galactic Center survey</h1>")
    out.append("<div class=muted>Final reduced mosaics, residual/model images, and "
               "PSF photometry catalogs.</div></header><main>")
    out.append("<p>Select a field. Image and catalog downloads are served via the "
               "<b>JWST root</b> Globus collection (free login required).</p>")

    # group fields into sections; Galactic Center (group None) first, then the
    # rest in a stable order. Single group -> no section header (back-compat).
    groups = {}
    for fi in fields_info:
        groups.setdefault(fi.get("group"), []).append(fi)
    if len(groups) <= 1:
        out += _field_cards(fields_info)
        # the overview maps the Galactic Centre fields, which in the single-group
        # case is the whole index -- so it still belongs directly after the cards
        if overview_html:
            out.append(overview_html)
    else:
        order = sorted(groups, key=lambda g: (g is not None, g or ""))
        for g in order:
            out.append(f"<h2>{html.escape(GROUP_TITLE.get(g, g or 'Other'))}</h2>")
            out += _field_cards(groups[g])
            # below the Galactic Centre section, above the other groups: the
            # panel maps CMZ fields only, so it must not float away from them
            if g is None and overview_html:
                out.append(overview_html)
    out.append("</main>")
    out.append(footer())
    out.append("</body></html>")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", nargs="+", default=["cloudc"])
    parser.add_argument("--version", default="v1.0-2026.06")
    parser.add_argument("--release-root",
                        default="/orange/adamginsburg/jwst/releases")
    parser.add_argument("--out", default="/orange/adamginsburg/jwst/releases/site")
    parser.add_argument("--cmz-hips",
                        help="site-relative URL of the CMZ two-color HiPS "
                             "(e.g. cmz/hips/CMZ_color). If given, writes "
                             "cmz_explorer.html with an Aladin Lite pane and links "
                             "it from the index.")
    parser.add_argument("--cmz-cat-hips",
                        help="site-relative URL of the catalog HiPS (optional)")
    parser.add_argument("--cmz-moc",
                        help="site-relative URL of the coverage MOC .fits (optional)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    def discover_versions(field):
        """All release versions of a field present on disk, latest first.
        (Version strings like v1.0-2026.06 / v1.1-2026.07 sort correctly as text.)"""
        root = Path(args.release_root)
        found = [p.name for p in root.iterdir() if p.is_dir()
                 and (field_release_dir(field, p.name, args.release_root) / "MANIFEST.json").is_file()]
        return sorted(found, reverse=True)

    fields_info = []
    overview_entries = []          # (field, latest release dir, index-relative href)
    for field in args.fields:
        versions = discover_versions(field)
        if not versions:
            print(f"skip {field}: no MANIFEST.json in any version")
            continue
        latest = versions[0]
        latest_dir = field_release_dir(field, latest, args.release_root)

        # Preview from the latest version that HAS one -- a re-stage that ships the
        # same mosaics under a new version usually has no preview/ of its own, and
        # falling off the latest version would blank the field's card on the index.
        preview_rel = None
        preview_channels = None
        preview_version = None
        previews = []
        for v in versions:
            vdir = latest_dir if v == latest else field_release_dir(field, v, args.release_root)
            previews = sorted((vdir / "preview").glob("*.jpg")) \
                if (vdir / "preview").is_dir() else []
            if previews:
                preview_version = v
                if v != latest:
                    print(f"  {field}: no preview in {latest}, using {v}'s")
                break
        # Curated images first -- the published renders, better than anything
        # the planner makes.
        #
        # They are NOT independent of the mosaics they were rendered from, and
        # an earlier version of this comment said they were.  gc2211 o050's
        # render predates the quarantine of its own source by one day, and the
        # page was promoting it as the field's primary image while the mosaic
        # it came from had been repudiated.
        #
        # Withhold on the OBSERVATION, not the band.  gc2211 is five
        # observations in one field and all five share F200W/F277W, so a
        # band-level test cannot express "o050 is repudiated, o049 is fine" --
        # it withholds o049, the one region whose astrometry is actually good
        # (~50 mas, against o050's 5.6").  The curated filenames and the
        # registry both carry the obs token.
        #
        # Deliberate: this reads the LATEST manifest whichever version's page is
        # being built, so an older version's page inherits the newest
        # repudiations.  A curated render is a property of the FIELD -- one
        # picture, linked from every version -- not of a release, so the newest
        # word on whether its mosaic is still good is the right one to obey.
        # The effect is that a frozen v1.0 page can lose a curated image on the
        # strength of a v1.2-era quarantine, which is the intended direction.
        try:
            _lm = json.loads((field_release_dir(field, latest, args.release_root)
                              / "MANIFEST.json").read_text())
        except (OSError, ValueError) as err:
            # Silence here disabled curated withholding entirely and published
            # every render, with nothing said.
            print(f"  {field}: WARNING cannot read the latest MANIFEST "
                  f"({err}); curated withholding is DISABLED for this field")
            _lm = None
        withheld_obs, withheld_bands_c, known_obs_c = \
            curated_withholding_inputs(_lm)

        curated_items = []
        curated_prov = []
        for entry in curated_images.for_field(field):
            _why = curated_withheld_reason(entry, withheld_obs,
                                           withheld_bands_c,
                                           known_obs=known_obs_c)
            if _why:
                print(f"  {field}: WITHHOLDING curated {entry['stem']} -- {_why}")
                continue
            dest = assets / f"curated_{entry['stem']}.jpg"
            try:
                web_jpeg(entry["file"], dest)
            except (OSError, ValueError) as err:
                print(f"  {field}: could not prepare {entry['stem']}: {err}")
                continue
            try:
                prov = curated_provenance(
                    entry, dest, sidecar=assets / f"curated_{entry['stem']}.json")
            except (OSError, ValueError) as err:
                # A record that cannot be written is a reason to say so, not to
                # delete a picture that rendered fine.  Sharing one handler with
                # web_jpeg meant an unwritable sidecar dropped the field's
                # primary image from the page.
                print(f"  {field}: WARNING no provenance for {entry['stem']}: "
                      f"{err}")
                prov = None
            if prov:
                curated_prov.append(prov)
            if prov and prov["avm"] == "absent":
                # not fatal -- some renders genuinely carry no AVM -- but it is
                # the difference between a picture and a positioned picture, so
                # it is said out loud rather than discovered later
                print(f"  {field}: curated {entry['stem']} published without AVM")
            curated_items.append((f"assets/{dest.name}", curated_images.caption(entry)))
        for gone in curated_images.missing(field):
            print(f"  {field}: curated image listed but not on disk: "
                  f"{os.path.basename(gone)}")
        # The provenance record for every curated render actually published.
        # A staged mosaic reaches the page through MANIFEST.json; a curated
        # render is not staged, so this file is the equivalent -- what it was
        # made from, the checksum of that source, the checksum of the bytes
        # served, and whether the AVM astrometry survived publication.
        if curated_prov:
            (out_dir / f"{field}_curated.json").write_text(
                json.dumps({"field": field, "images": curated_prov},
                           indent=1, sort_keys=True))

        preview_items = []
        preview_from_curated = False
        if curated_items:
            # the front page shows the beautified image, not a generated one
            shutil.copy2(assets / os.path.basename(curated_items[0][0]),
                         assets / f"{field}.jpg")
            preview_rel = f"assets/{field}.jpg"
            # ... and the FIELD page must not then render it a second time
            # under the generated preview's caption and provenance.
            preview_from_curated = True
        if previews:
            if not curated_items:
                shutil.copy2(previews[0], assets / f"{field}.jpg")
                preview_rel = f"assets/{field}.jpg"
            parts = previews[0].stem.split("_rgb_")
            if len(parts) == 2 and parts[1].count("_") == 2:
                preview_channels = [c.upper() for c in parts[1].split("_")]
            # The index card keeps ONE thumbnail; the field page shows them all
            # -- but in PLAN order and restricted to the plan, not whatever the
            # directory happens to hold. `preview/` is never emptied, so a glob
            # shows leftovers from an older plan (brick had 5 files for a
            # 4-preview plan) under a caption claiming the set is complete.
            by_stem = {src.stem: src for src in previews}
            # read the plan from the version the previews were rendered from
            specs = preview_plan.plan(field_release_dir(
                field, preview_version, args.release_root))
            planned = make_preview_rgb.planned_stems(field, specs)
            ordered = [next(iter(make_preview_rgb.planned_stems(field, [spec])))
                       for spec in specs]
            for stem in ordered:
                src = by_stem.get(stem)
                if src is None:
                    continue
                shutil.copy2(src, assets / f"{stem}.jpg")
                preview_items.append((f"assets/{stem}.jpg", stem))
            unplanned = sorted(set(by_stem) - planned)
            if unplanned:
                print(f"  {field}: {len(unplanned)} preview(s) not in the plan, "
                      f"not shown: {', '.join(unplanned)}")

        # one page per region for the latest (<field>.html) + one per older version
        for v in versions:
            manifest = json.loads(
                (field_release_dir(field, v, args.release_root) / "MANIFEST.json").read_text())
            stale_reasons = release_freshness.superseded_reasons(manifest)
            stale_files = sorted(stale_reasons)
            if stale_files and v == latest:
                counts = collections.Counter(stale_reasons.values())
                print(f"  {field}: WITHHOLDING {len(stale_files)} image(s) -- "
                      + ", ".join(f"{n} {state}" for state, n
                                  in sorted(counts.items())))
            page = render_field_page(field, manifest, preview_rel, preview_channels,
                                     superseded=stale_files,
                                     reasons=stale_reasons,
                                     curated=curated_items,
                                     curated_prov=curated_prov,
                                     preview_from_curated=preview_from_curated,
                                     all_versions=versions,
                                     preview_version=preview_version,
                                     previews=preview_items)
            fname = f"{field}.html" if v == latest else f"{field}.{v}.html"
            (out_dir / fname).write_text(page)
            if v == latest:
                def write_urls(suffix, cats):
                    urls = [f["url"] for f in manifest["files"]
                            if f.get("url") and (cats is None or f["category"] in cats)]
                    if urls:
                        (out_dir / f"{field}_{suffix}.txt").write_text("\n".join(urls) + "\n")
                write_urls("files", None)
                write_urls("images", {"image"})
                write_urls("catalogs", {"catalog"})
                files = manifest["files"]
                fields_info.append({
                    "field": field, "version": manifest["version"],
                    "group": manifest.get("group"), "preview": preview_rel,
                    "n_images": sum(1 for f in files if f["category"] == "image"),
                    "n_catalogs": sum(1 for f in files if f["category"] == "catalog"),
                })
                if manifest.get("group") is None:
                    # only the Galactic Centre group belongs on a CMZ map; the
                    # galactic_plane / globular_cluster fields are elsewhere on sky
                    overview_entries.append((field, latest_dir, f"{field}.html"))
        print(f"wrote {field}.html ({len(versions)} version(s): {', '.join(versions)})")

    cmz_explorer_link = None
    if args.cmz_hips:
        (out_dir / "cmz_explorer.html").write_text(render_cmz_explorer(
            args.cmz_hips, cat_hips_url=args.cmz_cat_hips, moc_url=args.cmz_moc))
        cmz_explorer_link = "cmz_explorer.html"
        print("wrote cmz_explorer.html (Aladin Lite pane)")

    # The on-sky overview reads each staged mosaic's HEADER only (i2d is a
    # rectified plain TAN grid, so WCS(header) is exact).  It is decorative: a
    # field whose mosaics cannot be read is simply left off the map, and if no
    # field yields geometry the panel is omitted entirely and the index is
    # exactly what it was before.
    overview_geoms = field_overview.collect(overview_entries)
    if overview_entries and not overview_geoms:
        print("note: no footprints readable -- on-sky overview omitted")
    overview_html = field_overview.section(overview_geoms)

    index_html = render_index(fields_info, overview_html=overview_html)
    if cmz_explorer_link:
        # surface the explorer at the top of the index (additive; no-op otherwise)
        index_html = index_html.replace(
            "<main>",
            "<main><p class=muted style='font-size:1.1em'>🔭 "
            f"<a href='{cmz_explorer_link}'>Open the CMZ explorer</a> "
            "(interactive two-color HiPS + catalog).</p>", 1)
    (out_dir / "index.html").write_text(index_html)
    (out_dir / "download_help.html").write_text(render_help())
    # .txt extension so the web server serves it as text (a .py 500s under CGI)
    (out_dir / "get_globus_token_helper.txt").write_text(TOKEN_HELPER)
    print(f"wrote index.html + download_help.html + get_globus_token_helper.txt "
          f"({len(fields_info)} fields) into {out_dir}")


if __name__ == "__main__":
    main()
