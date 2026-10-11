"""Index of the HiPS sky maps built from each release field's data.

The HiPS layers are not part of a release: they are rendered from the mosaics
by other tools (jwst_scripts, ACES_Aladin_tour) and served from a different
docroot on the same host (``htdocs/avm_images/``).  A field page therefore
cannot read them from its MANIFEST.  This module keeps the two halves of the
answer apart:

* which served layers belong to which field -- a name-pattern registry,
  ``FIELD_HIPS``, plus the views that span several fields (``CMZ_WIDE_HIPS``,
  ``AGGREGATE_VIEWS``) and the per-field Aladin Lite tours (``FIELD_TOURS``);
* which layers are actually served -- an inventory collected from the web
  server itself (``collect``), because the build host's copy of
  ``avm_images`` differs from the served one in both directions (test and
  stale renders exist only on the build host; several published layer sets
  exist only on the server).  Listing the build host's copy would link 404s
  and miss real layers.

The inventory is written into the site as ``hips_inventory.json``, so the
machine-readable index ships beside the human one (``hips_index.html``).

Usage::

    python scripts/release/hips_index.py --out /orange/.../releases/site/hips_inventory.json
    python scripts/release/hips_index.py --local-root /orange/.../avm_images --out inv.json
"""
import argparse
import datetime
import html
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.parse
from pathlib import Path

#: Public base URL of the served HiPS directories.  The server sends
#: ``Access-Control-Allow-Origin: *``, which the Aladin Lite links need: they
#: load the layer from aladin.cds.unistra.fr, a different origin.
HIPS_BASE = "https://starformation.astro.ufl.edu/avm_images/"
TOUR_BASE = "https://starformation.astro.ufl.edu/Aladin_tours/"
ALADIN_LITE = "https://aladin.cds.unistra.fr/AladinLite/"
REMOTE_HOST = os.environ.get("HIPS_WEB_HOST", "starformation")
REMOTE_DIR = os.environ.get(
    "HIPS_WEB_DIR",
    "/h/cnswww-starformation.astro/starformation.astro.ufl.edu/htdocs/avm_images")
INVENTORY_NAME = "hips_inventory.json"
INDEX_PAGE = "hips_index.html"

#: properties keys copied into the inventory
KEYS = ("obs_title", "hips_initial_ra", "hips_initial_dec", "hips_initial_fov",
        "hips_order", "hips_release_date")

#: Layer-name patterns per release field.  Hand-maintained: a new HiPS
#: target name needs a pattern here (and a title in jwst_scripts'
#: jwst_rgb/hips_naming.py, which sets the obs_title shown in the table).  Only renders of JWST data from the
#: field are listed: the ALMA, GTC, and MUSTANG layers that share a target
#: name (SgrB2M_RGB, MUBLO_*, w51_GTC_*, w51e2...) are not matched.  JWST
#: renders WITH a radio overlay (Brick_*_alma, GCTreasury_radio_*) are.
FIELD_HIPS = {
    "arches": (r"^arches_", r"^ArchesQuintuplet_"),
    "quintuplet": (r"^Quintuplet_", r"^ArchesQuintuplet_"),
    "brick": (r"^Brick(JWST)?_",),
    "cloudc": (r"^cloudcJWST_", r"^CloudC_"),
    "cloudef_controlfield": (r"^Cloudef_", r"^CloudefControl_"),
    "gc2211": (r"^GC2211_",),
    "sgra": (r"^SgrA_",),
    "sgrb2": (r"^SgrB2_(?!DS_alma)",),
    "sgrc": (r"^SGRC_",),
    "sickle": (r"^Sickle_",),
    "w51": (r"^w51_(RGB|starsub)_",),
    "wd1": (r"^wd1_",),
    "wd2": (r"^wd2_",),
    "gc-treasury": (r"^gc10678_", r"^gc_f770w_", r"^gctreasury_", r"^GCTreasury_",
                    r"^jwst_gc_treasury_", r"^jwst-(median|rc|red|star|stars)-"),
}

#: Superseded and diagnostic renders kept on disk beside the real ones.
EXCLUDE = re.compile(r"(?:^|_)(?:stale|superseded|tests?|check|flipped|broken)(?=_|\d|$)",
                     re.IGNORECASE)

#: Fields inside the Central Molecular Zone: their data are also part of the
#: survey-wide coadds below.
CMZ_FIELDS = frozenset({"arches", "brick", "cloudc", "cloudef_controlfield", "gc2211",
                        "quintuplet", "sgra", "sgrb2", "sgrc", "sickle", "gc-treasury"})

#: Survey-wide HiPS coadds that include every CMZ field.
CMZ_WIDE_HIPS = ("jwst_nir_hips", "jwst_miri_hips", "jwst_gc_treasury_hips",
                 "jwst_gc_treasury_miri_hips")

#: Aladin Lite tours of one field, as (page under TOUR_BASE, title).
FIELD_TOURS = {
    "arches": (("arches_wavelength_tour_linear.html", "Arches wavelength explorer (linear)"),),
    "brick": (("brick_wavelength_tour_linear.html", "Brick wavelength explorer (linear)"),
              ("brick_wavelength_tour_other.html", "Brick wavelength explorer (other)")),
    "cloudef_controlfield": (("cloudef_wavelength_tour_linear.html",
                              "Clouds E/F wavelength explorer (linear)"),),
    "sgra": (("sgra_wavelength_tour_linear.html", "Sgr A* wavelength explorer (linear)"),),
    "sgrb2": (("sgrb2_wavelength_tour_linear.html", "Sgr B2 wavelength explorer (linear)"),
              ("sgrb2_wavelength_tour_other.html", "Sgr B2 wavelength explorer (other)"),
              ("sgrb2_tour.html", "Sgr B2 JWST + ALMA tour")),
    "sgrc": (("sgrc_wavelength_tour_linear.html", "Sgr C wavelength explorer (linear)"),),
    "sickle": (("sickle_wavelength_tour_linear.html", "Sickle wavelength explorer (linear)"),
               ("sickle_wavelength_tour.html", "Sickle wavelength explorer")),
    "w51": (("w51_wavelength_tour_linear.html", "W51 wavelength explorer (linear)"),
            ("w51_wavelength_tour_linear_starsub.html",
             "W51 wavelength explorer, stars subtracted (linear)"),
            ("w51_wavelength_tour_other.html", "W51 wavelength explorer (other)"),
            ("w51_wavelength_tour.html", "W51 wavelength explorer"),
            ("w51_tour.html", "W51 star-forming region tour")),
    "wd2": (("wd2_wavelength_tour_linear.html", "Westerlund 2 wavelength explorer (linear)"),
            ("wd2_tour.html", "Westerlund 2 multi-wavelength tour")),
}

#: Views that combine several fields' HiPS, as (href, title, blurb, scope);
#: scope "cmz" lists the view on CMZ field pages only, "all" on every page.
AGGREGATE_VIEWS = (
    ("https://starformation.astro.ufl.edu/avm_images/jwst_gc_aladin.html",
     "HiPS sky viewer",
     "Pan and zoom the JWST CMZ and Treasury mosaics over VVV, Spitzer, and "
     "MeerKAT context.", "cmz"),
    (TOUR_BASE + "jwst_cmz_tour.html", "JWST Central Molecular Zone tour",
     "A guided Aladin Lite tour across the CMZ fields.", "cmz"),
    (TOUR_BASE, "All Aladin Lite tours",
     "Every tour and wavelength explorer, including the linear wavelength tours.",
     "all"),
    (INDEX_PAGE, "Index of all release HiPS",
     "Every HiPS layer made from the release fields, grouped by field.", "all"),
)


# ---------------------------------------------------------------- inventory

def parse_properties_dump(text):
    """Records from ``### <name>`` headers each followed by ``key = value`` lines."""
    layers, cur = [], None
    for line in text.splitlines():
        if line.startswith("### "):
            cur = {"name": line[4:].strip()}
            layers.append(cur)
        elif cur is not None and "=" in line:
            key, _, value = line.partition("=")
            if key.strip() in KEYS:
                cur[key.strip()] = value.strip()
    return sorted(layers, key=lambda r: r["name"])


def _dump_command(root):
    keys = "|".join(KEYS)
    return (f"cd {shlex.quote(str(root))} && for p in */properties; do "
            f"[ -f \"$p\" ] || continue; echo \"### ${{p%/properties}}\"; "
            f"grep -E '^({keys}) *=' \"$p\" || true; done")


def collect_remote(host=REMOTE_HOST, root=REMOTE_DIR, timeout=300):
    """The served layers, read from the web server over ssh."""
    run = subprocess.run(["ssh", "-n", host, _dump_command(root)],
                         capture_output=True, text=True, timeout=timeout, check=True)
    return parse_properties_dump(run.stdout)


def collect_local(root):
    """The layers in a local directory (for a mirror, or for testing)."""
    chunks = []
    for prop in sorted(Path(root).glob("*/properties")):
        chunks.append(f"### {prop.parent.name}\n{prop.read_text(errors='replace')}")
    return parse_properties_dump("\n".join(chunks))


def write_inventory(layers, out, source):
    out = Path(out)
    doc = {"collected": datetime.datetime.now(datetime.timezone.utc)
           .strftime("%Y-%m-%dT%H:%MZ"),
           "source": source, "base_url": HIPS_BASE, "layers": layers}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out)
    return doc


def load_inventory(path):
    """The inventory dict, or None when there is none to read."""
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        print(f"note: no HiPS inventory at {path} -- HiPS sections omitted")
    except json.JSONDecodeError as err:
        print(f"WARNING: unreadable HiPS inventory {path} ({err}) -- HiPS sections omitted")
    return None


# ---------------------------------------------------------------- selection

def field_layers(field, inventory):
    pats = [re.compile(p) for p in FIELD_HIPS.get(field, ())]
    return [layer for layer in (inventory or {}).get("layers", ())
            if any(p.search(layer["name"]) for p in pats)
            and not EXCLUDE.search(layer["name"])]


def cmz_wide_layers(inventory):
    by_name = {layer["name"]: layer for layer in (inventory or {}).get("layers", ())}
    return [by_name[n] for n in CMZ_WIDE_HIPS if n in by_name]


def layer_label(layer):
    """The layer's title, or its directory name when the title is a leftover
    path or filename rather than a description."""
    title = (layer.get("obs_title") or "").strip()
    if (not title or title.startswith("/") or title.endswith(".new")
            or title.lower().endswith((".png", ".jpg"))):
        return layer["name"]
    return title


def aladin_url(layer):
    url = HIPS_BASE + layer["name"]
    query = []
    if layer.get("hips_initial_ra") and layer.get("hips_initial_dec"):
        query.append("target=" + urllib.parse.quote(
            f"{layer['hips_initial_ra']} {layer['hips_initial_dec']}"))
    if layer.get("hips_initial_fov"):
        query.append("fov=" + urllib.parse.quote(layer["hips_initial_fov"]))
    query.append("survey=" + urllib.parse.quote(url, safe=":/"))
    return ALADIN_LITE + "?" + "&".join(query)


# ---------------------------------------------------------------- rendering

def _layer_rows(layers):
    out = ["<div style='overflow-x:auto'><table><tr><th>Layer</th><th>HiPS URL</th><th>Max order</th>"
           "<th>Built</th><th>Open</th></tr>"]
    for layer in layers:
        url = HIPS_BASE + layer["name"]
        built = (layer.get("hips_release_date") or "")[:10]
        out.append(
            f"<tr><td>{html.escape(layer_label(layer))}</td>"
            f"<td><code style='word-break:break-all'>{html.escape(url)}</code></td>"
            f"<td>{html.escape(layer.get('hips_order', ''))}</td>"
            f"<td>{html.escape(built)}</td>"
            f"<td><a href='{html.escape(aladin_url(layer))}'>Aladin Lite</a> · "
            f"<a href='{html.escape(url)}/'>preview</a> · "
            f"<a href='{html.escape(url)}/properties'>properties</a></td></tr>")
    out.append("</table></div>")
    return out


def _links(items):
    return ["<ul>"] + [f"<li><a href='{html.escape(href)}'>{html.escape(title)}</a>"
                       + (f" <span class=muted>{html.escape(blurb)}</span>" if blurb else "")
                       + "</li>" for href, title, blurb in items] + ["</ul>"]


def tour_links(field):
    return [(TOUR_BASE + page, title, "") for page, title in FIELD_TOURS.get(field, ())]


def aggregate_links(field):
    return [(href, title, blurb) for href, title, blurb, scope in AGGREGATE_VIEWS
            if scope == "all" or field in CMZ_FIELDS]


def field_section_html(field, inventory):
    """The "HiPS sky maps" section of a field page, or '' with no inventory."""
    if inventory is None:
        return ""
    layers = field_layers(field, inventory)
    out = ["<h2>HiPS sky maps</h2>",
           "<p class=muted>Hierarchical (HiPS) renderings of this field's mosaics, "
           "for Aladin Lite, Aladin Desktop, or any HiPS client: paste the HiPS URL, "
           f"or open the layer directly. Served layers as of "
           f"{html.escape(inventory.get('collected', '?'))}; the full list is in "
           f"<a href='{INDEX_PAGE}'>the HiPS index</a> and "
           f"<a href='{INVENTORY_NAME}'>{INVENTORY_NAME}</a>.</p>"]
    if layers:
        out += _layer_rows(layers)
    else:
        out.append("<p class=muted>No HiPS layer has been published from this "
                   "field's data yet.</p>")
    tours = tour_links(field)
    if tours:
        out.append("<h3>Tours of this field</h3>")
        out += _links(tours)
    if field in CMZ_FIELDS and cmz_wide_layers(inventory):
        out.append("<h3>Survey-wide HiPS that include this field</h3>")
        out += _layer_rows(cmz_wide_layers(inventory))
    out.append("<h3>Views across several fields</h3>")
    out += _links(aggregate_links(field))
    return "\n".join(out)


def render_index_page(fields, inventory, page_head, footer):
    """``hips_index.html``: every field's layers, then the multi-field views."""
    out = [page_head("HiPS index — JWST Galactic Center survey"),
           "<header><h1>HiPS index</h1><div class=muted><a href='index.html'>"
           "&larr; all fields</a></div></header><main>",
           "<p>Every HiPS sky map made from the release fields' data, grouped by "
           "field. Paste a HiPS URL into Aladin Lite, Aladin Desktop, or another "
           "HiPS client, or follow the Aladin Lite link. The same list is "
           f"available as <a href='{INVENTORY_NAME}'>{INVENTORY_NAME}</a>. "
           f"Collected {html.escape(inventory.get('collected', '?'))} from the "
           "web server.</p>",
           "<h2>Views across several fields</h2>"]
    out += _links([(h, t, b) for h, t, b, _ in AGGREGATE_VIEWS if h != INDEX_PAGE])
    wide = cmz_wide_layers(inventory)
    if wide:
        out.append("<h2>Survey-wide HiPS</h2>")
        out += _layer_rows(wide)
    for field in fields:
        layers = field_layers(field, inventory)
        tours = tour_links(field)
        if not layers and not tours:
            continue
        title = field
        out.append(f"<h2 id='{html.escape(field)}'><a href='{html.escape(field)}.html'>"
                   f"{html.escape(title)}</a> <span class=muted>({len(layers)} "
                   f"layer{'s' if len(layers) != 1 else ''})</span></h2>")
        if tours:
            out += _links(tours)
        if layers:
            out += _layer_rows(layers)
    out.append("</main>" + footer() + "</body></html>")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="inventory JSON to write")
    parser.add_argument("--local-root",
                        help="read this local directory instead of the web server")
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--remote-dir", default=REMOTE_DIR)
    args = parser.parse_args(argv)
    if args.local_root:
        layers, source = collect_local(args.local_root), args.local_root
    else:
        try:
            layers = collect_remote(args.host, args.remote_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as err:
            # keep the previous inventory: a stale list beats a page with no list
            print(f"HiPS inventory NOT refreshed ({err}); keeping {args.out}",
                  file=sys.stderr)
            return 1
        source = f"{args.host}:{args.remote_dir}"
    if not layers:
        print(f"HiPS inventory NOT written: no layers found in {source}", file=sys.stderr)
        return 1
    write_inventory(layers, args.out, source)
    print(f"wrote {args.out}: {len(layers)} layers from {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
