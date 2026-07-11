#!/usr/bin/env python
"""
Generate the static distribution website for staged JWST-GC release fields.

Reads each field's ``MANIFEST.json`` (written by stage_release.py) and emits a
self-contained site:

    <out>/index.html            field grid with preview thumbnails
    <out>/<field>.html          per-field page: preview + image/catalog tables
                                with Globus download links, sizes, checksums
    <out>/assets/<field>.jpg    field preview (copied from the release preview/)

Links point at the Globus HTTPS URLs recorded in each MANIFEST.  Deploy by
rsyncing <out>/ to starformation:.../htdocs/jwst-gc/.
"""
import argparse
import html
import json
import shutil
import urllib.parse
from pathlib import Path

from stage_release import field_release_dir

# Display label per group folder (None = Galactic Center, the default survey).
GROUP_LABEL = {
    None: "JWST Galactic Center survey",
    "galactic_plane": "JWST Galactic Plane fields",
}
GROUP_TITLE = {
    None: "Galactic Center",
    "galactic_plane": "Galactic Plane",
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
.card img { width:100%; display:block; background:#000; }
.card .body { padding:.8rem 1rem; }
.preview { width:100%; border:1px solid var(--border); border-radius:8px;
           margin:1rem 0; background:#000; }
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
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>")


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


def render_field_page(field, manifest, preview_rel, preview_channels=None, all_versions=None):
    files = manifest["files"]
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
    if all_versions and manifest['version'] != all_versions[0]:
        out.append(f"<p class=muted style='border:1px solid #b58900;padding:.5em'>"
                   f"You are viewing an <b>older</b> release ({html.escape(manifest['version'])}). "
                   f"The latest is <a href='{html.escape(field)}.html'>{html.escape(all_versions[0])}</a>.</p>")

    multi = any(f.get("observation") for f in files)

    if preview_rel:
        cap = (f"RGB preview (R={preview_channels[0]}, G={preview_channels[1]}, "
               f"B={preview_channels[2]})." if preview_channels else "Preview.")
        out.append(f"<img class=preview src='{html.escape(preview_rel)}' "
                   f"alt='{html.escape(field)} preview'>")
        out.append(f"<div class=muted>{html.escape(cap)} "
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


def render_index(fields_info):
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
    else:
        order = sorted(groups, key=lambda g: (g is not None, g or ""))
        for g in order:
            out.append(f"<h2>{html.escape(GROUP_TITLE.get(g, g or 'Other'))}</h2>")
            out += _field_cards(groups[g])
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
    for field in args.fields:
        versions = discover_versions(field)
        if not versions:
            print(f"skip {field}: no MANIFEST.json in any version")
            continue
        latest = versions[0]
        latest_dir = field_release_dir(field, latest, args.release_root)

        # preview from the latest version
        preview_rel = None
        preview_channels = None
        previews = sorted((latest_dir / "preview").glob("*.jpg")) \
            if (latest_dir / "preview").is_dir() else []
        if previews:
            shutil.copy2(previews[0], assets / f"{field}.jpg")
            preview_rel = f"assets/{field}.jpg"
            parts = previews[0].stem.split("_rgb_")
            if len(parts) == 2 and parts[1].count("_") == 2:
                preview_channels = [c.upper() for c in parts[1].split("_")]

        # one page per region for the latest (<field>.html) + one per older version
        for v in versions:
            manifest = json.loads(
                (field_release_dir(field, v, args.release_root) / "MANIFEST.json").read_text())
            page = render_field_page(field, manifest, preview_rel, preview_channels,
                                     all_versions=versions)
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
        print(f"wrote {field}.html ({len(versions)} version(s): {', '.join(versions)})")

    (out_dir / "index.html").write_text(render_index(fields_info))
    (out_dir / "download_help.html").write_text(render_help())
    # .txt extension so the web server serves it as text (a .py 500s under CGI)
    (out_dir / "get_globus_token_helper.txt").write_text(TOKEN_HELPER)
    print(f"wrote index.html + download_help.html + get_globus_token_helper.txt "
          f"({len(fields_info)} fields) into {out_dir}")


if __name__ == "__main__":
    main()
