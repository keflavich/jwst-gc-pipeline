#!/usr/bin/env python
"""Build two CARTA snippets for cloudef (category jw-gc):

  1. cloudef-final        : final combined pipeline i2d mosaics + science catalogs
  2. cloudef-photometry   : final data i2d + LAST-iteration photometry residual+model

Re-run after the pipeline finishes to pick up newly-written products (each snippet
is a snapshot of what exists at generation time).  Paths are relative to ROOT.
"""
import glob
import json
import os
import re

ROOT = '/orange/adamginsburg'
BASE = f'{ROOT}/jwst/cloudef'
FILTERS = ['F162M', 'F210M', 'F360M', 'F480M']
OBS = ['o002', 'o005']
CATS = f'{BASE}/catalogs'

# distinct overlay symbol per image-group / filter (numeric CatalogOverlayShape)
SHAPES = [2, 16, 18, 9, 7, 1, 3, 11]
COLORS = ['#00ff00', '#ff3b30', '#00d0ff', '#ffcc00', '#ff00ff', '#ffffff',
          '#ff8800', '#88ff00']


def rel(p):
    return os.path.relpath(p, ROOT)


def _iter_num(path):
    """Highest manual/dao iteration token in a filename (for 'last iteration')."""
    m = re.findall(r'_m([1-9])_', os.path.basename(path))
    if m:
        return max(int(x) for x in m)
    if 'iter3' in path:
        return 3
    if 'daoiterative' in path:
        return 2
    return 1


def _i2d_per_filter():
    """The 8 final combined (nrca+nrcb merged) i2d mosaics, in filter/obs order."""
    out = []
    for filt in FILTERS:
        for obs in OBS:
            g = sorted(glob.glob(
                f'{BASE}/{filt}/pipeline/jw02092-{obs}_t001_nircam_clear-'
                f'{filt.lower()}-merged_i2d.fits'))
            out += g
    return out


def _last_iter_products(kind):
    """Per filter/obs, the HIGHEST-iteration mergedcat <kind> i2d (kind in
    {'residual','model'}).  Falls back to none if absent (pipeline still running)."""
    out = []
    for filt in FILTERS:
        for obs in OBS:
            cands = glob.glob(
                f'{BASE}/{filt}/pipeline/jw02092-{obs}_t001_nircam_clear-'
                f'{filt.lower()}-*_mergedcat_{kind}_i2d.fits')
            if cands:
                out.append(max(cands, key=_iter_num))
    return out


def _header_js():
    """Common raster-scaling-match block."""
    return [
        '',
        'for (const frame of app.frames) {',
        '    app.setRasterScalingMatchingEnabled(frame, true);',
        '    frame.renderConfig.setScaling(1);',
        '    frame.renderConfig.setCustomScale(-1, 100);',
        '}',
        '',
    ]


def _catalog_js_helpers():
    return [
        # CARTA snippets run as new AsyncFunction(code) with no injected scope;
        # CatalogOverlayShape is referenced by numeric value.  Auto-detect the
        # sky columns: prefer skycoord(_ref) mixin serialisation, else ra/dec.
        f'const _shapes = {json.dumps(SHAPES)};',
        f'const _colors = {json.dumps(COLORS)};',
        'let _ci = 0;',
        'async function _addCatalog(dir, file, raCol, decCol) {',
        '    const fileId = await app.appendCatalog(dir, file, 100, 1);',
        '    try {',
        '        const w = app.catalogStore.getCatalogWidgetStore(fileId);',
        '        if (w) {',
        '            w.setxAxis(raCol); w.setyAxis(decCol);',
        '            w.setCatalogShape(_shapes[_ci % _shapes.length]);',
        '            w.setCatalogColor(_colors[_ci % _colors.length]);',
        '        }',
        '    } catch (e) { console.error("catalog style failed:", file, e); }',
        '    _ci++; return fileId;',
        '}',
    ]


def write_snippet(name, images, catalogs, category='jw-gc'):
    """catalogs: list of (path, raCol, decCol)."""
    lines = [f'await app.appendFile("{rel(p)}")' for p in images]
    lines += _header_js()
    if catalogs:
        lines += _catalog_js_helpers()
        for p, ra, dec in catalogs:
            d = '/' + os.path.relpath(os.path.dirname(p), ROOT) + '/'
            lines.append(f'await _addCatalog("{d}", "{os.path.basename(p)}", '
                         f'"{ra}", "{dec}")')
    snippet = {
        "$schema": "https://cartavis.org/schemas/snippet_schema_1.json",
        "categories": [category],
        "code": '\n'.join(lines) + '\n',
        "frontendVersion": "5.0.2",
        "snippetVersion": 1,
    }
    out = f'/home/adamginsburg/.carta/config/snippets/{name}.json'
    with open(out, 'w') as fh:
        json.dump(snippet, fh, indent=4)
    print(f'wrote {out}  ({len(images)} images, {len(catalogs)} catalogs)')
    for p in images:
        print('  img', rel(p))
    for p, _, _ in catalogs:
        print('  cat', os.path.basename(p))


# ---- snippet 1: final combined i2d + catalogs ----------------------------
data_i2d = _i2d_per_filter()
cats1 = []
combined = f'{CATS}/basic_merged_indivexp_photometry_tables_merged.fits'
if os.path.exists(combined):
    cats1.append((combined, 'skycoord_ref.ra', 'skycoord_ref.dec'))
for filt in FILTERS:
    pf = f'{CATS}/{filt.lower()}_merged_indivexp_merged_dao_basic.fits'
    if os.path.exists(pf):
        cats1.append((pf, 'skycoord.ra', 'skycoord.dec'))
write_snippet('cloudef-final', data_i2d, cats1)

# ---- snippet 2: final data i2d + last-iteration residuals + models -------
resid = _last_iter_products('residual')
model = _last_iter_products('model')
# order: data, then per-filter residual+model interleaved after the data block
images2 = data_i2d + resid + model
write_snippet('cloudef-photometry', images2, cats1)
