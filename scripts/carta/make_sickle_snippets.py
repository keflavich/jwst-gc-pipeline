#!/usr/bin/env python
"""Build a CARTA snippet for sickle (category jw-gc):

  sickle-photometry : per-filter final data i2d + LAST-iteration mergedcat
                      residual + model i2d, with all science catalogs overlaid.

Re-run after the manual pipeline finishes; the snippet is a snapshot of what
exists at generation time (missing products are skipped). Module B only (nrcb).
"""
import glob
import json
import os
import re

ROOT = '/orange/adamginsburg'
BASE = f'{ROOT}/jwst/sickle'
FILTERS = ['F187N', 'F210M', 'F335M', 'F470N', 'F480M']
MODULE = 'nrcb'
PROG = '03958'
OBS = 'o007'
CATS = f'{BASE}/catalogs'

SHAPES = [2, 16, 18, 9, 7, 1, 3, 11]
COLORS = ['#00ff00', '#ff3b30', '#00d0ff', '#ffcc00', '#ff00ff', '#ffffff',
          '#ff8800', '#88ff00']


def rel(p):
    return os.path.relpath(p, ROOT)


def _iter_num(path):
    m = re.findall(r'_m([1-9])_', os.path.basename(path))
    if m:
        return max(int(x) for x in m)
    if 'iter3' in path:
        return 3
    if 'daoiterative' in path:
        return 2
    return 1


def _data_i2d():
    out = []
    for filt in FILTERS:
        g = sorted(glob.glob(
            f'{BASE}/{filt}/pipeline/jw{PROG}-{OBS}_t001_nircam_clear-'
            f'{filt.lower()}-{MODULE}_data_i2d.fits'))
        out += g
    return out


def _last_iter_products(kind):
    """Per filter, the highest-iteration GROUP mergedcat <kind> i2d
    (kind in {'residual','model'})."""
    out = []
    for filt in FILTERS:
        cands = glob.glob(
            f'{BASE}/{filt}/pipeline/jw{PROG}-{OBS}_t001_nircam_clear-'
            f'{filt.lower()}-{MODULE}_*group*_mergedcat_{kind}_i2d.fits')
        if not cands:  # fall back to non-group if --group wasn't used
            cands = glob.glob(
                f'{BASE}/{filt}/pipeline/jw{PROG}-{OBS}_t001_nircam_clear-'
                f'{filt.lower()}-{MODULE}_*_mergedcat_{kind}_i2d.fits')
        if cands:
            out.append(max(cands, key=_iter_num))
    return out


def _radec_cols(path):
    """Pick (ra, dec) overlay column names for a FITS catalog: prefer a
    skycoord(_ref) mixin serialisation, else plain ra/dec."""
    try:
        from astropy.table import Table
        cols = set(Table.read(path).colnames)
    except Exception:
        cols = set()
    for base in ('skycoord_ref', 'skycoord'):
        if base in cols or f'{base}.ra' in cols:
            return (f'{base}.ra', f'{base}.dec')
    if 'ra' in cols and 'dec' in cols:
        return ('ra', 'dec')
    return ('skycoord.ra', 'skycoord.dec')  # CARTA-side best effort


def _per_filter_catalogs():
    out = []
    for filt in FILTERS:
        cands = glob.glob(
            f'{CATS}/{filt.lower()}_{MODULE}_indivexp_merged_*_dao_basic.fits')
        # exclude *_allcols / *_vetted; keep the plain merged science catalog
        cands = [c for c in cands
                 if not c.endswith(('_allcols.fits', '_vetted.fits'))]
        if cands:
            out.append(max(cands, key=_iter_num))
    return out


def _catalog_js_helpers():
    return [
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


def _header_js():
    return [
        '',
        'for (const frame of app.frames) {',
        '    app.setRasterScalingMatchingEnabled(frame, true);',
        '    frame.renderConfig.setScaling(1);',
        '    frame.renderConfig.setCustomScale(-1, 100);',
        '}',
        '',
    ]


def write_snippet(name, images, catalogs, category='jw-gc'):
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
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(snippet, fh, indent=4)
    print(f'wrote {out}  ({len(images)} images, {len(catalogs)} catalogs)')
    for p in images:
        print('  img', rel(p))
    for p, _, _ in catalogs:
        print('  cat', os.path.basename(p))


data = _data_i2d()
resid = _last_iter_products('residual')
model = _last_iter_products('model')
images = data + resid + model

cats = []
combined = f'{CATS}/basic_merged_indivexp_photometry_tables_merged.fits'
if os.path.exists(combined):
    cats.append((combined,) + _radec_cols(combined))
for pf in _per_filter_catalogs():
    cats.append((pf,) + _radec_cols(pf))

write_snippet('sickle-photometry', images, cats)
# Also (re)generate sickle-all in its existing category, pointing at the current
# products via the axis-setting _addCatalog form (the bare appendCatalog form
# CARTA cannot auto-map the dotted skycoord.ra/.dec coords -> "not supported").
write_snippet('sickle-all', images, cats, category='jwst-pipeline-june2026')
