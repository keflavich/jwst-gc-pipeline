#!/usr/bin/env python
"""Generate a CARTA snippet for a photometry-pipeline (cataloging.py) run.

Works for any run laid out as ``<base>/<FILTER>/pipeline/...`` with merged
catalogs under ``<base>/catalogs/`` -- full-frame OR cutout, single- or
multi-filter, any project (sickle / arches / ...).

The pipeline writes, per filter:
  * <...>_data_i2d.fits                                    (input data mosaic)
  * <...>_m{N}[_resbgsub]_..._mergedcat_residual_i2d.fits  (per-phase residual)
  * <...>_mergedcat_model_i2d.fits                         (model; incl satstars)
  * <...>_mergedcat_residual_smoothed_bg_i2d.fits          (per-phase background)
  * catalogs/<filt>_<mod>_indivexp_merged_{m2..m7}_dao_basic.fits

Loads data + per-phase residual/model/bg with matched constant flux scaling and
appends the merged BASIC catalogs, one symbol per phase, one colour per filter
(skycoord.ra / skycoord.dec forced as the overlay axes).

Usage:
  make_carta_snippet_manual.py --base <dir> --filters F210M,F480M \
      --category jw-gc --name sickle-mf-low_background_mf
  make_carta_snippet_manual.py --base /orange/adamginsburg/jwst/sickle \
      --filters F480M --category jw-sickle-manual --name sickle-480-fullframe
"""
import argparse
import glob
import json
import os

ROOT = '/orange/adamginsburg'   # CARTA's working dir; appendFile paths are rel to it

PHASE_TAGS = ('_m2_', '_m3_', '_m4_', '_resbgsub_m5_', '_resbgsub_m6_',
              '_resbgsub_m7_')
# CatalogOverlayShape numerics: CIRCLE_FILLED=2 CROSS_FILLED=16 X_FILLED=18
#   TRIANGLE_LINED_UP=9 RHOMB_LINED=7 BOX_LINED=1 -- one per phase (m2..m7)
PHASE_SHAPES = [2, 16, 18, 9, 7, 1]
# one colour per filter (cycled)
FILTER_COLORS = ['#00ff00', '#ff3b30', '#00d0ff', '#ffcc00', '#ff00ff', '#ffffff']


def rel(p):
    return os.path.relpath(p, ROOT)


def _iter_of(p):
    b = os.path.basename(p)
    for n in (2, 3, 4, 5, 6, 7):
        if f'_m{n}_' in b:
            return n
    return 9


def _kind(p):
    if '_mergedcat_model_i2d' in p:
        return 1
    if 'smoothed_bg' in p:
        return 2
    return 0   # residual


def collect_images(pipe):
    """Per-filter image list: data first, then (residual, model, bg) per phase."""
    resid = [p for p in sorted(glob.glob(f'{pipe}/*_mergedcat_residual_i2d.fits'))
             if 'smoothed_bg' not in p]
    models = sorted(glob.glob(f'{pipe}/*_mergedcat_model_i2d.fits'))
    bgmaps = sorted(glob.glob(f'{pipe}/*_mergedcat_residual_smoothed_bg_i2d.fits'))
    data = sorted(glob.glob(f'{pipe}/*_data_i2d.fits'))
    phase_imgs = sorted(resid + models + bgmaps, key=lambda p: (_iter_of(p), _kind(p)))
    return data + phase_imgs


def collect_catalogs(cats, filt):
    """Merged BASIC catalogs for one filter, ordered by phase (m2..m7)."""
    out = []
    for tag in PHASE_TAGS:
        out += [h for h in sorted(glob.glob(f'{cats}/{filt.lower()}_*_dao_basic.fits'))
                if tag in os.path.basename(h)
                and '_allcols' not in h and '_vetted' not in h
                and '_i2dseed' not in h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True,
                    help='run output base (cutout dir or full-frame basepath)')
    ap.add_argument('--filters', required=True, help='comma list, e.g. F210M,F480M')
    ap.add_argument('--category', default='jw-gc')
    ap.add_argument('--name', required=True, help='snippet name (and json filename)')
    args = ap.parse_args()

    base = args.base.rstrip('/')
    filters = [f.strip() for f in args.filters.split(',') if f.strip()]
    cats = f'{base}/catalogs'

    images, cat_files, cat_filt_idx = [], [], []
    for fi, filt in enumerate(filters):
        images += collect_images(f'{base}/{filt}/pipeline')
        for c in collect_catalogs(cats, filt):
            cat_files.append(c)
            cat_filt_idx.append(fi)

    lines = []
    for p in images:
        lines.append(f'await app.appendFile("{rel(p)}")')
    lines.append('')
    lines.append('for (const frame of app.frames) {')
    lines.append('    app.setRasterScalingMatchingEnabled(frame, true);')
    lines.append('    frame.renderConfig.setScaling(1);')
    lines.append('    frame.renderConfig.setCustomScale(-1, 100);')
    lines.append('}')
    lines.append('')
    # Catalog overlays.  CARTA snippets run as `new AsyncFunction(code)` with NO
    # injected scope -- `app` is a browser global and frontend enums are NOT in
    # scope, so CatalogOverlayShape is referenced by its numeric value.
    #   app.appendCatalog(dir, file, previewDataSize, fileType=1[FITS]) -> fileId
    #   app.catalogStore.getCatalogWidgetStore(fileId) -> CatalogWidgetStore
    #     .setxAxis(col)/.setyAxis(col)/.setCatalogShape(n)/.setCatalogColor(s)
    # The astropy SkyCoord mixin serialises to 'skycoord.ra'/'skycoord.dec'.
    # Shape encodes the phase (m2..m7); colour encodes the filter.
    lines.append(f'const _phaseShapes = {json.dumps(PHASE_SHAPES)};  // m2..m7')
    lines.append(f'const _filterColors = {json.dumps(FILTER_COLORS)};')
    lines.append('function _iterIdx(file) {')
    lines.append('    const m = file.match(/_m([234567])_/);')
    lines.append('    return m ? (parseInt(m[1]) - 2) : 0;')
    lines.append('}')
    lines.append('async function _addCatalog(dir, file, filterIdx) {')
    lines.append('    const fileId = await app.appendCatalog(dir, file, 100, 1);')
    lines.append('    try {')
    lines.append('        const w = app.catalogStore.getCatalogWidgetStore(fileId);')
    lines.append('        if (w) {')
    lines.append('            w.setxAxis("skycoord.ra");')
    lines.append('            w.setyAxis("skycoord.dec");')
    lines.append('            w.setCatalogShape(_phaseShapes[_iterIdx(file) % _phaseShapes.length]);')
    lines.append('            w.setCatalogColor(_filterColors[filterIdx % _filterColors.length]);')
    lines.append('        }')
    lines.append('    } catch (e) { console.error("catalog style failed:", file, e); }')
    lines.append('    return fileId;')
    lines.append('}')
    for p, fi in zip(cat_files, cat_filt_idx):
        d = '/' + os.path.relpath(os.path.dirname(p), ROOT) + '/'
        lines.append(f'await _addCatalog("{d}", "{os.path.basename(p)}", {fi})')

    snippet = {
        "$schema": "https://cartavis.org/schemas/snippet_schema_1.json",
        "categories": [args.category],
        "code": '\n'.join(lines) + '\n',
        "frontendVersion": "5.0.2",
        "snippetVersion": 1,
    }
    out = f'/home/adamginsburg/.carta/config/snippets/{args.name}.json'
    with open(out, 'w') as fh:
        json.dump(snippet, fh, indent=4)
    print(f'wrote {out}  ({len(images)} images, {len(cat_files)} catalogs, '
          f'category={args.category})')
    for p in images:
        print('  img', rel(p))
    for p in cat_files:
        print('  cat', os.path.basename(p))


if __name__ == '__main__':
    main()
