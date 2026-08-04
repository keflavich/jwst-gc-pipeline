"""Find the diagnostic images that already exist on disk, and draw the ones that
do not.

Two sources, deliberately separate:

* **Found** -- PNG/PDF diagnostics the pipeline, the QA layer or the paper
  already wrote (``astrometry_diag/``, ``audit_plots/``, ``figures/``,
  ``pngs/``, the paper's ``outputs/*/figures/``).  These are linked, not
  embedded: they are megabytes, and a page that inlined them would not load.
  Links are RELATIVE to a ``figures/`` directory that ``report.publish``
  populates, so they resolve on the served copy.

* **Drawn** -- small SVGs the monitor generates from the numbers it already
  read (per-tile residual map, per-detector offset quiver).  These are inlined,
  so they work everywhere including a published artifact where an external
  request would be blocked, and they exist for every field rather than only the
  ones somebody made figures for.

Coverage is uneven and that is reported rather than hidden: brick has ~1250
figures, gc2211 has none.  A field with no diagnostics says so.
"""
import glob
import html
import os


def _esc(text):
    """Escape a value bound into SVG markup.

    Detector names and visit ids are internal provenance strings, so this is
    hygiene rather than exposure -- but everything else in the package routes
    through an escape, and one unescaped path is how that stops being true.
    """
    return html.escape('' if text is None else str(text), quote=True)

#: Directories under a field that hold diagnostic imagery, most specific first.
FIGURE_DIRS = ('astrometry_diag', 'audit_plots', 'figures', 'pngs',
               'astrometry_analysis', 'diagnostics')

#: Cap per field: these directories run to thousands of files and the page only
#: needs a way in, not a gallery.
MAX_PER_DIR = 12
MAX_TOTAL = 40


def find_figures(base, filters=(), max_total=MAX_TOTAL):
    """``[{path, name, dir, filter, mtime}]`` -- diagnostics already on disk.

    Filenames are matched against the run's filters so a per-filter finding can
    offer the figures that actually concern it; unmatched figures are still
    returned (with ``filter=None``) because a field-level plot is often the
    relevant one.
    """
    found = []
    upper = {f.upper() for f in filters}
    for sub in FIGURE_DIRS:
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            continue
        names = []
        for ext in ('png', 'pdf', 'svg', 'jpg'):
            names += glob.glob(os.path.join(d, f'*.{ext}'))
        names.sort(key=lambda p: os.path.basename(p))
        for path in names[:MAX_PER_DIR]:
            name = os.path.basename(path)
            hit = next((f for f in upper if f in name.upper()), None)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            found.append({'path': path, 'name': name, 'dir': sub,
                          'filter': hit, 'mtime': mtime})
        if len(found) >= max_total:
            break
    return found[:max_total]


def paper_figures(paper_dir, max_total=12):
    """The astrometry paper's generated figures, newest analysis first."""
    out = []
    dirs = sorted(glob.glob(os.path.join(paper_dir, 'outputs', '*', 'figures')),
                  reverse=True)
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, '*'))):
            if not path.lower().endswith(('.pdf', '.png', '.svg')):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            out.append({'path': path, 'name': os.path.basename(path),
                        'dir': f'paper/{os.path.basename(os.path.dirname(d))}',
                        'filter': None, 'mtime': mtime})
            if len(out) >= max_total:
                return out
    return out


# --------------------------------------------------------------------------
# The per-field diagnostic writeup
# --------------------------------------------------------------------------

#: Each field carries a ``diagnostic_writeup/`` -- a compiled ``main.pdf`` plus a
#: fixed set of figures D1..D8.  Because the set is fixed and the same for every
#: field, a finding can be pointed at the figure that actually shows it instead of
#: dumping a directory listing.  Order matters: the first matching substring wins.
WRITEUP_DIR = 'diagnostic_writeup'
WRITEUP_FIGURES = {
    'D1': ('D1_overview', 'field overview'),
    'D2': ('D2_astrometry_internal', 'internal astrometric repeatability'),
    'D3': ('D3_astrometry_absolute', 'absolute tie to the reference frame'),
    'D4': ('D4_photometry_precision', 'photometric precision and depth'),
    'D5': ('D5_photometry_quality', 'fit quality'),
    'D6': ('D6_background_distributions', 'background distributions'),
    'D7': ('D7_background_spatial', 'background, spatial'),
    'D8': ('D8_color_diagrams', 'colour diagrams'),
}

#: Which writeup figure answers which finding.  Matched against the check name,
#: longest key first, so ``astrometry-worst-tile`` beats ``astrometry``.
FINDING_FIGURE = {
    'astrometry-worst-tile': 'D3',
    'astrometry-tile-contrast': 'D3',
    'astrometry-tie-unapplied': 'D3',
    'astrometry-tie-gross': 'D3',
    'astrometry-misaligned': 'D2',
    'astrometry-scatter': 'D2',
    'astrometry-consensus': 'D2',
    'astrometry-swept': 'D3',
    'astrometry-unverified': 'D3',
    'satstar-all-rejected': 'D8',
    'paper-certifiers-absent': 'D8',
    'unreduced': 'D1',
    'filteroffset-module-mismatch': 'D3',
    'crds-context-mixed': 'D3',
    'ladder-gap': 'D1',
    'ambiguous-catalogs': 'D1',
}


def writeup(base, link_base=None):
    """The field's diagnostic writeup: ``{main, figures: {code: href}}``.

    ``link_base`` is what the page should link to.  The served copy carries a
    ``diagnostics-<field>`` symlink into this directory, so passing that name
    yields hrefs that resolve with no extra publishing step; passing ``None``
    falls back to absolute paths, which are still useful on a shell.
    """
    d = os.path.join(base, WRITEUP_DIR)
    if not os.path.isdir(d):
        return None
    prefix = link_base.rstrip('/') if link_base else d
    out = {'dir': d, 'main': None, 'figures': {}, 'mtime': None}

    main = os.path.join(d, 'main.pdf')
    if os.path.exists(main):
        out['main'] = f'{prefix}/main.pdf'
        out['mtime'] = os.path.getmtime(main)

    figdir = os.path.join(d, 'figures')
    try:
        names = os.listdir(figdir)
    except OSError:
        names = []
    for code, (stem, label) in WRITEUP_FIGURES.items():
        hit = next((n for n in names if n.startswith(stem)), None)
        if hit:
            out['figures'][code] = {'href': f'{prefix}/figures/{hit}',
                                    'name': hit, 'label': label}
    return out if (out['main'] or out['figures']) else None


def figure_for_finding(name):
    """The writeup figure code that shows this finding, or ``None``."""
    for key in sorted(FINDING_FIGURE, key=len, reverse=True):
        if name.startswith(key):
            return FINDING_FIGURE[key]
    return None


# --------------------------------------------------------------------------
# Drawn diagnostics (inline SVG, no dependencies, no external requests)
# --------------------------------------------------------------------------

def _ramp(frac):
    """0..1 -> a colour on a neutral→amber→red ramp (safe in both themes)."""
    frac = max(0.0, min(1.0, frac))
    if frac <= 0.5:
        t = frac / 0.5
        h, s, light = 190 - 150 * t, 30 + 45 * t, 62 - 6 * t
    else:
        t = (frac - 0.5) / 0.5
        h, s, light = 40 - 30 * t, 75 + 15 * t, 56 - 10 * t
    return f'hsl({h:.0f} {s:.0f}% {light:.0f}%)'


def tile_map_svg(cells, tol_mas, worst_cell=None, size=168):
    """A per-tile residual map: WHERE across the mosaic the astrometry is bad.

    The whole reason this exists is that one number cannot separate the two
    cases that matter -- a single bad corner (an edge/coverage artefact) from a
    coherent gradient (a real frame problem) -- and they call for different
    responses.  Cells at or above ``tol_mas`` are outlined.
    """
    if not cells:
        return ''
    xs = [c['ix'] for c in cells if c.get('ix') is not None]
    ys = [c['iy'] for c in cells if c.get('iy') is not None]
    if not xs or not ys:
        return ''
    nx, ny = max(xs) + 1, max(ys) + 1
    offs = [c['off_mas'] for c in cells if c.get('off_mas') is not None]
    if not offs:
        return ''
    hi = max(max(offs), tol_mas)
    cw, ch = size / max(nx, 1), size / max(ny, 1)

    parts = []
    for c in cells:
        ix, iy, off = c.get('ix'), c.get('iy'), c.get('off_mas')
        if ix is None or iy is None:
            continue
        # iy grows upward on sky; SVG y grows downward, so flip it
        x, y = ix * cw, (ny - 1 - iy) * ch
        fill = '#8a949a22' if off is None else _ramp(off / hi if hi else 0)
        stroke = ('#c0392b' if (off is not None and off >= tol_mas)
                  else 'rgba(128,140,148,.35)')
        width = 1.6 if (off is not None and off >= tol_mas) else 0.5
        title = (f'cell ({ix},{iy}): '
                 + ('no tie' if off is None else f'{off:.1f} mas')
                 + (f', contrast {c["contrast"]:.0f}'
                    if c.get('contrast') is not None else ''))
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}">'
            f'<title>{title}</title></rect>')
    if worst_cell:
        try:
            wx, wy = worst_cell.strip('()').split(',')
            x = int(wx) * cw + cw / 2
            y = (ny - 1 - int(wy)) * ch + ch / 2
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(cw, ch) / 4:.1f}" '
                         f'fill="none" stroke="#c0392b" stroke-width="2"/>')
        except (ValueError, AttributeError):
            pass
    return (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
            f'role="img" aria-label="per-tile residual map" '
            f'style="border-radius:2px">{"".join(parts)}</svg>')


def quiver_svg(exposures, size=190, scale_mas=None):
    """Per-exposure offset vectors, coloured by detector.

    Answers "is this one detector or all of them" at a glance -- the difference
    between a detector-local defect and a whole-frame misalignment.
    """
    pts = [e for e in exposures
           if e.get('dra') is not None and e.get('ddec') is not None]
    if not pts:
        return ''
    mags = [max(abs(e['dra']), abs(e['ddec'])) for e in pts]
    hi = scale_mas or max(mags) or 1.0
    half = size / 2.0
    k = (half - 12) / hi

    detectors = sorted({e.get('detector') or '?' for e in pts})
    hues = {d: 200 - (i * 360 // max(len(detectors), 1)) % 340
            for i, d in enumerate(detectors)}

    parts = [f'<line x1="0" y1="{half}" x2="{size}" y2="{half}" '
             f'stroke="rgba(128,140,148,.3)" stroke-width="1"/>',
             f'<line x1="{half}" y1="0" x2="{half}" y2="{size}" '
             f'stroke="rgba(128,140,148,.3)" stroke-width="1"/>']

    # Only FLAGGED exposures get a vector and a tooltip.  The aligned ones sit
    # within a couple of mas of the origin, so their "vectors" are sub-pixel
    # stubs that add nothing to read -- but one <line> + <circle> + <title> each
    # is what took a 192-exposure filter to 92 kB of markup.  They are drawn as
    # bare dots instead, which is all they contribute: the background the
    # outliers stand against.
    dots, flagged = [], []
    for e in pts[:600]:
        (flagged if e.get('misaligned') else dots).append(e)

    by_colour = {}
    for e in dots:
        colour = f'hsl({hues.get(e.get("detector") or "?", 200)} 60% 45%)'
        by_colour.setdefault(colour, []).append(
            (half + e['dra'] * k, half - e['ddec'] * k))
    for colour, xy in by_colour.items():
        marks = ''.join(f'M{x:.0f} {y:.0f}h1' for x, y in xy)
        parts.append(f'<path d="{marks}" stroke="{colour}" stroke-width="1.6" '
                     f'opacity=".35" fill="none" stroke-linecap="round"/>')

    for e in flagged[:80]:
        x = half + e['dra'] * k
        y = half - e['ddec'] * k
        colour = f'hsl({hues.get(e.get("detector") or "?", 200)} 60% 45%)'
        parts.append(
            f'<line x1="{half}" y1="{half}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="1.1" opacity="0.95"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.7" fill="{colour}" '
            f'opacity="0.95"><title>{_esc(e.get("detector") or "?")} '
            f'visit {_esc(e.get("visit"))}: dRA {e["dra"]:.1f}, '
            f'dDec {e["ddec"]:.1f} mas</title></circle>')
    legend = ' '.join(
        f'<tspan fill="hsl({hues[d]} 60% 45%)">{_esc(d)}</tspan>'
        for d in detectors[:8])
    parts.append(f'<text x="4" y="{size - 4}" font-size="8" '
                 f'font-family="ui-monospace,monospace">{legend}</text>')
    parts.append(f'<text x="4" y="11" font-size="8" fill="rgba(128,140,148,.9)" '
                 f'font-family="ui-monospace,monospace">±{hi:.0f} mas</text>')
    return (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
            f'role="img" aria-label="per-exposure offset vectors" '
            f'style="border-radius:2px">{"".join(parts)}</svg>')


def detector_tally(exposures):
    """``[(detector, n_misaligned, n_total), ...]`` -- is it one chip or all?"""
    totals, bad = {}, {}
    for e in exposures:
        det = e.get('detector') or '?'
        totals[det] = totals.get(det, 0) + 1
        if e.get('misaligned'):
            bad[det] = bad.get(det, 0) + 1
    return sorted(((d, bad.get(d, 0), n) for d, n in totals.items()),
                  key=lambda r: (-r[1], r[0]))
