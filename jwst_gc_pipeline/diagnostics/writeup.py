r"""Generate the LaTeX write-up that accompanies a field's diagnostic figures.

The document is written from the *measurements* the figure builders returned,
not from a separate re-analysis, so the prose and the figures cannot drift
apart.  Where a number decides how a paragraph reads -- an astrometric floor
above the expected few mas, a propagated-to-formal error ratio well above
one, a background correlation that fails to materialise -- the generator
picks the sentence that matches the number rather than emitting a fixed
template.

Output is a self-contained ``main.tex`` plus a ``Makefile``, so each
field directory is an Overleaf-ready project on its own.
"""

import json
import os
from datetime import date

import numpy as np

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[hidelinks]{hyperref}
\usepackage{siunitx}
\usepackage{xcolor}

\newcommand{\code}[1]{\texttt{\small #1}}
\newcommand{\mas}{\si{\milli\arcsecond}}

\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
"""


def _fmt(value, digits=2, unit=''):
    """A number for prose, or ``--`` when it was not measurable."""
    if value is None:
        return '--'
    if isinstance(value, (int, np.integer)):
        return f'{value:,}{unit}'
    if not np.isfinite(value):
        return '--'
    return f'{value:.{digits}f}{unit}'


def _esc(text):
    """Escape the LaTeX specials that appear in file and column names."""
    return (str(text).replace('\\', r'\textbackslash{}')
            .replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
            .replace('#', r'\#').replace('$', r'\$'))


class Writeup:
    """Assemble the document for one field."""

    def __init__(self, inv, results, outdir):
        self.inv = inv
        self.results = {r.key: r for r in results if r is not None}
        self.outdir = outdir

    # ------------------------------------------------------------------ util

    def _figure(self, key, width=r'\textwidth'):
        res = self.results.get(key)
        if res is None:
            return ''
        return (
            '\\begin{figure}[htbp]\n\\centering\n'
            f'\\includegraphics[width={width}]{{{res.relpath}}}\n'
            f'\\caption{{{res.caption}}}\n'
            f'\\label{{{res.label}}}\n\\end{{figure}}\n\n')

    def _ref(self, key):
        res = self.results.get(key)
        return f'Figure~\\ref{{{res.label}}}' if res else 'the omitted figure'

    def _m(self, key, *path, default=None):
        """Dig a measurement out of a figure result."""
        res = self.results.get(key)
        if res is None:
            return default
        node = res.measurements
        for step in path:
            if not isinstance(node, dict) or step not in node:
                return default
            node = node[step]
        return node

    # -------------------------------------------------------------- sections

    def header(self):
        inv = self.inv
        proposals = ', '.join(inv.proposals)
        return (
            PREAMBLE +
            '\n\\begin{document}\n\n'
            f'\\title{{Diagnostic measurements for the JWST field '
            f'\\textsc{{{_esc(inv.name)}}}}}\n'
            f'\\author{{JWST Galactic Centre pipeline --- automated '
            f'diagnostic write-up}}\n'
            f'\\date{{{date.today().isoformat()}}}\n'
            '\\maketitle\n\n'
            '\\begin{abstract}\n'
            f'This document characterises the astrometric, photometric and '
            f'diffuse-background measurements delivered for '
            f'\\textsc{{{_esc(inv.name)}}} (JWST programme(s) {proposals}). '
            'It is an analysis of the finished data products, not a quality '
            'gate: it asks how well the measurements behave and where their '
            'limits are, rather than whether they pass a threshold. '
            'Every figure is generated from the products on disk by '
            '\\code{jwst\\_gc\\_pipeline.diagnostics}, and every number '
            'quoted in the text is read back from the same computation that '
            'drew the figure.\n'
            '\\end{abstract}\n\n')

    def introduction(self):
        inv = self.inv
        notes = ''
        if inv.notes:
            notes = ('\n\\paragraph{Product gaps.} '
                     + ' '.join(_esc(n) for n in inv.notes) + '\n')
        return (
            '\\section{Scope}\n'
            'Three related efforts characterise these data and it is worth '
            'saying at the outset which is which. The \\code{JWST-GC/data-qa} '
            'repository checks \\emph{initial} data products, close to the '
            'telescope, and is oriented towards catching a bad exposure early. '
            'The astrometry paper is a technique-development document: it '
            'follows the iterative work of establishing how to measure a '
            'position at all in these fields. This write-up is neither. It '
            'takes the finished, released products for one field and measures '
            'their properties comprehensively, so that a user of the catalogue '
            'can see what precision they are entitled to assume.\n'
            + notes +
            '\n\\section{Products measured}\n'
            + self.product_table() + '\n')

    def product_table(self):
        inv = self.inv
        rows = []
        for filt, cat, mos in inv.summary_rows():
            rows.append(
                f'{filt.upper()} & {_esc(cat) if cat else "--"} & '
                f'{_esc(mos) if mos else "--"} \\\\')
        cross = (_esc(os.path.basename(inv.crossband_catalog))
                 if inv.crossband_catalog else 'none')
        refs = ', '.join(_esc(os.path.basename(p))
                         for p in sorted(set(inv.reference_catalogs.values()))) \
            or 'none'
        return (
            'The field directory is \\code{' + _esc(inv.basepath) + '}. '
            f'The cross-band merge used here is \\code{{{cross}}} and the '
            f'astrometric reference is \\code{{{refs}}}.\n\n'
            '{\\footnotesize\n'
            '\\begin{longtable}{lll}\n\\toprule\n'
            'Filter & Per-filter catalogue & Mosaic \\\\\n\\midrule\n\\endhead\n'
            + '\n'.join(rows) +
            '\n\\bottomrule\n\\end{longtable}\n}\n')

    def overview_section(self):
        if 'D1_overview' not in self.results:
            return ''
        n = self._m('D1_overview', 'n_sources', default={}) or {}
        areas = self._m('D1_overview', 'area_arcsec2', default={}) or {}
        peak = self._m('D1_overview', 'peak_density')
        median = self._m('D1_overview', 'median_density')
        richest = self._m('D1_overview', 'richest_filter', default='')
        turn = self._m('D1_overview', 'turnover', default={}) or {}
        total = sum(n.values()) if n else 0
        area = areas.get(richest)
        crowding = ''
        if peak and np.isfinite(peak):
            crowding = (
                f'The peak surface density is {_fmt(peak, 2)} sources per '
                f'square arcsecond and the typical occupied cell holds '
                f'{_fmt(median, 2)}. ')
            if peak > 0.5:
                crowding += (
                    'At that density the mean separation between neighbours is '
                    'comparable to the point-spread function, so crowding, not '
                    'photon noise, sets the achievable precision over much of '
                    'the field; the qfit and propagated-error diagnostics '
                    'below should be read with that in mind. ')
        turn_txt = ''
        if turn:
            worst = min(turn, key=lambda k: turn[k])
            best = max(turn, key=lambda k: turn[k])
            turn_txt = (
                f'Number counts turn over between {_fmt(turn[worst], 2)} '
                f'({worst.upper()}) and {_fmt(turn[best], 2)} '
                f'({best.upper()}) magnitudes, which is the practical '
                'completeness limit of each band. ')
        return (
            '\\section{Overview}\n'
            f'{self._ref("D1_overview")} orients the rest of the document. '
            f'The per-filter catalogues hold {total:,} measurements in total, '
            f'the largest being {richest.upper()} with '
            f'{n.get(richest, 0):,} sources over roughly '
            f'{_fmt(area, 0)} square arcseconds. '
            + crowding + turn_txt + '\n\n'
            + self._figure('D1_overview'))

    def astrometry_section(self):
        if not ({'D2_astrometry_internal', 'D3_astrometry_absolute'}
                & set(self.results)):
            return ''
        body = ['\\section{Astrometry}\n']
        body.append(self._method_note_astrometry())

        floors = self._m('D2_astrometry_internal', 'floors_mas', default={}) or {}
        zeros = self._m('D2_astrometry_internal', 'n_zero_scatter',
                        default={}) or {}
        cross = self._m('D2_astrometry_internal', 'crossband', default={}) or {}

        if floors:
            finite = {k: v for k, v in floors.items() if v is not None
                      and np.isfinite(v)}
            body.append('\\subsection{Repeatability}\n')
            if finite:
                best = min(finite, key=finite.get)
                worst = max(finite, key=finite.get)
                body.append(
                    f'{self._ref("D2_astrometry_internal")} shows the '
                    'exposure-to-exposure scatter of each source about its '
                    'merged position. The bright-end floor ranges from '
                    f'{_fmt(finite[best], 2)}\\,\\mas{{}} in {best.upper()} to '
                    f'{_fmt(finite[worst], 2)}\\,\\mas{{}} in {worst.upper()}. '
                    'This floor is the single-exposure centroiding precision '
                    'convolved with whatever residual frame-to-frame '
                    'misregistration survived alignment; it is an upper limit '
                    'on the latter, and the merged position of a star seen in '
                    r'$N$ exposures is better than it by roughly $\sqrt{N}$. ')
                if max(finite.values()) > 15:
                    body.append(
                        'A floor above about 15\\,\\mas{} in any band is '
                        'larger than single-exposure centroiding alone '
                        'explains and points at a per-exposure registration '
                        'residual in that band rather than at measurement '
                        'noise. ')
            n_zero = sum(zeros.values()) if zeros else 0
            if n_zero:
                body.append(
                    f'\\ {n_zero:,} sources report a scatter of identically '
                    'zero. These are forced or seeded fits whose position was '
                    'copied from a fixed seed and never measured per exposure, '
                    'so their zero is an absence of measurement rather than '
                    'perfect precision; they are excluded from the curves. ')
            body.append('\n\n')

        if cross and cross.get('filters'):
            names = cross['filters']
            img = np.asarray(cross['median_sep_mas'], dtype=float)
            finite = img[np.isfinite(img)]
            if finite.size:
                body.append(
                    'The same stars measured independently in different '
                    'filters agree to a median of '
                    f'{_fmt(float(np.median(finite)), 2)}\\,\\mas{{}}, with the '
                    f'worst filter pair at {_fmt(float(np.max(finite)), 2)}'
                    '\\,\\mas{}. Because the two measurements share no '
                    'detector pixels, no distortion solution and no PSF, this '
                    'is a stringent internal check: it bounds the systematic '
                    'floor of the astrometry in a way that the '
                    'within-filter scatter, which shares all of those, cannot. ')
                if float(np.max(finite)) > 30:
                    body.append(
                        'The worst pair exceeds 30\\,\\mas{}, which is larger '
                        'than the repeatability floor and therefore indicates '
                        'a filter-dependent systematic --- a distortion or '
                        'filter-offset term --- rather than random error. ')
            body.append('\n\n')

        body.append(self._figure('D2_astrometry_internal'))

        bulk = self._m('D3_astrometry_absolute', 'bulk', default={}) or {}
        if bulk:
            body.append('\\subsection{Absolute tie}\n')
            offs, swept, weak, tiles = [], [], [], []
            n_swept_tiles = n_total_tiles = 0
            for filt, rec in bulk.items():
                b = rec.get('bulk') or {}
                if b.get('dra') is not None:
                    offs.append((filt, float(np.hypot(b['dra'], b['ddec']))))
                if b.get('swept'):
                    swept.append(filt)
                if b.get('ok') is False:
                    weak.append(filt)
                t = rec.get('tiles') or {}
                n_swept_tiles += t.get('n_swept', 0)
                n_total_tiles += t.get('n_total', 0)
                if t.get('worst_off_mas') is not None:
                    tiles.append((filt, t['worst_off_mas'], t['median_off_mas'],
                                  t.get('p95_off_mas', np.nan),
                                  t.get('n_above_50mas', 0),
                                  t.get('n_measured', 0)))
            if offs:
                worst = max(offs, key=lambda kv: kv[1])
                body.append(
                    f'{self._ref("D3_astrometry_absolute")} ties the catalogue '
                    'to its registered reference. Bulk offsets run from '
                    f'{_fmt(min(o[1] for o in offs), 1)} to '
                    f'{_fmt(worst[1], 1)}\\,\\mas{{}} '
                    f'(largest: {worst[0].upper()}). ')
            if tiles:
                wf, woff, _wmed, _wp95, _wn, _wm = max(tiles, key=lambda t: t[1])
                med_of_med = float(np.median([t[2] for t in tiles]))
                p95s = [t[3] for t in tiles if np.isfinite(t[3])]
                n_big = sum(t[4] for t in tiles)
                n_meas = sum(t[5] for t in tiles)
                body.append(
                    'Mapped per tile, the median tile offset across filters is '
                    f'{_fmt(med_of_med, 1)}\\,\\mas{{}}')
                if p95s:
                    body.append(
                        f', the 95th percentile {_fmt(float(np.median(p95s)), 1)}'
                        '\\,\\mas{}')
                body.append(f', and the worst single tile {_fmt(woff, 1)}'
                            f'\\,\\mas{{}} in {wf.upper()}. ')
                # A large maximum with a small 95th percentile is one bad tile;
                # a large 95th percentile is a property of the field.
                widespread = bool(p95s) and float(np.median(p95s)) > 50
                if widespread:
                    body.append(
                        'More than five per cent of tiles exceed 50\\,\\mas{}, '
                        'so this is not an isolated bad tile: it is a '
                        'spatially varying registration residual of the kind a '
                        'field-averaged number washes out, and it is exactly '
                        'what the per-tile map exists to catch. The overlap '
                        'regions of the affected bands should be inspected '
                        'before those positions are used at the tens of '
                        'milliarcsecond level. ')
                elif n_big:
                    body.append(
                        f'{n_big} of {n_meas} measured tiles exceed '
                        '50\\,\\mas{}. With the bulk tie and the tile median '
                        'both far below that, these are isolated tiles rather '
                        'than a coherent displaced region --- usually at the '
                        'field edge, where a tile is only partly filled --- '
                        'but they are worth a look in the map. ')
                else:
                    body.append(
                        'No tile departs far enough from the bulk value to '
                        'indicate a locally displaced region. ')
            if n_swept_tiles and n_total_tiles:
                frac = 100.0 * n_swept_tiles / n_total_tiles
                body.append(
                    f'{n_swept_tiles} of {n_total_tiles} tiles across all '
                    f'filters ({frac:.0f} per cent) reached a tie only after '
                    'the search window was widened. Those tiles hold too few '
                    'sources for a coherent peak at the nominal window, so '
                    'their offsets are excluded from the numbers above and '
                    'are marked rather than drawn in the figure; a high '
                    'fraction means the map is sparse, not that the field is '
                    'misregistered. ')
            if swept:
                body.append(
                    'The tie in ' + ', '.join(f.upper() for f in swept) +
                    ' was only found after the search window was widened, '
                    'which means the offset is large compared with the '
                    'nominal window; treat those bands as grossly shifted '
                    'until the alignment is revisited. ')
            if weak:
                body.append(
                    'No coherent peak was found for ' +
                    ', '.join(f.upper() for f in weak) +
                    ', so those bands have no measured absolute tie here. ')
            body.append('\n\n')
        body.append(self._figure('D3_astrometry_absolute'))
        return ''.join(body)

    def _method_note_astrometry(self):
        return (
            '\\paragraph{How the offsets were measured.} '
            'Every offset in this section comes from offset-histogram '
            'stacking: all pairs within a search window contribute to a '
            'two-dimensional histogram of their separations and the peak is '
            'the offset. The alternative --- matching each source to its '
            'nearest reference counterpart and taking the median --- is not '
            'used and is prohibited in this pipeline. When the true shift '
            'exceeds the reference catalogue\'s own nearest-neighbour '
            'spacing, nearest-neighbour matching pairs the wrong star, and '
            'the median of those wrong pairs collapses towards zero: the '
            'method reports agreement precisely when the frame is most '
            'badly misregistered. The search window is also swept rather '
            'than fixed, because a shift much larger than the window leaves '
            'no true pairs inside it and reads as noise rather than as a '
            'large offset.\n\n')

    def photometry_section(self):
        keys = {'D4_photometry_precision', 'D5_photometry_quality',
                'D8_color_diagrams'}
        if not (keys & set(self.results)):
            return ''
        body = ['\\section{Photometry}\n']

        depth = self._m('D4_photometry_precision', 'depth', default={}) or {}
        ratio = self._m('D4_photometry_precision', 'err_ratio', default={}) or {}
        finite_depth = {k: v for k, v in depth.items()
                        if v is not None and np.isfinite(v)}
        if finite_depth:
            deepest = max(finite_depth, key=finite_depth.get)
            body.append(
                '\\subsection{Precision and depth}\n'
                f'{self._ref("D4_photometry_precision")} shows the fractional '
                'flux uncertainty against brightness. The '
                r'$5\sigma$ depth --- where the median '
                r'$\sigma_F/F$ reaches $0.2$ --- is deepest in '
                f'{deepest.upper()} at {_fmt(finite_depth[deepest], 2)} mag, '
                f'and the full range across filters is '
                f'{_fmt(min(finite_depth.values()), 2)} to '
                f'{_fmt(max(finite_depth.values()), 2)} mag. ')
        if ratio:
            worst = max(ratio, key=ratio.get)
            median_ratio = float(np.median(list(ratio.values())))
            body.append(
                'The uncertainty propagated from the exposure-to-exposure '
                'scatter exceeds the fitter\'s formal covariance by a median '
                f'factor of {_fmt(median_ratio, 2)}, worst in '
                f'{worst.upper()} at {_fmt(ratio[worst], 2)}. ')
            if median_ratio > 1.5:
                body.append(
                    'A ratio well above unity means the exposures disagree by '
                    'more than the fit predicts, so the formal errors '
                    'understate the real uncertainty. In a crowded field the '
                    'usual cause is that the flux assigned to a source depends '
                    'on how its neighbours were fitted in that particular '
                    'exposure --- a term the single-exposure covariance has no '
                    'way to represent. The propagated error is the one to use. ')
            elif median_ratio < 0.8:
                body.append(
                    'A ratio below unity is the opposite problem: the '
                    'exposures agree with each other better than the fit '
                    'covariance predicts, so the formal error is '
                    'conservative. That happens when the same systematic --- '
                    'a shared neighbour model, a shared background --- moves '
                    'every exposure the same way, which suppresses the '
                    'scatter between them without making the measurement any '
                    'more correct. The propagated error should not be read '
                    'as the better estimate here; it is measuring '
                    'reproducibility, not accuracy. ')
            elif median_ratio < 1.1:
                body.append(
                    'The two agree closely, which means the noise model is '
                    'adequate and the fits are not being perturbed by '
                    'exposure-dependent crowding. ')
            body.append('\n\n')
        body.append(self._figure('D4_photometry_precision'))

        qfit = self._m('D5_photometry_quality', 'qfit', default={}) or {}
        census = self._m('D5_photometry_quality', 'census', default={}) or {}
        if qfit:
            meds = {k: v['median'] for k, v in qfit.items()}
            bad = {k: v['frac_above_warn'] for k, v in qfit.items()}
            worst = max(bad, key=bad.get)
            body.append(
                '\\subsection{Fit quality}\n'
                f'{self._ref("D5_photometry_quality")} shows the normalised '
                'PSF-fit residual. Median qfit ranges from '
                f'{_fmt(min(meds.values()), 3)} to {_fmt(max(meds.values()), 3)} '
                f'across filters; the fraction of sources above the '
                f'vetting threshold peaks at {100 * bad[worst]:.1f} per cent in '
                f'{worst.upper()}. ')
            if max(bad.values()) > 0.3:
                body.append(
                    'Where that fraction is large the catalogue is dominated '
                    'by sources whose profile the model does not reproduce, '
                    'which in these fields is normally blending rather than a '
                    'defective PSF: the qfit distribution rises towards the '
                    'faint end where neighbours are unresolved, not towards '
                    'the bright end where a bad PSF model would show. ')
        if census:
            sat = {k: 100.0 * v.get('is_saturated', 0) / max(v['total'], 1)
                   for k, v in census.items()}
            if sat:
                worst_sat = max(sat, key=sat.get)
                body.append(
                    f'Saturation touches at most {sat[worst_sat]:.1f} per cent '
                    f'of a band ({worst_sat.upper()}); the census panel breaks '
                    'that down into the flagged, core-replaced, gate-rejected '
                    'and clip-corrected populations, which is the accounting '
                    'needed before trusting any bright-star photometry. ')
        body.append('\n\n')
        body.append(self._figure('D5_photometry_quality'))

        if 'D8_color_diagrams' in self.results:
            body.append(
                '\\subsection{Colour diagrams}\n'
                f'{self._ref("D8_color_diagrams")} closes the photometric '
                'section. Colours are differential between filters and so '
                'reveal systematics that survive every per-band statistic '
                'above: a zero-point error displaces a sequence as a whole, a '
                'saturation-correction discontinuity breaks it at one '
                'magnitude, and a filter-dependent background error tilts it. '
                'A sequence that is continuous and has no spur is the '
                'strongest single piece of evidence that the cross-band merge '
                'assembled the same physical star in every band.\n\n')
            body.append(self._figure('D8_color_diagrams'))
        return ''.join(body)

    def background_section(self):
        keys = {'D6_background_distributions', 'D7_background_spatial'}
        if not (keys & set(self.results)):
            return ''
        body = ['\\section{Background and diffuse emission}\n',
                self._method_note_background()]

        per_filter = self._m('D6_background_distributions', 'per_filter',
                             default={}) or {}
        has_ms = self._m('D6_background_distributions', 'has_modelsub',
                         default={}) or {}
        if per_filter:
            meds = {k: v['median_local'] for k, v in per_filter.items()
                    if np.isfinite(v.get('median_local', np.nan))}
            excess = {k: v['bright_excess_local']
                      for k, v in per_filter.items()
                      if np.isfinite(v.get('bright_excess_local', np.nan))}
            if meds:
                hi = max(meds, key=meds.get)
                body.append(
                    '\\subsection{Distributions}\n'
                    f'{self._ref("D6_background_distributions")} shows both '
                    'estimators. The median annulus background ranges from '
                    f'{_fmt(min(meds.values()), 3)} to '
                    f'{_fmt(max(meds.values()), 3)} image units, highest in '
                    f'{hi.upper()}. ')
            n_ms = sum(1 for v in has_ms.values() if v)
            if n_ms == 0:
                body.append(
                    'The residual-footprint background is absent from every '
                    'catalogue here: it was introduced in July 2026 and these '
                    'products predate it. Re-running the cataloguing stage '
                    'would add it, and would make the background measurement '
                    'independent of which pipeline stage produced the '
                    'catalogue --- which the annulus estimator is not. ')
            elif n_ms < len(has_ms):
                body.append(
                    f'The residual-footprint background is present in {n_ms} '
                    f'of {len(has_ms)} filters, reflecting which bands have '
                    'been re-catalogued since it was introduced. ')
            if excess:
                worst = max(excess, key=lambda k: abs(excess[k]))
                body.append(
                    'Towards the bright end the annulus estimator rises: the '
                    'brightest one per cent of sources sit on a background '
                    f'that is {_fmt(excess[worst], 3)} image units above the '
                    f'field median in {worst.upper()}, the largest such offset '
                    'in this field. This is the star\'s own wings entering its '
                    'own background aperture, and it is a systematic rather '
                    'than noise --- it biases in one direction and grows with '
                    'brightness. Any use of the background column for bright '
                    'stars should account for it. ')
            body.append('\n\n')
        body.append(self._figure('D6_background_distributions'))

        corr = self._m('D7_background_spatial', 'correlations', default={}) or {}
        if corr:
            rhos = {k: v['spearman'] for k, v in corr.items()
                    if np.isfinite(v.get('spearman', np.nan))}
            if rhos:
                best = max(rhos, key=rhos.get)
                worst = min(rhos, key=rhos.get)
                median_rho = float(np.median(list(rhos.values())))
                body.append(
                    '\\subsection{Spatial structure}\n'
                    f'{self._ref("D7_background_spatial")} is the test of the '
                    'physical claim. The left column is a map of the diffuse '
                    'emission built entirely out of the star catalogue; the '
                    'right column sets the same per-source value against the '
                    'drizzled mosaic sampled at each star. The rank '
                    f'correlation between them has a median of '
                    f'{_fmt(median_rho, 3)} across filters, from '
                    f'{_fmt(rhos[worst], 3)} ({worst.upper()}) to '
                    f'{_fmt(rhos[best], 3)} ({best.upper()}). ')
                if median_rho > 0.6:
                    body.append(
                        'A correlation that strong means the per-source '
                        'background is measuring real extended emission, not '
                        'a fitting artefact: two independent measurements of '
                        'the same sky --- one from a per-star aperture, one '
                        'from the drizzled surface brightness --- agree in '
                        'rank across the field. The catalogue column can '
                        'therefore be used as a diffuse-emission tracer '
                        'sampled at the positions of stars, at a spatial '
                        'sampling set by the source density rather than by '
                        'the mosaic pixel scale. ')
                elif median_rho > 0.3:
                    body.append(
                        'That is a real but partial correlation. The '
                        'per-source background is picking up the extended '
                        'emission, but a substantial part of its variance '
                        'comes from something else --- most plausibly the '
                        'fitting residuals of neighbouring stars, which in a '
                        'crowded field contribute to the same aperture. ')
                else:
                    body.append(
                        'That is weak enough to say the per-source background '
                        'in this field is not primarily tracking the extended '
                        'emission. In a field whose mosaic is close to flat '
                        'this is the expected result --- there is little '
                        'diffuse structure to correlate with --- and the '
                        'mosaic panel should be inspected before reading '
                        'anything further into it. ')
                spread = max(rhos.values()) - min(rhos.values())
                if spread > 0.4:
                    body.append(
                        'The spread between filters is large, which is itself '
                        'informative: the bands where the correlation is '
                        'strongest are those whose diffuse emission is '
                        'brightest relative to the stellar contribution. ')
            body.append('\n\n')
        body.append(self._figure('D7_background_spatial'))
        return ''.join(body)

    def _method_note_background(self):
        return (
            '\\paragraph{Two different quantities.} '
            'The catalogues carry up to two background columns and they do '
            'not measure the same thing. \\code{local\\_bkg} is the '
            'sigma-clipped annulus median that was handed to the PSF fitter, '
            'evaluated on whatever image that stage fitted; in the later '
            'stages that image has already had a smoothed background removed, '
            'so the column measures the residual of that subtraction and '
            'tends towards zero. \\code{mean\\_modelsub\\_bkg} is the mean '
            r'over a $3\times3$ pixel box on the per-exposure residual after '
            'only the star model is subtracted, combined across exposures as '
            'a sigma-clipped weighted mean. Writing $D_i$ for the exposure '
            'data, $S_i$ for the smoothed background subtracted before '
            'fitting, $B_i$ for the annulus background and $P_i$ for the '
            'fitted PSF model,\n'
            '\\begin{equation}\n'
            'A_i = D_i - S_i - B_i, \\qquad R_i = D_i - S_i - P_i,\n'
            '\\end{equation}\n'
            'where $A_i$ is the array the fitter minimised against and $R_i$ '
            'is the image this column is measured on. Because $R_i$ removes '
            'only the star, the second quantity is invariant across pipeline '
            'stages while the first is not --- which is why they are plotted '
            'separately and never averaged together.\n\n')

    def implications(self):
        """A short, field-specific closing section."""
        lines = ['\\section{Implications}\n']
        items = []

        floors = self._m('D2_astrometry_internal', 'floors_mas', default={}) or {}
        finite = [v for v in floors.values() if v is not None and np.isfinite(v)]
        if finite:
            items.append(
                f'Positions in this field are repeatable to '
                f'{_fmt(float(np.median(finite)), 1)}\\,\\mas{{}} per exposure '
                'at the bright end, so a proper-motion programme against an '
                'earlier epoch is limited by the epoch baseline and by the '
                'reference frame, not by these measurements.')

        ratio = self._m('D4_photometry_precision', 'err_ratio', default={}) or {}
        if ratio:
            med = float(np.median(list(ratio.values())))
            if med > 1.2:
                items.append(
                    'Quote \\code{flux\\_err\\_prop} rather than '
                    '\\code{flux\\_err}: the propagated error is a median '
                    f'factor of {_fmt(med, 2)} larger here, and it is the one '
                    'that reflects how much the measurement actually moves '
                    'between exposures.')
            elif med < 0.8:
                items.append(
                    'Do not substitute \\code{flux\\_err\\_prop} for '
                    '\\code{flux\\_err} in this field: it is a median factor '
                    f'of {_fmt(med, 2)} \\emph{{smaller}}, which means the '
                    'exposures reproduce each other better than the fit '
                    'covariance allows --- a measure of consistency, not of '
                    'accuracy. The larger of the two is the safer error bar.')

        corr = self._m('D7_background_spatial', 'correlations', default={}) or {}
        rhos = [v['spearman'] for v in corr.values()
                if np.isfinite(v.get('spearman', np.nan))]
        if rhos and float(np.median(rhos)) > 0.5:
            items.append(
                'The per-source background column is usable as a '
                'diffuse-emission tracer in its own right, sampled wherever '
                'there is a star; that is a spatial sampling no mosaic-based '
                'measurement provides, because it is unaffected by the '
                'stellar light that dominates the mosaic in these fields.')

        bulk = self._m('D3_astrometry_absolute', 'bulk', default={}) or {}
        p95s = [rec['tiles'].get('p95_off_mas', np.nan)
                for rec in bulk.values() if rec.get('tiles')]
        p95s = [v for v in p95s if v is not None and np.isfinite(v)]
        if p95s and float(np.median(p95s)) > 50:
            items.append(
                'Local registration residuals reach '
                f'{_fmt(float(np.median(p95s)), 0)}\\,\\mas{{}} at the 95th '
                'percentile of tiles --- widespread rather than isolated. '
                'Resolve this before using the affected bands for anything '
                'positional at the tens of milliarcsecond level.')

        has_ms = self._m('D6_background_distributions', 'has_modelsub',
                         default={}) or {}
        if has_ms and not any(has_ms.values()):
            items.append(
                'Re-cataloguing this field would add the stage-invariant '
                'residual-footprint background, which the current products '
                'predate; until then the background column here cannot be '
                'compared with a field catalogued after July 2026.')

        if not items:
            return ''
        lines.append('\\begin{itemize}\n')
        lines += [f'\\item {t}\n' for t in items]
        lines.append('\\end{itemize}\n\n')
        return ''.join(lines)

    def reproducibility(self):
        from jwst_gc_pipeline.version import __version__ as pipeline_version
        return (
            '\\section{Reproducibility}\n'
            'Every figure in this document was produced by\n'
            '\\begin{quote}\\code{python scripts/analysis/'
            'make\\_diagnostic\\_writeup.py --field '
            + _esc(self.inv.name) + '}\\end{quote}\n'
            f'against \\code{{jwst\\_gc\\_pipeline}} version '
            f'\\code{{{_esc(pipeline_version)}}}. The numerical values quoted '
            'in the text are stored alongside the figures in '
            '\\code{measurements.json}, so the prose and the plots are '
            'generated from one computation and cannot disagree.\n\n')

    # ------------------------------------------------------------------ build

    def render(self):
        return (self.header()
                + self.introduction()
                + self.overview_section()
                + self.astrometry_section()
                + self.photometry_section()
                + self.background_section()
                + self.implications()
                + self.reproducibility()
                + '\\end{document}\n')

    def write(self):
        os.makedirs(self.outdir, exist_ok=True)
        tex = os.path.join(self.outdir, 'main.tex')
        with open(tex, 'w') as fh:
            fh.write(self.render())
        payload = {k: dict(caption=r.caption, section=r.section,
                           figure=r.relpath, measurements=r.measurements)
                   for k, r in self.results.items()}
        with open(os.path.join(self.outdir, 'measurements.json'), 'w') as fh:
            json.dump(payload, fh, indent=2, default=_jsonable)
        return tex


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)
