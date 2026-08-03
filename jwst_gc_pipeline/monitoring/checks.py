"""Turn a scan into a list of verdicts.

Every threshold used here is IMPORTED from the module that enforces it, never
copied as a literal.  A monitor that carries its own copy of ``5 mas`` drifts
away from the gate it claims to be watching and then reports green on a run the
pipeline would have refused, so the numbers have exactly one home.

Severities
----------
``fail``  the run is broken or its products are not usable as they stand.
``warn``  the run produced output that needs a human look before it is trusted.
``info``  a fact worth showing that is not a problem.
``skip``  the check does not apply here (no data, wrong instrument, cutout run).
"""
import os

from ..photometry.astrometry_offsets import DEFAULT_MIN_CONTRAST
from ..photometry.astrometry_checkpoint import (CROSSFILTER_TOL_MAS,
                                                LOCAL_CELL_TOL_MAS,
                                                STAGE_STABILITY_TOL_MAS)
from ..photometry.visit_consensus import (EXPOSURE_CONSENSUS_TOL_MAS,
                                          REFERENCE_AGREE_TOL_MAS,
                                          REFERENCE_CROSSCHECK_GROSS_MAS)

#: Environment variables that switch OFF a safety gate.  A run made with any of
#: these set produced products the pipeline would otherwise have refused, so the
#: monitor reports them as a property of the RUN, not of the data.
SAFETY_OVERRIDES = (
    'ASTROM_CHECKPOINT',                  # =0 disables the checkpoint ladder
    'ALLOW_LATE_STAGE_ASTROM_SHIFT',      # accepts a shift at m3-m6
    'ALLOW_CROSSFILTER_ASTROM_FAIL',      # accepts cross-filter disagreement at m7
    'ALLOW_MISSING_MERGEDCAT_MOSAIC',     # accepts a merge with no mosaic
    'FORCE_REALIGN_ON_DISAGREE',          # (opposite sense: stricter, reported as info)
)

#: A visit-consensus scatter far above the per-exposure tolerance means the
#: exposures do not agree with each other even after alignment.  There is no
#: single constant for this in the pipeline (the consensus is not gated on its
#: own scatter), so the monitor flags at a multiple of the exposure tolerance and
#: says so rather than implying an enforced gate.
CONSENSUS_SCATTER_WARN_MAS = 10 * EXPOSURE_CONSENSUS_TOL_MAS

#: Days after which an unfinished ladder is called stalled rather than running.
STALE_DAYS = 14

SEVERITIES = ('fail', 'warn', 'info', 'skip')


def _verdict(name, severity, summary, detail='', value=None, threshold=None,
             source='', cause='', evidence=None):
    """One finding.

    ``detail`` says what the number means; ``cause`` says how a run gets into
    this state and what to do; ``evidence`` carries the rows and drawings that
    show WHICH things are affected.  A red line that reports only a number
    ("misaligned: 183") cannot be acted on -- the reader still has to go and find
    out which 183, on which detector, and whether it is one chip or all of them.
    """
    return {'name': name, 'severity': severity, 'summary': summary,
            'detail': detail, 'value': value, 'threshold': threshold,
            'source': source, 'cause': cause, 'evidence': evidence or {}}


# --------------------------------------------------------------------------
# Astrometry
# --------------------------------------------------------------------------

def check_astrometry(run):
    """Verdicts from the m2 checkpoint records the pipeline wrote itself.

    Nothing is re-measured.  The checkpoint already ran the sanctioned
    offset-histogram machinery; recomputing an offset here would mean
    hand-rolling a crossmatch, and the ad-hoc version of that is the banned
    nearest-neighbour-median against a dense reference.
    """
    out = []
    astrom = run.get('astrometry') or {}
    if not astrom:
        out.append(_verdict(
            'astrometry-checkpoint', 'skip',
            'no m2 astrometry checkpoint on disk',
            'The field has not reached the m2 merge, or ran with '
            'ASTROM_CHECKPOINT=0.',
            source='astrometry_checkpoints/checkpoint_m2_*_latest.json'))
        return out

    for filt, rec in sorted(astrom.items()):
        src = os.path.basename(rec.get('path', ''))
        n_mis = rec.get('n_misaligned') or 0
        n_exp = rec.get('n_exposures') or 0
        # A checkpoint is written per FILTER, not per observation.  Where the
        # filter belongs to several observations of the field, the record cannot
        # be pinned to this one -- reporting it as this observation's failure
        # would raise the same alarm on every sibling observation, most of which
        # it does not describe.
        attributable = rec.get('attributable', True)
        # A checkpoint is a SNAPSHOT of the moment it ran.  At m2 a misalignment
        # corrects the offsets table and stops the run so the crf frames can be
        # regenerated -- so if the reduced frames are NEWER than the record, the
        # correction has already been applied and the record describes the state
        # before the fix.  Reporting that as a live failure would keep a resolved
        # problem permanently red.
        reduced_mtime = ((run.get('per_filter') or {}).get(filt, {})
                         .get('reduced') or {}).get('mtime')
        superseded = bool(reduced_mtime and rec.get('mtime')
                          and reduced_mtime > rec['mtime'])
        stale_note = (
            '' if not superseded else
            '  NOTE: the reduced frames are NEWER than this checkpoint, so the '
            'correction it triggered has probably already been applied — re-run '
            'the m2 checkpoint to confirm rather than treating this as current.')
        unattributed_note = (
            '' if attributable else
            f'  NOTE: {filt} belongs to more than one observation of this field '
            f'and the checkpoint file names only the filter, so this record '
            f'cannot be attributed to {run.get("proposal")}/o{run.get("obsid")} — '
            f'it describes whichever observation last ran m2.')
        if n_mis:
            from . import figures as _fig
            severity = 'fail' if (attributable and not superseded) else 'warn'
            suffix = (' (unattributed)' if not attributable
                      else ' (checkpoint predates the frames)' if superseded else '')
            bad = rec.get('misaligned_exposures') or []
            tally = _fig.detector_tally(rec.get('all_exposures') or [])
            hit = [row for row in tally if row[1]]
            concentrated = len(hit) == 1 and len(tally) > 1
            spread = f'{len(hit)}/{len(tally)} detectors'
            out.append(_verdict(
                f'astrometry-misaligned-{filt}', severity,
                f'{filt}: {n_mis}/{n_exp} exposures misaligned vs their visit '
                f'consensus' + suffix,
                f'An exposure is misaligned when its offset exceeds '
                f'{EXPOSURE_CONSENSUS_TOL_MAS} mas AND is significant against the '
                f'peak error bars.  At m2 this CORRECTS the offsets table and stops '
                f'the run; the crf frames must be regenerated before cataloging '
                f'continues.' + unattributed_note + stale_note,
                value=n_mis, threshold=0, source=src,
                cause=(
                    f'Affected: {spread}'
                    + (f' — confined to {hit[0][0]}, which points at a '
                       f'detector-local defect (distortion reference, a bad '
                       f'filteroffset, or one chip\'s frames never re-aligned) '
                       f'rather than a whole-frame misalignment.'
                       if concentrated else
                       ' — spread across detectors, so the frame as a whole moved: '
                       'look at the offsets table and whether these exposures were '
                       'regenerated from _cal after the table changed.')
                    + ' fix_alignment SKIPS a frame that already has a RAOFFSET '
                      'header, so correcting the table alone leaves stale frames in '
                      'place; regenerate the working copy from _cal instead of '
                      're-applying on top.'),
                evidence={
                    'quiver': _fig.quiver_svg(rec.get('all_exposures') or []),
                    'detector_tally': tally,
                    'rows': {
                        'columns': ['visit', 'detector', 'dRA (mas)',
                                    'dDec (mas)', 'off (mas)', 'contrast',
                                    'pairs', 'window ("")'],
                        'data': [[e.get('visit'), e.get('detector'),
                                  e.get('dra'), e.get('ddec'), e.get('off'),
                                  e.get('contrast'), e.get('npairs'),
                                  e.get('window_arcsec')] for e in bad[:40]],
                        'total': len(bad)},
                    'filter': filt}))
        elif n_exp:
            out.append(_verdict(
                f'astrometry-misaligned-{filt}', 'info',
                f'{filt}: all {n_exp} exposures within '
                f'{EXPOSURE_CONSENSUS_TOL_MAS} mas of consensus',
                value=0, threshold=0, source=src))

        contrast = rec.get('min_contrast')
        if contrast is not None and contrast < DEFAULT_MIN_CONTRAST:
            out.append(_verdict(
                f'astrometry-contrast-{filt}', 'fail',
                f'{filt}: lowest offset-peak contrast {contrast:.1f} '
                f'< {DEFAULT_MIN_CONTRAST}',
                'A low peak contrast means the offset histogram found no real tie. '
                'It does NOT mean "no offset": a shift larger than the search window '
                'has zero true pairs inside it, so a gross misalignment reads as low '
                'contrast (the brick-1182 v001 signature).',
                value=contrast, threshold=DEFAULT_MIN_CONTRAST, source=src))

        n_swept = rec.get('n_swept') or 0
        if n_swept:
            out.append(_verdict(
                f'astrometry-swept-{filt}', 'warn',
                f'{filt}: {n_swept} exposure tie(s) needed a swept search window',
                'measure_offset widens the window (3->10->30->60") when a narrow '
                'window finds no peak.  A swept tie means the frame is grossly '
                'shifted; check that the peak reproduces at an independent window '
                'rather than sitting on the window edge.',
                value=n_swept, threshold=0, source=src))

        n_unver = rec.get('n_unverified') or 0
        if n_unver:
            out.append(_verdict(
                f'astrometry-unverified-{filt}', 'warn',
                f'{filt}: {n_unver} exposure(s) could not be verified',
                'Recorded as could-not-verify: the reference tie was not coherent, '
                f'or the gross cross-check ({REFERENCE_CROSSCHECK_GROSS_MAS:g} mas) '
                'failed, so no correction was applied.',
                value=n_unver, threshold=0, source=src))

        for visit in rec.get('visits') or []:
            scatter = visit.get('scatter_mas')
            label = f'{filt} visit {visit.get("visit")}'

            # ⚠ The headline "N/N tiles ok" is NOT a tolerance statement.
            # measure_offset_grid runs with no max_off_mas, and astrometry_offsets
            # sets off_ok=True whenever that is None -- so a tile counts as ok if
            # its offset histogram had a coherent PEAK, however large the offset.
            # Measured on brick F182M: 36/36 tiles "ok" with a 29.1 mas worst
            # cell. Only worst_off_mas reports the thing the ladder gates on.
            worst = visit.get('worst_tile_mas')
            if worst is not None and worst > LOCAL_CELL_TOL_MAS:
                from . import figures as _fig
                cells = visit.get('cells') or []
                over = [c for c in cells
                        if (c.get('off_mas') or 0) > LOCAL_CELL_TOL_MAS]
                edge = [c for c in over
                        if c.get('ix') in (0, 5) or c.get('iy') in (0, 5)]
                on_edge = over and len(edge) == len(over)
                out.append(_verdict(
                    f'astrometry-worst-tile-{filt}-v{visit.get("visit")}', 'warn',
                    f'{label}: worst tile {worst:.1f} mas at cell '
                    f'{visit.get("worst_tile_cell") or "?"} '
                    f'(> {LOCAL_CELL_TOL_MAS:g} mas)',
                    f'{visit.get("tiles_ok")}/{visit.get("tiles_total")} tiles are '
                    f'reported "ok", but that counts tiles whose offset histogram '
                    f'had a coherent PEAK — not tiles within tolerance. The m7 '
                    f'cross-band gate is no significant 2" cell above '
                    f'{LOCAL_CELL_TOL_MAS:g} mas, and this cell exceeds it, so the '
                    f'bulk tie being ~0 does not mean the field is flat.',
                    value=worst, threshold=LOCAL_CELL_TOL_MAS, source=src,
                    cause=(
                        (f'{len(over)}/{len(cells)} cells exceed '
                         f'{LOCAL_CELL_TOL_MAS:g} mas' if cells else
                         'The per-cell list was not recorded in this checkpoint, '
                         'so only the worst value is available — the map below is '
                         'empty for that reason, not because the field is flat')
                        + (', and every one of them is on the mosaic EDGE — that is '
                           'usually thin coverage (few exposures, so few pairs per '
                           'cell) rather than a distortion error. Check the pair '
                           'counts in the table before treating it as a defect.'
                           if on_edge else
                           '. They are not confined to the edge, so this is an '
                           'interior residual: a distortion or per-detector '
                           'alignment problem, not a coverage artefact.')
                        + f' The bulk tie for this visit is '
                        + (f'{visit["tie_off_mas"]:.2f} mas'
                           if visit.get('tie_off_mas') is not None else 'unrecorded')
                        + ' — a small bulk value cannot cancel a local one.'),
                    evidence={
                        'tile_map': _fig.tile_map_svg(
                            cells, LOCAL_CELL_TOL_MAS,
                            visit.get('worst_tile_cell')),
                        'rows': {
                            'columns': ['cell', 'off (mas)', 'dRA', 'dDec',
                                        'contrast', 'pairs'],
                            'data': [[f'({c.get("ix")},{c.get("iy")})',
                                      c.get('off_mas'), c.get('dra'),
                                      c.get('ddec'), c.get('contrast'),
                                      c.get('npairs')]
                                     for c in sorted(
                                         over,
                                         key=lambda c: -(c.get('off_mas') or 0))[:36]],
                            'total': len(over)},
                        'filter': filt}))

            tile_contrast = visit.get('min_tile_contrast')
            if tile_contrast is not None and tile_contrast < DEFAULT_MIN_CONTRAST:
                out.append(_verdict(
                    f'astrometry-tile-contrast-{filt}-v{visit.get("visit")}', 'fail',
                    f'{label}: weakest tile peak contrast {tile_contrast:.0f} '
                    f'< {DEFAULT_MIN_CONTRAST:g}',
                    'At least one tile has no real tie, so its offset is noise.',
                    value=tile_contrast, threshold=DEFAULT_MIN_CONTRAST, source=src))

            if visit.get('tie_apply_ok') is False:
                out.append(_verdict(
                    f'astrometry-tie-unapplied-{filt}-v{visit.get("visit")}', 'warn',
                    f'{label}: reference tie recorded as could-not-verify '
                    f'(apply_ok=false)',
                    'A correction is applied only when the tie is coherent AND the '
                    f'gross cross-check ({REFERENCE_CROSSCHECK_GROSS_MAS:g} mas) '
                    'passes AND the per-tile map is clean. This one was not, so no '
                    'correction was made — the frame keeps whatever it had.',
                    source=src))
            elif visit.get('tie_gross_ok') is False:
                out.append(_verdict(
                    f'astrometry-tie-gross-{filt}-v{visit.get("visit")}', 'fail',
                    f'{label}: reference tie failed the gross cross-check '
                    f'(> {REFERENCE_CROSSCHECK_GROSS_MAS:g} mas vs sparse Gaia)',
                    'This is the guard that catches a spurious or window-limited '
                    'VIRAC peak — the brick-1182 v001 ~700 mas tell.',
                    source=src))

            if visit.get('consensus_ok') is False:
                out.append(_verdict(
                    f'astrometry-consensus-{filt}-v{visit.get("visit")}', 'fail',
                    f'{filt} visit {visit.get("visit")}: consensus not formed',
                    'The exposures of this visit could not be brought into a common '
                    'frame, so nothing downstream of m2 is trustworthy for it.',
                    source=src))
            elif scatter is not None and scatter > CONSENSUS_SCATTER_WARN_MAS:
                out.append(_verdict(
                    f'astrometry-scatter-{filt}-v{visit.get("visit")}', 'warn',
                    f'{filt} visit {visit.get("visit")}: consensus scatter '
                    f'{scatter:.1f} mas',
                    f'Above {CONSENSUS_SCATTER_WARN_MAS:g} mas '
                    f'({int(CONSENSUS_SCATTER_WARN_MAS / EXPOSURE_CONSENSUS_TOL_MAS)}x '
                    f'the {EXPOSURE_CONSENSUS_TOL_MAS} mas per-exposure tolerance). '
                    'This is a monitor heuristic, not a pipeline gate.',
                    value=scatter, threshold=CONSENSUS_SCATTER_WARN_MAS, source=src))
    return out


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def check_provenance(run):
    """One pipeline tag per phase, and no dirty-tree tags.

    A field whose filters were produced by different pipeline tags cannot be
    released: the image and the catalog must come from the same run.  A ``-dirty``
    tag means the tree had uncommitted changes when the product was written, so
    the tag does not identify reproducible code.
    """
    out = []
    prov = run.get('provenance') or {}
    if not prov:
        out.append(_verdict('provenance', 'skip', 'no provenance sidecars found',
                            'No *.prov.json next to the merged catalogs.',
                            source='catalogs/*.prov.json'))
        return out
    for phase, rec in sorted(prov.items()):
        tags = rec.get('tags') or {}
        if rec.get('n_distinct', 0) > 1:
            listing = ', '.join(f'{t} ({n})' for t, n in
                                sorted(tags.items(), key=lambda kv: -kv[1]))
            out.append(_verdict(
                f'provenance-mixed-{phase}', 'fail',
                f'{phase}: products come from {rec["n_distinct"]} different '
                f'pipeline tags',
                f'{listing}.  Image and catalog must be deployed from the SAME '
                f'pipeline run; a mixed-tag set is not releasable as one product.',
                value=rec['n_distinct'], threshold=1,
                source=f'catalogs/*{phase}*.prov.json',
                cause=(
                    'This happens when filters are re-run at different times and '
                    'only some are regenerated — a partial rerun, or a merge that '
                    'picked up an older per-filter product. The fix is to re-run '
                    'the lagging filters at the current tag, not to re-stamp: the '
                    'tag records which code wrote the numbers.'),
                evidence={'rows': {
                    'columns': ['pipeline tag', 'products'],
                    'data': sorted(tags.items(), key=lambda kv: -kv[1]),
                    'total': len(tags)}}))
        else:
            out.append(_verdict(
                f'provenance-{phase}', 'info',
                f'{phase}: single tag {next(iter(tags), "?")} '
                f'({rec.get("n_sidecars")} products)',
                value=1, threshold=1, source=f'catalogs/*{phase}*.prov.json'))
        if rec.get('n_dirty'):
            out.append(_verdict(
                f'provenance-dirty-{phase}', 'warn',
                f'{phase}: {rec["n_dirty"]} product(s) stamped with a -dirty tag',
                'The tree had uncommitted changes, so the tag does not identify '
                'the code that ran.  Fine for a development run; not for a release.',
                value=rec['n_dirty'], threshold=0,
                source=f'catalogs/*{phase}*.prov.json'))
    return out


# --------------------------------------------------------------------------
# Products / ladder
# --------------------------------------------------------------------------

def check_products(run):
    """Coverage and ordering problems in the stage ladder."""
    out = []
    per_filter = run.get('per_filter') or {}

    # Some registered observations are deliberately never globbed by the
    # reduction (wd1 lists o001+o003 but reads only 001; wd2 o003+o005 reads
    # 005).  Their directories are empty BY DESIGN, so every product check below
    # would fire on a field that is behaving correctly.
    if run.get('globbed') is False:
        return [_verdict(
            'not-globbed', 'info',
            f'o{run.get("obsid")} is registered but not globbed '
            f'(glob_obsid = {run.get("glob_obsid")!r})',
            'The reduction builds filenames from glob_obsid, so this observation '
            'has no products by design. Nothing below applies to it.',
            source='fields.yaml glob_obsid')]

    missing = run.get('filters_missing') or []
    if missing:
        out.append(_verdict(
            'filters-missing', 'warn',
            f'{len(missing)} registered filter(s) have no directory: '
            f'{", ".join(missing)}',
            'Registered in fields.yaml but absent from disk -- either not yet '
            'reduced, or reduced somewhere else.',
            value=len(missing), threshold=0, source='fields.yaml vs disk'))

    for filt, rows in sorted(per_filter.items()):
        crf = (rows.get('crf') or {}).get('n', 0)
        reduced = rows.get('reduced') or {}
        if crf and not reduced.get('n'):
            out.append(_verdict(
                f'unreduced-{filt}', 'fail',
                f'{filt}: {crf} crf frames but no destreak_/align_ working copies',
                'Cataloging reads the reduced working copy named by --each-suffix. '
                'A filter that never got one is catalogued against a suffix that '
                'matches nothing -- the wd1 F150W failure mode.',
                value=0, threshold=crf, source=f'{filt}/pipeline/*_crf.fits',
                cause=(
                    'Stage 1 produced crf frames for this filter but the '
                    'destreak/align step never ran on them, so --each-suffix '
                    'matches nothing and the filter is silently skipped by '
                    'cataloging — it does not error, it just produces no catalog. '
                    'Re-run the reduction for this filter, or set an '
                    '--each-suffix-override if it is meant to use the bare crf.'),
                evidence={'rows': {
                    'columns': ['product', 'count'],
                    'data': [['crf frames', crf],
                             ['destreak_/align_ working copies', 0]],
                    'total': 2}}))

        # Saturated stars: all-rejected is NOT the same as none-present.  A filter
        # whose current stage produced zero satstar catalogs but many rejected
        # files still ships saturated photometry -- carried over from an earlier
        # stage -- so the product mixes generations.  Measured on brick: F405N and
        # F410M have 0 catalogs / 48 rejected each while the m8 merge reports
        # thousands of replaced_saturated rows.
        sat = rows.get('satstar_frames') or {}
        if sat.get('rejected') and not sat.get('accepted'):
            out.append(_verdict(
                f'satstar-all-rejected-{filt}', 'fail',
                f'{filt}: 0 satstar catalogs but {sat["rejected"]} rejected frames',
                'Every saturated-star fit at this stage was rejected, so any '
                'saturated photometry in the merged catalog came from an earlier '
                'stage and the shipped product mixes generations. A '
                'present/absent count of satstar products cannot see this.',
                value=0, threshold=sat['rejected'],
                source=f'{filt}/pipeline/*_satstar_{{catalog,rejected}}.fits',
                cause=(
                    'Every saturated-star fit was rejected at this stage — usually '
                    'a too-tight gate (wing-fit tolerance, core radius, or a PSF '
                    'model that does not match the saturated profile). Because the '
                    'merge falls back to whatever satstar products already existed, '
                    'the shipped catalog keeps the OLD generation and looks '
                    'complete. Compare replaced_saturated counts in the merged '
                    'catalog against these zeros before trusting bright-end '
                    'photometry.'),
                evidence={'rows': {
                    'columns': ['satstar product', 'count'],
                    'data': [['accepted catalogs', sat.get('accepted', 0)],
                             ['rejected', sat.get('rejected', 0)],
                             ['wingcal calibrators', sat.get('wingcal', 0)]],
                    'total': 3}}))

        # a later cataloging phase present while an earlier one is absent
        seen = [(p, (rows.get(p) or {}).get('n', 0))
                for p in ('m12', 'm3', 'm4', 'm5', 'm6', 'm7')]
        present = [p for p, n in seen if n]
        if present:
            first = seen.index((present[0], dict(seen)[present[0]]))
            gaps = [p for p, n in seen[first:] if not n and p != present[-1]]
            holes = [p for p in gaps if any(
                n for q, n in seen if q in ('m12', 'm3', 'm4', 'm5', 'm6', 'm7')
                and ('m12', 'm3', 'm4', 'm5', 'm6', 'm7').index(q) >
                ('m12', 'm3', 'm4', 'm5', 'm6', 'm7').index(p))]
            if holes:
                out.append(_verdict(
                    f'ladder-gap-{filt}', 'warn',
                    f'{filt}: phase(s) {", ".join(holes)} missing below a phase '
                    f'that ran',
                    'A later merge exists without its input phase on disk. Either '
                    'the intermediate was cleaned up, or the later product is stale '
                    'and predates a rerun.',
                    source='catalogs/'))

    # One verdict for the whole run, not one per filter x phase: on a field like
    # gc2211 that would be ~30 identical rows and bury everything else.
    shared = [f for f in sorted(per_filter)
              if any((per_filter[f].get(p) or {}).get('scope') == 'ambiguous'
                     and (per_filter[f].get(p) or {}).get('n')
                     for p in ('m12', 'm3', 'm4', 'm5', 'm6', 'm7'))]
    if shared:
        out.append(_verdict(
            'ambiguous-catalogs', 'warn',
            f'{len(shared)} filter(s) have catalogs that cannot be attributed to '
            f'{run.get("proposal")}/o{run.get("obsid")}: {", ".join(shared)}',
            'These filters belong to more than one observation of this field, and '
            'the per-filter merged catalogs are written with no _o<obs> token, so '
            'the file on disk is whichever observation ran last. Counts for these '
            'filters describe the directory, not this observation.',
            value=len(shared), threshold=0, source='catalogs/'))
    return out


def check_headers(run):
    """WCS provenance read straight out of the frames' primary headers.

    Both checks come from the astrometry paper's WCS-provenance section and cost
    nothing but a header open -- no reprojection, no reference catalog.
    """
    from .scan import LW_FILTEROFFSET

    out = []
    headers = run.get('headers') or {}
    if not headers:
        return out

    bad = [(filt, m) for filt, rec in sorted(headers.items())
           for m in rec.get('filteroffset_mismatch') or []]
    if bad:
        listing = '; '.join(
            f'{filt} {m["module"]}: {m["r_filoff"]} (module {m["module"]} needs '
            f'…_{m["expected"]}.asdf)' for filt, m in bad[:6])
        out.append(_verdict(
            'filteroffset-module-mismatch', 'fail',
            f'{len(bad)} sampled LW frame(s) carry the OTHER module\'s '
            f'filteroffset reference',
            f'{listing}\n'
            f'The LW filteroffset reference is module-specific '
            f'(A→…_{LW_FILTEROFFSET["A"]}, B→…_{LW_FILTEROFFSET["B"]}). A frame '
            f'with the wrong one is displaced on sky by the difference between '
            f'the two filter offsets — up to ~26 mas per module for F410M, a '
            f'~52 mas A−B differential. The error is ANTI-SYMMETRIC between '
            f'modules, so mixing swapped and corrected frames manufactures an '
            f'apparent inter-module offset that no reference tie can diagnose.',
            value=len(bad), threshold=0,
            source='FITS primary header R_FILOFF vs MODULE',
            cause=(
                'The CRDS filteroffset rmap had the two LW modules swapped for a '
                'period; frames calibrated in that window carry the wrong '
                'reference. Because the pipeline reads whatever the glob picks up, '
                'a directory holding both generations can ship either. Re-run '
                'assign_wcs on these frames against a current CRDS context, and '
                'check that no stale generation is left where a glob can reach it.'),
            evidence={'rows': {
                'columns': ['frame', 'module', 'R_FILOFF', 'expected'],
                'data': [[m['file'], m['module'], m['r_filoff'],
                          f'…_{m["expected"]}.asdf'] for _f, m in bad[:40]],
                'total': len(bad)}}))

    contexts = set()
    versions = set()
    for rec in headers.values():
        contexts |= set(rec.get('crds_ctx') or {})
        versions |= set(rec.get('cal_ver') or {})
    if len(contexts) > 1:
        out.append(_verdict(
            'crds-context-mixed', 'warn',
            f'{len(contexts)} different CRDS contexts among the sampled frames: '
            f'{", ".join(sorted(contexts))}',
            'Frames calibrated against different CRDS contexts do not share one '
            'WCS solution — distortion and filteroffset references differ between '
            'them. A catalog built from the mixture carries whichever each frame '
            'got. (Sampled, so this is a lower bound on the spread.)',
            value=len(contexts), threshold=1, source='FITS header CRDS_CTX',
            cause=(
                'Frames from different reduction epochs are co-resident in the '
                'filter directory. Distortion and filteroffset references differ '
                'between contexts, so positions from the two generations do not '
                'share one solution — and an apparent inter-module or '
                'inter-visit offset can be entirely this. Re-reduce the older '
                'frames, or move them out of the globbed directory.'),
            evidence={'rows': {
                'columns': ['filter', 'CRDS_CTX', 'frames sampled'],
                'data': [[f, ctx, n]
                         for f, rec in sorted(headers.items())
                         for ctx, n in sorted((rec.get('crds_ctx') or {}).items())],
                'total': len(contexts)}}))
    if len(versions) > 1:
        out.append(_verdict(
            'calver-mixed', 'info',
            f'{len(versions)} different CAL_VER among the sampled frames: '
            f'{", ".join(sorted(versions))}',
            'The jwst package version alone does not change the WCS; it is '
            'reported because it usually correlates with a CRDS-context change.',
            value=len(versions), threshold=1, source='FITS header CAL_VER'))
    return out


def check_crossband(run):
    """Cross-band m7/m8 presence and observation attribution."""
    out = []
    cross = run.get('crossband') or {}
    if run.get('is_cutout'):
        out.append(_verdict(
            'crossband', 'skip', 'cutout run: cross-band merge does not apply',
            'm7 needs at least two filters and m8 the cross-band dedup; a '
            'one-filter cutout stops after m6, and the all-filter merge reads '
            '<basepath>/catalogs/, which a cutout never writes.'))
        return out
    for phase in ('m7', 'm8'):
        row = cross.get(phase) or {}
        if not row.get('n'):
            out.append(_verdict(
                f'crossband-{phase}', 'info', f'{phase} cross-band product absent',
                source='catalogs/basic_*_photometry_tables_merged_*.fits'))
        elif row.get('scope') == 'ambiguous' and run.get('multi_obs'):
            out.append(_verdict(
                f'crossband-{phase}-ambiguous', 'warn',
                f'{phase}: only an untagged cross-band product exists',
                'This field has several observations but the product carries no '
                '_o<obs> token, so it cannot be attributed to one of them.',
                source='catalogs/'))
        else:
            out.append(_verdict(
                f'crossband-{phase}', 'info',
                f'{phase} present ({row["n"]} file(s))',
                source='catalogs/'))
    return out


# --------------------------------------------------------------------------
# Run environment / jobs
# --------------------------------------------------------------------------

def check_overrides(environ=None):
    """Safety gates switched off in the environment the monitor is running in.

    This describes THIS shell, not the shell a past product was written in -- a
    product's own environment is not recorded in the sidecar (``inputs.env`` is
    empty in practice), so the monitor says what it can actually see rather than
    implying it audited the historical run.
    """
    environ = os.environ if environ is None else environ
    out = []
    for var in SAFETY_OVERRIDES:
        if var not in environ:
            continue
        value = environ[var]
        if var == 'FORCE_REALIGN_ON_DISAGREE':
            out.append(_verdict(f'override-{var}', 'info',
                                f'{var}={value} (stricter: hard-stops on a stale '
                                f'RAOFFSET rather than warning)', source='environ'))
        elif var == 'ASTROM_CHECKPOINT' and value not in ('0', 'false', 'False'):
            continue
        else:
            out.append(_verdict(
                f'override-{var}', 'warn',
                f'{var}={value} is set in this environment',
                'A run made with this set produces products the checkpoint ladder '
                'would otherwise have refused.', source='environ'))
    return out


def check_jobs(run, jobs_for_target, log_scans):
    """Live queue and recent log signatures for this field."""
    out = []
    active = [j for j in jobs_for_target if j.get('state') in ('RUNNING', 'PENDING')]
    if active:
        running = sum(1 for j in active if j['state'] == 'RUNNING')
        out.append(_verdict(
            'jobs-active', 'info',
            f'{len(active)} job(s) in the queue ({running} running)',
            ', '.join(sorted({j['name'] for j in active}))[:400], source='squeue'))

    # One array job fans out over 16 tasks and writes 16 near-identical logs; a
    # verdict per log would report the same failure sixteen times and push
    # everything else off the page.  Group by (job name, signature set) instead
    # and say how many tasks hit it.
    from . import jobs as _jobs
    groups = {}
    for scan in log_scans:
        if not scan or scan.get('worst') not in ('error', 'warn'):
            continue
        labels = tuple(sorted(k for k in scan['hits'] if k not in ('start', 'done')))
        if not labels:
            continue
        name = _jobs.log_job_name(scan['path']) or os.path.basename(scan['path'])
        key = (scan['worst'], name, labels)
        entry = groups.setdefault(key, {'n': 0, 'lines': [], 'paths': []})
        entry['n'] += 1
        entry['paths'].append(scan['path'])
        if not entry['lines']:
            entry['lines'] = [text for _, _, text in scan['lines'][:6]]

    for (severity, name, labels), entry in sorted(groups.items()):
        n_tasks = f' ({entry["n"]} tasks)' if entry['n'] > 1 else ''
        out.append(_verdict(
            'log-error' if severity == 'error' else 'log-warn',
            'fail' if severity == 'error' else 'warn',
            f'{"errors" if severity == "error" else "warnings"} in {name}'
            f'{n_tasks}: {", ".join(labels)}',
            '\n'.join(entry['lines']),
            source=os.path.basename(entry['paths'][0])))

    if log_scans and not groups and not active:
        out.append(_verdict('logs', 'info',
                            f'{len(log_scans)} recent log(s), no error signatures',
                            source='logs/'))
    return out


# --------------------------------------------------------------------------
# Roll-up
# --------------------------------------------------------------------------

#: A post-recat verdict older than this is reported as aged: the analysis runs on a
#: SLURM dependency after re-cataloging, so a verdict that has not been refreshed in
#: weeks is describing an older state of the tree.  Monitor-owned, not a paper gate.
PAPER_VERDICT_AGE_WARN_DAYS = 14


def check_paper(run, paper_summary):
    """Surface the astrometry paper's own validation verdict for its field.

    The paper's ``post_recat_validation.py`` already applied its gates (cross-filter
    vs anchor > 30 mas, p60/p90 mode flip > 10 mas, degenerate-pair drift >= 0.10
    mag) with the sanctioned offset estimator over full catalogs.  Those verdicts
    are REPORTED here, not recomputed -- a second copy of the thresholds would drift
    from the published numbers, and recomputing would mean the monitor crossmatching
    catalogs itself.

    What is added is whether the verdict still describes what is on disk.
    """
    from . import paper as _paper

    out = []
    if run.get('target') != _paper.PAPER_FIELD or run.get('is_cutout'):
        return out
    # No verdict at all is a SKIP, not a finding: "the paper has not validated
    # this" and "the paper validated it and something is missing" are different
    # statements, and only the second is about the run.
    if not paper_summary or not paper_summary.get('generated'):
        return [_verdict('paper', 'skip', 'no astrometry-paper verdict found',
                         f'Looked for outputs/*_postrecat/summary.json under '
                         f'{_paper.PAPER_DIR}.', source='astrometry_paper/')]

    src = os.path.basename(paper_summary.get('postrecat_dir') or '') + '/summary.json'
    generated = paper_summary.get('generated')
    program = str(run.get('proposal') or '')

    # The paper validates BOTH brick programmes in one pass and prefixes each
    # problem with "<program>/<band>". Scoping to this run's programme keeps a
    # 1182 failure off the 2221 card, which would otherwise show four failures
    # that are not about it.
    def _mine(text):
        return str(text).startswith(f'{program}/') if program else True

    for problem in paper_summary.get('problems') or []:
        if not _mine(problem):
            continue
        out.append(_verdict(
            'paper-problem', 'fail', f'paper validation: {problem}',
            'Reported by the paper\'s own post-recat validation '
            '(scripts/post_recat_validation.py), which applies its gates with the '
            'sanctioned window-swept offset histogram over the full vetted '
            'catalogs. Not recomputed here.',
            source=src))

    other = sum(1 for p in (paper_summary.get('problems') or []) if not _mine(p))
    if other:
        out.append(_verdict(
            'paper-problem-other', 'info',
            f'{other} further paper-validation problem(s) belong to the field\'s '
            f'other programme(s), not to {program}',
            source=src))

    freshness = [r for r in (paper_summary.get('freshness') or [])
                 if not program or str(r.get('program')) == program]
    rewritten = [f'{r["program"]}/{r["band"]}' for r in freshness
                 if r.get('rewritten_since_verdict')]
    if rewritten:
        out.append(_verdict(
            'paper-verdict-outdated', 'fail',
            f'{len(rewritten)} catalog(s) rewritten since the verdict was written: '
            f'{", ".join(rewritten)}',
            'The verdict certifies a product that no longer exists on disk. A stale '
            'pass is worse than no verdict — re-run post_recat_validation.py before '
            'trusting it.',
            value=len(rewritten), threshold=0, source=src))

    missing = [f'{r["program"]}/{r["band"]}' for r in freshness if not r.get('present')]
    if missing:
        out.append(_verdict(
            'paper-catalog-missing', 'fail',
            f'{len(missing)} validated catalog(s) no longer on disk: '
            f'{", ".join(missing)}',
            value=len(missing), threshold=0, source=src))

    stale = [f'{r["program"]}/{r["band"]}' for r in freshness
             if r.get('predates_min_catalog_date')]
    if stale:
        min_date = (paper_summary.get('config') or {}).get('min_catalog_date')
        out.append(_verdict(
            'paper-catalog-freshness', 'warn',
            f'{len(stale)} catalog(s) predate MIN_CATALOG_DATE={min_date}: '
            f'{", ".join(stale)}',
            'The paper\'s provenance.check_catalog_freshness RAISES on a catalog '
            'older than this, so the analysis would refuse to run on these. The '
            'guard exists for the broken-v001 1182 re-reduction; whether it should '
            'bind the other program is the analysis\'s call, not the monitor\'s.',
            value=len(stale), threshold=0, source='astrometry_paper/config.py'))

    age = _paper.age_days(generated)
    if age is not None and age > PAPER_VERDICT_AGE_WARN_DAYS:
        out.append(_verdict(
            'paper-verdict-age', 'warn',
            f'paper validation last ran {age:.0f} days ago ({generated[:10]})',
            'It runs on a SLURM dependency after re-cataloging, so a long gap means '
            'it has not seen the recent runs.',
            value=round(age, 1), threshold=PAPER_VERDICT_AGE_WARN_DAYS, source=src))

    # An ABSENT certifier is unknown, not passing.  The paper's validation only
    # records degenerate-pair flatness and saturation continuity when a merged
    # table with the right columns was found; the 2026-07-19 brick run has none.
    # Reading a missing key as a pass is how a release gate gets skipped.
    cert = paper_summary.get('certifiers') or {}
    if not [k for k in cert if k != 'table']:
        out.append(_verdict(
            'paper-certifiers-absent', 'warn',
            'photometric certifiers were not computed in the latest paper run',
            'Saturation continuity and degenerate-pair flatness are therefore '
            'UNKNOWN, not passing. The release gate refuses at >= 0.10 mag '
            '(stage_release.CONTINUITY_TOL_MAG); the paper additionally states a '
            '< 0.05 mag goal for degenerate-pair flatness — the two numbers differ '
            'and are reported as they stand.',
            source=src))

    prov = paper_summary.get('provenance') or {}
    dirty = [name for name, rec in (prov.get('git') or {}).items()
             if isinstance(rec, dict) and rec.get('dirty')]
    if dirty:
        out.append(_verdict(
            'paper-provenance-dirty', 'warn',
            f'paper analysis ran against dirty checkout(s): {", ".join(dirty)}',
            'The recorded git SHA does not identify the code that produced these '
            'numbers.', value=len(dirty), threshold=0,
            source=os.path.basename(paper_summary.get('postrecat_dir') or '')
                   + '/provenance.json'))

    if not out:
        out.append(_verdict(
            'paper', 'info',
            f'paper validation clean ({generated[:10] if generated else "?"})',
            source=src))
    return out


def run_checks(run, jobs_for_target=(), log_scans=(), paper_summary=None):
    """Every verdict for one observation, worst first."""
    if run.get('globbed') is False:
        # Everything else describes products this observation is not supposed to
        # have; running those checks would manufacture findings.
        return check_products(run)
    verdicts = (check_astrometry(run) + check_provenance(run)
                + check_products(run) + check_crossband(run)
                + check_headers(run)
                + check_paper(run, paper_summary)
                + check_jobs(run, list(jobs_for_target), list(log_scans)))
    order = {s: i for i, s in enumerate(SEVERITIES)}
    verdicts.sort(key=lambda v: (order.get(v['severity'], 9), v['name']))
    return verdicts


def worst_severity(verdicts):
    """The most serious severity present, or ``'skip'`` if there is nothing."""
    for severity in SEVERITIES:
        if any(v['severity'] == severity for v in verdicts):
            return severity
    return 'skip'


def tally(verdicts):
    """``{severity: count}``."""
    return {s: sum(1 for v in verdicts if v['severity'] == s) for s in SEVERITIES}
