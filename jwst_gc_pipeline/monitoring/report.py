"""Glue: scan + queue + checks -> the entries the renderer draws, and the files.

``build_entries`` is the one place that decides what a monitored "run" is: one
registered observation of one field, optionally inside a cutout subtree.  Every
other module takes that shape as given.
"""
import os
import shutil
import time

from . import schedule as _schedule
from . import (checks, figures as _figures, jobs as _jobs, paper as _paper,
               render, scan, skyview)

#: Where the per-field pages and the JSON snapshot are written by default.
DEFAULT_OUTDIR = os.environ.get('GC_MONITOR_OUTDIR',
                                '/orange/adamginsburg/jwst/monitor')


def _newest_mtime(run):
    """Most recent product mtime anywhere in the run, for the "touched" stamp."""
    best = None
    for rows in (run.get('per_filter') or {}).values():
        for row in rows.values():
            mtime = row.get('mtime') if isinstance(row, dict) else None
            if mtime and (best is None or mtime > best):
                best = mtime
    for row in (run.get('crossband') or {}).values():
        if row.get('mtime') and (best is None or row['mtime'] > best):
            best = row['mtime']
    for rec in (run.get('astrometry') or {}).values():
        if rec.get('mtime') and (best is None or rec['mtime'] > best):
            best = rec['mtime']
    return best


def build_entries(targets=None, instrument='nircam', cutout_label=None,
                  with_jobs=True, with_logs=True, log_dir=None):
    """``([entry, ...], unattributed_jobs)``.

    An entry is ``{run, verdicts, tally, worst, jobs, anchor, newest_mtime}``.
    Fields whose scan raises are still returned, carrying a single ``fail``
    verdict -- a field that cannot be scanned is a finding, not a gap in the page.
    """
    targets = list(targets or scan.all_targets())
    queue = _jobs.squeue_jobs() if with_jobs else []
    by_target = _jobs.jobs_by_target(queue)
    unattributed = [j for j in queue if not j.get('target')]
    # Read once: the paper's verdicts are field-wide, not per-observation.
    paper_summary = (_paper.summarize(_paper.read_verdicts())
                     if _paper.PAPER_FIELD in targets else None)

    entries = []
    for target in targets:
        try:
            runs = scan.scan_field(target, instrument,
                                   cutout_label=cutout_label)['runs']
        except scan.ScanError as ex:
            entries.append({
                'run': {'target': target, 'proposal': '?', 'obsid': '?',
                        'basepath': '', 'per_filter': {}, 'crossband': {},
                        'astrometry': {}, 'provenance': {},
                        'cutout_label': cutout_label,
                        'is_cutout': bool(cutout_label)},
                'verdicts': [{'name': 'scan', 'severity': 'fail',
                              'summary': f'cannot scan {target}',
                              'detail': str(ex), 'value': None,
                              'threshold': None, 'source': 'fields.yaml'}],
                'tally': {'fail': 1, 'warn': 0, 'info': 0, 'skip': 0},
                'worst': 'fail', 'jobs': [],
                'anchor': f'f-{target}', 'newest_mtime': None})
            continue

        target_jobs = by_target.get(target, [])

        for run in runs:
            log_scans = []
            if with_logs:
                # Scoped to the observation: a gc2211 o050 crash must not be
                # reported against o023, which shares only the field name.
                for path in _jobs.logs_for_target(
                        target, log_dir=log_dir, limit=8,
                        obsid=run['obsid'], proposal=run['proposal']):
                    got = _jobs.scan_log(path)
                    if got:
                        log_scans.append(got)
            # A job whose name carries an observation is attributed to it; one
            # that does not (the older 'brick-catalog' shape) is shown on every
            # observation of the field, marked by its name_kind, rather than
            # pinned to an observation it may not belong to.
            # Same pin as the logs: obsid alone puts ngc6334's 6778 jobs on the
            # 7213 card, since both observations are o001.
            mine = [j for j in target_jobs
                    if j.get('obsid') in (None, run['obsid'])
                    and j.get('proposal') in (None, run['proposal'])]
            verdicts = checks.run_checks(run, mine, log_scans, paper_summary)
            # Offer the diagnostics that already exist for this field, preferring
            # ones whose filename names the finding's filter.
            found = _figures.find_figures(run['basepath'],
                                          list(run.get('per_filter') or {}))
            if target == _paper.PAPER_FIELD:
                found += _figures.paper_figures(_paper.PAPER_DIR)
            # The per-field diagnostic writeup.  The served copy carries a
            # `diagnostics-<field>` symlink to it, so linking by that name
            # resolves with no extra publishing step.
            wu = _figures.writeup(run['basepath'], f'diagnostics-{target}')
            for v in verdicts:
                if v['severity'] not in ('fail', 'warn'):
                    continue
                v.setdefault('evidence', {})
                if found:
                    want = v['evidence'].get('filter')
                    picks = [f for f in found if want and f['filter'] == want][:6]
                    v['evidence']['figures'] = picks or found[:4]
                if wu:
                    code = _figures.figure_for_finding(v['name'])
                    v['evidence']['writeup'] = {
                        'main': wu['main'],
                        'figure': wu['figures'].get(code) if code else None}
            anchor = f"f-{run['target']}-{run['proposal']}-o{run['obsid']}"
            if cutout_label:
                anchor += f'-{cutout_label}'
            entries.append({'run': run, 'verdicts': verdicts,
                            'tally': checks.tally(verdicts),
                            'worst': checks.worst_severity(verdicts),
                            'jobs': mine, 'anchor': anchor,
                            'newest_mtime': _newest_mtime(run),
                            'paper': (paper_summary
                                      if target == _paper.PAPER_FIELD
                                      and not cutout_label else None)})

    order = {'fail': 0, 'warn': 1, 'info': 2, 'skip': 3}
    entries.sort(key=lambda e: (order.get(e['worst'], 9), e['run']['target']))
    return entries, unattributed


def collect_cutouts(targets=None):
    """Every cutout run on disk, as ``cutout_summary`` dicts."""
    out = []
    for target in (targets or scan.all_targets()):
        for label in scan.cutout_labels(target):
            out.append(scan.cutout_summary(target, label))
    return out


def write_report(outdir=DEFAULT_OUTDIR, targets=None, instrument='nircam',
                 cutout_label=None, show_skip=False, per_field=True,
                 with_cutouts=True, log_dir=None, schedule_program=None,
                 schedule_offline=False):
    """Build everything and write it.

    Returns ``{'aggregate': path, 'fragment': path, 'fields': {target: path},
    'entries': [...]}``.  The fragment is the same page without the HTML
    document wrapper, for publishing as an artifact.
    """
    entries, unattributed = build_entries(targets, instrument, cutout_label,
                                          log_dir=log_dir)
    # Footprint geometry for the sky view.  Absent is fine -- the section says
    # how to generate it rather than vanishing.
    footprints = skyview.load_footprints(
        os.path.join(outdir, skyview.FOOTPRINTS_JSON))
    roman = skyview.load_footprints(os.path.join(outdir, 'roman_gbtds.json'))
    rgps = skyview.load_footprints(os.path.join(outdir, skyview.RGPS_JSON))
    # The published observing schedule.  Fetched here rather than in the
    # renderer so the JSON is written next to the page and the network is
    # touched once per run, not once per page.  `load` never raises on a
    # network problem -- it falls back to its cache and says so.
    schedule = None
    if schedule_program:
        schedule = _schedule.load(outdir, program=schedule_program,
                                  offline=schedule_offline)
        _schedule.write_json(outdir, schedule)
    cutouts = collect_cutouts(targets) if with_cutouts else []
    generated = time.time()

    scope = 'all fields' if not targets else ', '.join(targets)
    subtitle = (f'{scope} · {time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(generated))}'
                + (f' · cutout {cutout_label}' if cutout_label else ''))

    stem = f'monitor{"_" + cutout_label if cutout_label else ""}'
    aggregate = os.path.join(outdir, f'{stem}.html')
    fragment = os.path.join(outdir, f'{stem}_fragment.html')

    suffix = f'_{cutout_label}' if cutout_label else ''

    def detail_href(entry):
        """Where a card on the front page points.

        Per *field*, not per observation, because that is the page that exists;
        the anchor picks out the observation within it. ``per_field=False``
        turns this off, and then the cards fall back to same-page anchors --
        which is why the front page keeps its detail in that mode.
        """
        return (f'fields/{entry["run"]["target"]}{suffix}.html'
                f'#{entry["anchor"]}')

    front = dict(entries=entries, cutouts=cutouts, subtitle=subtitle,
                 show_skip=show_skip, generated=generated,
                 unattributed_jobs=unattributed, footprints=footprints,
                 roman=roman, rgps=rgps, schedule=schedule,
                 # The front page is the overview: map, then status cards, then
                 # a link out per field. Inlining 18 fields' tables and evidence
                 # below the cards made the monitor's entry point its largest
                 # and slowest document.
                 include_detail=not per_field,
                 detail_href=detail_href if per_field else None)

    render.write_html(aggregate, render.render_page(standalone=True, **front))

    # The fragment is published as a single-file artifact, so it cannot split:
    # `fields/<target>.html` is not a document that exists there, and a card
    # linking to one would 404. It keeps the detail inline and the same-page
    # anchors that go with it -- bigger, but whole.
    render.write_html(fragment, render.render_page(
        standalone=False,
        **dict(front, include_detail=True, detail_href=None)))

    # Link every figure the pages reference into <outdir>/figures/ under its
    # basename, so the relative hrefs resolve wherever the page is served from.
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    seen = set()
    for entry in entries:
        for v in entry['verdicts']:
            for fig in (v.get('evidence') or {}).get('figures') or []:
                if fig['name'] in seen:
                    continue
                seen.add(fig['name'])
                _link(fig['path'], os.path.join(figdir, fig['name']))

    field_paths = {}
    if per_field:
        by_field = {}
        for entry in entries:
            by_field.setdefault(entry['run']['target'], []).append(entry)
        for target, group in by_field.items():
            path = os.path.join(outdir, 'fields', f'{target}{suffix}.html')
            render.write_html(path, render.render_page(
                group, [c for c in cutouts if c['target'] == target],
                title=f'{target} · jwst-gc pipeline monitor',
                subtitle=f'{target} · '
                         f'{time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(generated))}',
                standalone=True, show_skip=show_skip, generated=generated,
                # ../ because the per-field pages live in fields/ while the
                # assets sit beside the aggregate page.  Without these the
                # per-field pages told the reader to generate a file that was
                # already sitting next to them.
                footprints=footprints, roman=roman, rgps=rgps,
                include_schedule=False,
                asset_prefix='../',
                # The map is ~100 kB of identical inline geometry; carrying it
                # on all 18 field pages costs more than it tells anyone reading
                # about one field.
                include_skyview=False,
                home_href=f'../monitor{suffix}.html#overview'))
            field_paths[target] = path

    return {'aggregate': aggregate, 'fragment': fragment, 'fields': field_paths,
            'entries': entries, 'cutouts': cutouts, 'schedule': schedule,
            'unattributed_jobs': unattributed, 'generated': generated}


#: Files worth serving from a web directory.  The ``*_fragment.html`` outputs are
#: body fragments for an artifact publisher -- no doctype, charset or viewport --
#: so they are deliberately NOT published.
_PUBLISH_GLOBS = ('monitor*.html', 'monitor*.json')


def _copy_is_current(src_stat, dst_stat, dst_dev):
    """Is ``dst`` already the copy ``_link`` would make?

    Only ever true across filesystems. On the same one a hardlink is nearly
    free, and relinking is the whole point of re-running ``publish`` -- skipping
    it there would reintroduce the stale-page bug that function exists to
    prevent, because a rewritten page can keep both its size and its second.
    """
    return (src_stat.st_dev != dst_dev
            and src_stat.st_size == dst_stat.st_size
            and src_stat.st_mtime == dst_stat.st_mtime)   # copy2 preserves it


def _link(src, dst):
    """Hardlink ``src`` to ``dst``; **copy** across filesystems.

    A hardlink is preferred because the renderer rewrites a page in place
    (``open(path, 'w')``), which keeps the inode -- so the published copy tracks
    every regeneration with no second copy and no re-publish step.

    The cross-filesystem fallback is a copy rather than a symlink because a
    published directory is served by a web server, and a symlink whose target
    leaves the served tree is not a file the server will hand out: Apache at
    data.rc.ufl.edu returns **403** for every one of these. That failure is
    silent in exactly the wrong way -- the pages are fine and only the figures
    they link to are missing -- so the figure links looked correct and 403'd.
    Roughly half the figures come from ``/blue`` while the output lives on
    ``/orange``, so this is the normal case, not the exotic one.
    """
    # Publishing into the output directory itself would remove the page and then
    # link it to itself -- a one-character scrontab typo losing the report.
    #
    # `and not islink`: an existing SYMLINK to src also has src's realpath, so
    # the bare comparison reported 'same' and left it alone -- which meant the
    # switch from symlinking to copying never converted the symlinks already on
    # disk, and they kept on 403-ing.
    if (os.path.realpath(src) == os.path.realpath(dst)
            and not os.path.islink(dst)):
        return 'same'
    # An unchanged copy is left alone: re-copying ~110 MB of figures on an
    # hourly refresh would be most of the job's cost, all of it pointless.
    #
    # ONLY when a hardlink is impossible. On the same filesystem, relinking is
    # nearly free and is the whole point of re-running publish -- skipping it
    # there would reintroduce the stale-page bug this function exists to avoid,
    # since a rewritten page can keep its size and its second.
    if not os.path.islink(dst) and os.path.isfile(dst):
        try:
            if _copy_is_current(
                    os.stat(src), os.stat(dst),
                    os.stat(os.path.dirname(os.path.abspath(dst))).st_dev):
                return 'copy'
        except OSError:
            pass
    try:
        if os.path.lexists(dst):
            os.remove(dst)
    except OSError:
        return None
    try:
        os.link(src, dst)
        return 'hard'
    except OSError:
        pass
    try:
        shutil.copy2(src, dst)
        return 'copy'
    except (OSError, shutil.SameFileError):
        return None


def publish(outdir, publish_dir, index_from='monitor.html'):
    """Link the generated pages into a web directory.

    Re-run after every generation.  That is not redundant: it costs nothing when
    the inode is unchanged, and it is what keeps the published copy correct if
    the renderer ever moves to an atomic write (write-temp + rename), which
    replaces the inode and would otherwise leave the web copy frozen at whatever
    it linked to first -- a stale dashboard that still looks live.

    Returns ``{path: 'hard'|'copy'|'same'|None}``.
    """
    import fnmatch

    os.makedirs(os.path.join(publish_dir, 'fields'), exist_ok=True)
    done = {}
    for name in sorted(os.listdir(outdir)):
        if '_fragment' in name:
            continue
        if not any(fnmatch.fnmatch(name, pat) for pat in _PUBLISH_GLOBS):
            continue
        done[name] = _link(os.path.join(outdir, name),
                           os.path.join(publish_dir, name))

    fields = os.path.join(outdir, 'fields')
    if os.path.isdir(fields):
        for name in sorted(os.listdir(fields)):
            if name.endswith('.html'):
                done[f'fields/{name}'] = _link(
                    os.path.join(fields, name),
                    os.path.join(publish_dir, 'fields', name))

    # The figures the pages link to, hardlinked into figures/ so the relative
    # hrefs resolve on the served copy.  Written by write_report into
    # <outdir>/figures/ so publishing stays a pure link step.
    figdir = os.path.join(outdir, 'figures')
    if os.path.isdir(figdir):
        os.makedirs(os.path.join(publish_dir, 'figures'), exist_ok=True)
        for name in sorted(os.listdir(figdir)):
            done[f'figures/{name}'] = _link(
                os.path.join(figdir, name),
                os.path.join(publish_dir, 'figures', name))

    # The diagnostic-writeup symlinks the pages link into.  These previously
    # existed only in the live web directory, made out of band, so a fresh
    # --publish-dir had ~395 dead `diagnostics-<field>/...` links while the docs
    # said they resolved with no extra step.  Creating them here makes that true.
    for target in scan.all_targets():
        try:
            wu_dir = os.path.join(scan.basepath(target), _figures.WRITEUP_DIR)
        except scan.ScanError:
            continue
        if not os.path.isdir(wu_dir):
            continue
        link = os.path.join(publish_dir, f'diagnostics-{target}')
        if os.path.islink(link) or os.path.exists(link):
            try:
                if os.path.realpath(link) == os.path.realpath(wu_dir):
                    done[f'diagnostics-{target}'] = 'same'
                    continue
                os.remove(link)
            except OSError:
                continue
        try:
            # A directory cannot be hardlinked, so this one is always a symlink.
            os.symlink(os.path.abspath(wu_dir), link)
            done[f'diagnostics-{target}'] = 'sym'
        except OSError:
            done[f'diagnostics-{target}'] = None

    # Sky-view assets: the footprint data and a same-origin copy of Aladin Lite,
    # so the page depends on no third-party CDN.
    for name in (skyview.FOOTPRINTS_JSON, 'roman_gbtds.json',
                 skyview.RGPS_JSON, _schedule.SCHEDULE_JSON):
        src = os.path.join(outdir, name)
        if os.path.exists(src):
            done[name] = _link(src, os.path.join(publish_dir, name))
    if os.path.exists(skyview.ALADIN_SOURCE):
        done[skyview.ALADIN_LOCAL] = _link(
            skyview.ALADIN_SOURCE, os.path.join(publish_dir, skyview.ALADIN_LOCAL))

    src_index = os.path.join(outdir, index_from)
    if os.path.exists(src_index):
        done['index.html'] = _link(src_index, os.path.join(publish_dir, 'index.html'))
    return done


def summarize(entries):
    """One text line per entry, for the CLI and for a quick eyeball in a log."""
    lines = []
    for entry in entries:
        run = entry['run']
        tally = entry['tally']
        lines.append(
            f"{entry['worst']:>4}  {run['target']:<11s} "
            f"{run['proposal']}/o{run['obsid']:<4s} "
            f"fail={tally['fail']:<2d} warn={tally['warn']:<2d} "
            f"filters={len(run.get('per_filter') or {}):<2d} "
            f"jobs={len(entry.get('jobs') or [])}")
    return '\n'.join(lines)
