"""Glue: scan + queue + checks -> the entries the renderer draws, and the files.

``build_entries`` is the one place that decides what a monitored "run" is: one
registered observation of one field, optionally inside a cutout subtree.  Every
other module takes that shape as given.
"""
import os
import time

from . import checks, figures as _figures, jobs as _jobs, paper as _paper, render, scan

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
                 with_cutouts=True, log_dir=None):
    """Build everything and write it.

    Returns ``{'aggregate': path, 'fragment': path, 'fields': {target: path},
    'entries': [...]}``.  The fragment is the same page without the HTML
    document wrapper, for publishing as an artifact.
    """
    entries, unattributed = build_entries(targets, instrument, cutout_label,
                                          log_dir=log_dir)
    cutouts = collect_cutouts(targets) if with_cutouts else []
    generated = time.time()

    scope = 'all fields' if not targets else ', '.join(targets)
    subtitle = (f'{scope} · {time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(generated))}'
                + (f' · cutout {cutout_label}' if cutout_label else ''))

    stem = f'monitor{"_" + cutout_label if cutout_label else ""}'
    aggregate = os.path.join(outdir, f'{stem}.html')
    fragment = os.path.join(outdir, f'{stem}_fragment.html')

    render.write_html(aggregate, render.render_page(
        entries, cutouts, subtitle=subtitle, standalone=True,
        show_skip=show_skip, generated=generated,
        unattributed_jobs=unattributed))
    render.write_html(fragment, render.render_page(
        entries, cutouts, subtitle=subtitle, standalone=False,
        show_skip=show_skip, generated=generated,
        unattributed_jobs=unattributed))

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
            path = os.path.join(outdir, 'fields', f'{target}{"_" + cutout_label if cutout_label else ""}.html')
            render.write_html(path, render.render_page(
                group, [c for c in cutouts if c['target'] == target],
                title=f'{target} · jwst-gc pipeline monitor',
                subtitle=f'{target} · '
                         f'{time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(generated))}',
                standalone=True, show_skip=show_skip, generated=generated))
            field_paths[target] = path

    return {'aggregate': aggregate, 'fragment': fragment, 'fields': field_paths,
            'entries': entries, 'cutouts': cutouts,
            'unattributed_jobs': unattributed, 'generated': generated}


#: Files worth serving from a web directory.  The ``*_fragment.html`` outputs are
#: body fragments for an artifact publisher -- no doctype, charset or viewport --
#: so they are deliberately NOT published.
_PUBLISH_GLOBS = ('monitor*.html', 'monitor*.json')


def _link(src, dst):
    """Hardlink ``src`` to ``dst``; fall back to a symlink across filesystems.

    A hardlink is preferred because the renderer rewrites a page in place
    (``open(path, 'w')``), which keeps the inode -- so the published copy tracks
    every regeneration with no second copy and no re-publish step.
    """
    # Publishing into the output directory itself would remove the page and then
    # symlink it to itself -- a one-character scrontab typo losing the report.
    if os.path.realpath(src) == os.path.realpath(dst):
        return 'same'
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
        os.symlink(os.path.abspath(src), dst)
        return 'sym'
    except OSError:
        return None


def publish(outdir, publish_dir, index_from='monitor.html'):
    """Link the generated pages into a web directory.

    Re-run after every generation.  That is not redundant: it costs nothing when
    the inode is unchanged, and it is what keeps the published copy correct if
    the renderer ever moves to an atomic write (write-temp + rename), which
    replaces the inode and would otherwise leave the web copy frozen at whatever
    it linked to first -- a stale dashboard that still looks live.

    Returns ``{path: 'hard'|'sym'|None}``.
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
