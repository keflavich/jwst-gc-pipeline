"""One cataloging phase's satstar catalog per exposure.

Every cataloging phase (m12, m3, m4, m5, m6, m7) re-runs the saturated-star
fit on every exposure and writes its own per-exposure catalog,
``..._crf[_resbgsub]_m<N>_satstar_catalog.fits`` (plus
``..._satstar_rejected.fits`` when the gates rejected anything).  Nothing
removes an earlier phase's files, so a tree that has been re-cataloged holds
one catalog per phase per exposure, from whichever code version last wrote
each phase.

``merge_catalogs.load_satstar_catalog`` used to read all of them, and its
brightest-first dedup then kept the brightest fit among an exposure's re-fits.
Two consequences:

* A re-catalog with fixed satstar code still shipped the OLD code's flux
  wherever an old phase's fit happened to be brighter.  Measured on
  gc-treasury o111 F480M (an m7 re-run with the #972 fixes, the older m12..m6
  catalogs left in place): 387 of 1120 consolidated rows kept an m12 fit.
  Against a run with no old-phase files, flux_fit was +2.5% median and +7.4%
  p90, >2% off on 222 rows and >10% on 18.  Where the old fit won, old/F_psf
  was 1.113 against 1.053 for the new fit, and the new fit was closer to
  F_psf in 142 of 143 cases.
* Brightest-of-N over re-fits of the same pixels biases the flux high even
  when every phase ran the same code.

Within one run the phases fit identical inputs and agree to 0.000e+00
(``satstar_cache``), so reading one phase loses nothing.  The rule here: each
exposure contributes ONE phase's catalog -- the phase being merged when the
caller names it, else the latest phase the exposure has.

Phase order is the pipeline's, m12 < m3 < m4 < m5 < m6 < m7 < m8.  ``m12`` is
the FIRST phase (it fits m1 and m2 and names its satstar catalogs ``_m12``;
see ``satstar_consensus.satstar_stage_label``), so it ranks below ``m3`` even
though 12 > 3.  The ``_resbgsub``/``_bgsub`` variants of a phase share its
rank.

``SATSTAR_POOL_PHASES=all`` restores the pooled read (every phase of every
exposure), for bisecting only.
"""
import hashlib
import os
import re
from collections import Counter

#: Tokens written by the first cataloging phase.  All three are rank 2, so a
#: caller merging ``'m1'`` or ``'m2'`` reads the ``_m12`` satstar catalogs.
_FIRST_PHASE_TOKENS = ('m1', 'm2', 'm12')
_FIRST_PHASE_RANK = 2

#: The phase token at the end of a caller's iteration label: ``'m7'``,
#: ``'_m7'``, ``'resbgsub_m7'``.
_PHASE_LABEL_RE = re.compile(r'(?:^|_)(m\d+)$')

#: The phase token of a per-exposure satstar catalog or rejected file.
_PHASE_FILE_RE = re.compile(r'_(m\d+)_satstar_(?:catalog|rejected)\.fits$')

#: Everything after the frame stem of a per-exposure satstar product: the
#: background-subtraction token(s) and the phase token.
_PHASE_TAIL_RE = re.compile(
    r'(?:_bgsub)?(?:_resbgsub)?_m\d+_satstar_(?:catalog|rejected)\.fits$')

#: Observation + visit group + exposure + detector: one image.  The same rule
#: as ``merge_catalogs.satstar_exposure_key`` (a test pins the two together).
_JW_EXPOSURE_RE = re.compile(r'^(jw\d+_\d+_\d+_[A-Za-z0-9]+)')

#: Environment switch.  ``all`` pools every phase (the pre-fix behaviour).
POOL_PHASES_ENV = 'SATSTAR_POOL_PHASES'


def satstar_phase_rank(label):
    """Pipeline order of a satstar phase label, or ``None``.

    Parameters
    ----------
    label : str or None
        A file token (``'m12'``, ``'m3'``) or a caller's iteration label
        (``'m7'``, ``'_m7'``, ``'resbgsub_m7'``, ``'m2'``).

    Returns
    -------
    int or None
        2 for ``m1``/``m2``/``m12`` (the first phase), N for any other
        ``m<N>``, ``None`` when the label names no phase (``None``, ``''``,
        ``'iter2'``).
    """
    if label in (None, ''):
        return None
    match = _PHASE_LABEL_RE.search(str(label).strip())
    if match is None:
        return None
    token = match.group(1)
    if token in _FIRST_PHASE_TOKENS:
        return _FIRST_PHASE_RANK
    return int(token[1:])


def satstar_file_phase_token(path):
    """``'m12'``/``'m3'``/... of a per-exposure satstar product, else ``''``."""
    match = _PHASE_FILE_RE.search(os.path.basename(str(path)))
    return match.group(1) if match is not None else ''


def satstar_phase_group_key(path):
    """The exposure a per-exposure satstar product belongs to, phase-free.

    For an observatory name this is the image (observation, visit group,
    exposure, detector), so an exposure fit on two frame variants
    (``_align_`` and ``_destreak_`` both exist in several trees) is ONE
    exposure, as the ensemble statistics already count it.  Any other name is
    grouped by its frame stem: the name with the background and phase tokens
    removed.
    """
    base = os.path.basename(str(path))
    match = _JW_EXPOSURE_RE.match(base)
    if match is not None:
        return match.group(1)
    return _PHASE_TAIL_RE.sub('', base)


def pool_all_phases():
    """True when ``SATSTAR_POOL_PHASES=all`` asks for the pre-fix pooled read."""
    return os.environ.get(POOL_PHASES_ENV, '').strip().lower() == 'all'


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return float('-inf')


def _phase_sort_key(token):
    rank = satstar_phase_rank(token)
    return (rank if rank is not None else -1, token)


def select_satstar_phase_files(paths, phase=None):
    """Keep one cataloging phase's per-exposure satstar file per exposure.

    Parameters
    ----------
    paths : sequence of str
        Per-exposure ``*_m<N>_satstar_catalog.fits`` (or
        ``*_m<N>_satstar_rejected.fits``) paths, already observation-scoped
        and filtered to those carrying a phase token.
    phase : str, optional
        The phase being merged, as the merge's iteration label (``'m7'``,
        ``'m2'`` for the first phase, ``'resbgsub_m6'`` ...).  Each exposure
        then uses that phase's file, or its latest EARLIER phase when it has
        none.  A later phase's file is never used: in a phase-P merge it can
        only be left over from a previous run.  ``None``, or a label that names
        no phase (``'iter2'``), selects the latest phase each exposure has.

    Returns
    -------
    selected : list of str
        Sorted.
    report : dict
        ``mode`` (``'one'`` or ``'all'``), ``phase`` (the label as given),
        ``rank`` (its rank or ``None``), and ``used`` / ``superseded`` /
        ``ahead`` -- `collections.Counter` of phase token -> file count.
        ``superseded`` is an exposure's earlier phases (and same-phase
        duplicates: another frame variant or background token); ``ahead`` is
        a phase later than the one being merged.

    Notes
    -----
    Ties within one rank (``_m5`` beside ``_resbgsub_m5``, or one exposure
    fit on two frame variants) go to the most recently written file, then to
    the name, so the choice is deterministic.
    """
    paths = sorted(str(p) for p in paths)
    rank_req = satstar_phase_rank(phase)
    report = {'mode': 'one', 'phase': phase, 'rank': rank_req,
              'used': Counter(), 'superseded': Counter(), 'ahead': Counter(),
              'tie_losers': []}
    if pool_all_phases():
        report['mode'] = 'all'
        report['used'].update(satstar_file_phase_token(p) for p in paths)
        return paths, report
    groups = {}
    for path in paths:
        groups.setdefault(satstar_phase_group_key(path), []).append(path)
    selected = []
    for members in groups.values():
        eligible = []
        for path in members:
            token = satstar_file_phase_token(path)
            rank = satstar_phase_rank(token)
            if rank is None or (rank_req is not None and rank > rank_req):
                report['ahead'][token] += 1
            else:
                eligible.append((rank, _mtime(path), path, token))
        if not eligible:
            continue
        best = max(eligible)
        selected.append(best[2])
        report['used'][best[3]] += 1
        for item in eligible:
            if item is not best:
                report['superseded'][item[3]] += 1
                if item[0] == best[0]:
                    # Same phase, another frame variant or background token:
                    # the newer file won.  Name the loser so a surprising
                    # choice can be traced from the log.
                    report['tie_losers'].append(
                        (os.path.basename(best[2]), os.path.basename(item[2])))
    return sorted(selected), report


def select_rejected_for_catalogs(rejected_paths, chosen_catalogs, phase=None):
    """Gate-rejected files that belong to the chosen per-exposure catalogs.

    A rejected file is written only when the gates rejected something, so the
    latest rejected file of an exposure can be an OLD phase's while its chosen
    accepted catalog is a newer phase with no rejections.  An exposure whose
    accepted catalog was chosen therefore keeps only the rejected file of that
    same run (same frame, background token and phase), or none.  An exposure
    with no accepted catalog at all falls back to
    :func:`select_satstar_phase_files` over its own rejected files.

    Parameters
    ----------
    rejected_paths : sequence of str
        ``*_m<N>_satstar_rejected.fits`` paths.
    chosen_catalogs : sequence of str
        The accepted catalogs :func:`select_satstar_phase_files` chose.
    phase : str, optional
        As for :func:`select_satstar_phase_files`.

    Returns
    -------
    selected : list of str
    report : dict
        As for :func:`select_satstar_phase_files`.
    """
    rejected_paths = sorted(str(p) for p in rejected_paths)
    if pool_all_phases():
        return select_satstar_phase_files(rejected_paths, phase=phase)
    siblings = {os.path.basename(str(c)).replace('_satstar_catalog.fits',
                                                 '_satstar_rejected.fits')
                for c in chosen_catalogs}
    covered = {satstar_phase_group_key(c) for c in chosen_catalogs}
    rank_req = satstar_phase_rank(phase)
    selected, orphans = [], []
    report = {'mode': 'one', 'phase': phase, 'rank': rank_req,
              'used': Counter(), 'superseded': Counter(), 'ahead': Counter(),
              'tie_losers': []}
    for path in rejected_paths:
        if satstar_phase_group_key(path) not in covered:
            orphans.append(path)
            continue
        token = satstar_file_phase_token(path)
        rank = satstar_phase_rank(token)
        if os.path.basename(path) in siblings:
            selected.append(path)
            report['used'][token] += 1
        elif rank is None or (rank_req is not None and rank > rank_req):
            report['ahead'][token] += 1
        else:
            report['superseded'][token] += 1
    if orphans:
        extra, sub = select_satstar_phase_files(orphans, phase=phase)
        selected.extend(extra)
        for key in ('used', 'superseded', 'ahead'):
            report[key].update(sub[key])
        report['tie_losers'].extend(sub['tie_losers'])
    return sorted(selected), report


def satstar_phase_selection_signature(selected):
    """Digest of the per-exposure files a consolidated catalog is built from.

    Stored as the consolidated cache's ``SATPHSEL`` so a cache built from a
    different set -- in particular one built before this selection existed,
    from every phase pooled -- is rebuilt rather than served.  Names only:
    the cache's mtime and count checks already cover rewritten files.

    Returns a 16-character hex string; ``''`` for an empty input.
    """
    names = sorted(os.path.basename(str(p)) for p in selected)
    if not names:
        return ''
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()[:16]


def format_phase_report(report, what='satstar catalog'):
    """One log line: files used per phase, and how many were left out."""
    def _fmt(counter):
        if not counter:
            return 'none'
        return ', '.join(f'{tok or "?"}: {counter[tok]}'
                         for tok in sorted(counter, key=_phase_sort_key))
    n_used = sum(report['used'].values())
    if report['mode'] == 'all':
        return (f"{POOL_PHASES_ENV}=all: pooling every phase, {n_used} "
                f"{what} file(s) [{_fmt(report['used'])}]")
    if report['rank'] is not None:
        head = f"one phase per exposure (merging {report['phase']})"
    elif report['phase'] not in (None, ''):
        head = (f"one phase per exposure (label {report['phase']!r} names no "
                f"phase; latest per exposure)")
    else:
        head = "one phase per exposure (latest per exposure)"
    line = (f"{head}: using {n_used} {what} file(s) [{_fmt(report['used'])}]; "
            f"excluded {sum(report['superseded'].values())} superseded "
            f"[{_fmt(report['superseded'])}] and "
            f"{sum(report['ahead'].values())} from a later phase "
            f"[{_fmt(report['ahead'])}]")
    losers = report.get('tie_losers') or []
    if losers:
        shown = '; '.join(f'{win} over {lose}' for win, lose in losers[:3])
        more = f' (+{len(losers) - 3} more)' if len(losers) > 3 else ''
        line += (f"; {len(losers)} same-phase tie(s) went to the newer file: "
                 f"{shown}{more}")
    return line
