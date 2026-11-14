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

Frame variants.  Several trees hold the satstar catalogs of one exposure fit
on two frame variants (``_align_o002_crf`` and ``_destreak_o002_crf``).  The
merge places those rows beside daophot rows measured on ONE variant, so a
caller that knows which frames it reads passes their variant tokens
(``frame_variants``) and only that variant's catalogs are eligible; without
them a same-phase tie between variants goes to the newer file, and the report
says so as a WARNING (#1194).

``SATSTAR_POOL_PHASES=all`` restores the pooled read (every phase of every
exposure), for bisecting only.
"""
import hashlib
import os
import re
from collections import Counter

from astropy.io import fits

#: Tokens written by the first cataloging phase.  All three are rank 2, so a
#: caller merging ``'m1'`` or ``'m2'`` reads the ``_m12`` satstar catalogs.
_FIRST_PHASE_TOKENS = ('m1', 'm2', 'm12')
_FIRST_PHASE_RANK = 2

#: The phase token at the end of a caller's iteration label: ``'m7'``,
#: ``'_m7'``, ``'resbgsub_m7'``.
_PHASE_LABEL_RE = re.compile(r'(?:^|_)(m\d+)$')

#: The per-exposure products one satstar fit writes that a later reader pools
#: across exposures: the accepted catalog, the gate-rejected candidates, and
#: the wing self-calibration measurements (``build_pooled_wingcal``).
_PHASE_PRODUCT = r'(?:satstar_(?:catalog|rejected)|wingcal_calibrators)\.fits$'

#: The phase token of a per-exposure satstar product.
_PHASE_FILE_RE = re.compile(r'_(m\d+)_' + _PHASE_PRODUCT)

#: Everything after the frame stem of a per-exposure satstar product: the
#: background-subtraction token(s) and the phase token.
_PHASE_TAIL_RE = re.compile(
    r'(?:_bgsub)?(?:_resbgsub)?_m\d+_' + _PHASE_PRODUCT)

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


def satstar_frame_variant(path, use_header=False):
    """The frame variant token of a frame or per-exposure satstar product.

    The part of the stem that follows the exposure name
    (``jw<prog><obs><visit>_<group>_<exp>_<detector>``), without the leading
    underscore and without the background/phase/product tail of a satstar
    product.  The frame
    ``jw02221002001_06201_00001_nrca1_destreak_o002_crf.fits`` and its catalog
    ``..._destreak_o002_crf_resbgsub_m6_satstar_catalog.fits`` both give
    ``'destreak_o002_crf'``; the ``_align_`` frame gives ``'align_o002_crf'``.

    Parameters
    ----------
    path : str
        A frame or product path or basename.
    use_header : bool
        When the name yields no token and ``path`` is an existing satstar
        product, read the frame name from the ``FILENAME`` card of extension 1
        and parse that.

    Returns
    -------
    str
        The token, or ``''`` when none can be read (a name that is not an
        observatory exposure name, or a frame name with nothing after the
        detector).
    """
    base = os.path.basename(str(path))
    if _PHASE_TAIL_RE.search(base):
        stem = _PHASE_TAIL_RE.sub('', base)
        is_product = True
    else:
        stem = base[:-5] if base.endswith('.fits') else base
        is_product = False
    match = _JW_EXPOSURE_RE.match(stem)
    token = stem[match.end():].lstrip('_') if match is not None else ''
    if token or not (use_header and is_product and os.path.exists(str(path))):
        return token
    try:
        with fits.open(str(path)) as hdul:
            frame = hdul[1].header.get('FILENAME', '')
    except (OSError, IndexError, KeyError):
        return ''
    return satstar_frame_variant(frame) if frame else ''


def normalise_frame_variants(frame_variants):
    """``frame_variants`` as a frozenset of tokens, or ``None`` when unset.

    ``None``, an empty collection and a collection of empty tokens all mean
    that the caller could not name a variant, so the selection is unfiltered.
    """
    if not frame_variants:
        return None
    if isinstance(frame_variants, str):
        frame_variants = [t for t in frame_variants.split(',')]
    tokens = frozenset(str(t).strip() for t in frame_variants if str(t).strip())
    return tokens or None


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


def _new_report(phase, rank_req, frame_variants):
    return {'mode': 'one', 'phase': phase, 'rank': rank_req,
            'used': Counter(), 'superseded': Counter(), 'ahead': Counter(),
            'tie_losers': [], 'frame_variants': sorted(frame_variants or ()),
            'other_variant': Counter(), 'no_variant': Counter(),
            'unparsed_variant': 0}


def select_satstar_phase_files(paths, phase=None, frame_variants=None):
    """Keep one cataloging phase's per-exposure satstar file per exposure.

    Parameters
    ----------
    paths : sequence of str
        Per-exposure ``*_m<N>_satstar_catalog.fits`` (or
        ``*_m<N>_satstar_rejected.fits``, or
        ``*_m<N>_wingcal_calibrators.fits``) paths of ONE product type,
        already observation-scoped where the caller scopes, and filtered to
        those carrying a phase token.
    phase : str, optional
        The phase being merged, as the merge's iteration label (``'m7'``,
        ``'m2'`` for the first phase, ``'resbgsub_m6'`` ...).  Each exposure
        then uses that phase's file, or its latest EARLIER phase when it has
        none.  A later phase's file is never used: in a phase-P merge it can
        only be left over from a previous run.  ``None``, or a label that names
        no phase (``'iter2'``), selects the latest phase each exposure has.
    frame_variants : collection of str, optional
        Frame variant tokens (:func:`satstar_frame_variant`) of the frames the
        caller reads.  Only files of those variants are eligible, so the phase
        rule then picks the latest phase AT OR BEFORE ``phase`` within the read
        variant.  An exposure with no file of a read variant at any eligible
        phase is left out and counted in ``report['no_variant']``.  A file
        whose variant cannot be read passes through (``unparsed_variant``).
        ``None`` keeps every variant eligible.

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
        a phase later than the one being merged.  ``frame_variants`` lists the
        variants asked for; ``other_variant`` counts files left out for
        belonging to another variant (by phase token), ``no_variant`` the
        exposures left out because none of their files is of a read variant
        (by the variants they do have).

    Notes
    -----
    Ties within one rank and one variant (``_m5`` beside ``_resbgsub_m5``) go
    to the most recently written file, then to the name, so the choice is
    deterministic.  With ``frame_variants=None`` that rule also settles a tie
    between two frame variants of one exposure, which the report flags.
    """
    paths = sorted(str(p) for p in paths)
    rank_req = satstar_phase_rank(phase)
    wanted = normalise_frame_variants(frame_variants)
    report = _new_report(phase, rank_req, wanted)
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
        if wanted is not None:
            kept = []
            for path in members:
                variant = satstar_frame_variant(path, use_header=True)
                if not variant:
                    report['unparsed_variant'] += 1
                    kept.append(path)
                elif variant in wanted:
                    kept.append(path)
                else:
                    report['other_variant'][satstar_file_phase_token(path)] += 1
            if not kept:
                report['no_variant'][', '.join(sorted(
                    {satstar_frame_variant(p) for p in members}))] += 1
                continue
            members = kept
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
                    # Same phase, another background token or (without
                    # frame_variants) another frame variant: the newer file
                    # won.  Name the loser so a surprising choice can be
                    # traced from the log.
                    report['tie_losers'].append(
                        (os.path.basename(best[2]), os.path.basename(item[2])))
    return sorted(selected), report


def select_rejected_for_catalogs(rejected_paths, chosen_catalogs, phase=None,
                                 frame_variants=None):
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
    phase, frame_variants : optional
        As for :func:`select_satstar_phase_files`; ``frame_variants`` applies
        to the orphan exposures only, the others follow their chosen catalog.

    Returns
    -------
    selected : list of str
    report : dict
        As for :func:`select_satstar_phase_files`.
    """
    rejected_paths = sorted(str(p) for p in rejected_paths)
    if pool_all_phases():
        return select_satstar_phase_files(rejected_paths, phase=phase,
                                          frame_variants=frame_variants)
    siblings = {os.path.basename(str(c)).replace('_satstar_catalog.fits',
                                                 '_satstar_rejected.fits')
                for c in chosen_catalogs}
    covered = {satstar_phase_group_key(c) for c in chosen_catalogs}
    rank_req = satstar_phase_rank(phase)
    selected, orphans = [], []
    report = _new_report(phase, rank_req, normalise_frame_variants(frame_variants))
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
        extra, sub = select_satstar_phase_files(
            orphans, phase=phase, frame_variants=frame_variants)
        selected.extend(extra)
        for key in ('used', 'superseded', 'ahead', 'other_variant',
                    'no_variant'):
            report[key].update(sub[key])
        report['unparsed_variant'] += sub['unparsed_variant']
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
        cross = [(w, l) for w, l in losers
                 if satstar_frame_variant(w) != satstar_frame_variant(l)]
        line += (f"; {len(losers)} same-phase tie(s) went to the newer file: "
                 f"{shown}{more}")
        if cross:
            line += (f"; WARNING: {len(cross)} of them chose between frame "
                     f"variants by file time alone, so the satstar rows may "
                     f"come from frames the merge does not read (#1194)")
    wanted = report.get('frame_variants') or []
    if wanted:
        line += (f"; frame variant(s) read: {', '.join(wanted)}; "
                 f"{sum(report['other_variant'].values())} file(s) of another "
                 f"variant left out [{_fmt(report['other_variant'])}]")
        if len(wanted) > 1:
            line += ("; WARNING: more than one frame variant is read, so an "
                     "exposure that has both still ties by file time")
        if report.get('unparsed_variant'):
            line += (f"; {report['unparsed_variant']} file(s) with no "
                     f"readable frame variant passed through")
        if report.get('no_variant'):
            n_nov = sum(report['no_variant'].values())
            have = '; '.join(f'{k or "?"}: {v}'
                             for k, v in sorted(report['no_variant'].items()))
            line += (f"; WARNING: {n_nov} exposure(s) have no satstar catalog "
                     f"of the frame variant(s) read and contribute no satstar "
                     f"rows (variants on disk: {have})")
    return line
