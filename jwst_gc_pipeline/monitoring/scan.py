"""Scan a field's directory tree and report where its pipeline run stands.

The monitor answers three questions from disk alone, for a full field or for a
``--cutout-region`` run:

* **how far has it got** -- the stage ladder, one row per pipeline phase;
* **is what it produced coherent** -- provenance tags, astrometry checkpoints;
* **what is in flight** -- live SLURM jobs and their logs (``jobs.py``).

Two rules shape every glob here, both learned from bugs:

1. **Never glob a stage without pinning the observation.**  A field with more
   than one observation (cloudef o002+o005, gc2211 o023/o028/o046/o049/o050,
   sickle, ngc6334's two proposals) has products from every observation in one
   directory.  A pattern that omits the observation token reports one
   observation's products under another's name.  Where the pipeline itself
   writes a name that carries no observation token -- the per-filter merged
   catalogs are named ``<filter>_<module>_indivexp_merged_...``, with no ``_o``
   for every proposal except the per-obs-merged ones (10678/gc-treasury, which
   spell ``<filter>_<module>_o<obs>_indivexp_merged_...``;
   naming.PER_OBS_MERGED_PROPOSALS) -- the ambiguity is REPORTED
   (``scope='ambiguous'``) rather than resolved by guessing.

2. **Follow symlinks.**  ``brick``, ``cloudc`` and ``wd1`` under
   ``/orange/adamginsburg/jwst`` are symlinks into ``/blue`` and
   ``/orange/adamginsburg/westerlund``; a scan that does not resolve them
   silently reports the flagship fields as empty.
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

from .. import fields as _fields

#: Detector-frame exposure names encode the observation: ``jw<PPPPP><OOO><VVV>_...``
#: so ``jw02221001001_...`` is proposal 02221, observation 001, visit 001.
_EXPNAME_RE = re.compile(r'^jw(?P<proposal>\d{5})(?P<obsid>\d{3})(?P<visit>\d{3})_')
#: Mosaic/association products name the observation explicitly: ``jw02221-o001_...``
_MOSAIC_RE = re.compile(r'^jw(?P<proposal>\d{5})-o(?P<obsid>\d+)[_-]')

#: The reduced per-exposure product the cataloging stage reads.  Fields differ in
#: which one they carry (``destreak_`` for most, ``align_`` for w51/ngc6334/wd2),
#: so both are accepted and which one was found is reported.
_REDUCED_SUFFIXES = ('destreak', 'align')

#: Cataloging phases in run order.  ``m12`` is the combined m1+m2 per-exposure
#: phase; m8 is the cross-band dedup.  Kept as an ordered tuple because the
#: ladder's meaning is the ORDER -- a later stage present with an earlier one
#: missing is itself a finding.
CATALOG_PHASES = ('m12', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8')

#: Phases whose per-filter product is written with ``resbgsub`` in the name (the
#: residual-background-subtracted iteration); m3/m4 predate it.  An explicit set,
#: not a ``>= 'm5'`` compare -- that is lexicographic over phase NAMES, so a
#: future ``m10`` would sort below ``m5`` and silently lose its bg token.
_RESBGSUB_PHASES = frozenset({'m5', 'm6', 'm7', 'm8'})


class ScanError(ValueError):
    """A field cannot be scanned (unknown target, unreadable root)."""


def _finite(value):
    """``value`` as a float, or ``None`` if it is missing or non-finite."""
    import math
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------
# Locating a field
# --------------------------------------------------------------------------

def basepath(target, cutout_label=None):
    """The directory to scan for ``target``, symlinks resolved.

    With ``cutout_label`` the cutout subtree ``<base>/cutouts/<label>`` is
    returned instead -- the same layout, one level down, which is why the whole
    scanner works unchanged on a cutout run.
    """
    try:
        base = _fields.fields_basepath(target)
    except (_fields.FieldRegistryError, KeyError) as ex:
        raise ScanError(f'unknown target {target!r}') from ex
    base = os.path.realpath(base)
    if cutout_label:
        base = os.path.join(base, 'cutouts', cutout_label)
    return base


def observations(target, instrument='nircam'):
    """``[(proposal, obsid), ...]`` registered for ``target``.

    Read from ``fields.yaml`` rather than from disk, so an observation whose
    products are entirely missing still appears in the monitor as a row of
    pending stages instead of vanishing.
    """
    field = _fields.BY_NAME.get(target)
    if field is None:
        raise ScanError(f'unknown target {target!r}')
    out = []
    for obs in field.observations:
        for obsid in obs.obsids.get(instrument, ()):
            out.append((obs.proposal, obsid))
    return out


def registered_filters(target, proposal, instrument='nircam'):
    """Upper-case filter names the registry lists for one observation.

    ``fields.yaml`` stores ONE flat filter list per proposal covering every
    instrument (w51/6151 lists F140M..F480M *and* F770W/F1280W/F2100W), so the
    list is split here against ``naming.MIRI_FILTERS`` -- the pipeline's own
    single source of truth.  Without the split a NIRCam run grows permanently
    empty rows for the MIRI filters, which read as missing products.
    """
    from ..photometry.naming import MIRI_FILTERS
    field = _fields.BY_NAME.get(target)
    if field is None:
        return []
    for obs in field.observations:
        if obs.proposal != str(proposal):
            continue
        names = [f.lower() for f in obs.filters]
        if instrument == 'miri':
            names = [f for f in names if f in MIRI_FILTERS]
        elif instrument == 'nircam':
            names = [f for f in names if f not in MIRI_FILTERS]
        elif instrument == 'niriss':
            names = [f.lower() for f in (obs.niriss_filters or obs.filters)]
        return [f.upper() for f in names]
    return []


def is_globbed(target, proposal, obsid, instrument='nircam'):
    """Does the pipeline actually build filenames for this observation?

    Several fields register observations the reduction does NOT glob: wd1 lists
    o001 and o003 but ``glob_obsid`` is ``001``, wd2 lists o003/o005 with
    ``005``, w51 MIRI lists o001/o002 with ``002``.  Those observations have no
    products by design, and reporting them as a field with nothing done is a
    false alarm -- so the ladder shows them as *not globbed*, not as pending.
    ``glob_obsid`` of ``'*'`` (sickle, cloudef) means every observation is read.
    """
    pattern = _fields.glob_obsid(target, proposal, instrument)
    if pattern in (None, '*'):
        return True
    return str(pattern).lstrip('0') == str(obsid).lstrip('0')


#: The observation token in a checkpoint record filename.  Accepts the
#: registered JOINT forms (sgrb2 `o002-998`, sickle `o001-002`) as well as the
#: per-proposal `j` form (ngc6334's 7213/6778 share an obsid).
_OBS_TOKEN_RE = re.compile(
    r'checkpoint_m2_[^_]+(_(?:o[\d-]{3,}|j\d{4,5}))_latest')


def shared_filters(target, instrument='nircam'):
    """Filters registered to MORE THAN ONE observation of ``target``.

    Only these are genuinely ambiguous when a product name carries no
    ``_o<obs>`` token.  Brick's two observations use disjoint filter sets
    (1182: F115W/F200W/F356W/F444W; 2221: F182M/F187N/F212N/F405N/...), so a
    brick per-filter catalog is unambiguous despite the field being multi-obs --
    flagging all of them would bury the real cases (cloudef o002/o005 and the
    gc2211 observations share their whole filter list, and ngc6334's two
    proposals share F200W/F470N).
    """
    field = _fields.BY_NAME.get(target)
    if field is None:
        return set()
    seen, shared = {}, set()
    for obs in field.observations:
        # `obs.filters` is ONE list per registry entry, shared by NIRCam and
        # MIRI.  Iterating `obs.obsids[instrument]` while reading it credits
        # every filter on the entry to whichever instrument is asked, and
        # sgrb2 is the case: 3 MIRI obsids made all 14 of its filters "shared",
        # including 11 NIRCam-only bands, while its NIRCam side has exactly one
        # observation.  Ten sgrb2 records were then refused with a message
        # false on its own terms -- "more than one observation of this field
        # images F212N", which is not a MIRI filter at all.
        #
        # The registry carries no per-instrument filter list, but the band
        # names encode wavelength in units of 0.01 um and the split is hard:
        # MIRI imaging starts at F560W.  Scope by that.
        def _belongs(filt, inst=instrument):
            wl = _fields._wavelength_key(filt)[0]
            if wl >= 10 ** 6:               # unparseable: do not exclude
                return True
            return (wl >= 500) if inst == 'miri' else (wl < 500)

        # One registry entry can carry SEVERAL observation ids of one proposal
        # (gc2211's 2211 lists o023/o028/o046/o049/o050 against a single filter
        # list).  Expanding them is the whole point: those five observations all
        # write F150W catalogs into one directory under one name.
        for obsid in obs.obsids.get(instrument, ()):
            # An observation the reduction never globs cannot have written a
            # catalog, so it cannot make one ambiguous.  wd1 registers o001 and
            # o003 with identical filter lists but globs only 001; counting o003
            # flags all 11 of its filters, and crying wolf on the largest field
            # trains the reader past the marker that protects cloudef, gc2211
            # and ngc6334.
            if not is_globbed(target, obs.proposal, obsid, instrument):
                continue
            if obsid == _fields.WILDCARD_OBSID:
                # This entry claims EVERY observation of the proposal, so each
                # of its filters is written by many observations into one
                # <basepath>/<FILTER>/ tree under one name -- ambiguous by
                # definition.  Counted as a single token it matched nothing and
                # no gc-treasury filter was ever marked.
                shared |= {f.upper() for f in obs.filters if _belongs(f)}
                continue
            token = (obs.proposal, obsid)
            for filt in obs.filters:
                if not _belongs(filt):
                    continue
                key = filt.upper()
                if key in seen and seen[key] != token:
                    shared.add(key)
                seen[key] = token
    return shared


def filter_dirs(base):
    """Filter subdirectories that exist on disk under ``base``.

    Disk is the authority here (not the registry): a filter directory that
    exists but is not registered is worth seeing, and one that is registered but
    absent shows up as a missing stage.
    """
    if not os.path.isdir(base):
        return []
    out = []
    with os.scandir(base) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=True) and re.fullmatch(r'F\d{3,4}[A-Z]?\d*', entry.name):
                out.append(entry.name)
    return sorted(out)


# --------------------------------------------------------------------------
# Observation-safe file counting
# --------------------------------------------------------------------------

def _obs_of(filename):
    """``('02221', '001')`` from an exposure or mosaic filename, else ``None``."""
    name = os.path.basename(filename)
    for rx in (_EXPNAME_RE, _MOSAIC_RE):
        m = rx.match(name)
        if m:
            return m.group('proposal').lstrip('0'), m.group('obsid').lstrip('0') or '0'
    return None


def _matches_obs(filename, proposal, obsid):
    """Does ``filename``'s embedded observation match ``(proposal, obsid)``?

    Returns ``None`` when the name carries no observation at all -- the caller
    must then decide whether the ambiguity matters, never assume a match.

    ``obsid`` is ``fields.WILDCARD_OBSID`` for a field that claims every
    observation of its proposal (gc-treasury/10678).  Compared as a literal it
    equals no observation number, so a genuine ``jw10678037001_..._o037_crf``
    frame read as a non-match and every per-observation count for the largest
    programme came back zero.  Under the wildcard the proposal alone decides.
    """
    got = _obs_of(filename)
    if got is None:
        return None
    if str(obsid) == _fields.WILDCARD_OBSID:
        return got[0] == str(proposal).lstrip('0')
    return got == (str(proposal).lstrip('0'), str(obsid).lstrip('0') or '0')


#: The observation token as the reduced products spell it: ``_o037_``, and the
#: joint ``_o002-998_`` form sgrb2/sickle write when one product covers several
#: observations.  Used only under the wildcard, where there is no single number
#: to spell into the tail.
_OBS_TOKEN_PATTERN = r'_o\d{3}(?:-\d{3})*'


def crf_tail_predicate(obsid, suffix=None):
    """``name -> bool`` for a ``[_<suffix>]_o<obs>_crf.fits`` tail.

    A concrete ``obsid`` spells the tail literally.  ``fields.WILDCARD_OBSID``
    has no number to spell, and the literal ``_o*_crf.fits`` it used to build
    matched nothing on disk, so the observation token is read as a SHAPE there
    and every reduced frame of the proposal counts.  The caller still scopes by
    proposal through ``_matches_obs``.
    """
    lead = f'_{suffix}' if suffix else ''
    if str(obsid) == _fields.WILDCARD_OBSID:
        rx = re.compile(re.escape(lead) + _OBS_TOKEN_PATTERN + r'_crf\.fits$')
        return lambda name: rx.search(name) is not None
    tail = f'{lead}_o{obsid}_crf.fits'
    return lambda name: name.endswith(tail)


#: Directory NAMES only, one ``scandir`` per directory, cached for the life of a
#: scan.  Names are nearly free; ``stat`` is not.  A single filter's
#: ``pipeline/`` here holds ~33,000 files, and statting them all costs ~5 s per
#: directory on this NFS -- about a minute for one field, many minutes for the
#: archive -- while listing their names costs ~0.05 s.  So names drive the
#: counts, and ``stat`` is spent only on a bounded sample, for timestamps.
_LISTING_CACHE = {}

#: How many matched files a bucket stats to date itself.  A bucket is normally a
#: set of products written by ONE job within minutes of each other, so a sample
#: dates it correctly; where it does not, the reported time is a LOWER bound and
#: the result says so (``mtime_sampled``) rather than implying it stat'd them all.
MTIME_SAMPLE = 48


def listing(dirpath):
    """``(realpath, [name, ...])`` for one directory; cached.  Empty if absent."""
    key = os.path.realpath(dirpath)
    cached = _LISTING_CACHE.get(key)
    if cached is not None:
        return cached
    # os.listdir, NOT scandir + entry.is_file(): on NFS the directory entries
    # come back with d_type unknown, so is_file() falls back to a stat per entry
    # -- reintroducing the ~5 s per directory this cache exists to avoid.  Every
    # predicate below matches on a file suffix, so a stray subdirectory cannot be
    # miscounted as a product.
    try:
        names = os.listdir(key)
    except OSError:
        names = []
    cached = (key, names)
    _LISTING_CACHE[key] = cached
    return cached


def clear_cache():
    """Forget cached directory listings (call between successive scans)."""
    _LISTING_CACHE.clear()


def _sample_mtime(dirpath, names, sample=MTIME_SAMPLE):
    """``(newest_mtime, sampled)`` over at most ``sample`` of ``names``."""
    if not names:
        return None, False
    if len(names) <= sample:
        chosen, sampled = names, False
    else:
        step = len(names) / float(sample)
        chosen = [names[int(i * step)] for i in range(sample)]
        chosen.append(names[-1])
        sampled = True
    best = None
    for name in chosen:
        try:
            mtime = os.stat(os.path.join(dirpath, name)).st_mtime
        except OSError:
            continue
        if best is None or mtime > best:
            best = mtime
    return best, sampled


def count_matching(entries, predicate, proposal=None, obsid=None):
    """``{'n', 'mtime', 'mtime_sampled', 'files', 'scope'}`` over a cached listing.

    ``scope`` is ``'obs'`` when every counted file carried a matching observation
    token, ``'ambiguous'`` when some carried none (so the count may mix
    observations), and ``'none'`` when nothing matched.  The renderer shows
    ``ambiguous`` explicitly; it is never silently treated as ``obs``.
    """
    dirpath, names = entries
    kept, ambiguous = [], False
    for name in names:
        if not predicate(name):
            continue
        if proposal is not None and obsid is not None:
            verdict = _matches_obs(name, proposal, obsid)
            if verdict is False:
                continue
            if verdict is None:
                ambiguous = True
        kept.append(name)
    if not kept:
        return {'n': 0, 'mtime': None, 'mtime_sampled': False, 'files': [],
                'scope': 'none'}
    kept.sort()
    mtime, sampled = _sample_mtime(dirpath, kept)
    return {'n': len(kept), 'mtime': mtime, 'mtime_sampled': sampled,
            'files': kept[:5], 'scope': 'ambiguous' if ambiguous else 'obs'}


def count_products(pattern, proposal=None, obsid=None):
    """``count_matching`` for callers that have a glob rather than a predicate."""
    import fnmatch
    dirpath, base = os.path.split(pattern)
    return count_matching(listing(dirpath),
                          lambda n: fnmatch.fnmatch(n, base), proposal, obsid)


# --------------------------------------------------------------------------
# The stage ladder
# --------------------------------------------------------------------------

def _reduction_stages(base, filt, proposal, obsid):
    """Reduction-side rows for one filter: uncal -> cal -> crf -> reduced -> i2d."""
    entries = listing(os.path.join(base, filt, 'pipeline'))
    rows = {}
    rows['uncal'] = count_matching(entries, lambda n: n.endswith('_uncal.fits'),
                                   proposal, obsid)
    rows['cal'] = count_matching(entries, lambda n: n.endswith('_cal.fits'),
                                 proposal, obsid)

    # The bare `_o<obs>_crf.fits` is stage-3 output; the destreak_/align_ variants
    # are the reduced working copies cataloging actually reads.  They must be
    # counted SEPARATELY -- `*_o<obs>_crf.fits` also matches
    # `..._destreak_o<obs>_crf.fits`, and folding them together hides "reduction
    # ran but destreaking never did", the wd1 F150W failure.
    reduced_hits = tuple(crf_tail_predicate(obsid, s) for s in _REDUCED_SUFFIXES)
    crf_hit = crf_tail_predicate(obsid)
    rows['crf'] = count_matching(
        entries,
        lambda n: crf_hit(n) and not any(hit(n) for hit in reduced_hits),
        proposal, obsid)

    reduced = {'n': 0, 'mtime': None, 'files': [], 'scope': 'none', 'variant': None}
    for suffix, hit in zip(_REDUCED_SUFFIXES, reduced_hits):
        got = count_matching(entries, lambda n, h=hit: h(n),
                             proposal, obsid)
        if got['n'] > reduced['n']:
            reduced = dict(got, variant=suffix)
    rows['reduced'] = reduced

    # Only true drizzled mosaics: `jw<prop>-o<obs>_..._i2d.fits`.  The per-exposure
    # `*_outlier_i2d.fits` and the residual mosaics share the suffix but are not
    # the stage's product, and counting them makes a filter look mosaicked when
    # its mosaic was never built.
    rows['i2d'] = count_matching(
        entries,
        lambda n: (n.endswith('_i2d.fits') and 'outlier' not in n
                   and _MOSAIC_RE.match(n) is not None),
        proposal, obsid)
    return rows


def _catalog_phase_patterns(catdir, filt):
    """``{phase: glob}`` for one filter's per-filter merged catalogs.

    These names carry NO observation token, which is why the caller marks them
    ambiguous for a multi-observation field.
    """
    low = filt.lower()
    out = {}
    for phase in CATALOG_PHASES:
        if phase == 'm8':
            continue                      # m8 is cross-band only, handled separately
        bg = 'resbgsub_' if phase in _RESBGSUB_PHASES else ''
        out[phase] = os.path.join(
            catdir, f'{low}_*_indivexp_merged_{bg}{phase.replace("m12", "m2")}_dao_basic.fits')
    return out


def _catalog_stages(base, filt, ambiguous):
    """Per-filter cataloging rows m12..m7 for one filter.

    ``ambiguous`` marks the rows as unattributable: the pipeline writes these
    names without an observation token, so where the SAME filter belongs to more
    than one observation of the field the file on disk is whichever observation
    ran last.
    """
    import fnmatch
    catdir = os.path.join(base, 'catalogs')
    entries = listing(catdir)
    rows = {}
    for phase, pattern in _catalog_phase_patterns(catdir, filt).items():
        base_pat = os.path.basename(pattern)
        got = count_matching(entries, lambda n, p=base_pat: fnmatch.fnmatch(n, p))
        if ambiguous and got['n']:
            got['scope'] = 'ambiguous'
        rows[phase] = got
    # saturated-star products are a separate track that gates m12
    satname = f'{filt.lower()}_consolidated_satstar_catalog.fits'
    rows['satstar'] = count_matching(entries, lambda n: n == satname)
    return rows


def satstar_frames(base, filt, proposal, obsid):
    """Per-frame satstar products: accepted catalogs vs rejected files.

    Counted separately on purpose.  A filter with ZERO satstar catalogs but many
    *rejected* files, whose merged catalog nonetheless reports
    ``replaced_saturated`` rows, is shipping saturated-star photometry produced at
    an EARLIER stage -- the current stage rejected everything and the old products
    were carried forward.  A single "satstar present?" count cannot see that.
    """
    dirpath, names = listing(os.path.join(base, filt, 'pipeline'))
    accepted = rejected = wingcal = 0
    for name in names:
        if _matches_obs(name, proposal, obsid) is False:
            continue
        if name.endswith('_satstar_catalog.fits'):
            accepted += 1
        elif name.endswith('_satstar_rejected.fits'):
            rejected += 1
        elif name.endswith('_wingcal_calibrators.fits'):
            wingcal += 1
    return {'accepted': accepted, 'rejected': rejected, 'wingcal': wingcal}


def _crossband_stages(base, proposal, obsid):
    """Cross-band m7/m8 merged catalogs for one observation.

    The cross-band products DO carry an ``_o<obs>`` token on multi-observation
    fields (``..._m7_o023.fits``) but not on single-observation ones, so both
    spellings are tried and which one matched is reported.
    """
    catdir = os.path.join(base, 'catalogs')
    rows = {}
    for phase, stem in (('m7', 'resbgsub_m7'), ('m8', 'resbgsub_m8_dedup')):
        tagged = count_products(
            os.path.join(catdir, f'basic_*_photometry_tables_merged_{stem}_o{obsid}.fits'))
        if tagged['n']:
            rows[phase] = dict(tagged, scope='obs')
            continue
        plain = count_products(
            os.path.join(catdir, f'basic_*_photometry_tables_merged_{stem}.fits'))
        rows[phase] = dict(plain, scope='ambiguous' if plain['n'] else 'none')
    return rows


# --------------------------------------------------------------------------
# Astrometry checkpoints
# --------------------------------------------------------------------------

def astrometry_checkpoints(base, filters=None, ambiguous_filters=()):
    """Parse ``<base>/astrometry_checkpoints/checkpoint_m2_<FILT>_latest.json``.

    These are the pipeline's OWN verdicts (written by the m2 checkpoint), so the
    monitor reports them rather than re-measuring -- re-measuring would mean
    re-implementing the offset-histogram machinery, and an ad-hoc reimplementation
    is exactly the banned nearest-neighbour-median path.

    Returns ``{filter: summary}`` where summary carries the per-exposure worst
    case: how many exposures were flagged misaligned, the lowest peak contrast,
    whether any tie needed a swept window, and the consensus scatter.

    A checkpoint file is named for its FILTER only (``checkpoint_m2_F150W_latest``)
    and its ``context`` field names the target, not the observation.  So where a
    filter belongs to several observations of the field, the record cannot be
    attributed to one of them; those carry ``attributable=False`` and must not be
    reported as a failure OF a particular observation.
    """
    ckdir = os.path.join(base, 'astrometry_checkpoints')
    if not os.path.isdir(ckdir):
        return {}
    out = {}
    for path in sorted(glob.glob(os.path.join(ckdir, 'checkpoint_m2_*_latest.json'))):
        # Records are keyed on the observation (issue #281), so two of them can
        # describe the same filter.  `out[filt]` is last-wins, which silently
        # discards one observation's verdict -- key on the token as well.
        # NB `bname`, not `base` -- `base` is this function's own parameter (the
        # field basepath), and rebinding it inside the loop shadowed it.
        bname = os.path.basename(path)
        # joint obsids are registered (sgrb2 o002-998, sickle o001-002), so a
        # bare o\d{3} misses them and the keys collide back to last-wins
        # ONE pattern, imported rather than re-spelled.  A second literal copy
        # of a regex that already got joint obsids wrong once is how it drifts
        # back: sgrb2 registers o002-998 and sickle o001-002, and a bare
        # `o\d{3}` misses both.
        _tokm = _OBS_TOKEN_RE.search(bname)
        _tok = _tokm.group(1) if _tokm else ''
        filt = bname.split('_')[2]
        if filters and filt.upper() not in {f.upper() for f in filters}:
            continue
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        exposures, visits = [], []
        for visit in rec.get('visits', []):
            cons = visit.get('consensus') or {}
            tie = visit.get('reference_tie') or {}
            tiles = tie.get('per_tile') or {}
            worst_cell = tiles.get('worst_off_cell') or {}
            visits.append({
                'visit': visit.get('visit'),
                'n_stars': cons.get('n_stars'),
                'scatter_mas': _finite(cons.get('median_scatter_mas')),
                'consensus_ok': cons.get('consensus_ok'),
                'skipped': len(cons.get('skipped') or []),
                # The reference tie and its per-tile map.  `n_ok` counts tiles
                # with a COHERENT PEAK, not tiles within tolerance --
                # measure_offset_grid is called with no max_off_mas, and
                # astrometry_offsets sets off_ok=True whenever that is None.  So
                # `36/36 tiles ok` is compatible with a 29 mas worst cell, and
                # only worst_off_mas says so.
                'tie_off_mas': _finite(tie.get('off_mas')),
                'tie_source': tie.get('bulk_source'),
                'tie_apply_ok': tie.get('apply_ok'),
                'tie_gross_ok': tie.get('cross_reference_gross_ok'),
                'tie_swept': tie.get('swept'),
                'tiles_ok': tiles.get('n_ok'),
                'tiles_total': tiles.get('n_total'),
                'tiles_clean': tiles.get('clean'),
                'worst_tile_mas': _finite(tiles.get('worst_off_mas')),
                'worst_tile_cell': (f"({worst_cell.get('ix')},{worst_cell.get('iy')})"
                                    if worst_cell else None),
                'min_tile_contrast': _finite(tiles.get('min_contrast_seen')),
                # The cells themselves, so the page can draw the residual map and
                # say WHERE the field is bad rather than only how bad its worst
                # point is.  A single scalar cannot distinguish "one bad corner"
                # from "a gradient across the mosaic", and those have different
                # causes.
                # `off_mas` is the canonical per-cell offset key: every cell
                # producer in astrometry_offsets writes it (issue #267).  The
                # `off` fallback is ONLY for checkpoints recorded before that
                # change -- those files are never rewritten, so the reader has
                # to keep tolerating the old spelling.  Reading one spelling
                # only used to yield an empty map, and with it a derived claim
                # that the field was flat.
                'cells': [{'ix': c.get('ix'), 'iy': c.get('iy'),
                           'off_mas': _finite(c.get('off_mas', c.get('off'))),
                           'dra': _finite(c.get('dra')),
                           'ddec': _finite(c.get('ddec')),
                           'contrast': _finite(c.get('contrast')),
                           'npairs': c.get('npairs')}
                          for c in (tiles.get('cells') or [])],
            })
            for exposure in visit.get('exposures') or []:
                exposure = dict(exposure)
                exposure['_visit'] = visit.get('visit')
                exposures.append(exposure)
        # The checkpoint writer emits bare `NaN` (JSON's parser accepts it,
        # JavaScript's does not, and NaN silently poisons min/max).  Drop
        # non-finite values here so the page and the --json dump both stay valid.
        contrasts = [v for v in (_finite(e.get('contrast')) for e in exposures)
                     if v is not None]
        offs = [v for v in (_finite(e.get('off')) for e in exposures)
                if v is not None]
        out[filt.upper() + _tok] = {
            'path': path,
            # The dict KEY carries the token so two observations' verdicts do
            # not overwrite each other; consumers that need to look the filter
            # up elsewhere (`run['per_filter']`, which is keyed on bare filter
            # names) must use this, not the key.
            'filter': filt.upper(),
            'obs_token': _tok,
            'date': rec.get('date'),
            'stage': rec.get('stage'),
            'context': rec.get('context'),
            'correcting': rec.get('correcting'),
            # The record's OWN verdict and the reasons behind it.  A checkpoint
            # can fail before it ever measures an exposure -- a visit whose
            # consensus refuses to build (duplicate exposure identity) ingests
            # nothing, so every per-exposure counter below is 0 and the summary
            # is indistinguishable from a clean field that has not been
            # cataloged.  ngc6334 F090W has been in exactly that state since
            # 2026-07-29 (issue #407): `passed: false`, three visits refused,
            # zero exposures, and the monitor produced no verdict at all.
            'passed': rec.get('passed'),
            'failures': [str(f) for f in (rec.get('failures') or [])],
            'n_visits': len(rec.get('visits') or []),
            # A record that NAMES its observation is attributable to it, shared
            # filter or not -- that is the whole point of the token.  Only an
            # untokened record on a filter more than one observation images
            # cannot be pinned down.
            'attributable': bool(_tok) or filt.upper() not in {
                f.upper() for f in ambiguous_filters},
            'n_exposures': len(exposures),
            'n_misaligned': sum(1 for e in exposures if e.get('misaligned')),
            'n_unverified': sum(1 for e in exposures if e.get('unverified')),
            'n_not_ok': sum(1 for e in exposures if e.get('ok') is False),
            'n_swept': sum(1 for e in exposures if e.get('swept')),
            # Keep the offending exposures themselves.  "183 misaligned" is a
            # number; the page has to be able to say WHICH 183 -- which visit,
            # which detector, by how much -- because a misalignment concentrated
            # in one detector, one visit, or one dither component has a different
            # cause from one spread evenly.
            'misaligned_exposures': [
                {'visit': e.get('_visit'),
                 'key': e.get('key'),
                 'detector': (e.get('key') or [None, None, None])[2]
                             if isinstance(e.get('key'), list) else None,
                 'dra': _finite(e.get('dra')), 'ddec': _finite(e.get('ddec')),
                 'off': _finite(e.get('off')),
                 'contrast': _finite(e.get('contrast')),
                 'npairs': e.get('npairs'),
                 'swept': e.get('swept'),
                 'window_arcsec': _finite(e.get('window_arcsec')),
                 'component': e.get('component'),
                 'raoffset_meta': _finite(e.get('raoffset_meta')),
                 'deoffset_meta': _finite(e.get('deoffset_meta'))}
                for e in exposures if e.get('misaligned')],
            'all_exposures': [
                {'visit': e.get('_visit'),
                 'detector': (e.get('key') or [None, None, None])[2]
                             if isinstance(e.get('key'), list) else None,
                 'dra': _finite(e.get('dra')), 'ddec': _finite(e.get('ddec')),
                 'off': _finite(e.get('off')),
                 'misaligned': bool(e.get('misaligned'))}
                for e in exposures],
            'min_contrast': min(contrasts) if contrasts else None,
            'max_off_mas': max(offs) if offs else None,
            'med_off_mas': (sorted(offs)[len(offs) // 2] if offs else None),
            'visits': visits,
            'mtime': os.path.getmtime(path) if os.path.exists(path) else None,
        }
    return out


# --------------------------------------------------------------------------
# Per-frame WCS provenance (FITS primary headers only)
# --------------------------------------------------------------------------

#: The LW filteroffset reference is MODULE-SPECIFIC: module A must use
#: ``jwst_nircam_filteroffset_0007.asdf`` and module B ``0008``.  A frame carrying
#: the other module's reference is displaced on sky by the difference between the
#: two filter offsets -- up to ~26 mas per module for F410M (a 52 mas A-B
#: differential), ~11 mas for F444W/F405N.  The error is ANTI-SYMMETRIC between
#: modules, so a run that mixes swapped and corrected frames manufactures an
#: apparent inter-module offset that no reference tie can diagnose.
#: (astrometry_paper/wcs_provenance.tex:57-59 mapping, :63-65 amplitudes.)
LW_FILTEROFFSET = {'A': '0007', 'B': '0008'}

#: Frames sampled per filter for the header checks.  These are per-file opens on
#: NFS, so the sample is small; the failure modes it looks for (a swapped
#: reference, a stale CRDS context) affect whole batches of frames, not one.
HEADER_SAMPLE = 4

_HEADER_KEYS = ('CRDS_CTX', 'CAL_VER', 'MODULE', 'DETECTOR', 'R_FILOFF', 'FILTER')


def _read_header(path):
    """The primary-header keys of interest, or ``None``."""
    try:
        from astropy.io import fits
        head = fits.getheader(path, 0)
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return {k: head.get(k) for k in _HEADER_KEYS}


def frame_provenance(base, filt, proposal, obsid, each_suffix_variant=None,
                     sample=HEADER_SAMPLE):
    """WCS provenance sampled from a filter's reduced frames.

    Returns counts of ``CRDS_CTX``/``CAL_VER`` seen and any LW frame whose
    ``R_FILOFF`` disagrees with its ``MODULE``.  Header-only: nothing is
    re-projected, no reference catalog is touched.
    """
    entries = listing(os.path.join(base, filt, 'pipeline'))
    dirpath, names = entries
    hits = tuple(crf_tail_predicate(obsid, s) for s in _REDUCED_SUFFIXES)
    if each_suffix_variant:
        hits = (crf_tail_predicate(obsid, each_suffix_variant),)
    kept = [n for n in names
            if any(hit(n) for hit in hits)
            and _matches_obs(n, proposal, obsid) is not False]
    if not kept:
        return None
    kept.sort()
    step = max(1, len(kept) // sample)
    chosen = kept[::step][:sample]

    ctx, cal, mismatched, n_read = Counter(), Counter(), [], 0
    for name in chosen:
        head = _read_header(os.path.join(dirpath, name))
        if head is None:
            continue
        n_read += 1
        if head.get('CRDS_CTX'):
            ctx[str(head['CRDS_CTX'])] += 1
        if head.get('CAL_VER'):
            cal[str(head['CAL_VER'])] += 1
        detector = str(head.get('DETECTOR') or '')
        module = str(head.get('MODULE') or '').upper()
        filoff = str(head.get('R_FILOFF') or '')
        if detector.upper().endswith('LONG') and module in LW_FILTEROFFSET and filoff:
            want = LW_FILTEROFFSET[module]
            if want not in filoff:
                mismatched.append({'file': name, 'module': module,
                                   'r_filoff': os.path.basename(filoff),
                                   'expected': want})
    return {'filter': filt, 'n_sampled': n_read, 'n_frames': len(kept),
            'crds_ctx': dict(ctx), 'cal_ver': dict(cal),
            'filteroffset_mismatch': mismatched}


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def provenance(base, phases=('m7', 'm8'), multi_obs=False):
    """Pipeline tags recorded in the ``*.prov.json`` sidecars, per phase.

    Every stage output carries a sidecar naming the pipeline tag that produced
    it.  A field whose filters were produced by DIFFERENT tags is not
    releasable -- image and catalog must come from the same run -- so the
    monitor reports the tag SET per phase, not just the newest one.  A tag
    ending ``-dirty`` means the tree had uncommitted changes.
    """
    catdir = os.path.join(base, 'catalogs')
    out = {}
    for phase in phases:
        tags = Counter()
        dirty = 0
        files = glob.glob(os.path.join(catdir, f'*{phase}*.prov.json'))
        for path in files:
            try:
                with open(path) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            if rec.get('stage') != phase:
                continue
            tag = rec.get('tag') or '(untagged)'
            tags[tag] += 1
            if tag.endswith('-dirty'):
                dirty += 1
        if tags:
            out[phase] = {'tags': dict(tags), 'n_sidecars': sum(tags.values()),
                          'n_dirty': dirty, 'n_distinct': len(tags),
                          # Sidecars are globbed without an observation pin --
                          # the per-filter names carry none.  Marked like every
                          # other unpinned count rather than being the one
                          # unpinned number that still asserts a failure.
                          'scope': 'ambiguous' if multi_obs else 'obs'}
    return out


# --------------------------------------------------------------------------
# Cutouts
# --------------------------------------------------------------------------

#: A cutout frame is written as ``<stem>_cutout_<label>.fits``.
_CUTOUT_FRAME_RE = re.compile(r'_cutout_(?P<label>[^/]+)\.fits$')


def cutout_labels(target):
    """Every cutout label present under ``<base>/cutouts/`` for ``target``."""
    try:
        cutdir = os.path.join(basepath(target), 'cutouts')
    except ScanError:
        return []
    if not os.path.isdir(cutdir):
        return []
    out = []
    with os.scandir(cutdir) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=True):
                out.append(entry.name)
    return sorted(out)


def cutout_summary(target, label):
    """What one cutout run produced: frames, catalogs, filters, when.

    Used both for the existing hand-made experiment cutouts and for the 5-arcsec
    probe cutouts the monitor's own test matrix creates.
    """
    base = basepath(target, cutout_label=label)
    frames = glob.glob(os.path.join(base, '*', 'pipeline', f'*_cutout_{label}.fits'))
    cats = (glob.glob(os.path.join(base, 'catalogs', '*.fits'))
            + glob.glob(os.path.join(base, 'catalogs', '*.ecsv')))
    # Bounded globs, never `**`: a recursive walk of every cutout tree (w51 has
    # 46, sickle 45) dominated the whole scan.  A cutout only ever writes into
    # <label>/<FILTER>/pipeline/ and <label>/catalogs/.
    pngs = (glob.glob(os.path.join(base, '*', 'pipeline', '*.png'))
            + glob.glob(os.path.join(base, 'catalogs', '*.png')))
    per_filter = Counter()
    for path in frames:
        parts = path[len(base):].strip(os.sep).split(os.sep)
        if parts:
            per_filter[parts[0]] += 1

    # A cutout tree is small enough to stat exhaustively, and two storage-fault
    # signatures only show up here: a product left at ZERO bytes, and an orphan
    # `tmp*` file -- the atomic-write temp that was never renamed because the
    # write died.  Both look like a completed run to a count-based ladder.
    mtimes, empty = [], []
    for path in frames + cats:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        mtimes.append(stat.st_mtime)
        if stat.st_size == 0:
            empty.append(os.path.basename(path))
    orphan_tmp = [os.path.basename(p) for p in
                  (glob.glob(os.path.join(base, '*', 'pipeline', 'tmp*'))
                   + glob.glob(os.path.join(base, 'catalogs', 'tmp*')))]

    return {'target': target, 'label': label, 'base': base,
            'n_frames': len(frames), 'n_catalogs': len(cats), 'n_pngs': len(pngs),
            'filters': dict(per_filter),
            'empty_files': sorted(empty)[:10], 'n_empty': len(empty),
            'orphan_tmp': sorted(orphan_tmp)[:10], 'n_orphan_tmp': len(orphan_tmp),
            'mtime': max(mtimes) if mtimes else None,
            'exists': os.path.isdir(base)}


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def scan_observation(target, proposal, obsid, instrument='nircam',
                     cutout_label=None, filters=None, with_headers=True):
    """Everything the monitor knows about one observation of one field."""
    base = basepath(target, cutout_label=cutout_label)
    on_disk = filter_dirs(base)
    registered = registered_filters(target, proposal, instrument)
    # The registry is authoritative about WHICH filters belong to THIS
    # observation; disk is not (one directory holds every observation of the
    # field).  Fall back to disk only for an observation the registry does not
    # list filters for, and report the difference either way.
    # A cutout run deliberately covers ONE filter, so the registry's full list
    # would render a dozen empty rows that look like missing products; disk is
    # the right authority there.
    if cutout_label:
        use = list(filters or on_disk)
    else:
        use = list(filters or registered or on_disk)
    # A wildcard field registers ONE entry -- the literal '*' -- for every
    # observation of its proposal, so counting rows reads a 139-observation
    # campaign as single-observation and suppresses the untagged-product
    # ambiguity warning (checks.py `scope == 'ambiguous' and multi_obs`).
    multi_obs = (len(observations(target, instrument)) > 1
                 or _fields.claims_every_observation(target, instrument))
    ambiguous_filters = shared_filters(target, instrument)

    per_filter = {}
    headers = {}
    for filt in use:
        row = _reduction_stages(base, filt, proposal, obsid)
        row.update(_catalog_stages(base, filt, filt.upper() in ambiguous_filters))
        row['satstar_frames'] = satstar_frames(base, filt, proposal, obsid)
        per_filter[filt] = row
        if with_headers and row.get('reduced', {}).get('n'):
            got = frame_provenance(base, filt, proposal, obsid,
                                   row['reduced'].get('variant'))
            if got:
                headers[filt] = got

    return {
        'target': target,
        'proposal': str(proposal),
        'obsid': str(obsid),
        'instrument': instrument,
        'basepath': base,
        'cutout_label': cutout_label,
        'is_cutout': bool(cutout_label),
        'multi_obs': multi_obs,
        'globbed': is_globbed(target, proposal, obsid, instrument),
        'glob_obsid': _fields.glob_obsid(target, proposal, instrument),
        'shared_filters': sorted(ambiguous_filters),
        'filters_on_disk': on_disk,
        'filters_registered': registered,
        'filters_missing': sorted(set(registered) - set(on_disk)),
        'filters_unregistered': sorted(set(on_disk) - set(registered)),
        'per_filter': per_filter,
        'headers': headers,
        'crossband': _crossband_stages(base, proposal, obsid),
        'astrometry': astrometry_checkpoints(base, use, ambiguous_filters),
        'provenance': provenance(base, multi_obs=multi_obs),
    }


def scan_field(target, instrument='nircam', cutout_label=None,
               with_headers=True):
    """Scan every registered observation of ``target``."""
    runs = []
    for proposal, obsid in observations(target, instrument):
        runs.append(scan_observation(target, proposal, obsid, instrument,
                                     cutout_label=cutout_label,
                                     with_headers=with_headers))
    return {'target': target, 'instrument': instrument,
            'cutout_label': cutout_label,
            'basepath': basepath(target, cutout_label=cutout_label),
            'runs': runs,
            'cutouts': cutout_labels(target) if not cutout_label else []}


def all_targets():
    """Every field name in the registry, alphabetically."""
    return sorted(_fields.BY_NAME)
