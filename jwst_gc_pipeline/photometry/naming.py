"""Filename-token helpers shared across the photometry pipeline.

These build the small, unambiguous tokens embedded in per-frame / merged output
filenames (iteration label, background-subtraction mode, spatial seed chunk) so
that products from different iterations and modes never collide and can be
glob-matched exactly.

Factored out of ``crowdsource_catalogs_long.py`` (2026-06-09 restructure); the
old module now imports these names from here so there is a single source of
truth.  Pure string/regex helpers with no astronomy dependencies.
"""
import os
import re

# Match _chunkXXofYY (any width) when stripping the chunk suffix from an
# iteration_label or filename component.
_CHUNK_TOKEN_RE = re.compile(r'_chunk\d+of\d+')

# Single source of truth for which JWST filters are MIRI vs NIRCam.  Lives here
# (a heavy-import-free module) so merge_catalogs.py can import it without pulling
# in crowdsource_catalogs_long.py's webbpsf chain.
MIRI_FILTERS = frozenset(['f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w',
                          'f1500w', 'f1800w', 'f2100w', 'f2550w'])

# Canonical instrument names as used downstream (PSF branch, SVO filterID,
# filename token = lowercased).
_CANONICAL_INSTRUMENT = {'nircam': 'NIRCam', 'niriss': 'NIRISS', 'miri': 'MIRI'}


def _instrument_override():
    """Process-global instrument override.

    NIRISS shares filter names with NIRCam (F158M/F200W/F356W/F480M), so the
    instrument CANNOT be derived from the filter name.  A single reduction/
    cataloging process only ever handles ONE instrument, so -- exactly like
    ``GC_BASEPATH_OVERRIDE`` (jwst_gc_pipeline.scratch_basepath) -- a process-wide
    env override is safe and avoids threading an ``instrument`` argument through
    every ``_inst_token`` / ``_svo_filter_id`` / ``_instrument_from_filter`` call
    site.  Set ``GC_INSTRUMENT_OVERRIDE=niriss`` (via ``--instrument niriss``) to
    force the NIRISS branch.  Empty/unset -> filter-name heuristic (historical
    NIRCam/MIRI behavior)."""
    val = os.environ.get('GC_INSTRUMENT_OVERRIDE', '').strip().lower()
    return _CANONICAL_INSTRUMENT.get(val) if val else None


def _instrument_from_filter(filtername, instrument=None):
    """Return 'MIRI', 'NIRCam', or 'NIRISS'.

    Precedence: explicit ``instrument`` arg > ``GC_INSTRUMENT_OVERRIDE`` env >
    filter-name heuristic (MIRI filter set -> MIRI, else NIRCam).  NIRISS can only
    arrive via the arg or the env (its filter names are shared with NIRCam)."""
    if instrument is not None:
        return _CANONICAL_INSTRUMENT.get(str(instrument).lower(), instrument)
    override = _instrument_override()
    if override is not None:
        return override
    return 'MIRI' if str(filtername).lower() in MIRI_FILTERS else 'NIRCam'


def _inst_token(filtername, instrument=None):
    """Lowercased instrument token used in JWST i2d filename conventions."""
    return _instrument_from_filter(filtername, instrument=instrument).lower()


def _svo_filter_id(filtername, instrument=None):
    """SVO FPS filterID (e.g. 'JWST/NIRCam.F480M', 'JWST/NIRISS.F200W')."""
    return f'JWST/{_instrument_from_filter(filtername, instrument=instrument)}.{filtername.upper()}'


def _chunk_token(chunk_index, n_seed_chunks):
    """Filename token for spatial seed chunking.

    Returns '' when n_seed_chunks <= 1; otherwise '_chunk{i:02d}of{n:02d}'.
    Two-digit fields keep the token sortable and unambiguous.
    """
    n = int(n_seed_chunks) if n_seed_chunks else 1
    if n <= 1:
        return ''
    i = int(chunk_index)
    if i < 0 or i >= n:
        raise ValueError(
            f'seed_chunk_index={i} out of range for n_seed_chunks={n}')
    return f'_chunk{i:02d}of{n:02d}'


def _strip_chunk(label):
    """Remove a trailing/embedded ``_chunkXXofYY`` token from a string.

    Used to recover the *base* iteration label (e.g. ``'iter3'``) from
    a chunk-suffixed compound label (e.g. ``'iter3_chunk03of08'``) so
    semantic checks like ``is_iter3`` continue to fire when chunking is on.
    """
    if label is None:
        return None
    return _CHUNK_TOKEN_RE.sub('', str(label))


def _iteration_token(iteration_label):
    if iteration_label in (None, ''):
        return ''

    token = str(iteration_label)
    if token.startswith('_'):
        return token
    return f'_{token}'


def _bgsub_token_from_flags(bgsub, resbgsub=False):
    """Filename token for the background-subtraction mode(s) in effect.

    * ``--bgsub`` (global Background2D subtraction)        -> ``_bgsub``
    * ``--use-iter3-residual-bg`` (iter3 residual-smoothed
      background subtraction)                             -> ``_resbgsub``

    Both can be set; the tokens concatenate in a fixed order so output
    catalog/residual/model/diagnostic filenames are unambiguous and the
    skip-if-done prediction (_predict_output_tokens) stays in sync with the
    names actually written by do_photometry_step.  ``_bgsub`` is never a
    substring of ``_resbgsub`` so exact-token matching does not collide.

    This is the canonical flags-based form; ``merge_catalogs`` imports it as
    ``_bgsub_token`` (it works with explicit booleans) and the producer side
    uses the ``_bgsub_token(options)`` wrapper below.
    """
    token = '_bgsub' if bgsub else ''
    if resbgsub:
        token += '_resbgsub'
    return token


def _bgsub_token(options):
    """``_bgsub_token_from_flags`` reading the flags off an options object."""
    return _bgsub_token_from_flags(
        getattr(options, 'bgsub', False),
        getattr(options, 'use_iter3_residual_bg', False))


# --- residual-i2d product-name family --------------------------------------
# The cataloging products follow a fixed residual-i2d naming convention.  These
# centralize the (otherwise scattered, inline ``.replace(...)``) transforms so
# the convention has a single source of truth.

def residual_to_smoothed_bg_i2d(residual_i2d_path):
    """``..._residual_i2d.fits`` -> ``..._residual_smoothed_bg_i2d.fits``."""
    return residual_i2d_path.replace('_residual_i2d.fits',
                                     '_residual_smoothed_bg_i2d.fits')


def smoothed_bg_to_detection_i2d(smoothed_bg_i2d_path):
    """``..._residual_smoothed_bg_i2d.fits`` -> ``..._residual_i2d.fits`` (the
    detection image sits next to the smoothed-bg, differing only by the infix)."""
    return smoothed_bg_i2d_path.replace('_smoothed_bg_i2d.fits', '_i2d.fits')


def residual_to_model_i2d(residual_i2d_path):
    """``..._residual_i2d.fits`` -> ``..._model_i2d.fits``."""
    return residual_i2d_path.replace('_residual_i2d.fits', '_model_i2d.fits')


def residual_to_infilled_i2d(residual_i2d_path):
    """``..._residual_i2d.fits`` -> ``..._residual_infilled_i2d.fits``."""
    return residual_i2d_path.replace('_residual_i2d.fits',
                                     '_residual_infilled_i2d.fits')


def vetted_to_i2dseed(vetted_path):
    """``..._vetted.fits`` -> ``..._i2dseed.fits``."""
    return vetted_path.replace('_vetted.fits', '_i2dseed.fits')


# --- observation scoping ---------------------------------------------------
# Which proposals put more than one observation under ONE basepath with
# RESTARTED (visit, vgroup, exposure) numbering, so per-frame catalog names
# need the ``_o{field}`` disambiguator ``crowdsource_catalogs_long.obs_token``
# inserts.  2211 = gc2211 (5 GC pointings); 10678 = the GC Treasury program
# (139 tiles sharing the gc-treasury tree; issue #416).  Lives here (a
# heavy-import-free module) so merge_catalogs.py can consult it without a
# circular import of crowdsource_catalogs_long.
#: 9438 (Schlafly): seven Galactic-plane pointings, l = 3 to 54 deg, under one
#: proposal.  Same shape as 2211 and 10678 -- different sky per observation, so
#: an untokened per-frame name could belong to any of the seven.
MULTIOBS_PROPOSALS = ('2211', '10678', '9438')

#: The subset whose MERGED catalogs are per-observation too.
#:
#: 10678 (gc-treasury): 139 tiles share one tree, and pooling another tile's
#: frames into a merge is itself the corruption mode.
#:
#: 2211 (gc2211): its five observations are DIFFERENT targets -- different
#: parts of the sky, observed at different times -- that happen to share a
#: parent observation id and therefore a directory.  They were pooled into one
#: untokened merged catalog, which this entry ends; each observation now merges
#: to its own ``_o{field}`` catalog.  The five have since been split into
#: separate fields (``gc2211_o023`` ... ``gc2211_o050``, #469), which is what
#: exposed the mismatch: the per-frame writer stamps ``_o023`` while the merge
#: still globbed by the literal target name ``'gc2211'``, matched nothing under
#: the new name, and every m12 finalize died with::
#:
#:     ValueError: No tables found matching
#:     /orange/adamginsburg/jwst/gc2211_o023//F200W/f200w_nrca...._dao_basic.fits
#:
#: after its 8 fan-out shards had written 192 per-frame tables (2026-08-22).
#: 9438 joins for the same reason as 2211: its seven observations are seven
#: DIFFERENT targets, so one untokened merged catalog would pool unrelated sky.
PER_OBS_MERGED_PROPOSALS = ('10678', '2211', '9438')

#: What a ``field`` may look like inside an observation token: an observation
#: number, or several joined by ``-`` for a joint registration ('002-998').
_OBSERVATION_FIELD_RE = re.compile(r'\d+(?:-\d+)*')


class ObservationFieldError(ValueError):
    """``field`` does not name an observation, so no token can be built."""


def observation_field_token(field):
    """``field`` normalised to the three-digit spelling used in filenames.

    ``'1'``, ``'01'`` and ``'001'`` all name observation 001, and MAST, the
    reduction products and ``naming.OBS_TOKEN_PATTERN`` (``o\\d{3}``) all spell
    it ``001``.  An unpadded ``--field`` wrote ``_o1`` -- a name every reader of
    the token skips.  Joint registrations ('002-998') normalise part by part.

    A field that does not name an observation raises.  ``fields.py`` registers
    a program whose observation numbers land as the campaign executes with the
    wildcard obsid ``'*'``, and ``default_field_token`` hands that wildcard back
    when ``--field`` is omitted; ``f'_o{field}'`` then put a literal ``*``
    inside every catalog name the run wrote, where every ``_o*`` glob in the
    tree matches it.  A run that cannot name its observation stops here.
    """
    text = str(field).strip()
    if not _OBSERVATION_FIELD_RE.fullmatch(text):
        raise ObservationFieldError(
            f'field={field!r} does not name an observation, so the observation '
            f'token this run stamps into its catalog names cannot be built.  A '
            f"wildcard ('*') arrives when the field is registered obsids: "
            f"{{'nircam': '*'}} and --field was omitted; pass --field <NNN> "
            f'naming the observation under reduction.')
    return '-'.join(f'{int(part):03d}' for part in text.split('-'))


def merge_field_for_proposal(proposal_id, field):
    """The ``field`` to hand ``merge_individual_frames``, or ``None``.

    Per-obs-merged proposals scope the merge to one observation (glob and
    output both carry ``_o{field}``); every other proposal passes ``None``, so
    gc2211's all-obs pooling (``_o*`` glob, untokened output) and the
    single-obs targets' untokened names are unchanged.

    A field-less call on a per-obs-merged proposal raises HERE, at the point
    the observation is decided.  ``merge_individual_frames`` refuses such a
    call too, but that refusal arrives inside the cutout run's
    print-and-continue handler, where it becomes one printed line and a cutout
    tree with no merged catalog.  Raising at the decision keeps the two callers
    that resolve the field before entering a handler (``observation_merge``)
    loud.
    """
    if str(proposal_id) in PER_OBS_MERGED_PROPOSALS and field in (None, ''):
        raise ObservationFieldError(
            f'proposal {proposal_id} merges per observation, and this run has '
            f'no field to name one, so the merge would pool tiles or write a '
            f'name no reader spells.  Pass --field <NNN>.')
    return field if merged_catalog_obs_token(proposal_id, field) else None


def vetted_obs_tokens(proposal_id, field, filtername=None, module=None):
    """``(vetted_token, combined_token)`` for the END of a catalog name.

    The vetted catalog is per-observation where one basepath holds several
    observations of a filter: MIRI multi-obs targets (cloudef obs 002+005) vet
    each observation against its own ``data_i2d``, and gc2211's five NIRCam
    pointings share one tree and reuse ``(visit, vgroup, exp)`` tuples, so a
    single vetting pass would carry sources outside each pointing's footprint.
    gc2211's COMBINED (post-vet) catalog is per-observation as well; MIRI's
    stays all-obs.  Single-obs NIRCam targets get ``('', '')``.

    Per-obs-MERGED proposals (10678) get ``('', '')`` too: their merged name
    already carries ``_o{field}`` after the module, the vetted and combined
    names inherit that token, and an end-slot token would double it -- the m7
    seed reader and ``merge_daophot``'s input glob spell the module-slot form.
    """
    if field in (None, ''):
        return '', ''
    if merged_catalog_obs_token(proposal_id, field):
        return '', ''
    token = f'_o{observation_field_token(field)}'
    multiobs = str(proposal_id) in MULTIOBS_PROPOSALS
    miri = (str(module).lower() == 'mirimage'
            or _instrument_from_filter(filtername) == 'MIRI')
    return (token if (miri or multiobs) else ''), (token if multiobs else '')


def merged_catalog_obs_token(proposal_id, field):
    """Observation token baked into the MERGED catalog names, post-module slot.

    Only ``PER_OBS_MERGED_PROPOSALS`` get one: 10678's 139 tiles share the
    gc-treasury tree, and pooling another tile's frames into a merge is the
    corruption class the obs scoping exists to prevent, so every per-filter
    merged catalog is scoped to one observation
    (``{filt}_{module}_o{field}_indivexp_merged...``).  2211 is here for the
    same reason: its five pointings are five different fields, so pooling them
    into one merged catalog mixes unrelated sky.  (They previously kept
    UNTOKENED merged names and were scoped only afterwards, at the vetting
    step.)  Writers (``merge_individual_frames``' ``out_obs_``) and every reader of a
    merged-catalog name (``cataloging.merged_catalog_path``, the m7 seed
    reader, ``merge_daophot``'s input glob) must agree on this token.

    ``field`` goes through ``observation_field_token``, which normalises the
    spelling and refuses a field that does not name an observation.
    """
    if str(proposal_id) in PER_OBS_MERGED_PROPOSALS and field not in (None, ''):
        return f'_o{observation_field_token(field)}'
    return ''


#: ngc6334 is imaged by two proposals that share one target directory, one
#: filter list, one observation number AND the same ``(visit, vgroup, exp)``
#: tuples, so their per-frame catalog names collide.  Tagged by proposal id.
SHARED_TREE_PROPOSALS = ('7213', '6778')


def perframe_obs_token(proposal_id, field):
    """The token the PER-FRAME catalog writer stamps between detector and visit.

    This is the single source for that token.  ``crowdsource_catalogs_long
    .obs_token`` (the writer) delegates here, and every consumer that has to
    predict a per-frame name -- above all ``merge_individual_frames``' input
    glob -- calls it rather than re-deriving the rule.  Re-deriving it is what
    issue #316 is about: a glob that spells the token differently from the
    writer does not raise, it matches nothing, and the caller reports "no
    per-frame catalogs for this filter" and moves on.

    Two spellings the hand-rolled versions got wrong:

    * ``_o{field}`` unpadded.  The writer normalises through
      ``observation_field_token``, so ``--field 23`` writes ``_o023`` while a
      raw f-string globs ``_o23``.
    * a token on a single-observation proposal.  Only
      ``MULTIOBS_PROPOSALS`` and ``SHARED_TREE_PROPOSALS`` are tokened; every
      other target's per-frame names carry none, whatever ``field`` says.
    """
    if str(proposal_id) in MULTIOBS_PROPOSALS and field not in (None, ''):
        return f'_o{observation_field_token(field)}'
    if str(proposal_id) in SHARED_TREE_PROPOSALS:
        return f'_j{proposal_id}'
    return ''


def merged_catalog_module_token(proposal_id, field):
    """The whole module-slot token of a MERGED catalog name.

    ``_j{proposal}`` for the shared ngc6334 tree, plus
    ``merged_catalog_obs_token``.  The merge's output name and every reader of
    it (``cataloging.merged_catalog_path``, the m7 seed reader,
    ``merge_daophot``'s glob) are one contract, so they take it from here.
    """
    jtok = (f'_j{proposal_id}'
            if str(proposal_id) in SHARED_TREE_PROPOSALS else '')
    return jtok + merged_catalog_obs_token(proposal_id, field)


# --- reading a per-frame catalog name back --------------------------------
# Every reader that parses `{band}_{detector}_visit{NNN}_vgroup...` out of a
# per-frame catalog name has to allow for the per-observation token `obs_token`
# inserts BETWEEN the detector and the visit (`_o023`, `_j6778`).  A pattern
# that requires `_visit` immediately after the detector silently SKIPS every
# tokened frame -- it does not raise, so the reader just solves on a subset.
# Measured on the live trees (2026-08-07):
#
#     gc2211  F200W   592 globbed   192 parsed   400 skipped   (68%)
#     ngc6334 F200W   560 globbed   280 parsed   280 skipped   (50%)
#
# `cataloging.py::_DETECTOR_TOKEN_RE` was fixed for this in #302; these are the
# same property in the readers #316 lists.

#: Optional `_o023` / `_j6778` segment, as a non-capturing regex group.
OBS_TOKEN_PATTERN = r'(?:_(?:o\d{3}|j\d{4,5}))?'

#: Same segment, but CAPTURING the token (or None when absent).  Readers that
#: KEY on the frame identity need this rather than OBS_TOKEN_PATTERN: gc2211's
#: five pointings reuse the same (visit, vgroup, exposure) tuples, so admitting
#: tokened names into a key that cannot hold the token trades "silently skips
#: those frames" for "silently overwrites them", which is not an improvement.
OBS_TOKEN_CAPTURE = r'(?:_(o\d{3}|j\d{4,5}))?'

#: Glob fragment covering the same optional segment.  `*` rather than the
#: regex: a glob cannot express "optional", and a bare `*` here would also span
#: `_visit001_vgroup...`, so the caller must keep the following literal.
OBS_TOKEN_GLOB = '*'


def perframe_name_re(band_pattern=r'[a-z0-9]+',
                     detector_pattern=r'nrc[ab](?:[0-9]|long)'):
    """Compiled regex for ``{band}_{detector}[_obs]_visit{NNN}_vgroup{G}_exp{N}``.

    Groups: ``(band, detector, visit, vgroup, exposure)``.  The observation
    token is matched but not captured -- readers that need it should use
    ``obs_token``-aware parsing rather than re-deriving it here.
    """
    return re.compile(
        rf'({band_pattern})_({detector_pattern}){OBS_TOKEN_PATTERN}'
        rf'_visit(\d+)_vgroup(\w+)_exp(\d+)')


def frame_identity(path, field=None):
    """``(visit_id, vgroup_id, exposure_id, detector)`` of a per-exposure product.

    The four tokens every per-frame writer names its output from, counted as
    underscore-separated fields of ``jw<PPPPP><OOO><VVV>_<vgroup>_<exp>_<det>_...``.

    Read from the BASENAME, always.  The indices count underscores from the
    left, so parsing the full path shifts every one of them as soon as the
    DIRECTORY contains an underscore -- and underscored field directories are
    ordinary now (the gc2211 per-observation split, ``cloudef_controlfield``).
    Measured with a basepath of ``/orange/adamginsburg/jwst/gc2211_o023/``, the
    path split gave visit ``'211'`` (off ``.../jwst/gc2211``), vgroup
    ``'o023/F200W/pipeline/jw02211023001'`` and detector ``'00001'``, so all
    four exposures of a filter shared one identity.  Three fields lost every m12
    shard to it (issue #472); this helper exists so the expression has one home
    instead of five copies, four of which still had the bug when it was fixed in
    the fifth (issue #477).

    ``field`` folds the observation number into ``vgroup_id`` for a JOINT
    multi-obs run (a field like ``'002-998'``), where two observations can share
    visit+vgroup+exposure -- sgrb2 obs998 reused obs002's mosaic tile numbers --
    and would otherwise write one another's outputs.  Left ``None``, or any
    field without a ``-``, the vgroup is returned unchanged.
    """
    base = os.path.basename(path)
    parts = base.split('_')
    visit_id = parts[0][-3:]
    vgroup_id = parts[1]
    exposure_id = parts[2]
    detector = parts[3]
    if field is not None and '-' in str(field):
        vgroup_id = f'{parts[0][-6:-3]}{vgroup_id}'
    return visit_id, vgroup_id, exposure_id, detector
