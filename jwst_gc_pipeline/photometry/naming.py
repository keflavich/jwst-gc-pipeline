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
#: What the registry writes for "every observation of this proposal".
WILDCARD_OBSID = '*'

MULTIOBS_PROPOSALS = ('2211', '10678')


def proposal_is_multiobs(proposal_id):
    """Does this proposal put more than one observation under ONE basepath?

    DERIVED from the field registry, with ``MULTIOBS_PROPOSALS`` as a floor.
    The hand-maintained tuple listed 2211 and 10678 because those are the two
    someone noticed; the registry says EIGHT more proposals register several
    obsids against a single field, and every one of them writes per-frame
    catalogs whose names carry visit, vgroup, exposure and detector but not the
    observation:

        2092 cloudef   002 004 005 006 008
        5365 sgrb2     001 002 998
        3958 sickle    001 002 007
        6151 w51       001 002        1905 wd1   001 003
        3523 wd2       003 005        1979 m4    002 003
        2221           001 002   (brick and cloudc, separate basepaths)

    cloudef is the proven case and the cost is data loss, not a nuisance:
    obs 002 and obs 005 both use visit 001 / vgroup 02101, so the later run
    overwrote the earlier one.  Of ~64 obs-005 m2 catalogs per short-wavelength
    filter, 8 survive; F480M has none, and its offsets table cannot be rebuilt
    at all because `build_virac2_offsets` refuses to relabel obs 002's catalogs
    as obs 005's.

    "Multi-obs" is narrower than "multi-pointing".  1182 is ONE observation
    (004) with two visits -- separate pointings on different dates, the pair
    behind the brick-1182 v001 ~20" offset -- and it needs no token because the
    per-frame name already carries `visit001` / `visit002`.  What cloudef hit is
    two OBSERVATIONS that both restart at visit 001 and vgroup 02101, so the
    visit field does not separate them and nothing else in the name does either.

    Keeping the tuple as a floor means a proposal that is multi-obs in practice
    but not yet in the registry keeps its token.

    The registry import is deferred: this module is deliberately import-light so
    `merge_catalogs` can consult it without pulling in
    `crowdsource_catalogs_long`.
    """
    prop = str(proposal_id)
    if prop in MULTIOBS_PROPOSALS:
        return True
    try:
        from jwst_gc_pipeline import fields as _fields
    except ImportError:
        return False
    for fld in getattr(_fields, 'FIELDS', ()):
        for obs in getattr(fld, 'observations', ()):
            if str(getattr(obs, 'proposal', '')) != prop:
                continue
            seen = set()
            for ids in (getattr(obs, 'obsids', None) or {}).values():
                seen |= {str(i) for i in ids}
            for ids in (getattr(obs, 'glob_obsids', None) or {}).values():
                seen |= {str(i) for i in ids}
            # A wildcard entry claims every observation of the proposal, which
            # is the multi-obs case by construction.
            if WILDCARD_OBSID in seen or len(seen - {WILDCARD_OBSID}) > 1:
                return True
    return False

#: The subset whose MERGED catalogs are per-observation too.  gc2211 is multi-
#: obs at the per-frame level but pools all five pointings into one untokened
#: merged catalog by design; at 139 tiles that pooling is itself the corruption
#: mode, so 10678 scopes the merged catalogs to one observation as well.
PER_OBS_MERGED_PROPOSALS = ('10678',)

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
    multiobs = proposal_is_multiobs(proposal_id)
    miri = (str(module).lower() == 'mirimage'
            or _instrument_from_filter(filtername) == 'MIRI')
    return (token if (miri or multiobs) else ''), (token if multiobs else '')


def merged_catalog_obs_token(proposal_id, field):
    """Observation token baked into the MERGED catalog names, post-module slot.

    Only ``PER_OBS_MERGED_PROPOSALS`` get one: 10678's 139 tiles share the
    gc-treasury tree, and pooling another tile's frames into a merge is the
    corruption class the obs scoping exists to prevent, so every per-filter
    merged catalog is scoped to one observation
    (``{filt}_{module}_o{field}_indivexp_merged...``).  gc2211 keeps its
    all-obs UNTOKENED merged names ('' here): its five pointings are pooled at
    merge by design (the ``_o*`` glob in ``merge_individual_frames``) and
    scoped afterwards at the vetting step (``_vtok`` in cataloging.py).
    Writers (``merge_individual_frames``' ``out_obs_``) and every reader of a
    merged-catalog name (``cataloging.merged_catalog_path``, the m7 seed
    reader, ``merge_daophot``'s input glob) must agree on this token.

    ``field`` goes through ``observation_field_token``, which normalises the
    spelling and refuses a field that does not name an observation.
    """
    if str(proposal_id) in PER_OBS_MERGED_PROPOSALS and field not in (None, ''):
        return f'_o{observation_field_token(field)}'
    return ''


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
