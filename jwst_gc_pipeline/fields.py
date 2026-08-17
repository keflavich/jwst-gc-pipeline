"""The fields this pipeline knows about, loaded from ``fields.yaml``.

Adding a target used to mean editing eight to ten dictionaries scattered across
six files, four of them inside functions where they could not be imported or
overridden. Nothing checked that they agreed, and several had drifted apart.
They are now one YAML file, and this module turns it into the lookups the
pipeline uses.

To add a target, edit ``fields.yaml``. Nothing here needs changing.

**Order in the YAML file means nothing.** Proposals come back sorted
numerically and filters sorted by wavelength, so where you write an entry has
no effect on what the pipeline does — including the SLURM array index, which is
a position in the merge job list.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import yaml

#: Instruments a field can be observed with, in the order views report them.
INSTRUMENTS = ('nircam', 'miri', 'niriss')

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'fields.yaml')


class FieldRegistryError(ValueError):
    """The registry file says something the pipeline cannot act on."""


#: An observation block may declare ``obsids: {nircam: '*'}``: every
#: observation of the proposal belongs to this field for that instrument.
#:
#: Written for 10678, the GC Treasury, where one field owns the whole
#: proposal.  The observation numbers ARE published --
#: ``Observations.query_criteria(proposal_id='10678')`` returns 1668 planned
#: exposure-level rows carrying 139 distinct observation numbers, 001..139
#: contiguous, the same set for NIRCAM/IMAGE and MIRI/IMAGE (checked
#: 2026-08-17; ``t_min`` is NaN on every row, so none has executed).  The
#: wildcard records "one field owns them all", which stays true through a
#: replan that renumbers or adds observations, and spares the registry a
#: 139-entry list that each replan re-issues.  The cost it accepts is that an
#: obsid outside the plan resolves to this field where a list would raise, so
#: a proposal whose observations are split between fields enumerates them.
WILDCARD_OBSID = '*'

#: A wildcard says "every observation of this proposal", and the registry does
#: not record how many that is.  Where a count is used only as "one
#: observation, or several?" -- ``filter_observation_count``, whose one caller
#: is the m2 (second merge iteration) foreign-observation filter -- the answer
#: for a wildcard is
#: "several", so it contributes this rather than ``len(('*',)) == 1``.
WILDCARD_OBSERVATION_COUNT = 2

#: What a concrete observation number looks like: three digits, or several of
#: them joined by '-' for a joint-obsid token ('002-998').  Every obsid in
#: fields.yaml has this shape.  The wildcard resolves only keys that match, so
#: a typo or a module name ('nrcb', '0001') raises ``KeyError`` from the
#: mapping instead of being absorbed into the catch-all owner.
OBSID_RE = re.compile(r'\d{3}(?:-\d{3})*')


def is_obsid(token):
    """True when ``token`` has the shape of an observation number.

    Shape only.  A wildcard field declares no obsid list, so shape is all the
    registry gives to validate a key against.
    """
    return bool(OBSID_RE.fullmatch(str(token)))


class WildcardObsidMap(dict):
    """``{obsid: target}`` whose lookups fall back to a ``'*'`` catch-all owner.

    The reduce drivers hard-index ``mapping[obsid]`` with concrete observation
    numbers, so a wildcard-owning proposal has to resolve those through an
    ordinary dict interface: ``[]``, ``get`` and ``in`` all fall back to the
    wildcard owner when the concrete key is absent.  An explicit entry from
    another field still wins over the wildcard.

    The fallback applies only to obsid-SHAPED keys (``is_obsid``).  Resolving
    anything at all would make ``'nrcb' in mapping`` true and
    ``target_for_obsid(proposal, 'typo')`` answer the wildcard owner, so a
    misspelling would be absorbed rather than raised.
    """

    def __init__(self, mapping=(), wildcard_target=None):
        super().__init__(mapping)
        self.wildcard_target = wildcard_target

    def _wildcard_for(self, key):
        """The wildcard owner if it claims ``key``, else None."""
        if self.wildcard_target is None or not is_obsid(key):
            return None
        return self.wildcard_target

    def __missing__(self, key):
        owner = self._wildcard_for(key)
        if owner is None:
            raise KeyError(key)
        return owner

    def __contains__(self, key):
        return super().__contains__(key) or self._wildcard_for(key) is not None

    def get(self, key, default=None):
        if super().__contains__(key):
            return super().__getitem__(key)
        owner = self._wildcard_for(key)
        return default if owner is None else owner

    def copy(self):
        """A copy that is still wildcard-resolving.

        ``dict.copy`` returns a plain ``dict``, which silently drops the
        fallback and restores the ``KeyError`` this class exists to prevent.
        """
        return WildcardObsidMap(self, wildcard_target=self.wildcard_target)

    __copy__ = copy


def _wavelength_key(filtername):
    """Sort key putting filters in wavelength order: f115w, f405n, f2550w.

    Alphabetical would read f1130w before f187n, which is neither the order a
    person expects nor the order a colour-ordered catalog wants.
    """
    match = re.match(r'f(\d+)', filtername.lower())
    return (int(match.group(1)) if match else 10 ** 6, filtername.lower())


@dataclass(frozen=True)
class Obs:
    """One proposal's observations of one field."""

    proposal: str
    #: Every observation number, per instrument, that images this field.
    #: ``('*',)`` claims every observation of the proposal for that instrument;
    #: at most one field may hold the wildcard per (proposal, instrument).
    obsids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Tokens naming several observations cataloged in one run, e.g. '002-998'.
    joint_obsids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: The observation number the merge builds file globs from, per instrument.
    #: Defaults to the only entry in ``obsids``; ``'*'`` matches several.
    glob_obsids: Dict[str, str] = field(default_factory=dict)
    nvisits: Optional[int] = None
    filters: Tuple[str, ...] = ()
    niriss_filters: Tuple[str, ...] = ()
    #: Astrometric frame token (VIRAC2 / Gaia).  Names the offsets table.
    reference_frame: Optional[str] = None
    #: Observation number -> reference catalog files, relative to the field
    #: directory, in preference order.  Different observations of one proposal
    #: can sit at different epochs, so this is per observation; MIRI and NIRISS
    #: list several candidates and take the first one present on disk.
    reference_catalogs: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: The rare per-filter override of the above: obsid -> filter -> file.
    reference_catalogs_by_filter: Dict[str, Dict[str, str]] = field(default_factory=dict)
    #: Reference catalog files consulted for any observation that has no exact
    #: ``reference_catalogs`` key, in preference order.  What makes a
    #: wildcard-obsid proposal tie-able: it declares no obsid list, so there
    #: is nothing to hang per-obsid keys on.
    default_reference_catalog: Tuple[str, ...] = ()
    #: Path to the measured astrometric offsets, relative to the field
    #: directory. Measured from the data once and then fixed.
    offsets_table: Optional[str] = None

    def glob_obsid(self, instrument='nircam'):
        """The observation number to build a filename glob from.

        ``None`` when this proposal did not observe the field with this
        instrument, which callers must treat as "no data", never as a value to
        interpolate: ``jw02526-oNone_*`` matches nothing silently.
        """
        instrument = instrument.lower()
        if instrument in self.glob_obsids:
            return self.glob_obsids[instrument]
        seen = self.obsids.get(instrument, ())
        if len(seen) == 1:
            return seen[0]
        if len(seen) > 1:
            return '*'
        return None


@dataclass(frozen=True)
class Field:
    """One target, and every observation of it."""

    name: str
    root: str
    observations: Tuple[Obs, ...] = ()
    fov_region: Optional[str] = None
    #: The roots block this field was loaded with.  Carried rather than read
    #: from the module global, so a registry loaded from another file resolves
    #: against its own roots.
    roots: Optional[Dict[str, str]] = None

    @property
    def basepath(self):
        roots = self.roots if self.roots is not None else ROOTS
        return f'{roots[self.root]}/{self.name}/'

    def observation(self, proposal):
        """The ``Obs`` for one proposal, or ``None``."""
        for obs in self.observations:
            if obs.proposal == str(proposal):
                return obs
        return None


def _load(path=REGISTRY_PATH):
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    roots = dict(raw['roots'])
    loaded = []
    for name, spec in raw['fields'].items():
        if spec['root'] not in roots:
            raise FieldRegistryError(
                f"field {name!r} has root {spec['root']!r}; "
                f"the roots block defines {sorted(roots)}")
        observations = []
        for proposal, obs in sorted((spec.get('observations') or {}).items(),
                                    key=lambda kv: int(kv[0])):
            # `nircam: '*'` is a scalar; sorted('*') would still give ('*',)
            # but only by the accident of it being one character.  Any OTHER
            # scalar is refused rather than silently exploded: `nircam: '001'`
            # would load as ('0', '0', '1'), and the wildcard documented above
            # makes a scalar look like a supported spelling.
            obsids = {}
            for inst, ids in (obs.get('obsids') or {}).items():
                if ids == WILDCARD_OBSID:
                    obsids[inst.lower()] = (WILDCARD_OBSID,)
                elif isinstance(ids, (str, bytes, int)):
                    raise FieldRegistryError(
                        f'{name}/{proposal} obsids.{inst} is the scalar '
                        f'{ids!r}; write a list ([{ids!r}]) or the wildcard '
                        f"{WILDCARD_OBSID!r}.  A bare string loads as its "
                        f'individual characters.')
                else:
                    obsids[inst.lower()] = tuple(sorted(ids))
            unknown = set(obsids) - set(INSTRUMENTS)
            if unknown:
                raise FieldRegistryError(
                    f'{name}/{proposal} lists unknown instrument(s) '
                    f'{sorted(unknown)}; known: {list(INSTRUMENTS)}')
            default_refcat = obs.get('default_reference_catalog') or ()
            if not isinstance(default_refcat, (list, tuple)):
                default_refcat = (default_refcat,)
            observations.append(Obs(
                proposal=str(proposal),
                obsids=obsids,
                joint_obsids={inst.lower(): tuple(sorted(ids)) for inst, ids
                              in (obs.get('joint_obsids') or {}).items()},
                glob_obsids={inst.lower(): str(v) for inst, v
                             in (obs.get('glob_obsid') or {}).items()},
                nvisits=obs.get('nvisits'),
                filters=tuple(sorted(obs.get('filters') or (),
                                     key=_wavelength_key)),
                niriss_filters=tuple(sorted(obs.get('niriss_filters') or (),
                                            key=_wavelength_key)),
                reference_frame=obs.get('reference_frame'),
                reference_catalogs={
                    str(k): tuple(v) if isinstance(v, (list, tuple)) else (v,)
                    for k, v in (obs.get('reference_catalog') or {}).items()},
                reference_catalogs_by_filter={
                    str(k): {f.lower(): v for f, v in fd.items()}
                    for k, fd in (obs.get('reference_catalog_by_filter') or {}).items()},
                default_reference_catalog=tuple(default_refcat),
                offsets_table=obs.get('offsets_table'),
            ))
        loaded.append(Field(name=name, root=spec['root'],
                            observations=tuple(observations),
                            fov_region=spec.get('fov_region'),
                            roots=roots))
    _assert_one_wildcard_owner(loaded)
    return roots, tuple(sorted(loaded, key=lambda f: f.name))


def _assert_one_wildcard_owner(loaded):
    """At most one field may claim ``'*'`` per (proposal, instrument).

    ``docs/FIELDS.md`` states this as a property of the registry, so it is
    checked when the registry loads.  Checking it only inside
    ``field_to_reg_mapping`` makes it a property of one lookup: a file with two
    wildcard owners imports clean, every instrument nobody asked about stays
    silent, and the contradiction surfaces on whichever run happens to query
    the clashing instrument first.
    """
    owners = {}
    for field in loaded:
        for obs in field.observations:
            for inst, ids in obs.obsids.items():
                if WILDCARD_OBSID not in ids:
                    continue
                key = (obs.proposal, inst)
                if key in owners and owners[key] != field.name:
                    raise FieldRegistryError(
                        f'proposal {obs.proposal} ({inst}) has two wildcard '
                        f'obsid owners: {owners[key]!r} and {field.name!r}. '
                        f"'*' claims every observation, which only one field "
                        f'can do.')
                owners[key] = field.name


ROOTS, FIELDS = _load()
BY_NAME = {f.name: f for f in FIELDS}


# --------------------------------------------------------------------------
# Views.  Each one answers a question the pipeline used to answer with its own
# hand-maintained dictionary.
# --------------------------------------------------------------------------

def fields_basepath(target):
    """The data directory for one target.

    An unregistered target gets the blue tree, which is what the ``if target in
    (...)`` branches this replaces did in their ``else``.
    """
    known = BY_NAME.get(target)
    if known is not None:
        return known.basepath
    return f"{ROOTS['blue']}/{target}/"


basepath = fields_basepath


def obs_filters(instrument='nircam'):
    """``{target: {proposal: [filters]}}``.

    NIRISS reuses NIRCam filter names on a different pixel scale, so it has its
    own set and its own products; asking for it returns only the fields that
    have one.
    """
    if instrument.lower() == 'niriss':
        return {f.name: {o.proposal: list(o.niriss_filters)
                         for o in f.observations if o.niriss_filters}
                for f in FIELDS
                if any(o.niriss_filters for o in f.observations)}
    return {f.name: {o.proposal: list(o.filters)
                     for o in f.observations if o.filters}
            for f in FIELDS if any(o.filters for o in f.observations)}


#: The instruments that share a field's per-observation ``filters`` list and
#: live in the same ``{BASE}/{field}/{FILTER}/pipeline/`` tree.  NIRISS declares
#: its bands separately AND reduces to a different layout, so a caller that
#: enumerates that tree must not be handed NIRISS names.
NIRCAM_MIRI = ('nircam', 'miri')


def declared_filters(target, instruments=NIRCAM_MIRI):
    """Filters ``fields.yaml`` declares for ``target``, upper-cased, across all
    of its observations.  Empty set for an unregistered target.

    Instrument-aware, and the default is deliberate.  NIRCam and MIRI share the
    per-obs ``filters`` list and the ``{FILTER}/pipeline/`` layout; NIRISS has
    its own ``niriss_filters`` list and its own layout.  Unioning them makes a
    NIRISS-only band look like a NIRCam band whose directory is missing --
    sgrc declares F158M/F200W/F356W for NIRISS and none of them can ever have a
    NIRCam pipeline directory, so a caller enumerating that tree would block
    sgrc at release with no reduction able to clear it.

    Used to tell a real band whose reduction produced nothing (declared, must
    block) from an undeclared leftover directory (skip): a name absent here was
    never expected on disk, so an empty directory for it is not a band.
    """
    fobj = BY_NAME.get(target)
    if fobj is None:
        return set()
    want = set(instruments)
    out = set()
    for o in fobj.observations:
        if want & set(NIRCAM_MIRI):
            out |= {f.upper() for f in o.filters}
        if 'niriss' in want:
            out |= {f.upper() for f in o.niriss_filters}
    return out


def filter_observation_count(target, filtername, instrument=None):
    """How many ``(proposal, observation)`` pairs of ``target`` declare
    ``filtername``.

    The per-frame catalog basename carries an observation token only on files
    written since that token was introduced.  A pre-token basename is ambiguous
    ONLY when more than one observation could have produced it -- gc2211 has one
    proposal over five observations that all image F200W, so an untokened
    ``f200w_..._visit001_...`` could be any of them.  ngc6334 F090W is the
    opposite case: 6778 declares it and 7213 does not, so every F090W catalog on
    disk is 6778's whatever its name, and discarding the untokened ones would
    throw away real exposures (nrca exists ONLY under the pre-token name).

    A ``'*'`` (wildcard) obsid list contributes
    ``WILDCARD_OBSERVATION_COUNT`` (2), not 1: it claims every observation of
    the proposal, so the count is "several" and the registry does not record
    how many.  The exact number would matter only to a caller asking "how
    many", and the one caller (``cataloging._drop_foreign_obs_duplicates``)
    asks "one, or several?".

    ``instrument`` defaults to the one the filter NAME implies (MIRI bands are
    counted against MIRI's observations, everything else against NIRCam's);
    pass it explicitly for NIRISS.

    Returns 0 for an unregistered target or a filter nothing declares.
    """
    from jwst_gc_pipeline.photometry.naming import MIRI_FILTERS

    fobj = BY_NAME.get(target)
    if fobj is None:
        return 0
    want = str(filtername or '').upper()
    # `Observation.filters` is the shared NIRCAM_MIRI list, so the filter NAME is
    # what says which instrument's observations to count -- the same split
    # `monitoring.scan` makes.  Counting a MIRI filter against the NIRCam obsids
    # gets it wrong in both directions (sgrb2 F770W: 1 NIRCam obs vs 3 MIRI;
    # cloudef F770W: 2 vs 3).  An explicit `instrument` still wins, so a NIRISS
    # caller can ask for the NIRISS list rather than inherit a NIRCam count for
    # a band whose name both instruments use.
    if instrument is None:
        # MIRI_FILTERS is spelled lower-case; `want` is upper-cased above.
        instrument = 'miri' if want.lower() in MIRI_FILTERS else 'nircam'
    n = 0
    for o in fobj.observations:
        names = {f.upper() for f in (o.niriss_filters if instrument == 'niriss'
                                     else o.filters)}
        if want not in names:
            continue
        # `joint_obsids` is deliberately NOT added: its members are already in
        # `obsids` (sgrb2 miri lists 001/002/998 and joins 002-998), so adding
        # it would count the same observation twice.
        ids = tuple(o.obsids.get(instrument, ()))
        if WILDCARD_OBSID in ids:
            # '*' claims EVERY observation of the proposal, so this filter is
            # shared by definition -- `len(('*',)) == 1` would read as "one
            # observation images it", which switches the m2 foreign-observation
            # filter off and lets one tile's per-frame catalogs stand in for
            # another's (the gc2211 #259/#298 class).
            n += WILDCARD_OBSERVATION_COUNT
            continue
        # A proposal that declares the filter but lists no obsid for this
        # instrument still counts as one observation: the registry cannot say
        # how many, and 1 means "not shared", which keeps every catalog.
        n += len(ids) or 1
    return n


def claims_every_observation(target, instrument='nircam'):
    """Does ``target`` claim every observation of one of its proposals?

    True when any observation block declares ``obsids: {<instrument>: '*'}``.

    Callers that enumerate ``obsids`` to ask "does this field have several
    observations?" get ONE entry from a wildcard -- the literal ``'*'`` -- and
    read that as "a single observation", which is the answer that switches
    ambiguity handling off.  This says so directly, without pretending to know
    how many.
    """
    known = BY_NAME.get(target)
    if known is None:
        return False
    instrument = instrument.lower()
    return any(WILDCARD_OBSID in o.obsids.get(instrument, ())
               for o in known.observations)


def glob_obsid(target, proposal, instrument='nircam'):
    """The observation number to build a filename glob from, or ``None``.

    Instrument-aware on purpose. Proposal 2221 numbers its NIRCam and MIRI
    observations of the same two fields in opposite order (brick is NIRCam 001
    and MIRI 002; cloudc is the reverse), so one number per (target, proposal)
    cannot be right for both.
    """
    field = BY_NAME.get(target)
    if field is None:
        return None
    obs = field.observation(proposal)
    return None if obs is None else obs.glob_obsid(instrument)


def project_obsnum():
    """``{target: {proposal: obsid}}`` for NIRCam.

    A proposal that did not observe a field with NIRCam is omitted rather than
    given ``None``: consumers interpolate this into a glob, where ``None``
    would match nothing without saying so.
    """
    out = {}
    for field in FIELDS:
        entries = {o.proposal: o.glob_obsid('nircam') for o in field.observations}
        entries = {k: v for k, v in entries.items() if v is not None}
        if entries:
            out[field.name] = entries
    return out


def nvisits():
    """``{proposal: {target: n}}`` -- the transpose of the other two, which is
    a thing a reader had to remember and a view does not."""
    out = {}
    for field in FIELDS:
        for obs in field.observations:
            if obs.nvisits is not None:
                out.setdefault(obs.proposal, {})[field.name] = obs.nvisits
    return out


def field_to_reg_mapping(proposal, instrument='nircam'):
    """``{obsid: target}`` for one proposal and instrument.

    The reduce and catalog drivers use this to name the field an observation
    belongs to.

    A field declaring the ``'*'`` wildcard owns every observation of the
    proposal for that instrument, and the returned mapping resolves any
    concrete obsid through it, so the drivers' ``mapping[obsid]`` works for
    observation numbers that land after the registry was written.  An explicit
    entry from another field still wins; two fields claiming the wildcard for
    one (proposal, instrument) is an error.
    """
    instrument = instrument.lower()
    out = {}
    wildcard_owner = None
    for field in FIELDS:
        obs = field.observation(proposal)
        if obs is None:
            continue
        for obsid in (tuple(obs.obsids.get(instrument, ()))
                      + tuple(obs.joint_obsids.get(instrument, ()))):
            if obsid == WILDCARD_OBSID:
                if wildcard_owner is not None and wildcard_owner != field.name:
                    raise FieldRegistryError(
                        f'proposal {proposal} ({instrument}) has two wildcard '
                        f'obsid owners: {wildcard_owner!r} and {field.name!r}. '
                        f"'*' claims every observation, which only one field "
                        f'can do.')
                wildcard_owner = field.name
                continue
            if obsid in out:
                raise FieldRegistryError(
                    f'proposal {proposal} observation {obsid} ({instrument}) '
                    f'is claimed by both {out[obsid]!r} and {field.name!r}')
            out[obsid] = field.name
    if wildcard_owner is not None:
        out[WILDCARD_OBSID] = wildcard_owner
    return WildcardObsidMap(out, wildcard_target=wildcard_owner)


def target_for_obsid(proposal, obsid, instrument='nircam'):
    """Which field an observation belongs to.

    ``obsid`` may name several observations cataloged together, as
    ``'001-002'``; every part must belong to the same field.
    """
    mapping = field_to_reg_mapping(proposal, instrument)
    parts = str(obsid).split('-')
    targets = {mapping[p] for p in parts if p in mapping}
    if len(targets) == 1:
        return targets.pop()
    if not targets:
        raise KeyError(
            f'proposal {proposal} observation {obsid!r} ({instrument}) is not '
            f'in fields.yaml; known observations: {sorted(mapping)}')
    raise FieldRegistryError(
        f'joint observation {obsid!r} spans more than one field: '
        f'{sorted(targets)}')


def default_field_token(target, proposal, instrument='nircam'):
    """The --field value to use for a target when none was given.

    A joint token wins: Sgr B2's MIRI observations 002 and 998 tile the field
    between them and are cataloged as '002-998', so picking either one alone
    would catalog half a mosaic.  Relying on which key an inverted dict happened
    to keep is how that used to be decided.

    The ``'*'`` wildcard is filtered out, so a field that declares only the
    wildcard answers ``None``.  ``'*'`` is a registry token, not an
    observation number, and the caller interpolates this value into product
    names as ``-o{field}``: the literal ``-o*`` would be WRITTEN into mosaic
    and catalog filenames that no reader globs back.  ``None`` sends the
    caller to its "name the observation" error instead.
    """
    known = BY_NAME.get(target)
    if known is None:
        return None
    obs = known.observation(proposal)
    if obs is None:
        return None
    instrument = instrument.lower()
    joint = obs.joint_obsids.get(instrument, ())
    if joint:
        return joint[0]
    seen = tuple(o for o in obs.obsids.get(instrument, ())
                 if o != WILDCARD_OBSID)
    return seen[0] if seen else None


def field_token_for_run(target, proposal, instrument='nircam'):
    """The observation token a run names its products with, or raise.

    The catalog driver interpolates this into ~40 product-name f-strings as
    ``-o{token}``, so it has to be a real observation number.  Two sources,
    in order: the registry's default for this (target, proposal, instrument),
    then the field name's entry in the inverted ``field_to_reg_mapping`` --
    the historical fallback, with the ``'*'`` key removed so a wildcard-owning
    proposal cannot supply it.

    Raises ``FieldRegistryError`` when neither answers.  A field that claims
    every observation of its proposal has no default by construction, and the
    value it used to yield was the literal ``'*'``: names like
    ``jw010678-o*_t001_nircam_..._i2d.fits`` written to disk.
    """
    token = default_field_token(target, proposal, instrument)
    if token:
        return str(token)
    inverted = {v: k for k, v in field_to_reg_mapping(proposal, instrument).items()
                if k != WILDCARD_OBSID}
    token = inverted.get(target)
    if token:
        return str(token)
    raise FieldRegistryError(
        f'{target}/{proposal} ({instrument}) does not name one observation to '
        f'work on, so --field is required.  A field that claims every '
        f"observation of its proposal (fields.yaml obsids: '*') has no "
        f'default: pass --field <obsid>, e.g. --field 042.')


def reference_frame(proposal):
    """The astrometric frame token for one proposal, or ``None``.

    The token names the offsets table and the realignment gate.  The catalog
    the alignment is tied TO is ``reference_catalog_path``.

    Keyed by proposal because the offsets-table filename it builds is, and
    every field sharing a proposal shares that frame today.
    """
    seen = {}
    for known in FIELDS:
        obs = known.observation(proposal)
        if obs is not None and obs.reference_frame is not None:
            seen.setdefault(obs.reference_frame, []).append(known.name)
    if len(seen) > 1:
        raise FieldRegistryError(
            f'proposal {proposal} is given more than one reference frame: '
            f'{ {k: sorted(v) for k, v in seen.items()} }.  The frame token '
            f'names the offsets table, which is per proposal.')
    return next(iter(seen), None)


def reference_catalog_candidates(proposal, obsid, filtername=None,
                                 basepath=None, target=None,
                                 instrument='nircam'):
    """Every registered reference catalog for one observation, best first.

    MIRI and NIRISS register several and take the first present on disk; NIRCam
    registers one.  An exact ``reference_catalog`` key for the obsid wins;
    ``default_reference_catalog`` answers for any observation without one,
    which is how a wildcard-obsid proposal registers a catalog for observations
    whose numbers were unknown when the registry was written.
    """
    obsid = str(obsid)
    if target is None:
        # Instrument-dependent: proposal 2221 observation 001 is brick under
        # NIRCam and cloudc under MIRI.
        target = target_for_obsid(proposal, obsid, instrument)
    known = BY_NAME.get(target)
    obs = None if known is None else known.observation(proposal)
    if obs is None:
        raise FieldRegistryError(f'{target}/{proposal} is not in fields.yaml')
    relative = ()
    if filtername is not None:
        one = (obs.reference_catalogs_by_filter
               .get(obsid, {}).get(str(filtername).lower()))
        if one is not None:
            relative = (one,)
    if not relative:
        relative = obs.reference_catalogs.get(obsid, ())
    if not relative:
        relative = obs.default_reference_catalog
    if not relative:
        raise FieldRegistryError(
            f'no reference catalog registered for {target} proposal {proposal} '
            f'observation {obsid}.  Add it to fields.yaml under '
            f'fields.{target}.observations.{proposal!r}:\n'
            f'        reference_catalog:\n'
            f'          {obsid!r}: catalogs/<your-refcat>.fits')
    if basepath is None:
        basepath = fields_basepath(target)
    return [os.path.join(basepath, r) for r in relative]


def reference_catalog_path(proposal, obsid, filtername=None, basepath=None,
                           target=None, instrument='nircam'):
    """The reference catalog for one observation: the first candidate that
    exists on disk, else the best-registered one.

    Returning the best-registered path when none exists lets the caller report
    the file it wanted.
    """
    candidates = reference_catalog_candidates(
        proposal, obsid, filtername=filtername, basepath=basepath,
        target=target, instrument=instrument)
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def fov_region(target):
    """Path to the field-of-view region file, relative to the field directory.

    Read by the NIRCam driver.
    """
    known = BY_NAME.get(target)
    return None if known is None else known.fov_region


def offsets_table_relpath(target, proposal):
    """A field's offsets table, relative to its directory, or ``None``.

    Relative on purpose: the caller joins it to the basepath it is actually
    using, so a run redirected by ``GC_BASEPATH_OVERRIDE`` reads the table in
    the tree it is working in.
    """
    known = BY_NAME.get(target)
    if known is None:
        return None
    obs = known.observation(proposal)
    return None if obs is None else obs.offsets_table


def offsets_table_path(target, proposal, basepath=None):
    """Path to a field's measured offsets table, or ``None``.

    ``basepath`` is the directory the caller is working in; it defaults to the
    field's registered one.  Returns a path, not a read table: reading it is
    the caller's job, and doing it eagerly made every run of every target
    depend on one field's file.
    """
    relative = offsets_table_relpath(target, proposal)
    if relative is None:
        return None
    if basepath is None:
        known = BY_NAME.get(target)
        if known is None:
            return None
        basepath = known.basepath
    return os.path.join(basepath, relative)


def merge_jobs(target, instrument='nircam'):
    """``[(proposal, filter), ...]`` for one target, in a fixed order.

    This list is the SLURM array: task *n* runs entry *n*. The order is derived
    (proposals numerically, filters by wavelength) rather than taken from the
    file, so editing the registry cannot move a task onto a different filter.
    """
    per_target = obs_filters(instrument).get(target)
    if not per_target:
        raise KeyError(
            f'{target!r} has no {instrument} filters in fields.yaml, so there '
            f'is nothing to merge.  Registered: '
            f'{sorted(obs_filters(instrument))}')
    return [(proposal, filtername)
            for proposal in sorted(per_target, key=int)
            for filtername in per_target[proposal]]
