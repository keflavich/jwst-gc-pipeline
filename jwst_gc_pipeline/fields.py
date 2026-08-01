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
            obsids = {inst.lower(): tuple(sorted(ids))
                      for inst, ids in (obs.get('obsids') or {}).items()}
            unknown = set(obsids) - set(INSTRUMENTS)
            if unknown:
                raise FieldRegistryError(
                    f'{name}/{proposal} lists unknown instrument(s) '
                    f'{sorted(unknown)}; known: {list(INSTRUMENTS)}')
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
                offsets_table=obs.get('offsets_table'),
            ))
        loaded.append(Field(name=name, root=spec['root'],
                            observations=tuple(observations),
                            fov_region=spec.get('fov_region'),
                            roots=roots))
    return roots, tuple(sorted(loaded, key=lambda f: f.name))


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
    """
    instrument = instrument.lower()
    out = {}
    for field in FIELDS:
        obs = field.observation(proposal)
        if obs is None:
            continue
        for obsid in (tuple(obs.obsids.get(instrument, ()))
                      + tuple(obs.joint_obsids.get(instrument, ()))):
            if obsid in out:
                raise FieldRegistryError(
                    f'proposal {proposal} observation {obsid} ({instrument}) '
                    f'is claimed by both {out[obsid]!r} and {field.name!r}')
            out[obsid] = field.name
    return out


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
    seen = obs.obsids.get(instrument, ())
    return seen[0] if seen else None


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
    registers one.
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
