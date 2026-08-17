"""Locate the data products a field's diagnostic write-up needs.

Every field on disk follows the same layout, so discovery is a matter of
globbing and then *ranking* the candidates -- there are usually several
generations of a given product and the write-up must describe the most
recent one, not whichever the glob happened to return first.

Naming conventions this module encodes (see ``PHOTOMETRY_PIPELINE.md``):

cross-band merge
    ``catalogs/basic_<module>_indivexp_photometry_tables_merged[_resbgsub]_m<N>.fits``
per-filter merge
    ``catalogs/<filt>_<module>_indivexp_merged[_resbgsub]_m<N>_dao_basic.fits``
mosaic
    ``<FILT>/pipeline/jw<prop>-o<obs>_t*_<instr>_*-<filt>-<module>_i2d.fits``

Higher merge stage wins, then ``resbgsub`` over the plain token, then the
``merged`` module over a single-module product, then a per-proposal
``_j<proposal>`` collision-fix variant over the un-tokenized collision
product, then mtime.  Files matching ``_DERIVATIVE_RE`` (``_qualcuts``,
``_vetted``, ``_allcols``, ``_i2dseed``, ...) are post-hoc derivatives and
are skipped.  ``_dedup`` is *not* in that list: it is a merge-stage product,
matched by ``_CROSSBAND_RE`` and *preferred* at m8 (the m8 de-duplication is
part of the merge, not a filter applied after it).
"""

import os
import re
from dataclasses import dataclass, field as _dcfield

from jwst_gc_pipeline import fields as _fields
from jwst_gc_pipeline.mast_names import jw_prefix

# Post-hoc derivatives that are never the canonical product.
_DERIVATIVE_RE = re.compile(
    r'_(qualcuts|vetted|allcols|i2dseed|nirspeccand|filtered|seshat|partial'
    r'|abfix|ok\d|backup)', re.IGNORECASE)

_CROSSBAND_RE = re.compile(
    r'^basic_(?P<module>[a-z0-9-]+)_indivexp_photometry_tables_merged'
    r'(?P<resbg>_resbgsub)?(?:_m(?P<stage>\d+))?(?P<dedup>_dedup)?\.fits$')

# The module token can carry a per-proposal ``_j<proposal>`` suffix (e.g.
# ``nrca_j7213``), the products written by the shared-filter-collision fix.
# Without the optional ``_j\d+`` the module group ``[a-z0-9-]+`` cannot match
# the underscore and those products are invisible, leaving only the
# un-tokenized collision product on disk to be picked.
_PERFILTER_RE = re.compile(
    r'^(?P<filt>f\d{3}[a-z]\d?)_(?P<module>[a-z0-9-]+(?:_j\d+)?)_indivexp_merged'
    r'(?P<resbg>_resbgsub)?_m(?P<stage>\d+)_dao_basic\.fits$')


def _rank(match, path):
    """Sort key; larger is better.  See the module docstring."""
    stage = int(match.group('stage') or 0)
    resbg = 1 if match.group('resbg') else 0
    dedup = 1 if (match.groupdict().get('dedup') and stage >= 8) else 0
    module = match.group('module') if 'module' in match.groupdict() else ''
    merged = 1 if module == 'merged' else 0
    # Prefer a per-proposal collision-fix variant over the un-tokenized
    # collision product it was written to replace.
    jtok = 1 if '_j' in (module or '') else 0
    return (stage, dedup, resbg, merged, jtok, os.path.getmtime(path))


def _module_siblings(catdir, filt, chosen_module):
    """Per-filter module tokens on disk for *filt*, excluding *chosen_module*."""
    if not os.path.isdir(catdir):
        return set()
    mods = set()
    for name in os.listdir(catdir):
        if _DERIVATIVE_RE.search(name):
            continue
        m = _PERFILTER_RE.match(name)
        if m is not None and m.group('filt') == filt:
            mods.add(m.group('module'))
    mods.discard(chosen_module)
    return mods


def _best(catdir, regex, **constraints):
    """Highest-ranked non-derivative file in *catdir* matching *regex*."""
    if not os.path.isdir(catdir):
        return None
    best = None
    for name in os.listdir(catdir):
        if _DERIVATIVE_RE.search(name):
            continue
        match = regex.match(name)
        if match is None:
            continue
        if any(match.group(k) != v for k, v in constraints.items()):
            continue
        path = os.path.join(catdir, name)
        key = _rank(match, path)
        if best is None or key > best[0]:
            best = (key, path, match)
    if best is None:
        return None
    return best[1], best[2]


@dataclass
class FieldInventory:
    """What is on disk for one field, and what the write-up can therefore say."""

    name: str
    basepath: str
    filters: tuple = ()
    proposals: tuple = ()
    crossband_catalog: str = None
    crossband_stage: int = None
    crossband_resbgsub: bool = False
    per_filter_catalogs: dict = _dcfield(default_factory=dict)
    mosaics: dict = _dcfield(default_factory=dict)
    reference_catalogs: dict = _dcfield(default_factory=dict)
    notes: list = _dcfield(default_factory=list)

    @property
    def has_crossband(self):
        return self.crossband_catalog is not None

    @property
    def measured_filters(self):
        """Filters with a per-filter catalog, in wavelength order."""
        return tuple(f for f in self.filters if f in self.per_filter_catalogs)

    def summary_rows(self):
        """(filter, catalog, mosaic) presence table, for the LaTeX write-up."""
        return [(f,
                 os.path.basename(self.per_filter_catalogs[f])
                 if f in self.per_filter_catalogs else None,
                 os.path.basename(self.mosaics[f])
                 if f in self.mosaics else None)
                for f in self.filters]


# Synthetic mosaics written alongside the science one by the cataloguing
# stage.  They share the ``_i2d.fits`` suffix but are model/residual images,
# not data, and must never be picked up as "the mosaic".
_SYNTHETIC_MOSAIC_RE = re.compile(
    r'_(model|residual|smoothed_bg|mergedcat|starless)', re.IGNORECASE)


# The science mosaic is ``...-<filt>-<module>_i2d.fits`` and nothing else.
# Anything with an extra token before ``_i2d`` -- ``_data``, ``_model``,
# ``_residual``, a stage tag -- was written by the cataloguing stage from the
# mosaic rather than being it.
_MOSAIC_RE_TEMPLATE = r'-{filt}-(?P<module>[a-z0-9]+)_i2d\.fits$'


def _mosaic_for(basepath, filtername, proposals):
    """Science mosaic for *filtername*, preferring the merged module."""
    import glob
    filtdir = os.path.join(basepath, filtername.upper(), 'pipeline')
    pattern = re.compile(_MOSAIC_RE_TEMPLATE.format(filt=filtername.lower()))
    hits = [h for h in glob.glob(
        os.path.join(filtdir, f'*-{filtername.lower()}-*_i2d.fits'))
        if pattern.search(os.path.basename(h))
        and not _SYNTHETIC_MOSAIC_RE.search(os.path.basename(h))]
    # Restrict to this field's proposals so a stale, mis-filed product from a
    # neighbouring field cannot be picked up (see the brick-2221 retraction).
    scoped = [h for h in hits
              if any(jw_prefix(p) in os.path.basename(h) for p in proposals)]
    hits = scoped or hits
    if not hits:
        return None
    # merged first; otherwise the largest single-module mosaic, which is the
    # one covering the most sky.
    def rank(path):
        module = pattern.search(os.path.basename(path)).group('module')
        return (module == 'merged', os.path.getsize(path))
    return max(hits, key=rank)


def inventory(fieldname):
    """Build a :class:`FieldInventory` for *fieldname*."""
    if fieldname not in _fields.BY_NAME:
        raise KeyError(f'field {fieldname!r} is not in the registry; '
                       f'known: {", ".join(known_fields())}')
    basepath = _fields.BY_NAME[fieldname].basepath
    obs_list = list(_fields.BY_NAME[fieldname].observations)

    filters, proposals = [], []
    for obs in obs_list:
        proposals.append(obs.proposal)
        for filt in obs.filters:
            if filt not in filters:
                filters.append(filt)

    inv = FieldInventory(name=fieldname, basepath=basepath,
                         filters=tuple(filters), proposals=tuple(proposals))

    catdir = os.path.join(basepath, 'catalogs')
    hit = _best(catdir, _CROSSBAND_RE)
    if hit is not None:
        inv.crossband_catalog, match = hit
        inv.crossband_stage = int(match.group('stage') or 0) or None
        inv.crossband_resbgsub = bool(match.group('resbg'))
    else:
        inv.notes.append(
            'No cross-band merged catalogue: colour-colour and '
            'cross-band astrometry figures are omitted.')

    for filt in filters:
        hit = _best(catdir, _PERFILTER_RE, filt=filt)
        if hit is not None:
            inv.per_filter_catalogs[filt] = hit[0]
            chosen_mod = hit[1].group('module')
            # If the picked catalogue is a single module and other modules for
            # the same filter exist on disk, the write-up would describe part of
            # the field as if it were the whole; say so.
            if chosen_mod != 'merged':
                sibs = _module_siblings(catdir, filt, chosen_mod)
                if sibs:
                    inv.notes.append(
                        f'{filt.upper()}: no merged-module catalogue; using '
                        f'module {chosen_mod!r}, which covers part of the field '
                        f'only. Other modules on disk '
                        f'({", ".join(sorted(sibs))}) were not combined.')
        mosaic = _mosaic_for(basepath, filt, proposals)
        if mosaic is not None:
            inv.mosaics[filt] = mosaic

    missing = [f for f in filters if f not in inv.per_filter_catalogs]
    if missing:
        inv.notes.append(
            'No per-filter catalogue for ' + ', '.join(sorted(missing)).upper() + '.')

    for obs in obs_list:
        for obsid in obs.obsids.get('nircam', ()):
            # reference_catalog_path returns the best-registered path even when
            # nothing exists, so the existence test has to happen here; and it
            # raises outright when the observation has no registered catalogue
            # at all, which for this read-only survey is just "nothing to plot".
            try:
                path = _fields.reference_catalog_path(
                    obs.proposal, obsid, basepath=basepath, target=fieldname)
            except _fields.FieldRegistryError:
                continue
            if path and os.path.exists(path):
                inv.reference_catalogs[f'{obs.proposal}/{obsid}'] = path

    if not inv.reference_catalogs:
        inv.notes.append(
            'No reference catalogue on disk: the absolute astrometric tie '
            'figure is omitted and only internal (frame-to-frame) astrometry '
            'is characterised.')

    return inv


def known_fields():
    """Every field name in the registry, sorted."""
    return tuple(sorted(_fields.BY_NAME))
