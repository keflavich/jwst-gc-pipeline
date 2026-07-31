"""One registry of the fields this pipeline knows about.

PROPOSAL, not yet wired in.  See ``docs/FIELD_REGISTRY_PROPOSAL.md``.

Adding a target currently means editing seven dictionaries in five files, three
of them inside functions where they cannot be imported or overridden.  Nothing
checks that they agree, and today two of them do not: `cloudc/2526` is in
`obs_filters` but not `project_obsnum` (that merge raises `KeyError`), and
`w51/1182` is the reverse.

This module holds the same information once.  The views at the bottom reproduce
each existing dictionary exactly, so call sites can move over one at a time
instead of in a single sweep.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

ROOTS = {'orange': '/orange/adamginsburg/jwst',
         'blue': '/blue/adamginsburg/adamginsburg/jwst'}


@dataclass(frozen=True)
class Obs:
    """One observation of one field, under one proposal."""

    proposal: str
    obsid: Optional[str] = None            # -> project_obsnum
    nvisits: Optional[int] = None          # -> nvisits
    filters: Tuple[str, ...] = ()          # -> obs_filters
    offsets_table: Optional[str] = None    # -> offsets_tables


@dataclass(frozen=True)
class Field:
    """One target, and every observation of it."""

    name: str
    root: str                         # 'orange' or 'blue' -- which /.../jwst tree
    observations: Tuple[Obs, ...] = ()

    @property
    def basepath(self):
        return f'{ROOTS[self.root]}/{self.name}/'


FIELDS = (
    Field('arches', root='orange', observations=(
        Obs('2045', obsid='001', nvisits=1,
            filters=('f212n', 'f323n')),
    )),
    Field('brick', root='blue', observations=(
        Obs('2221', obsid='001', nvisits=2,
            filters=('f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f2550w')),
        Obs('1182', obsid='004', nvisits=2,
            offsets_table=('/blue/adamginsburg/adamginsburg/jwst/brick/offsets/'
                           'Offsets_JWST_Brick1182_F444ref.csv'),
            filters=('f444w', 'f356w', 'f200w', 'f115w')),
    )),
    Field('cloudc', root='blue', observations=(
        Obs('2221', obsid='002', nvisits=2,
            filters=('f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f2550w')),
        Obs('2526', obsid=None, nvisits=1,
            filters=('f770w',)),
    )),
    Field('cloudef', root='orange', observations=(
        Obs('2092', obsid='*', nvisits=1,
            filters=('f162m', 'f210m', 'f360m', 'f480m', 'f770w', 'f2100w')),
    )),
    Field('gc2211', root='orange', observations=(
        Obs('2211', obsid='*', nvisits=1,
            filters=('f150w', 'f200w', 'f277w')),
    )),
    Field('m4', root='orange', observations=(
        Obs('1979', obsid='002', nvisits=1,
            filters=('f150w2', 'f322w2')),
    )),
    Field('m92', root='orange', observations=(
        Obs('1334', obsid='001', nvisits=1,
            filters=('f090w', 'f150w', 'f277w', 'f444w')),
    )),
    Field('ngc6334', root='orange', observations=(
        Obs('7213', obsid='001', nvisits=2,
            filters=('f115w', 'f162m', 'f182m', 'f200w', 'f356w', 'f405n', 'f444w', 'f470n')),
        Obs('6778', obsid='001', nvisits=3,
            filters=('f090w', 'f187n', 'f200w', 'f277w', 'f335m', 'f470n')),
    )),
    Field('ngc6397', root='orange', observations=(
        Obs('1979', obsid='001', nvisits=1,
            filters=('f150w2', 'f322w2')),
    )),
    Field('quintuplet', root='orange', observations=(
        Obs('2045', obsid='003', nvisits=1,
            filters=('f212n', 'f323n')),
    )),
    Field('sgra', root='orange', observations=(
        Obs('1939', obsid='001', nvisits=1,
            filters=('f115w', 'f212n', 'f405n')),
    )),
    Field('sgrb2', root='orange', observations=(
        Obs('5365', obsid='001', nvisits=1,
            filters=('f150w', 'f182m', 'f187n', 'f210m', 'f212n', 'f300m', 'f360m', 'f405n', 'f410m', 'f466n', 'f480m', 'f770w', 'f1280w', 'f2550w')),
    )),
    Field('sgrc', root='orange', observations=(
        Obs('4147', obsid='012', nvisits=1,
            filters=('f115w', 'f162m', 'f182m', 'f212n', 'f360m', 'f405n', 'f470n', 'f480m')),
    )),
    Field('sickle', root='orange', observations=(
        Obs('3958', obsid='*', nvisits=1,
            filters=('f187n', 'f210m', 'f335m', 'f470n', 'f480m', 'f770w', 'f1130w', 'f1500w')),
    )),
    Field('w51', root='orange', observations=(
        Obs('6151', obsid='001', nvisits=2,
            filters=('f140m', 'f162m', 'f182m', 'f187n', 'f210m', 'f335m', 'f360m', 'f405n', 'f410m', 'f480m', 'f770w', 'f1280w', 'f2100w')),
    )),
    Field('wd1', root='blue', observations=(
        Obs('1905', obsid='001', nvisits=3,
            filters=('f115w', 'f150w', 'f164n', 'f187n', 'f200w', 'f212n', 'f277w', 'f323n', 'f405n', 'f444w', 'f466n')),
    )),
    Field('wd2', root='blue', observations=(
        Obs('3523', obsid='005', nvisits=1,
            filters=('f115w', 'f150w', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f212n', 'f250m', 'f277w', 'f300m', 'f323n', 'f335m', 'f405n', 'f410m', 'f444w', 'f466n')),
    )),
)


BY_NAME = {f.name: f for f in FIELDS}


def obs_filters():
    """`{target: {proposal: [filters]}}` -- as in merge_catalogs."""
    return {f.name: {o.proposal: list(o.filters) for o in f.observations}
            for f in FIELDS}


def project_obsnum():
    """`{target: {proposal: obsid}}` -- as in merge_catalogs."""
    return {f.name: {o.proposal: o.obsid for o in f.observations
                     if o.obsid is not None}
            for f in FIELDS}


def nvisits():
    """`{proposal: {target: n}}` -- note this one is the TRANSPOSE of the other
    two, which is a thing a human has to remember and a view does not."""
    out = {}
    for f in FIELDS:
        for o in f.observations:
            if o.nvisits is not None:
                out.setdefault(o.proposal, {})[f.name] = o.nvisits
    return out


def offsets_tables():
    """`{proposal: path-or-None}` -- as in merge_catalogs.main().

    Today's dict omits 1905, 3523 and 2526 entirely, so `offsets_tables[progid]`
    raises KeyError for every wd1/wd2 per-filter merge.  This view covers every
    registered proposal, which is the same bug class as cloudc/2526 and is fixed
    by construction rather than by remembering.
    """
    return {o.proposal: o.offsets_table
            for f in FIELDS for o in f.observations}


def basepath(target):
    """The data root for one target, replacing the `if target in (...)` branch.

    An unregistered target gets the blue tree, which is what that branch's
    `else` does -- not a KeyError.  Registering a new field is what moves it.
    """
    known = BY_NAME.get(target)
    if known is not None:
        return known.basepath
    return f"{ROOTS['blue']}/{target}/"
