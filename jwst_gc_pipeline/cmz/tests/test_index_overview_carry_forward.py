"""A subset `--fields` build must not truncate the on-sky overview map (#812).

The index CARDS were given a roster sidecar (`_fields_index.json`) when a
one-field rebuild was found reducing the front page from fifteen cards to one.
The overview MAP got no equivalent and went on regressing the same way, in
silence.  Publishing `cloudef_controlfield` alone:

    viewBox     0 0 1068.0 322.4  ->  0 0 641.0 518.0
    axis range  l = -0.5 .. 0.75  ->  l = 0.36 .. 0.46
    polygons    nine `<a class="ov-field">`  ->  the control field's, alone

and the build log said `16 on the index`, which was true of the cards and false
of the map.  The failure produced no output of its own.

`resolve_overview_geoms` is the rule, split out of `main` so it can be asserted
here rather than by reading the source: which SOURCE a field's footprint comes
from is as much of a decision as drawing it.
"""
import importlib.util
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_REL = os.path.join(_REPO, 'scripts', 'release')


def _load(name, path):
    import sys
    if _REL not in sys.path:
        sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mw():
    return _load('make_webpage', os.path.join(_REL, 'make_webpage.py'))


def _geom(field, poly=((266.0, -28.9), (266.1, -28.9), (266.1, -28.8))):
    return {'field': field, 'href': f'{field}.html', 'polys': [list(map(list, poly))]}


def _never_read(field, href):
    raise AssertionError(f'went to disk for {field}, which was cached')


#: Card entries as `_fields_index.json` holds them.  `group` None is the CMZ;
#: `galactic_plane` / `globular_clusters` fields are elsewhere on sky.
CMZ = ['arches', 'brick', 'cloudef_controlfield', 'quintuplet', 'sgra']
ROSTER = {f: {'field': f, 'group': None} for f in CMZ}


def test_a_one_field_run_keeps_every_other_footprint_on_the_map():
    """REGRESSION.  The map went from nine footprints to one."""
    mw = _mw()
    fresh = {'cloudef_controlfield': _geom('cloudef_controlfield')}
    cached = {f: _geom(f) for f in CMZ if f != 'cloudef_controlfield'}
    got = mw.resolve_overview_geoms(ROSTER, {'cloudef_controlfield'},
                                    fresh, cached, _never_read)
    assert [g['field'] for g in got] == sorted(CMZ)


def test_the_rebuilt_field_is_drawn_from_this_run_not_the_cache():
    """The one field the run DID rebuild must show its new geometry."""
    mw = _mw()
    new = _geom('arches', poly=((1.0, 1.0), (2.0, 1.0), (2.0, 2.0)))
    got = mw.resolve_overview_geoms(ROSTER, {'arches'}, {'arches': new},
                                    {f: _geom(f) for f in CMZ}, _never_read)
    assert next(g for g in got if g['field'] == 'arches')['polys'] == new['polys']


def test_the_sidecar_is_a_cache_and_disk_is_the_authority():
    """A site built before the sidecar existed has no sidecar, so keying solely
    off it would let the very first partial build truncate exactly as before --
    the same reasoning the roster records for the cards."""
    mw = _mw()
    reads = []

    def read(field, href):
        reads.append(field)
        return _geom(field)

    got = mw.resolve_overview_geoms(ROSTER, {'arches'},
                                    {'arches': _geom('arches')}, {}, read)
    assert [g['field'] for g in got] == sorted(CMZ)
    assert sorted(reads) == sorted(f for f in CMZ if f != 'arches')


def test_a_cached_field_is_not_re_read_from_disk():
    """The point of the sidecar: a subset run does not pay for every other
    field's mosaic headers."""
    mw = _mw()
    cached = {f: _geom(f) for f in CMZ if f != 'arches'}
    got = mw.resolve_overview_geoms(ROSTER, {'arches'},
                                    {'arches': _geom('arches')}, cached,
                                    _never_read)
    assert len(got) == len(CMZ)


def test_a_rebuilt_field_with_no_readable_geometry_is_left_off():
    """Not filled in from the cache.  `collect` returning nothing for a field
    this run rebuilt means its mosaics are unreadable NOW -- redrawing the
    previous footprint would have the map assert a coverage the release has
    just stopped shipping (a withheld or replaced mosaic)."""
    mw = _mw()
    cached = {f: _geom(f) for f in CMZ}
    got = mw.resolve_overview_geoms(ROSTER, {'arches'}, {}, cached, _never_read)
    assert 'arches' not in {g['field'] for g in got}
    assert len(got) == len(CMZ) - 1


def test_a_field_dropped_from_the_roster_leaves_the_map():
    """Carrying geometry forward must not resurrect a field the release no
    longer has -- the same rule the cards follow for a page that is gone."""
    mw = _mw()
    cached = {f: _geom(f) for f in CMZ + ['retired_field']}
    got = mw.resolve_overview_geoms(ROSTER, set(), {}, cached, _never_read)
    assert 'retired_field' not in {g['field'] for g in got}


def test_a_cached_entry_with_no_polygons_falls_through_to_disk():
    """An empty `polys` is not geometry.  Accepting it would pin a field off
    the map for every later build, since nothing would ever re-read it."""
    mw = _mw()
    reads = []

    def read(field, href):
        reads.append(field)
        return _geom(field)

    cached = {f: _geom(f) for f in CMZ}
    cached['brick'] = {'field': 'brick', 'href': 'brick.html', 'polys': []}
    got = mw.resolve_overview_geoms(ROSTER, set(), {}, cached, read)
    assert reads == ['brick']
    assert len(got) == len(CMZ)


def test_a_carried_entry_gets_this_site_s_href():
    """The sidecar is keyed by field, and the href is derived, so a sidecar
    copied between out-dirs cannot carry a stale link."""
    mw = _mw()
    stale = _geom('brick')
    stale['href'] = 'somewhere/else.html'
    got = mw.resolve_overview_geoms({'brick': {'field': 'brick', 'group': None}},
                                    set(), {}, {'brick': stale},
                                    _never_read)
    assert got[0]['href'] == 'brick.html'


def test_a_grouped_field_never_reaches_the_cmz_map():
    """REGRESSION, caught by a real build rather than by this file.

    The per-field append already skipped `group is not None` -- a
    `galactic_plane` / `globular_clusters` field is elsewhere on sky and the
    panel is a CMZ strip.  The carry-forward paths reach fields that append
    never sees, so the first build to resolve w51, wd1 and wd2 from disk put
    all three on the map: l spanned 284..49 deg, and every real CMZ footprint
    flattened to nothing (viewBox height 415.6 -> 170.2, 10 fields -> 13).
    """
    mw = _mw()
    roster = dict(ROSTER)
    roster['w51'] = {'field': 'w51', 'group': 'galactic_plane'}
    roster['m92'] = {'field': 'm92', 'group': 'globular_clusters'}
    cached = {f: _geom(f) for f in CMZ + ['w51', 'm92']}
    got = mw.resolve_overview_geoms(roster, set(), {}, cached, _never_read)
    assert [g['field'] for g in got] == sorted(CMZ)


def test_a_grouped_field_is_not_even_read_from_disk():
    """The group rule is applied before the source choice, so an off-map field
    costs no mosaic-header reads on any build."""
    mw = _mw()
    roster = dict(ROSTER, w51={'field': 'w51', 'group': 'galactic_plane'})
    reads = []

    def read(field, href):
        reads.append(field)
        return _geom(field)

    mw.resolve_overview_geoms(roster, set(), {}, {}, read)
    assert 'w51' not in reads
    assert sorted(reads) == sorted(CMZ)
