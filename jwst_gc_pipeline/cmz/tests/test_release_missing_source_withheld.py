"""A staged image whose source has VANISHED must not be published (#837).

`release_freshness.source_state` has four answers -- `live`, `quarantined`,
`rebuilt`, `missing` -- and the page builder was reading the set that means
"did the source change since staging", which is the first three minus one.
`MISSING` was in neither `SUPERSEDED_STATES` nor anything else, so a staged
image whose source had gone was served exactly as if it had been verified.

The shape that surfaced it, rebuilding w51's page on 2026-09-10:

    ten real bands      source present, `*_im0_badastrom.fits` twin beside it
                        -> quarantined -> withheld, correctly (m2 stale-tagged
                        410 mosaics pending the m7/m8 rebuild)
    two PHANTOM bands   sources moved out of tree to
                        `w51/_phantom_paired_band_20260909T031239Z/` as
                        fabricated paired-filter products (#829/#830)
                        -> missing -> PUBLISHED

`w51_images.txt` came out with 2 entries where the field ships 12, and both of
the survivors were the fabricated ones.  The correct data was withheld, the
fabricated data was kept, and the notice explained the withholding of the ten
while saying nothing about the two.
"""
import importlib.util
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_REL = os.path.join(_REPO, 'scripts', 'release')


def _load(name, path):
    import sys
    if _REL not in sys.path:            # scripts/release siblings import each other
        sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rf():
    return _load('release_freshness', os.path.join(_REL, 'release_freshness.py'))


def _mw():
    return _load('make_webpage', os.path.join(_REL, 'make_webpage.py'))


def _entry(dest, src, size, filt):
    # `kind` / `iteration` / `instrument` are what the images TABLE reads; the
    # freshness check needs none of them, but `main()` renders the table, so a
    # manifest entry here carries the same keys `stage_release` writes.
    return {'dest': dest, 'src': src, 'size_bytes': size, 'category': 'image',
            'filter': filt, 'observation': 'o001', 'kind': 'science',
            'iteration': None, 'instrument': 'NIRCam',
            'url': f'https://example.invalid/{dest}'}


def _manifest(files, built='2026-09-10T00:00:00'):
    return {'field': 'w51', 'version': 'v1.1', 'group': None,
            'release_path': '/releases/v1.1/w51', 'built': built,
            'mode': 'copy', 'globus_collection_id': 'x',
            'globus_https_base': 'https://example.invalid', 'files': files}


def _live(tmp_path, name, nbytes=2048):
    p = tmp_path / name
    p.write_bytes(b'\0' * nbytes)
    return str(p)


def _quarantined(tmp_path, name, nbytes=2048):
    """A source still on disk with a twin created AFTER staging: the m2 shape."""
    src = _live(tmp_path, name, nbytes)
    twin = tmp_path / (name[:-5] + '_im0_badastrom.fits')
    twin.write_bytes(b'\0')
    os.utime(twin, (2 ** 31, 2 ** 31))          # 2038: later than any `built`
    return src


def _gone(tmp_path, name):
    """A source moved out of tree -- no file, no twin.  The phantom-band shape."""
    return str(tmp_path / '_phantom_paired_band_20260909T031239Z' / name)


# ---- the state sets -------------------------------------------------------

def test_missing_withholds_without_being_called_superseded():
    """Both sets exist, and MISSING belongs to exactly one of them.

    Widening `SUPERSEDED_STATES` instead would have been the easy fix and the
    wrong one: "n withheld as bad astrometry" is a claim about what the m2
    checkpoint ruled, and the checkpoint never ruled on a file it cannot see.
    """
    rf = _rf()
    assert rf.is_withheld(rf.MISSING)
    assert not rf.is_superseded(rf.MISSING)
    assert not rf.is_withheld(rf.LIVE)
    for state in (rf.QUARANTINED, rf.REBUILT):
        assert rf.is_superseded(state) and rf.is_withheld(state), state


def test_withheld_reasons_reports_a_vanished_source_and_superseded_does_not(tmp_path):
    rf = _rf()
    manifest = _manifest([
        _entry('a.fits', _live(tmp_path, 'a_i2d.fits'), 2048, 'F182M'),
        _entry('b.fits', _quarantined(tmp_path, 'b_i2d.fits'), 2048, 'F210M'),
        _entry('c.fits', _gone(tmp_path, 'c_i2d.fits'), 2048, 'F150W'),
    ])
    assert rf.withheld_reasons(manifest) == {'b.fits': rf.QUARANTINED,
                                             'c.fits': rf.MISSING}
    # unchanged: the narrower question still answers only what it can observe
    assert rf.superseded_reasons(manifest) == {'b.fits': rf.QUARANTINED}
    assert rf.withheld_files(manifest) == ['b.fits', 'c.fits']


# ---- the w51 shape, end to end -------------------------------------------

def _w51_manifest(tmp_path):
    real = ['F140M', 'F162M', 'F182M', 'F187N', 'F210M',
            'F335M', 'F360M', 'F405N', 'F410M', 'F480M']
    files = [_entry(f'w51_{b}.fits', _quarantined(tmp_path, f'w51_{b}_i2d.fits'),
                    2048, b) for b in real]
    files += [_entry(f'w51_{b}.fits', _gone(tmp_path, f'w51_{b}_i2d.fits'),
                     2048, b) for b in ('F150W', 'F444W')]
    return _manifest(files)


def test_w51_publishes_neither_its_real_bands_nor_its_phantom_ones(tmp_path):
    """REGRESSION.  The published set was 2 of 12 -- the two fabricated ones.

    Ten withheld and two published was not a partial failure: it inverted the
    release, which exists to be evidence the astrometry is right.
    """
    rf, mw = _rf(), _mw()
    manifest = _w51_manifest(tmp_path)
    reasons = rf.withheld_reasons(manifest)
    assert len(reasons) == 12
    assert sorted(s for s in set(reasons.values())) == [rf.MISSING, rf.QUARANTINED]
    assert mw.published_urls(manifest, rf.withheld_files(manifest)) == []


def test_the_download_list_drops_a_missing_source(tmp_path):
    """`<field>_images.txt` is linked FROM the notice, so anything the page
    withholds and the list keeps is still published -- one click further away,
    to a `wget -i` that never shows the notice."""
    rf, mw = _rf(), _mw()
    manifest = _manifest([
        _entry('live.fits', _live(tmp_path, 'live_i2d.fits'), 2048, 'F182M'),
        _entry('phantom.fits', _gone(tmp_path, 'phantom_i2d.fits'), 2048, 'F150W'),
    ])
    urls = mw.published_urls(manifest, rf.withheld_files(manifest))
    assert urls == ['https://example.invalid/live.fits']


# ---- what the page SAYS about it -----------------------------------------

def test_the_notice_gives_a_vanished_source_its_own_sentence(tmp_path):
    """Three states, three sentences.

    Falling through to the `rebuilt` bucket would have asserted "the sources
    are no longer the files they were copied from", which is an observation
    about bytes that are THERE.  For a source that is gone there are no bytes
    to have observed -- the same class of public-facing false statement as
    naming a quarantine that did not happen.
    """
    rf, mw = _rf(), _mw()
    manifest = _manifest([
        _entry('phantom.fits', _gone(tmp_path, 'phantom_i2d.fits'), 2048, 'F150W'),
    ])
    page = mw.render_field_page('w51', manifest, None,
                                superseded=['phantom.fits'],
                                reasons={'phantom.fits': rf.MISSING})
    assert '<b>1 withheld: source no longer on disk.</b>' in page
    assert 'cannot be re-verified' in page
    assert 'F150W' in page
    # and it claims neither of the other two causes
    assert 'withheld as superseded' not in page
    assert 'withheld as bad astrometry' not in page


def test_each_bucket_keeps_its_own_count(tmp_path):
    """The w51 notice has to carry both halves without either absorbing the
    other -- the ten are an astrometry statement, the two are an absence."""
    rf, mw = _rf(), _mw()
    manifest = _w51_manifest(tmp_path)
    reasons = rf.withheld_reasons(manifest)
    page = mw.render_field_page('w51', manifest, None,
                                superseded=sorted(reasons), reasons=reasons)
    assert '<b>10 withheld as bad astrometry.</b>' in page
    assert '<b>2 withheld: source no longer on disk.</b>' in page
    assert 'withheld as superseded' not in page


# ---- the call site, driven through main() --------------------------------

def _staged_release(tmp_path, field='testfield', version='v1.0-2026.06'):
    """A release tree on disk, with one live image and one whose source is gone.

    Everything above this line hands `published_urls` and `render_field_page`
    the set they are meant to withhold, so it pins that they honour what they
    are GIVEN.  Which set the page ASKS for is a different line -- one word at
    `make_webpage.py`, `withheld_reasons(manifest)` -- and #837 was exactly that
    word being `superseded_reasons`.  Driving `main()` is what covers it: the
    only input is the tree, so the builder has to choose the question itself.
    """
    import json
    sources = tmp_path / 'sources'
    sources.mkdir(exist_ok=True)
    d = tmp_path / 'rel' / version / field
    d.mkdir(parents=True, exist_ok=True)
    files = [_entry('live.fits', _live(sources, 'live_i2d.fits'), 2048, 'F182M'),
             _entry('phantom.fits', _gone(sources, 'phantom_i2d.fits'), 2048,
                    'F150W')]
    manifest = _manifest(files)
    manifest.update(field=field, version=version,
                    release_path=f'/releases/{version}/{field}')
    (d / 'MANIFEST.json').write_text(json.dumps(manifest))
    return str(tmp_path / 'rel'), str(tmp_path / 'site'), field


def test_the_builder_asks_the_wider_question_of_its_own_accord(tmp_path):
    """REGRESSION on the call site: `main()` over a tree with a vanished source.

    Reverting the builder to `superseded_reasons` leaves `MISSING` out of
    `stale_files`, so this field's `_images.txt` ships the phantom URL and the
    page carries no notice about it -- which is #837 as the site served it.
    """
    mw = _mw()
    root, out, field = _staged_release(tmp_path)
    mw.main(['--fields', field, '--release-root', root, '--out', out])

    published = (open(os.path.join(out, f'{field}_images.txt')).read().split()
                 if os.path.isfile(os.path.join(out, f'{field}_images.txt')) else [])
    assert published == ['https://example.invalid/live.fits'], published

    page = open(os.path.join(out, f'{field}.html')).read()
    assert '<b>1 withheld: source no longer on disk.</b>' in page
    assert 'https://example.invalid/phantom.fits' not in page
