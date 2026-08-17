"""Detector-frame exposures offered beside a released mosaic.

The thing under test is a provenance chain, so the fixtures build the real one:
a mosaic whose ``ASNTABLE`` names an association on disk, whose members name
frames on disk.  Every assertion below is about a step in that chain being read
rather than guessed -- the failure this guards against is a release offering
frames that are not the ones behind its own mosaic.
"""
import importlib.util
import json
import os

import pytest

# .../jwst_gc_pipeline/cmz/tests/test_exposure_bundle.py -> repo root (4 up)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_REL = os.path.join(_REPO, 'scripts', 'release')


def _load(name, path):
    import sys
    if _REL not in sys.path:            # scripts/release siblings import each other
        sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def eb():
    return _load('exposure_bundle', os.path.join(_REL, 'exposure_bundle.py'))


def _fits(path, **cards):
    from astropy.io import fits
    hdu = fits.PrimaryHDU()
    for key, value in cards.items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)
    return str(path)


def _asn(path, asn_id, expnames):
    path.write_text(json.dumps({
        'asn_id': asn_id,
        'products': [{'name': 'p', 'members': [{'expname': e, 'exptype': 'science'}
                                               for e in expnames]}],
    }))
    return str(path)


@pytest.fixture
def nircam_field(tmp_path):
    """A NIRCam-shaped field: mosaic -> asn -> `_destreak` members with `_crf` twins."""
    pipe = tmp_path / 'F212N' / 'pipeline'
    pipe.mkdir(parents=True)
    stems = ['jw01234001001_02101_00001_nrca1', 'jw01234001001_02101_00001_nrca2']
    for stem in stems:
        _fits(pipe / f'{stem}_destreak.fits')
        _fits(pipe / f'{stem}_destreak_o001_crf.fits')
    _asn(pipe / 'image3_asn.json', 'o001', [f'{s}_destreak.fits' for s in stems])
    mosaic = _fits(pipe / 'mos-f212n-merged_i2d.fits', ASNTABLE='image3_asn.json')
    return {'root': tmp_path, 'pipeline': pipe, 'mosaic': mosaic, 'stems': stems}


# ---- which file is "the final detector-frame version" ----
def test_prefers_the_crf_twin_over_the_association_member(eb, nircam_field):
    """The association names the Stage-3 INPUT; the release must offer the OUTPUT.

    Offering the `_destreak` member is the plausible-looking mistake -- it is
    the name literally written in the association -- and it silently ships
    frames without the outlier/CR flags the mosaic was actually built with.
    """
    frames, problem = eb.exposures_for_mosaic(nircam_field['mosaic'])
    assert problem is None
    assert [os.path.basename(f) for f in frames] == [
        'jw01234001001_02101_00001_nrca1_destreak_o001_crf.fits',
        'jw01234001001_02101_00001_nrca2_destreak_o001_crf.fits']


def test_falls_back_to_the_member_when_no_crf_was_written(eb, tmp_path):
    """wd1 F150W: drizzled straight from `_cal`, no `_crf` ever produced.

    A rule that only ever emits `_crf` frames would ship nothing for that band
    while its mosaic sits on the page.
    """
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    _fits(pipe / 'jw01234001001_02101_00001_nrca1_cal.fits')
    _asn(pipe / 'a.json', 'o001', ['jw01234001001_02101_00001_nrca1_cal.fits'])
    mosaic = _fits(pipe / 'm_i2d.fits', ASNTABLE='a.json')
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert [os.path.basename(f) for f in frames] == \
        ['jw01234001001_02101_00001_nrca1_cal.fits']
    assert problem is None


def test_member_that_is_already_a_crf_is_taken_as_is(eb, tmp_path):
    """MIRI's Stage-3 association lists `_crf` frames directly, with asn_id
    `a3001`.  Appending the asn_id anyway would look for
    `..._o002_crf_a3001_crf.fits`, a name that has never existed, and the band
    would silently ship no frames."""
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    frame = _fits(pipe / 'jw05365002001_02101_00001_mirimage_o002_crf.fits')
    _asn(pipe / 'a.json', 'a3001', [frame])          # absolute, as MIRI writes it
    mosaic = _fits(pipe / 'm_i2d.fits', ASNTABLE='a.json')
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert [str(f) for f in frames] == [frame]
    assert problem is None


def test_relative_member_resolves_against_the_association_not_the_cwd(eb,
                                                                     nircam_field,
                                                                     tmp_path,
                                                                     monkeypatch):
    """NIRCam members are bare filenames.  Resolving them against the process's
    working directory finds nothing wherever the stager is run from."""
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    frames, _ = eb.exposures_for_mosaic(nircam_field['mosaic'])
    assert len(frames) == 2
    assert all(os.path.isfile(f) for f in frames)


# ---- what happens when the chain cannot be read ----
def test_missing_association_is_reported_and_emits_nothing(eb, tmp_path):
    """Never guessed around: globbing the pipeline directory for
    plausible-looking frames would offer another observation's exposures for a
    multi-pointing field, since they share that directory."""
    mosaic = _fits(tmp_path / 'm_i2d.fits', ASNTABLE='gone_asn.json')
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert frames == []
    assert 'gone_asn.json' in problem and 'not found' in problem


def test_no_asntable_header_is_reported(eb, tmp_path):
    mosaic = _fits(tmp_path / 'm_i2d.fits')
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert frames == [] and 'no ASNTABLE' in problem


def test_unreadable_association_is_reported(eb, tmp_path):
    (tmp_path / 'a.json').write_text('{not json')
    mosaic = _fits(tmp_path / 'm_i2d.fits', ASNTABLE='a.json')
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert frames == [] and 'unreadable association' in problem


def test_partly_absent_member_list_still_ships_and_says_so(eb, nircam_field):
    """A shipped mosaic some of whose frames have been removed since it was
    drizzled is a fact about the release.  Dropping the frame silently would
    present a partial input list as a complete one."""
    os.remove(nircam_field['pipeline'] /
              'jw01234001001_02101_00001_nrca2_destreak_o001_crf.fits')
    os.remove(nircam_field['pipeline'] /
              'jw01234001001_02101_00001_nrca2_destreak.fits')
    frames, problem = eb.exposures_for_mosaic(nircam_field['mosaic'])
    assert len(frames) == 1
    assert '1 of 2' in problem


def test_association_with_nothing_on_disk_emits_nothing(eb, tmp_path):
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    _asn(pipe / 'a.json', 'o001', ['gone_destreak.fits'])
    mosaic = _fits(pipe / 'm_i2d.fits', ASNTABLE='a.json')
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert frames == [] and 'none of them on disk' in problem


def test_association_found_under_search_root_when_mosaic_was_moved(eb, tmp_path):
    """brick's MIRI mosaic sits in `brick/images/` while its association stayed
    in a `*/pipeline/` directory."""
    pipe = tmp_path / 'F770W' / 'pipeline'
    pipe.mkdir(parents=True)
    frame = _fits(pipe / 'jw05365002001_02101_00001_mirimage_o002_crf.fits')
    _asn(pipe / 'a.json', 'a3001', [frame])
    images = tmp_path / 'images'
    images.mkdir()
    mosaic = _fits(images / 'm_i2d.fits', ASNTABLE='a.json')
    assert eb.exposures_for_mosaic(mosaic)[0] == []          # not beside it
    frames, problem = eb.exposures_for_mosaic(mosaic, search_root=tmp_path)
    assert [str(f) for f in frames] == [frame] and problem is None


# ---- manifest items ----
def _science(src, **kw):
    item = {'category': 'image', 'kind': 'science', 'filter': 'F212N',
            'iteration': None, 'observation': None, 'instrument': 'NIRCam',
            'src': src}
    item.update(kw)
    return item


def test_items_inherit_the_mosaic_and_record_their_parent(eb, nircam_field):
    items = eb.discover_exposures([_science(nircam_field['mosaic'],
                                            observation='o023')])
    assert len(items) == 2
    for it in items:
        assert it['category'] == eb.EXPOSURE_CATEGORY
        assert it['kind'] == eb.EXPOSURE_KIND
        assert (it['filter'], it['observation'], it['instrument']) == \
            ('F212N', 'o023', 'NIRCam')
        assert it['parent_src'] == nircam_field['mosaic']


def test_a_frame_shared_by_two_mosaics_is_emitted_once(eb, nircam_field):
    """Two manifest rows for one file would also mean two identical
    destinations, i.e. one staged link claimed by two entries."""
    mosaic = nircam_field['mosaic']
    items = eb.discover_exposures([_science(mosaic), _science(mosaic)])
    assert len(items) == 2
    assert len({it['src'] for it in items}) == 2


def test_problems_accumulate_across_mosaics(eb, nircam_field, tmp_path):
    bad = _fits(tmp_path / 'bad_i2d.fits', ASNTABLE='gone.json')
    problems = []
    items = eb.discover_exposures([_science(nircam_field['mosaic']),
                                   _science(bad)], problems=problems)
    assert len(items) == 2                 # the good mosaic is unaffected
    assert len(problems) == 1 and 'gone.json' in problems[0]


def test_link_parents_fills_parent_dest(eb, nircam_field):
    mosaic = _science(nircam_field['mosaic'])
    mosaic['dest'] = 'images/F212N/mos.fits'
    items = [mosaic] + eb.discover_exposures([mosaic])
    eb.link_parents(items)
    assert all(it['parent_dest'] == 'images/F212N/mos.fits'
               for it in items if it['category'] == eb.EXPOSURE_CATEGORY)


def test_suffix_histogram_names_the_product(eb):
    items = [{'category': eb.EXPOSURE_CATEGORY,
              'src': '/x/jw01234001001_02101_00001_nrca1_destreak_o001_crf.fits'},
             {'category': eb.EXPOSURE_CATEGORY,
              'src': '/x/jw01234001001_02101_00001_nrca2_cal.fits'},
             {'category': 'image', 'src': '/x/ignored_i2d.fits'}]
    assert eb.suffix_histogram(items) == {'destreak_o001_crf': 1, 'cal': 1}


# ---- staging ----
@pytest.fixture(scope='module')
def sr():
    return _load('stage_release', os.path.join(_REL, 'stage_release.py'))


def test_exposure_layout_mirrors_the_images_layout(sr, eb):
    """`exposures/<...>/<FILTER>` beside `images/<...>/<FILTER>` is what makes
    one Globus folder link per group a usable bulk download, and what lets a
    reader find the frames behind a mosaic without a lookup."""
    def dest(**kw):
        item = {'category': eb.EXPOSURE_CATEGORY, 'filter': 'F212N',
                'observation': None, 'instrument': 'NIRCam', 'src': '/x/e.fits'}
        item.update(kw)
        return sr.assign_dest(item, 'brick').as_posix()

    assert dest() == 'exposures/F212N/e.fits'
    assert dest(observation='o023') == 'exposures/o023/F212N/e.fits'
    assert dest(instrument='MIRI', filter='F770W') == 'exposures/MIRI/F770W/e.fits'
    # ...and the mosaic side is unchanged by any of this
    assert sr.assign_dest({'category': 'image', 'filter': 'F212N',
                           'observation': 'o023', 'instrument': 'NIRCam',
                           'src': '/x/m_i2d.fits'}, 'brick').as_posix() == \
        'images/o023/F212N/m_i2d.fits'


def _stage(sr, eb, tmp_path, nircam_field, mode='copy'):
    from pathlib import Path
    mosaic = _science(nircam_field['mosaic'])
    mosaic['dest'] = str(sr.assign_dest(mosaic, 'f'))
    items = [mosaic] + eb.discover_exposures([mosaic])
    for it in items:
        it.setdefault('dest', str(sr.assign_dest(it, 'f')))
        it['size_bytes'] = os.path.getsize(it['src'])
    eb.link_parents(items)
    root = tmp_path / 'releases'
    original = sr.GLOBUS_COLLECTION_ROOT
    sr.GLOBUS_COLLECTION_ROOT = Path(root)
    try:
        field_dir = sr.stage(items, 'f', 'v9-test', root, mode=mode,
                             do_checksum=True, continuity_gate='test')
    finally:
        sr.GLOBUS_COLLECTION_ROOT = original
    return field_dir, json.loads((field_dir / 'MANIFEST.json').read_text())


def test_exposures_stay_symlinks_under_copy(sr, eb, tmp_path, nircam_field):
    """--copy exists to make a release survive its sources being regenerated.
    Applying it to exposures would grow the frozen tree ~40x (a field-filter is
    ~20 GB of frames against ~500 MB of mosaics) for data whose whole point is
    that following the link gets the current frame."""
    field_dir, manifest = _stage(sr, eb, tmp_path, nircam_field)
    assert manifest['mode'] == 'copy' and manifest['exposure_mode'] == 'symlink'
    for entry in manifest['files']:
        path = field_dir / entry['dest']
        if entry['category'] == eb.EXPOSURE_CATEGORY:
            assert path.is_symlink(), entry['dest']
            assert os.path.realpath(path) == os.path.realpath(entry['src'])
        else:
            assert not path.is_symlink(), entry['dest']


def test_exposures_are_not_in_the_checksum_manifest(sr, eb, tmp_path, nircam_field):
    """CHECKSUMS.sha256 is an integrity claim about frozen bytes.  A hash of a
    symlink whose target is expected to be rewritten is not that claim, and
    listing one invites `sha256sum -c` to fail on a healthy release."""
    field_dir, manifest = _stage(sr, eb, tmp_path, nircam_field)
    exposures = [f for f in manifest['files']
                 if f['category'] == eb.EXPOSURE_CATEGORY]
    images = [f for f in manifest['files'] if f['category'] == 'image']
    assert exposures and images
    assert all('sha256' not in f for f in exposures)
    assert all('sha256' in f for f in images)
    listed = {line.split('  ', 1)[1]
              for line in (field_dir / 'CHECKSUMS.sha256').read_text().splitlines()
              if line.strip()}
    assert listed == {f['dest'] for f in images}


def test_readme_states_the_exposures_are_not_frozen(sr, eb, tmp_path, nircam_field):
    field_dir, _ = _stage(sr, eb, tmp_path, nircam_field)
    readme = (field_dir / 'README.md').read_text()
    assert '## Detector-frame exposures' in readme
    assert 'SYMLINKS' in readme
    assert 'not covered by `CHECKSUMS.sha256`' in readme


# ---- the page ----
@pytest.fixture(scope='module')
def mw():
    return _load('make_webpage', os.path.join(_REL, 'make_webpage.py'))


def _manifest(sr, eb, nircam_field, observation=None):
    mosaic = _science(nircam_field['mosaic'], observation=observation)
    mosaic['dest'] = str(sr.assign_dest(mosaic, 'f'))
    items = [mosaic] + eb.discover_exposures([mosaic])
    base = '/releases/v9-test/f'
    for it in items:
        it.setdefault('dest', str(sr.assign_dest(it, 'f')))
        it['size_bytes'] = 1024
        it['version'] = 'v9-test'
        it['url'] = sr.GLOBUS_HTTPS_BASE + f"{base}/{it['dest']}"
    eb.link_parents(items)
    return {'field': 'f', 'version': 'v9-test', 'group': None,
            'release_path': base, 'built': '2026-08-17T00:00:00-04:00',
            'mode': 'copy', 'exposure_mode': 'symlink',
            'globus_collection_id': sr.GLOBUS_COLLECTION_ID,
            'globus_https_base': sr.GLOBUS_HTTPS_BASE, 'files': items}


def test_page_offers_a_bundle_and_a_link_per_frame(mw, sr, eb, nircam_field):
    """"Links to each file" and "one click for all of them" are both asked for,
    and a table of 700 near-identical names is not a download interface -- so
    the group row carries the bundle and the per-frame links are collapsed
    underneath it."""
    page = mw.render_field_page('f', _manifest(sr, eb, nircam_field), None)
    assert 'Detector-frame exposures' in page
    # the bundle button transfers the group's own folder, not the field root
    assert 'origin_path=/releases/v9-test/f/exposures/F212N/' in page
    # every individual frame is linked
    for stem in nircam_field['stems']:
        assert f'{stem}_destreak_o001_crf.fits</a>' in page
    assert '<details class=frames>' in page


def test_page_states_the_product_and_that_they_are_not_frozen(mw, sr, eb,
                                                              nircam_field):
    page = mw.render_field_page('f', _manifest(sr, eb, nircam_field), None)
    assert 'destreak_o001_crf' in page
    assert 'symlinks into the live pipeline tree' in page
    assert 'CHECKSUMS.sha256' in page


def test_withholding_a_mosaic_withholds_the_frames_behind_it(mw, sr, eb,
                                                             nircam_field):
    """The pipeline's astrometric correction is baked into the frame's WCS --
    it is what `resample` reads.  Pulling a mosaic for a superseded solution
    and leaving its input frames up publishes that solution one level down."""
    rf = _load('release_freshness', os.path.join(_REL, 'release_freshness.py'))
    manifest = _manifest(sr, eb, nircam_field)
    bad = next(f['dest'] for f in manifest['files'] if f['category'] == 'image')
    page = mw.render_field_page('f', manifest, None, superseded=[bad],
                                reasons={bad: rf.QUARANTINED})
    assert '/exposures/F212N/' not in page
    assert 'Detector-frame exposures' not in page


def test_withheld_notice_counts_mosaics_not_frames(mw, sr, eb, nircam_field):
    """Only mosaics are audited, so only a mosaic has a state; a frame is
    withheld by association.  Pooling them reports the frame count as if the
    audit had found that many superseded products."""
    rf = _load('release_freshness', os.path.join(_REL, 'release_freshness.py'))
    manifest = _manifest(sr, eb, nircam_field)
    bad = next(f['dest'] for f in manifest['files'] if f['category'] == 'image')
    page = mw.render_field_page('f', manifest, None, superseded=[bad],
                                reasons={bad: rf.QUARANTINED})
    assert '<b>1 withheld as bad astrometry.</b>' in page
    assert '2 detector-frame exposures behind those mosaics are withheld' in page
    # The discriminator: a frame has no entry in `reasons`, so pooling it with
    # the mosaics does not just inflate the quarantined count -- it invents a
    # SECOND bucket, reporting the field's own input frames as "withheld as
    # superseded ... they have been rebuilt or replaced since", which is a
    # public statement about files nothing was measured on.
    assert 'withheld as superseded' not in page


def test_url_lists_withhold_what_the_page_withholds(mw, sr, eb, nircam_field):
    """`<field>_files.txt` is linked from the page that carries the withholding
    notice.  Leaving a withheld product in it publishes the file anyway, to a
    `wget -i` that never shows the notice."""
    manifest = _manifest(sr, eb, nircam_field)
    bad = next(f['dest'] for f in manifest['files'] if f['category'] == 'image')
    assert len(mw.published_urls(manifest)) == 3          # 1 mosaic + 2 frames
    assert mw.published_urls(manifest, [bad]) == []       # and the frames go too
    assert mw.published_urls(manifest, categories={'image'}) == \
        [f['url'] for f in manifest['files'] if f['category'] == 'image']
    assert len(mw.published_urls(manifest,
                                 categories={mw.EXPOSURE_CATEGORY})) == 2


def test_a_page_with_no_exposures_gains_no_section(mw, sr, eb, nircam_field):
    manifest = _manifest(sr, eb, nircam_field)
    manifest['files'] = [f for f in manifest['files']
                         if f['category'] != eb.EXPOSURE_CATEGORY]
    page = mw.render_field_page('f', manifest, None)
    assert 'Detector-frame exposures' not in page
    assert '/exposures/' not in page
