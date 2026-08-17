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


# ---- adding frames to an already-staged release ----
#
# A full re-stage re-derives the deliverable set and re-runs every mosaic gate,
# so a field that cannot currently ship a mosaic cannot receive its frames
# either.  arches is exactly that: F212N sits in an m2 correct-and-requarantine
# cycle with no live product, so the listed-source gate refuses the field, while
# its already-published v1.2 release sits on disk gated and frozen.

def _staged_release(sr, eb, tmp_path, nircam_field, mode='copy'):
    """A minimal staged release: one science mosaic, no exposures yet."""
    import shutil
    field_dir = tmp_path / 'releases' / 'v9-test' / 'f'
    mosaic = _science(nircam_field['mosaic'])
    mosaic['dest'] = str(sr.assign_dest(mosaic, 'f'))
    mosaic['size_bytes'] = os.path.getsize(mosaic['src'])
    mosaic['sha256'] = 'deadbeef'
    dest = field_dir / mosaic['dest']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'copy':
        shutil.copy2(mosaic['src'], dest)
    else:
        dest.symlink_to(mosaic['src'])
    manifest = {'field': 'f', 'version': 'v9-test', 'mode': mode,
                'built': '2026-08-04T12:58:26-04:00',
                'release_path': '/releases/v9-test/f',
                'globus_collection_id': sr.GLOBUS_COLLECTION_ID,
                'globus_https_base': sr.GLOBUS_HTTPS_BASE,
                'files': [mosaic]}
    (field_dir / 'MANIFEST.json').write_text(json.dumps(manifest))
    (field_dir / 'CHECKSUMS.sha256').write_text(f"deadbeef  {mosaic['dest']}\n")
    return field_dir


def test_add_to_release_reads_the_asn_from_the_staged_copy(sr, eb, tmp_path,
                                                           nircam_field):
    """The whole point: the ORIGINAL mosaic can be gone -- quarantined by the m2
    checkpoint, or re-drizzled under a new name -- and the frames must still
    resolve, because the release tree holds a frozen copy of the exact mosaic
    that shipped and that copy names its own association."""
    field_dir = _staged_release(sr, eb, tmp_path, nircam_field)
    os.remove(nircam_field['mosaic'])          # the pipeline product goes away
    problems = []
    exposures, manifest = eb.add_to_release(
        field_dir, lambda it: sr.assign_dest(it, 'f'),
        tmp_path / 'releases', sr.GLOBUS_HTTPS_BASE,
        search_root=nircam_field['root'], problems=problems)
    assert not problems, problems
    assert len(exposures) == 2
    assert all(e['url'].startswith(sr.GLOBUS_HTTPS_BASE) for e in exposures)
    # ...and they still point at the mosaic that is IN the manifest, so the
    # page's withholding keys on the right thing
    assert {e['parent_dest'] for e in exposures} == {manifest['files'][0]['dest']}


def test_add_to_release_falls_back_to_src_for_a_symlink_release(sr, eb, tmp_path,
                                                                nircam_field):
    field_dir = _staged_release(sr, eb, tmp_path, nircam_field, mode='symlink')
    exposures, _ = eb.add_to_release(
        field_dir, lambda it: sr.assign_dest(it, 'f'),
        tmp_path / 'releases', sr.GLOBUS_HTTPS_BASE,
        search_root=nircam_field['root'])
    assert len(exposures) == 2


def test_add_to_release_reports_an_unreadable_mosaic_rather_than_guessing(
        sr, eb, tmp_path, nircam_field):
    """A dangling staged symlink whose source is also gone.  It must say the
    mosaic is not readable -- 'no ASNTABLE header' would send someone looking at
    a header that was never opened."""
    field_dir = _staged_release(sr, eb, tmp_path, nircam_field, mode='symlink')
    os.remove(nircam_field['mosaic'])
    problems = []
    exposures, _ = eb.add_to_release(
        field_dir, lambda it: sr.assign_dest(it, 'f'),
        tmp_path / 'releases', sr.GLOBUS_HTTPS_BASE, problems=problems)
    assert exposures == []
    assert len(problems) == 1 and 'not readable' in problems[0]


def test_exposures_only_preserves_built_and_the_frozen_deliverables(
        sr, eb, tmp_path, nircam_field, monkeypatch):
    """`built` is what `release_freshness` compares a quarantine twin's mtime
    against.  Stamping a fresh one here would make every existing twin older
    than "staging" and silently flip this field's QUARANTINED images back to
    LIVE -- re-publishing, as a side effect of adding a symlink, the very
    mosaics the astrometry checkpoint pulled.  arches would have had both
    superseded F212N mosaics returned to its page."""
    field_dir = _staged_release(sr, eb, tmp_path, nircam_field)
    before = json.loads((field_dir / 'MANIFEST.json').read_text())
    checksums_before = (field_dir / 'CHECKSUMS.sha256').read_text()
    mosaic_ino = os.stat(field_dir / before['files'][0]['dest']).st_ino

    monkeypatch.setattr(sr, 'GLOBUS_COLLECTION_ROOT', tmp_path / 'releases')
    monkeypatch.setattr(sr, 'field_release_dir',
                        lambda field, version, root: field_dir)
    monkeypatch.setitem(sr.FIELDS, 'f', {'data_dir': nircam_field['root']})
    out_dir, n = sr.stage_exposures_only('f', 'v9-test', tmp_path / 'releases')

    assert out_dir == field_dir and n == 2
    after = json.loads((field_dir / 'MANIFEST.json').read_text())
    assert after['built'] == before['built']
    assert after['exposures_added']            # recorded separately
    assert after['exposure_mode'] == 'symlink'
    # the frozen deliverables are untouched, byte for byte and inode for inode
    assert (field_dir / 'CHECKSUMS.sha256').read_text() == checksums_before
    assert os.stat(field_dir / before['files'][0]['dest']).st_ino == mosaic_ino
    assert [f for f in after['files'] if f['category'] == 'image'] == \
        before['files']
    frames = [f for f in after['files'] if f['category'] == eb.EXPOSURE_CATEGORY]
    assert len(frames) == 2
    assert all((field_dir / f['dest']).is_symlink() for f in frames)


def test_exposures_only_is_idempotent(sr, eb, tmp_path, nircam_field,
                                      monkeypatch):
    """Re-running must not accumulate duplicate manifest rows for one file."""
    field_dir = _staged_release(sr, eb, tmp_path, nircam_field)
    monkeypatch.setattr(sr, 'GLOBUS_COLLECTION_ROOT', tmp_path / 'releases')
    monkeypatch.setattr(sr, 'field_release_dir',
                        lambda field, version, root: field_dir)
    monkeypatch.setitem(sr.FIELDS, 'f', {'data_dir': nircam_field['root']})
    sr.stage_exposures_only('f', 'v9-test', tmp_path / 'releases')
    sr.stage_exposures_only('f', 'v9-test', tmp_path / 'releases')
    after = json.loads((field_dir / 'MANIFEST.json').read_text())
    dests = [f['dest'] for f in after['files']]
    assert len(dests) == len(set(dests)) == 3      # 1 mosaic + 2 frames


def test_add_to_release_needs_search_root_because_the_asn_is_not_beside_the_copy(
        sr, eb, tmp_path, nircam_field):
    """The association stays in the pipeline directory the mosaic was drizzled
    in, so it is NEVER beside the staged copy.  `search_root` is a fallback on
    the normal staging path and a REQUIREMENT on this one; without it every
    mosaic reports its ASNTABLE as not found and the field silently gains no
    frames at all."""
    field_dir = _staged_release(sr, eb, tmp_path, nircam_field)
    problems = []
    exposures, _ = eb.add_to_release(
        field_dir, lambda it: sr.assign_dest(it, 'f'),
        tmp_path / 'releases', sr.GLOBUS_HTTPS_BASE, problems=problems)
    assert exposures == []
    assert len(problems) == 1 and 'not found on disk' in problems[0]


# ---- enumeration with no mosaic at all ----
#
# Detector frames are a DEPENDENCY of the mosaic -- Stage 2/3 write them first --
# so a field can release them before anything is drizzled, and should: a
# two-filter program mid-reduction has its frames on disk and nothing about them
# is waiting on a drizzle.  Two-filter programs are the norm (JWST 10678).

def _disk_field(tmp_path, prop='02045', obs='001', filters=('F212N', 'F323N'),
                product='destreak_o{obs}_crf', n=2):
    for filt in filters:
        pipe = tmp_path / filt / 'pipeline'
        pipe.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            tail = product.format(obs=obs)
            _fits(pipe / f'jw{prop}{obs}001_02101_{i:05d}_nrca1_{tail}.fits')
    return {'data_dir': tmp_path,
            'proposal_prefix': f'jw{prop}-o{obs}_t001_nircam_clear'}


def test_enumerates_frames_with_no_mosaic_present(eb, tmp_path):
    cfg = _disk_field(tmp_path)
    found = eb.enumerate_field_exposures(cfg, 'arches')
    assert sorted(found) == [(None, 'F212N'), (None, 'F323N')]
    assert all(len(v) == 2 for v in found.values())


def test_enumeration_is_scoped_to_this_release_observations(eb, tmp_path):
    """A multi-pointing field keeps every observation's frames in ONE pipeline
    directory.  An unscoped glob hands o023's release o046's exposures, and the
    two are different sky."""
    pipe = tmp_path / 'F200W' / 'pipeline'
    pipe.mkdir(parents=True)
    for obs in ('023', '046', '099'):
        _fits(pipe / f'jw02211{obs}001_02201_00001_nrca1_destreak_o{obs}_crf.fits')
    cfg = {'data_dir': tmp_path, 'proposal_prefix': 'jw02211',
           'observations': ['o023', 'o046']}
    found = eb.enumerate_field_exposures(cfg, 'gc2211')
    # split per observation, and o099 -- not in this release -- is not there
    assert sorted(found) == [('o023', 'F200W'), ('o046', 'F200W')]
    assert all(len(v) == 1 for v in found.values())


def test_enumeration_picks_one_product_per_filter(eb, tmp_path):
    """Never a `_crf` for one exposure and a `_cal` for another: the most
    processed product present wins for the whole filter."""
    pipe = tmp_path / 'F212N' / 'pipeline'
    pipe.mkdir(parents=True)
    for tail in ('cal', 'destreak', 'destreak_o001_crf'):
        _fits(pipe / f'jw02045001001_02101_00001_nrca1_{tail}.fits')
    _fits(pipe / 'jw02045001001_02101_00002_nrca1_cal.fits')   # crf-less exposure
    cfg = {'data_dir': tmp_path,
           'proposal_prefix': 'jw02045-o001_t001_nircam_clear'}
    found = eb.enumerate_field_exposures(cfg, 'arches')
    names = [p.name for p in found[(None, 'F212N')]]
    assert names == ['jw02045001001_02101_00001_nrca1_destreak_o001_crf.fits']


def test_enumeration_falls_back_to_cal_when_that_is_all_there_is(eb, tmp_path):
    """Mid-reduction: Stage 2 has run, Stage 3 has not.  This is the state a
    field is in when its frames should already be releasable."""
    cfg = _disk_field(tmp_path, product='cal')
    found = eb.enumerate_field_exposures(cfg, 'arches')
    assert sorted(found) == [(None, 'F212N'), (None, 'F323N')]
    assert all(p.name.endswith('_cal.fits') for v in found.values() for p in v)


def test_enumeration_honours_the_destreak_policy_token(eb, tmp_path):
    """w51 does not destreak, so its frames are `_align_*`, not `_destreak_*`.
    Hard-coding either token would find nothing on half the survey."""
    cfg = _disk_field(tmp_path, filters=('F212N',), product='align_o{obs}_crf')
    found = eb.enumerate_field_exposures(cfg, 'w51')       # in EXTENDED_EMISSION_FIELDS
    assert len(found[(None, 'F212N')]) == 2
    # ...and the same files are NOT claimed for a field that DOES destreak
    assert not eb.enumerate_field_exposures(cfg, 'arches')


def test_observation_keys_read_both_registry_shapes(eb):
    assert eb.field_observation_keys(
        {'proposal_prefix': 'jw02045-o001_t001_nircam_clear'}) == {('02045', '001')}
    assert eb.field_observation_keys(
        {'proposal_prefix': 'jw02211', 'observations': ['o023', 'o046']}) == \
        {('02211', '023'), ('02211', '046')}
    assert eb.field_observation_keys(
        {'proposal_prefix': ['jw01182-o004_t001_nircam_clear',
                             'jw02221-o001_t001_nircam_clear']}) == \
        {('01182', '004'), ('02221', '001')}


# ---- a partial --fields build must not truncate the index ----

def test_partial_field_build_carries_the_rest_of_the_index(mw, tmp_path):
    """`--fields <subset>` is the obvious way to refresh one field after
    re-staging it, and it rewrote index.html from that subset alone: a one-field
    rebuild reduced the front page from fifteen cards to one, with every other
    field's page still on disk and unreachable from it.  The per-field pages are
    written independently, so only the index had this coupling, and nothing in
    the output said so."""
    out = tmp_path / 'site'
    out.mkdir()
    for field in ('arches', 'm92', 'quintuplet'):
        (out / f'{field}.html').write_text('<html></html>')
    (out / '_fields_index.json').write_text(json.dumps([
        {'field': f, 'version': 'v1', 'group': None, 'preview': None,
         'n_images': 4, 'n_catalogs': 0}
        for f in ('arches', 'm92', 'quintuplet')]))

    # a run that rebuilt ONLY arches
    roster = {e['field']: e for e in
              json.loads((out / '_fields_index.json').read_text())}
    rebuilt = [{'field': 'arches', 'version': 'v2', 'group': None,
                'preview': None, 'n_images': 4, 'n_catalogs': 0}]
    for info in rebuilt:
        roster[info['field']] = info
    roster = {f: e for f, e in roster.items() if (out / f'{f}.html').is_file()}
    index = mw.render_index([roster[f] for f in sorted(roster)])
    for field in ('arches', 'm92', 'quintuplet'):
        assert f"{field}.html" in index, field
    assert 'v2' in index          # ...and the rebuilt field's new version shows


def test_index_drops_a_field_whose_page_is_gone(mw, tmp_path):
    """Carrying entries forward must not resurrect a field dropped from the
    release: a card pointing at a 404 is worse than no card."""
    out = tmp_path / 'site'
    out.mkdir()
    (out / 'arches.html').write_text('<html></html>')
    roster = {f: {'field': f, 'version': 'v1', 'group': None, 'preview': None,
                  'n_images': 1, 'n_catalogs': 0}
              for f in ('arches', 'retired')}
    roster = {f: e for f, e in roster.items() if (out / f'{f}.html').is_file()}
    index = mw.render_index([roster[f] for f in sorted(roster)])
    assert 'arches.html' in index
    assert 'retired.html' not in index


# ---- HDRTAB: what resample actually consumed ----
#
# The earlier rule read the association and then preferred a
# `<stem>_<asn_id>_crf.fits` twin if one existed on disk.  It named the wrong
# file for 25 of the 170 live staged mosaics.  Each shape below is one of those.

def _mosaic_with_hdrtab(path, inputs, **cards):
    from astropy.io import fits
    import numpy as np
    col = fits.Column(name='FILENAME', format='A80',
                      array=np.array(inputs, dtype='U80'))
    hdu = fits.PrimaryHDU()
    for key, value in cards.items():
        hdu.header[key] = value
    tab = fits.BinTableHDU.from_columns([col], name='HDRTAB')
    fits.HDUList([hdu, tab]).writeto(path, overwrite=True)
    return str(path)


def test_hdrtab_wins_over_the_association(eb, tmp_path):
    """Both are present and they disagree; HDRTAB is resample's own record."""
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    for tail in ('destreak', 'destreak_o001_crf'):
        _fits(pipe / f'jw01234001001_02101_00001_nrca1_{tail}.fits')
    _asn(pipe / 'a.json', 'o001', ['jw01234001001_02101_00001_nrca1_destreak.fits'])
    mosaic = _mosaic_with_hdrtab(
        pipe / 'm_i2d.fits',
        ['jw01234001001_02101_00001_nrca1_destreak.fits'],
        ASNTABLE='a.json', S_OUTLIR='SKIPPED')
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert [os.path.basename(f) for f in frames] == \
        ['jw01234001001_02101_00001_nrca1_destreak.fits']
    assert problem is None


def test_crf_that_replaces_the_cal_suffix_is_found(eb, tmp_path):
    """wd1 F150W.  The pipeline REPLACES `_cal` when it writes the Stage-3
    frame; the twin rule APPENDED, looked for `..._nrca1_cal_o001_crf.fits`,
    missed, and fell back to `_cal` -- shipping 96 frames with no outlier/CR
    flags while 96 correct ones sat in the same directory."""
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    _fits(pipe / 'jw01905001001_02101_00001_nrca1_cal.fits')
    _fits(pipe / 'jw01905001001_02101_00001_nrca1_o001_crf.fits')
    mosaic = _mosaic_with_hdrtab(
        pipe / 'm_i2d.fits', ['jw01905001001_02101_00001_nrca1_o001_crf.fits'],
        S_OUTLIR='COMPLETE')
    frames, _ = eb.exposures_for_mosaic(mosaic)
    assert [os.path.basename(f) for f in frames] == \
        ['jw01905001001_02101_00001_nrca1_o001_crf.fits']


def test_outlier_skipped_mosaic_is_not_given_a_crf(eb, tmp_path):
    """arches F323N, quintuplet F212N/F323N, four sickle bands.  A `_crf` for
    that exposure exists, but it belongs to the MERGED association while the
    shipped product is the per-module one."""
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    _fits(pipe / 'jw02045001001_02101_00001_nrcalong_destreak.fits')
    _fits(pipe / 'jw02045001001_02101_00001_nrcalong_destreak_o001_crf.fits')
    mosaic = _mosaic_with_hdrtab(
        pipe / 'm-f323n-nrca_i2d.fits',
        ['jw02045001001_02101_00001_nrcalong_destreak.fits'],
        S_OUTLIR='SKIPPED')
    frames, _ = eb.exposures_for_mosaic(mosaic)
    assert [os.path.basename(f) for f in frames] == \
        ['jw02045001001_02101_00001_nrcalong_destreak.fits']


def test_skymatch_inputs_are_offered_as_the_frames(eb, tmp_path):
    """sickle: 96 distinct `*_<n>_skymatch.fits`, one per exposure.  Not the
    per-exposure names they look nothing like, but they ARE what resample was
    handed and what that field's mosaic came from."""
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    names = [f'jw03958-o007_t001_nircam_clear-f187n-nrcb_{i}_skymatch.fits'
             for i in range(3)]
    for n in names:
        _fits(pipe / n)
    mosaic = _mosaic_with_hdrtab(pipe / 'm-f187n-nrcb_i2d.fits', names)
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert sorted(os.path.basename(f) for f in frames) == sorted(names)
    assert problem is None


def test_partial_hdrtab_list_says_partial(eb, tmp_path):
    """sgrb2 F2550W had 10 of 20 inputs elsewhere.  The old message said the
    frames were NOT offered while staging every frame it did find."""
    pipe = tmp_path / 'pipeline'
    pipe.mkdir()
    _fits(pipe / 'a_o002_crf.fits')
    mosaic = _mosaic_with_hdrtab(pipe / 'm_i2d.fits',
                                 ['a_o002_crf.fits', 'b_o002_crf.fits'])
    frames, problem = eb.exposures_for_mosaic(mosaic)
    assert len(frames) == 1
    assert problem.count('PARTIAL') == 1 and '1 of 2' in problem


def test_a_quarantined_copy_is_never_offered(eb, tmp_path):
    """A backup directory holds a file of the RIGHT name and the WRONG
    generation -- sgrb2 keeps one under `stale_oldcrf_2026-07-11/` -- and
    `sorted` reaches it before the live tree."""
    live = tmp_path / 'NB' / 'F2550W' / 'pipeline'
    stale = tmp_path / 'F2550W' / 'pipeline' / 'stale_oldcrf_2026-07-11'
    live.mkdir(parents=True)
    stale.mkdir(parents=True)
    name = 'jw05365002001_02105_00001_mirimage_o002_crf.fits'
    _fits(live / name)
    _fits(stale / name)
    images = tmp_path / 'images'
    images.mkdir()
    mosaic = _mosaic_with_hdrtab(images / 'm_i2d.fits', [name])
    frames, problem = eb.exposures_for_mosaic(mosaic, search_root=tmp_path)
    assert len(frames) == 1 and problem is None
    assert 'stale' not in str(frames[0])


def test_association_search_prefers_the_mosaics_own_filter(eb, tmp_path):
    """Association filenames collide across filter directories.  An
    unconstrained `*/pipeline/` glob returns whichever sorts first, which handed
    gc2211's F277W mosaic the F200W association: 32 SW frames for 8 LW ones."""
    for filt, det, n in (('F200W', 'nrca1', 2), ('F277W', 'nrcalong', 1)):
        pipe = tmp_path / filt / 'pipeline'
        pipe.mkdir(parents=True)
        names = [f'jw02211023001_02201_{i:05d}_{det}_destreak_o023_crf.fits'
                 for i in range(n)]
        for nm in names:
            _fits(pipe / nm)
        _asn(pipe / 'shared_asn.json', 'o023', names)
    mosaic = _fits(tmp_path / 'F277W' / 'pipeline' / 'x-f277w-merged_i2d.fits',
                   ASNTABLE='shared_asn.json')
    os.remove(tmp_path / 'F277W' / 'pipeline' / 'shared_asn.json')
    frames, _ = eb.exposures_for_mosaic(mosaic, search_root=tmp_path)
    # F277W's own directory has no association left, so the fallback runs -- and
    # must not silently return F200W's.  Either it finds nothing, or it finds
    # LW frames; it must never offer the SW ones.
    assert all('nrca1' not in os.path.basename(f) for f in frames), frames


# ---- astrometry provenance ----
#
# The module shipped with zero test references. Its load-bearing rule is which
# reference leg a reader should quote, and the first version of that rule was a
# one-field generalisation that told readers to cite a 19-ARCSECOND offset.

@pytest.fixture(scope='module')
def ap():
    return _load('astrometry_provenance',
                 os.path.join(_REL, 'astrometry_provenance.py'))


def test_low_contrast_leg_is_never_the_one_to_quote(ap):
    """sgrc F115W, real values. The sparse leg reads 18968 mas at contrast 9
    while the dense leg reads 4.44 mas at contrast 325. Fixed text saying
    "quote the sparse number" published the 19-arcsecond one."""
    leg, why = ap.preferred_leg({'sparse_mas': 18968.11, 'sparse_contrast': 9.0,
                                 'dense_mas': 4.44, 'dense_contrast': 325.0})
    assert leg == 'dense', why


def test_sparse_is_chosen_where_sparse_is_the_solid_leg(ap):
    """arches F212N: the case the fixed rule was generalised from. It must still
    come out sparse -- the fix is choosing per filter, not flipping the default."""
    leg, _ = ap.preferred_leg({'sparse_mas': 1.22, 'sparse_contrast': 51.0,
                               'dense_mas': 9.27, 'dense_contrast': 41.0})
    assert leg == 'sparse'


def test_no_leg_is_quoted_when_none_reaches_the_floor(ap):
    """sgrc F182M: sparse 9801 mas at contrast 4.7, dense 11.68 at 10.8. Neither
    is a measurement, so neither is published."""
    leg, why = ap.preferred_leg({'sparse_mas': 9801.45, 'sparse_contrast': 4.67,
                                 'dense_mas': 11.68, 'dense_contrast': 10.83})
    assert leg is None
    assert 'contrast' in why


def test_a_leg_with_no_contrast_recorded_is_not_quoted(ap):
    """Absent contrast cannot clear a contrast floor. Treating it as passing
    would reinstate exactly the unguarded case."""
    assert ap.preferred_leg({'sparse_mas': 1.0})[0] is None


def test_summary_names_the_three_registration_states(ap, tmp_path):
    for state, expect in (("unregistered", "not registered"),
                          ("no-table", "no offsets table"),
                          ("table", "Pointing corrections")):
        rec = {"state": state, "ties": {},
               "table": {"name": "t.csv", "exists": True, "size_bytes": 10,
                         "modified": "2026-08-17T00:00:00", "sha256": "ab"}}
        assert expect in "\n".join(ap.summary_lines(rec)), state


def test_unregistered_summary_warns_rather_than_implying_a_tie(ap):
    text = "\n".join(ap.summary_lines({"state": "unregistered", "ties": {}}))
    assert "raw" in text and "assign_wcs" in text
    assert "Do not assume" in text


def test_highest_contrast_wins_when_both_legs_clear_the_floor(ap):
    """arches F323N: sparse contrast 58, dense 107, both above the floor. Taking
    the first usable leg rather than the strongest returns sparse here and
    passes every other case in this file, so it needs its own."""
    leg, why = ap.preferred_leg({'sparse_mas': 2.10, 'sparse_contrast': 58.0,
                                 'dense_mas': 9.33, 'dense_contrast': 107.0})
    assert leg == 'dense', why
    assert '107' in why
