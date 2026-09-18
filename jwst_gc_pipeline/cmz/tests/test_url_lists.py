"""URL lists beside the bundle buttons, and what the lists are allowed to hold."""
import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REL = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'scripts', 'release'))


@pytest.fixture(scope='module')
def mw():
    import sys
    sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(
        'make_webpage', os.path.join(_REL, 'make_webpage.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(obs, filt, name, link_mode='hardlink', url=True):
    return {'dest': f'exposures/{obs}/{filt}/{name}',
            'category': 'exposure', 'observation': obs, 'filter': filt,
            'link_mode': link_mode, 'size_bytes': 1024,
            'url': f'https://example.invalid/{obs}/{filt}/{name}' if url else None}


def test_a_symlinked_frame_is_not_put_in_a_list_the_page_refuses_to_link(mw):
    """The page renders a symlinked frame's name unlinked and carries a notice
    saying its per-file URL 404s -- and the same URL sat in `<field>_files.txt`
    for a `wget -i` that never sees the notice. 2,400 of the published entries
    are symlinks."""
    manifest = {'files': [
        _frame('o127', 'F212N', 'a.fits'),
        _frame('o004', 'F115W', 'b.fits', link_mode='symlink'),
    ]}
    urls = mw.published_urls(manifest)
    assert [u.rsplit('/', 1)[-1] for u in urls] == ['a.fits']

    # and the full set is still reachable for a caller that wants every path
    everything = mw.published_urls(manifest, servable_only=False)
    assert len(everything) == 2


def test_the_list_name_is_one_rule_for_the_page_and_the_writer(mw):
    """A link to a list that was never written is a 404; a list nothing links
    to is dead weight. Both sides call the same function."""
    assert mw.group_url_file('gc-treasury', 'o127', 'F212N') == \
        'gc-treasury_exposures_o127_F212N.txt'
    # a single-observation field has no observation to name
    assert mw.group_url_file('brick', '', 'F115W') == 'brick_exposures_F115W.txt'
    # and a filter with a slash or space cannot escape the filename
    assert '/' not in mw.group_url_file('f', 'o1', 'F212N/F480M')


def test_each_group_gets_its_own_frames_and_only_those(mw):
    """The list behind a row is that row's frames: a reader who clicks URLs
    next to o127 F212N and gets the whole field has been handed 20 GB they did
    not ask for."""
    exposures = [
        _frame('o127', 'F212N', 'a.fits'),
        _frame('o127', 'F212N', 'b.fits'),
        _frame('o127', 'F480M', 'c.fits'),
        _frame('o129', 'F212N', 'd.fits'),
    ]
    lists = mw.exposure_group_urls('gc-treasury', exposures)
    assert set(lists) == {'gc-treasury_exposures_o127_F212N.txt',
                          'gc-treasury_exposures_o127_F480M.txt',
                          'gc-treasury_exposures_o129_F212N.txt'}
    assert len(lists['gc-treasury_exposures_o127_F212N.txt']) == 2
    assert all('o127/F212N' in u
               for u in lists['gc-treasury_exposures_o127_F212N.txt'])


def test_a_withheld_frame_is_withheld_from_its_group_list_too(mw):
    """The lists withhold exactly what the page withholds, per group as well
    as per field."""
    exposures = [_frame('o127', 'F212N', 'a.fits'),
                 _frame('o127', 'F212N', 'bad.fits')]
    lists = mw.exposure_group_urls('f', exposures,
                                   superseded={'exposures/o127/F212N/bad.fits'})
    assert len(lists['f_exposures_o127_F212N.txt']) == 1


def test_a_group_of_symlinks_gets_no_list_at_all(mw):
    """An empty list file beside a bundle button reads as "nothing here",
    which is wrong -- the frames exist and transfer fine."""
    exposures = [_frame('o004', 'F115W', 'a.fits', link_mode='symlink')]
    assert mw.exposure_group_urls('brick', exposures) == {}


def test_the_row_links_the_list_and_a_symlinked_row_says_transfer_only(mw):
    """The button and the list are the same group for two different tools; a
    group that HTTPS cannot serve says so instead of linking a file that would
    be empty."""
    def app_link(subpath, label, cls='btn'):
        return f"<a class='{cls}' href='globus:{subpath}'>{label}</a>"

    html = mw.render_exposures('gc-treasury',
                               [_frame('o127', 'F212N', 'a.fits')],
                               'releases/v1/gc-treasury', app_link, multi=True)
    assert "href='gc-treasury_exposures_o127_F212N.txt'" in html
    assert '>URLs</a>' in html

    sym = mw.render_exposures('brick',
                              [_frame('o004', 'F115W', 'a.fits',
                                      link_mode='symlink')],
                              'releases/v1/brick', app_link, multi=False)
    assert 'transfer only' in sym
    assert '_exposures_' not in sym


def test_the_help_says_the_bundle_button_is_not_a_command_line_tool(mw):
    """The question a reader arrives with. The button is a link into the Globus
    web file manager: pasted into wget it fetches HTML, and there is no archive
    behind it."""
    page = mw.render_help()
    assert 'Can the <b>bundle</b> buttons be used from a terminal?' in page
    assert 'app.globus.org/file-manager' in page
    assert 'origin_id' in page and 'origin_path' in page
    # it points at the two routes rather than restating either recipe: the
    # transfer command is quoted as a link, and the runnable block that
    # carries the real collection id stays in one place
    assert page.count('<pre><code>pip install globus-cli') == 1
    assert '#command-line-with-globus-recommended-one-tool-no-token' in page
    section = page.split('Can the <b>bundle</b>')[1].split('<h2')[0]
    assert '<pre>' not in section
