"""The release index has to reach the analyses that sit beside it.

``make_webpage.py`` lives in ``scripts/release`` and is not importable as a
package, so it is loaded by path the way the other release tests do it.
"""
import importlib.util
import os
import sys

import pytest

_REL = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'scripts', 'release')


def _make_webpage():
    if _REL not in sys.path:
        sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(
        'make_webpage', os.path.join(_REL, 'make_webpage.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIELDS = [{'field': 'brick', 'preview': '', 'n_images': 2, 'n_catalogs': 1,
           'version': 'v1', 'group': None}]


def test_the_index_links_the_monitor_and_the_hips_viewer():
    mw = _make_webpage()
    html = mw.render_index(FIELDS)
    assert 'Quicklook analyses' in html
    assert "href='monitor/'" in html
    assert 'avm_images/jwst_gc_aladin.html' in html


def test_the_monitor_link_is_relative_and_the_viewer_link_is_absolute():
    """The monitor is a subdirectory of this site, so a relative link survives
    a staging copy. The HiPS viewer is under a different docroot, which a
    relative link reaches only from the server."""
    mw = _make_webpage()
    hrefs = dict((title, href) for href, title, _ in mw.QUICKLOOKS)
    assert not hrefs['Observing monitor'].startswith('http')
    assert hrefs['HiPS sky viewer'].startswith('https://')


def test_the_cmd_explorer_card_is_absent_until_the_page_is_built():
    """A card pointing at a 404 is worse than no card: the explorer is built by
    a separate script that needs catalogs, and it can legitimately be missing."""
    mw = _make_webpage()
    assert 'cmd_explorer.html' not in mw.render_index(FIELDS)
    with_card = mw.render_index(FIELDS,
                                quicklooks=mw.QUICKLOOKS + (mw.CMD_VIEWER_CARD,))
    assert "href='cmd_explorer.html'" in with_card


@pytest.mark.parametrize('groups', [
    [{'field': 'a', 'preview': '', 'n_images': 1, 'n_catalogs': 1,
      'version': 'v1', 'group': None}],
    [{'field': 'a', 'preview': '', 'n_images': 1, 'n_catalogs': 1,
      'version': 'v1', 'group': None},
     {'field': 'b', 'preview': '', 'n_images': 1, 'n_catalogs': 1,
      'version': 'v1', 'group': 'galactic_plane'}],
])
def test_the_links_appear_whether_or_not_the_index_is_sectioned(groups):
    """``render_index`` has two branches -- one section or several -- and the
    quicklook block was easy to attach to only one of them."""
    mw = _make_webpage()
    html = mw.render_index(groups)
    assert html.count('Quicklook analyses') == 1
    assert "href='monitor/'" in html


def test_the_block_sits_inside_main():
    mw = _make_webpage()
    html = mw.render_index(FIELDS)
    assert html.index('Quicklook analyses') < html.index('</main>')


def _run_main(out_dir, release_root):
    """Drive `make_webpage.main()` the way the deploy does.

    The test above passes the card into `render_index` by hand, which exercises
    the renderer and never the branch in `main()` that DECIDES whether to pass
    it -- so replacing that branch with an unconditional append left the file
    green. These run the decision.
    """
    mw = _make_webpage()
    mw.main(['--out', str(out_dir), '--release-root', str(release_root)])
    return (out_dir / 'index.html').read_text()


def _empty_release(tmp_path):
    root = tmp_path / 'releases'
    root.mkdir()
    return root


def test_main_omits_the_card_when_the_explorer_has_not_been_built(tmp_path):
    out = tmp_path / 'site'
    out.mkdir()
    html = _run_main(out, _empty_release(tmp_path))
    assert 'cmd_explorer.html' not in html
    assert 'Quicklook analyses' in html          # the other two are still there
    assert "href='monitor/'" in html


def test_main_adds_the_card_once_the_explorer_is_there(tmp_path):
    out = tmp_path / 'site'
    out.mkdir()
    (out / 'cmd_explorer.html').write_text('<!doctype html><title>x</title>')
    html = _run_main(out, _empty_release(tmp_path))
    assert "href='cmd_explorer.html'" in html
    assert html.count('Colour-magnitude explorer') == 1
