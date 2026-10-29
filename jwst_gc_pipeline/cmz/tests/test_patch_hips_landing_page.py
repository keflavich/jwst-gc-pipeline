"""`scripts/release/patch_hips_landing_page.py` -- the HiPS landing-page patcher.

Hipsgen's `INDEX` action writes an `index.html` into every HiPS directory it
builds, with its own CDS `<script src=".../hips-templates/hips-landing-page.js">`
tag -- the same failure mode as a direct Aladin Lite CDS tag (CDS down ->
page hangs, then `buildLandingPage` is undefined).  This patches that tag to
a self-hosted-first fallback chain.  The logic lives in
jwst_gc_pipeline.aladin_loader; the script is its command-line wrapper and
re-exports it, so loading the script from its file path (same pattern as
test_release_integration.py) exercises both.
"""
import importlib.util
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_REL = os.path.join(_REPO, 'scripts', 'release')


def _load():
    spec = importlib.util.spec_from_file_location(
        'patch_hips_landing_page', os.path.join(_REL, 'patch_hips_landing_page.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CDS_TAG = ('<script src="https://aladin.cds.unistra.fr/hips-templates/'
            'hips-landing-page.js"></script>')


def _page(body):
    return f"<!doctype html><html><head></head><body>\n{body}\n</body></html>"


def test_patches_the_cds_tag():
    m = _load()
    new, status = m.patch(_page(_CDS_TAG))
    assert status == 'patched'
    assert m.MARKER in new
    assert new.count(m.CDS) == 1
    assert new.index(m.MIRROR) < new.index(m.SELFHOSTED) < new.index(m.CDS)
    assert 'buildLandingPage' in new


def test_idempotent_on_a_second_pass():
    m = _load()
    once, _ = m.patch(_page(_CDS_TAG))
    twice, status = m.patch(once)
    assert status == 'already'
    assert twice == once


def test_no_cds_tag_leaves_the_page_alone():
    m = _load()
    page = _page('<p>nothing to see here</p>')
    new, status = m.patch(page)
    assert status == 'no-cds-tag'
    assert new == page


def test_more_than_one_tag_raises():
    m = _load()
    with pytest.raises(ValueError, match='more than one'):
        m.patch(_page(_CDS_TAG + '\n' + _CDS_TAG))


def test_also_matches_the_old_u_strasbg_host():
    m = _load()
    old = _CDS_TAG.replace('aladin.cds.unistra.fr', 'aladin.u-strasbg.fr')
    new, status = m.patch(_page(old))
    assert status == 'patched'


def test_patch_file_writes_in_place(tmp_path):
    m = _load()
    path = tmp_path / 'index.html'
    path.write_text(_page(_CDS_TAG), encoding='utf-8')
    status = m.patch_file(str(path))
    assert status == 'patched'
    assert m.MARKER in path.read_text(encoding='utf-8')


# ------------------------------------------------------------- hipsgen wiring

def test_hipsgen_build_mono_hips_patches_the_written_index_html(monkeypatch, tmp_path):
    from jwst_gc_pipeline.cmz import hipsgen

    jar = tmp_path / 'Hipsgen.jar'
    jar.write_text('not a real jar', encoding='utf-8')
    monkeypatch.setattr(hipsgen.shutil, 'which', lambda _x: '/usr/bin/java')
    monkeypatch.setenv('HIPSGEN_JAR', str(jar))

    out_dir = tmp_path / 'out_hips'
    out_dir.mkdir()
    (out_dir / 'index.html').write_text(_page(_CDS_TAG), encoding='utf-8')

    calls = []
    monkeypatch.setattr(hipsgen, '_run', lambda cmd: calls.append(cmd))

    hipsgen.build_mono_hips('in_dir', str(out_dir))

    assert calls, 'Hipsgen was never invoked'
    patched = (out_dir / 'index.html').read_text(encoding='utf-8')
    m = _load()
    assert m.MARKER in patched


def test_hipsgen_build_mono_hips_is_a_noop_without_an_index_html(monkeypatch, tmp_path):
    """RGB-only Hipsgen runs may not write an index.html at all; patching
    must not invent one or raise."""
    from jwst_gc_pipeline.cmz import hipsgen

    jar = tmp_path / 'Hipsgen.jar'
    jar.write_text('not a real jar', encoding='utf-8')
    monkeypatch.setattr(hipsgen.shutil, 'which', lambda _x: '/usr/bin/java')
    monkeypatch.setenv('HIPSGEN_JAR', str(jar))
    out_dir = tmp_path / 'out_hips'
    out_dir.mkdir()
    monkeypatch.setattr(hipsgen, '_run', lambda cmd: None)

    hipsgen.build_mono_hips('in_dir', str(out_dir))

    assert not (out_dir / 'index.html').exists()
