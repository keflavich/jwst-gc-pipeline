"""The HiPS distributor's gate, verification and layer list.

Publishing was a hand-run `rsync ... jwst_* ...`: the NIRCam layer sat ~5 h and
~3700 tiles behind its build, data.rc's copies were a day stale with nothing
scheduled to touch them, and the glob meant any new `jwst_*` directory became a
public layer the next time anyone ran it.
"""
import importlib.util
import os

import pytest

_MOD = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'scripts', 'monitoring', 'publish_hips_layers.py'))
_spec = importlib.util.spec_from_file_location('publish_hips_layers', _MOD)
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)


def _props(date):
    return ('hips_order = 14\n'
            + (f'hips_release_date = {date}\n' if date else '')
            + 'hips_frame = galactic\n')


# --- the gate ---------------------------------------------------------------

def test_a_newer_build_publishes_and_an_older_one_does_not():
    assert ph.needs_publish(_props('2026-09-13T22:02Z'), _props('2026-09-13T17:03Z'))
    assert not ph.needs_publish(_props('2026-09-13T17:03Z'),
                                _props('2026-09-13T17:03Z'))
    assert not ph.needs_publish(_props('2026-09-13T01:19Z'),
                                _props('2026-09-13T17:03Z'))


def test_the_gate_reads_the_release_date_not_the_file_mtime():
    """A copy rewrites mtime, so an mtime gate republishes on every tick
    forever.  `hips_release_date` is a property of the BUILD."""
    assert ph.release_date(_props('2026-09-13T22:02Z')) == '2026-09-13T22:02Z'
    assert ph.release_date('hips_order = 14\n') is None
    assert ph.release_date('') is None
    assert ph.release_date(None) is None


def test_an_unreadable_destination_publishes_rather_than_looking_current():
    """The silent no-op this avoids: a destination whose properties cannot be
    read compares equal under any equality test, and the layer then never
    updates again."""
    assert ph.needs_publish(_props('2026-09-13T22:02Z'), None)
    assert ph.needs_publish(_props('2026-09-13T22:02Z'), '')
    assert ph.needs_publish(_props('2026-09-13T22:02Z'), 'hips_order = 14\n')


def test_an_unreadable_source_also_publishes_rather_than_silently_skipping():
    assert ph.needs_publish(None, _props('2026-09-13T17:03Z'))


def test_iso_dates_compare_correctly_as_strings():
    """The gate compares them as text; that only works because HiPS writes
    zero-padded ISO-8601 UTC."""
    assert ph.needs_publish(_props('2026-09-13T09:00Z'), _props('2026-09-13T08:59Z'))
    assert not ph.needs_publish(_props('2026-09-13T08:59Z'),
                                _props('2026-09-13T09:00Z'))
    assert ph.needs_publish(_props('2026-10-01T00:00Z'), _props('2026-09-30T23:59Z'))


# --- the staged verification ------------------------------------------------

def test_a_good_staged_copy_is_accepted():
    assert ph.verify(_props('2026-09-13T22:02Z'), True, 4631, 4631) is None


@pytest.mark.parametrize('props,n3,tiles,expect,why', [
    (_props(None), True, 10, 10, 'properties'),
    (None, True, 10, 10, 'properties'),
    (_props('x'), False, 10, 10, 'Norder3'),
    (_props('x'), True, 9, 10, 'tile count'),
    (_props('x'), True, 0, 10, 'tile count'),
])
def test_a_bad_staged_copy_is_refused_with_a_reason(props, n3, tiles, expect, why):
    """Checked on the STAGED tree rather than on rsync's exit code: a truncated
    transfer can exit 0 on a killed connection, and what would then be renamed
    over a good layer is a partial pyramid."""
    got = ph.verify(props, n3, tiles, expect)
    assert got is not None and why in got


def test_a_short_transfer_is_caught_even_though_it_has_tiles():
    """The dangerous case is not zero tiles, it is nearly all of them."""
    assert ph.verify(_props('x'), True, 4630, 4631) is not None


# --- the layer list ---------------------------------------------------------

def test_layers_are_listed_explicitly_and_not_globbed():
    """`rsync ... avm_images/jwst_* ...` published whatever appeared on disk.
    That is how the bg-matched MIRI layer would have gone public as a side
    effect rather than as a decision."""
    import inspect
    import re as _re
    src = inspect.getsource(ph)
    assert isinstance(ph.LAYERS, dict) and ph.LAYERS
    # A glob CALL, not the token: the module docstring quotes `jwst_*` and the
    # word "glob" precisely to explain why neither is used, so a substring test
    # fails on its own explanation.
    assert not _re.search(r'\bglob\.(glob|iglob)\s*\(', src)
    assert not _re.search(r'^\s*import glob', src, _re.M)
    # and every registered source is a literal path, not a pattern
    for name, path in ph.LAYERS.items():
        assert '*' not in path and '?' not in path, (name, path)


def test_both_serving_destinations_are_covered():
    """data.rc serves the docroot and starformation is a separate copy.
    Publishing one and not the other is what left them a day apart."""
    assert ph.DOCROOT.endswith('/avm_images')
    assert ph.WEB_DIR.endswith('/htdocs/avm_images')
    assert ph.WEB_HOST


def test_every_published_layer_is_registered_explicitly():
    """The list is a diff, never a glob -- adding a public layer is a decision.

    The CMZ overview coadds are here for a different reason from the treasury
    ones: they are BUILT in the docroot, so their local step is a no-op and
    what they need is the second destination. starformation had no scheduled
    path to them at all, and its `jwst_nir_hips` fell 14 months behind.
    """
    assert set(ph.LAYERS) == {
        'jwst_gc_treasury_hips',
        'jwst_gc_treasury_miri_hips',
        'jwst_gc_treasury_miri_bgmatch_hips',
        'jwst_gc_treasury_vminmax_hips',
        'jwst_gc_treasury_log_hips',
        'jwst_nir_hips',
        'jwst_miri_hips',
    }
    for name, src in ph.LAYERS.items():
        assert src.endswith('/' + name), (name, src)


def test_a_docroot_built_layer_does_not_copy_over_itself():
    """For the CMZ coadds the source IS the docroot copy, so the local publish
    must be a no-op rather than an rm -rf and a re-copy of a tree onto itself.
    `needs_publish` comparing a properties file against itself is what makes
    that safe, so it is asserted rather than assumed."""
    for name in ('jwst_nir_hips', 'jwst_miri_hips'):
        assert ph.LAYERS[name] == f'{ph.DOCROOT}/{name}'
    same = 'hips_release_date = 2026-09-15T20:15Z\n'
    assert not ph.needs_publish(same, same)


def test_an_unknown_layer_name_is_an_error_not_a_silent_no_op(capsys):
    with pytest.raises(SystemExit):
        ph.main(['--layer', 'jwst_gc_treasury_typo'])
    assert 'unknown layer' in capsys.readouterr().err


# --- tile counting ----------------------------------------------------------

def test_only_tile_files_are_counted():
    tree = [('/r', ['Norder3'], ['properties', 'index.html']),
            ('/r/Norder3', [], ['Npix1.png', 'Npix2.png', 'Moc.fits'])]
    # Moc.fits counts as a tile file by extension; the point of the count is a
    # like-for-like comparison of the same walker over source and destination,
    # so any consistent rule works.
    assert ph.count_tiles(lambda root: iter(tree), '/r') == 3
    assert ph.count_tiles(lambda root: iter([('/r', [], [])]), '/r') == 0


def test_a_layer_mid_rebuild_is_refused_rather_than_shipped_partial():
    """A coadd being rebuilt has no `properties` until the build finishes, and
    the tree on disk at that moment is a strict subset of the right one.

    This is not hypothetical: a manual --coadd raced the hourly cron on
    2026-09-15 and left jwst_gc_treasury_hips at 2,007 of 11,426 tiles with a
    zero exit status. `needs_publish` says yes (unknown means yes, so a layer
    never silently stops updating), and `verify` is what stops it.
    """
    assert ph.needs_publish(None, 'hips_release_date = 2026-09-15T12:21Z\n')
    assert ph.verify(None, True, 8535, 8535) == 'no readable properties'
    # and a tree that grew under the rsync is caught by the count, not waved
    # through because rsync exited 0
    assert ph.verify('hips_release_date = 2026-09-15T20:12Z\n', True,
                     2007, 11426) == 'tile count 2007 != source 11426'


def test_a_failed_transfer_says_so_and_leaves_no_staging(tmp_path, capsys,
                                                         monkeypatch):
    """A publish that moved no bytes must not look like one with nothing to do.

    Measured on 2026-09-15: an rsync of 11,430 tiles was killed by a caller's
    `timeout`, and the run printed only the docroot line. starformation stayed
    eight hours behind with a 1.2 GB partial `.new` beside the live layer, and
    the only way to notice was to go and look at the served properties.
    """
    src = tmp_path / 'src'
    (src / 'Norder3').mkdir(parents=True)
    (src / 'properties').write_text('hips_release_date = 2026-09-15T20:12Z\n')
    (src / 'Norder3' / 'Npix1.png').write_bytes(b'x')

    calls = []

    def fake_run(cmd, dry=False):
        calls.append(cmd)
        return 124 if cmd[0] == 'rsync' else 0        # 124 = timeout's rc

    monkeypatch.setattr(ph, '_run', fake_run)
    removed = []
    monkeypatch.setattr(ph.subprocess, 'call', lambda cmd: removed.append(cmd) or 0)
    monkeypatch.setattr(ph, '_read_remote', lambda *a: None)

    rc = ph.publish_remote('zz_layer', str(src), host='zz_host',
                           web_dir='/zz')
    assert rc == 124
    err = capsys.readouterr().err
    assert 'TRANSFER FAILED' in err and 'rc=124' in err
    assert 'live layer untouched' in err
    # the partial staging tree is removed rather than left to be mistaken for
    # progress by the next person who looks
    assert any('rm -rf' in ' '.join(c) and '.new' in ' '.join(c)
               for c in removed), removed
    # and nothing was swapped
    assert not any('mv ' in ' '.join(c) for c in removed), removed
