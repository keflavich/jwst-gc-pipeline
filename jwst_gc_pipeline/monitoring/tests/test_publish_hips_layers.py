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


@pytest.mark.parametrize('where', ['docroot', 'remote'])
def test_a_failed_transfer_says_so_and_leaves_no_staging(where, tmp_path,
                                                         capsys, monkeypatch):
    """A publish that moved no bytes must not look like one with nothing to do.

    Measured on 2026-09-15: an rsync of 11,430 tiles was killed by a caller's
    `timeout`, and the run printed only its docroot line and exited 0.
    starformation stayed eight hours behind with a 1.2 GB partial `.new` beside
    the live layer, and the only way to notice was to fetch the served
    properties by hand.

    Both call sites, because the docroot one is not the milder case: its
    staging path is `dst + '.new'` INSIDE the directory the host serves, so a
    killed rsync leaves a partial pyramid next to the live layer on the machine
    publishing it.
    """
    src = tmp_path / 'src'
    (src / 'Norder3').mkdir(parents=True)
    (src / 'properties').write_text('hips_release_date = 2026-09-15T20:12Z\n')
    (src / 'Norder3' / 'Npix1.png').write_bytes(b'x')

    def fake_run(cmd, dry=False):
        return 124 if cmd[0] == 'rsync' else 0        # 124 = timeout's rc

    monkeypatch.setattr(ph, '_run', fake_run)
    removed = []
    monkeypatch.setattr(ph.subprocess, 'call', lambda cmd: removed.append(cmd) or 0)

    if where == 'remote':
        monkeypatch.setattr(ph, '_read_remote', lambda *a: None)
        rc = ph.publish_remote('zz_layer', str(src), host='zz_host',
                               web_dir='/zz')
        stage = '/zz/zz_layer.new'
    else:
        monkeypatch.setattr(ph, 'DOCROOT', str(tmp_path / 'docroot'))
        monkeypatch.setattr(ph, '_read_local', lambda path: (
            None if 'docroot' in path else (src / 'properties').read_text()))
        rc = ph.publish_local('zz_layer', str(src))
        stage = str(tmp_path / 'docroot' / 'zz_layer.new')

    assert rc == 124
    err = capsys.readouterr().err
    assert 'TRANSFER FAILED' in err and 'rc=124' in err
    assert 'live layer untouched' in err
    # the partial staging tree is removed rather than left to be mistaken for
    # progress by the next person who looks
    assert any(stage in ' '.join(c) and 'rm' in ' '.join(c)
               for c in removed), removed
    # and nothing was swapped
    assert not any('mv ' in ' '.join(c) for c in removed), removed


@pytest.mark.parametrize('where', ['docroot', 'remote'])
def test_a_pre_clean_failure_is_named_as_itself(where, tmp_path, capsys,
                                                monkeypatch):
    """`rc = _run(rm) or _run(rsync)` called a failed pre-clean a failed
    TRANSFER. The two want different words -- one says the destination could
    not be cleared, the other that the bytes did not arrive -- and someone
    debugging the first while being told the second loses real time.

    Asserted behaviourally. The version of this test that read
    `inspect.getsource` and checked two literal forms were absent passed
    happily on a recombination that kept the phrase in a comment and moved the
    `or` to the next line: it caught the edit it was written against rather
    than the behaviour. What the source could not express is the last
    assertion here -- that the transfer never ran at all.
    """
    src = tmp_path / 'src'
    (src / 'Norder3').mkdir(parents=True)
    (src / 'properties').write_text('hips_release_date = 2026-09-15T20:12Z\n')
    (src / 'Norder3' / 'Npix1.png').write_bytes(b'x')

    ran = []

    def fake_run(cmd, dry=False):
        ran.append(cmd)
        return 1 if ('rm' in cmd[0] or 'rm -rf' in ' '.join(cmd)) else 0

    monkeypatch.setattr(ph, '_run', fake_run)
    monkeypatch.setattr(ph.subprocess, 'call', lambda cmd: 0)

    if where == 'remote':
        monkeypatch.setattr(ph, '_read_remote', lambda *a: None)
        rc = ph.publish_remote('zz_layer', str(src), host='zz_host',
                               web_dir='/zz')
    else:
        monkeypatch.setattr(ph, 'DOCROOT', str(tmp_path / 'docroot'))
        monkeypatch.setattr(ph, '_read_local', lambda path: (
            None if 'docroot' in path else (src / 'properties').read_text()))
        rc = ph.publish_local('zz_layer', str(src))

    assert rc == 1
    err = capsys.readouterr().err
    assert 'could not clear the staging path' in err
    assert 'TRANSFER FAILED' not in err
    # the thing the source-reading version could not say
    assert not any(c[0] == 'rsync' for c in ran), ran

# --- --force: fresh content under a stale date -------------------------------

SAME = 'hips_release_date = 2026-09-15T04:19Z\n'


def test_the_date_gate_skips_an_unchanged_date_by_default():
    """What keeps the hourly path from re-copying 10,000 tiles for nothing."""
    from scripts.monitoring import publish_hips_layers as P
    assert P.needs_publish(SAME, SAME) is False


def test_force_publishes_anyway():
    """The date is the BUILDER's claim about itself, and a builder can rewrite
    every tile without advancing it. On 2026-09-15 the MIRI treasury coadd was
    rebuilt with the corrected AVM in all 34 fields and kept the 04:19 date
    already published, so the corrected layer was unpublishable by its own
    publisher."""
    from scripts.monitoring import publish_hips_layers as P
    assert P.needs_publish(SAME, SAME, force=True) is True


def test_force_does_not_change_the_normal_newer_case():
    from scripts.monitoring import publish_hips_layers as P
    newer = 'hips_release_date = 2026-09-15T06:00Z\n'
    assert P.needs_publish(newer, SAME) is True
    assert P.needs_publish(newer, SAME, force=True) is True


def test_force_reaches_both_destinations(tmp_path, monkeypatch):
    """Publishing one host and not the other is the split the tool exists to
    prevent; a flag that only reached one would reintroduce it.

    Driven through `main()` with both publishers stubbed, rather than by
    matching source text. The string this used to assert --
    `'force=args.force) or rc'` -- is satisfied by the LOCAL call site on its
    own, so deleting `force` from the remote call left the suite green while
    shipping a `--force` that republishes data.rc and silently skips
    starformation: the exact split named above.
    """
    layer = tmp_path / 'zz_layer'
    layer.mkdir()
    monkeypatch.setattr(ph, 'LAYERS', {'zz_layer': str(layer)})
    seen = {}
    monkeypatch.setattr(ph, 'publish_local',
                        lambda name, src, dry=False, force=False:
                        seen.setdefault('local', force) and 0 or 0)
    monkeypatch.setattr(ph, 'publish_remote',
                        lambda name, src, dry=False, force=False:
                        seen.setdefault('remote', force) and 0 or 0)

    assert ph.main(['--layer', 'zz_layer', '--force']) == 0
    assert seen == {'local': True, 'remote': True}, seen

    seen.clear()
    assert ph.main(['--layer', 'zz_layer']) == 0
    assert seen == {'local': False, 'remote': False}, seen


def test_force_still_verifies_before_swapping():
    """It bypasses the DATE gate, not the staging and verification -- a forced
    publish of a half-written tree must still refuse."""
    import inspect
    from scripts.monitoring import publish_hips_layers as P
    for fn in (P.publish_local, P.publish_remote):
        body = inspect.getsource(fn)
        assert 'verify(' in body, fn.__name__
        assert '.new' in body, fn.__name__
