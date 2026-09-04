"""``--seed`` and ``--pool`` are incoherent together, and must say so.

``--pool`` exists because module-family rows cannot express a per-detector
shift and un-pooled corrections are SUMMED onto the shared row.  The seeder
takes no ``pool`` argument, so a run passing both would get no pooling and no
warning -- a silent no-op on the one flag whose purpose is preventing summed
rows.  These pin the refusal, and pin that it costs nothing to reach: it fires
at parse time, before any catalog is read.
"""
import importlib.util
import pathlib

import pytest

_CLI = (pathlib.Path(__file__).parents[2] / 'scripts' / 'reduction'
        / 'run_astrometry_checkpoint.py')


def _load():
    spec = importlib.util.spec_from_file_location('_ckpt_cli', _CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def cli():
    return _load()


def test_seed_with_pool_is_refused(cli, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['--seed', '--pool', '--apply', '--stage', 'm2',
                  '--offsets-table', '/nonexistent/Offsets.csv',
                  '--proposal-id', '9438', '--obsid', '001',
                  '--basepath', '/nonexistent'])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert '--seed cannot be combined with --pool' in err
    assert 'Seed first' in err, 'the refusal must say what to do instead'


def test_refusal_precedes_any_catalog_read(cli, monkeypatch):
    """The pair is wrong on its face, so it must not cost a catalog load.

    Anything that reads a table before the check would make the refusal depend
    on inputs that have nothing to do with it -- and on a real invocation that
    is minutes of glob and Table.read before the user is told they used two
    flags that cannot go together.
    """
    def _boom(*a, **k):                      # pragma: no cover - must not run
        raise AssertionError('read a catalog before rejecting --seed --pool')

    monkeypatch.setattr(cli.Table, 'read', staticmethod(_boom))
    monkeypatch.setattr(cli.glob, 'glob', _boom)
    with pytest.raises(SystemExit):
        cli.main(['--seed', '--pool', '--catalog-glob', '/nonexistent/*.fits'])


def test_seed_without_pool_still_reaches_its_own_guards(cli, capsys):
    """The refusal is scoped to the pair, not to --seed.

    --seed alone must still fail on ITS missing prerequisites, which is a
    different message; a guard that swallowed --seed entirely would pass this
    file's other test while breaking the feature.
    """
    with pytest.raises(SystemExit) as exc:
        cli.main(['--seed', '--apply', '--stage', 'm2',
                  '--offsets-table', '/nonexistent/Offsets.csv',
                  '--catalog-glob', '/nonexistent/*.fits'])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert '--seed cannot be combined with --pool' not in err


def test_pool_alone_is_untouched(cli, monkeypatch):
    """--pool without --seed is the ordinary cycle-N path and must not refuse."""
    def _no_catalogs(*a, **k):
        return []

    monkeypatch.setattr(cli.glob, 'glob', _no_catalogs)
    with pytest.raises(SystemExit) as exc:
        cli.main(['--pool', '--apply', '--stage', 'm2',
                  '--offsets-table', '/nonexistent/Offsets.csv',
                  '--catalog-glob', '/nonexistent/*.fits'])
    # it gets as far as the catalog glob, i.e. past every flag-pair check
    assert exc.value.code == 2
