"""The m8 dedup options are read at option-parse time and reach both dedup paths.

``cataloging._maybe_dedup_m8`` runs the dedup inside a handler that logs a
failure and carries on, so a malformed ``--m8-dedup-band-resolve-mas`` used to
cost the dedup product only after the whole m8 fill (#1278).  Both entry points
(``crowdsource_catalogs_long.main`` and ``m8_merge_partials.py``) now reject a
bad value when the command line is read, and ``m8_merge_partials.py`` forwards
the two options so a re-merge from partials dedups like the cataloging run.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import dedup_catalog

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / 'scripts' / 'reduction' / 'm8_merge_partials.py')
_M7 = 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'
_PARTIAL = 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_partial_F187N.fits'


def _load():
    spec = importlib.util.spec_from_file_location('m8_merge_partials', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path):
    m7 = Table({'id': np.arange(3), 'flux_f187n': np.zeros(3)})
    m7.write(tmp_path / _M7)
    Table({'flux_f187n': np.ones(3),
           'forced_filled_f187n': np.ones(3, bool)}).write(tmp_path / _PARTIAL)
    return ['--m7', str(tmp_path / _M7), '--filters', 'F187N', '--execute',
            '--no-spike-flag']


@pytest.fixture
def dedup_calls(monkeypatch):
    calls = []

    def fake(src, dst, *, link_radius, winner_emag_ratio, band_resolve_mas):
        calls.append({'winner_emag_ratio': winner_emag_ratio,
                      'band_resolve_mas': band_resolve_mas})
        return dst
    monkeypatch.setattr(dedup_catalog, 'dedup_merged_catalog', fake)
    return calls


def test_merge_partials_forwards_the_dedup_options(tmp_path, dedup_calls):
    argv = _tree(tmp_path)
    _load().main(argv + ['--dedup-winner-emag-ratio', '0.8',
                         '--dedup-band-resolve-mas', 'f187n=60'])
    assert dedup_calls == [{'winner_emag_ratio': 0.8,
                            'band_resolve_mas': {'f187n': 60.0}}]


def test_merge_partials_dedup_options_default_off(tmp_path, dedup_calls):
    _load().main(_tree(tmp_path))
    assert dedup_calls == [{'winner_emag_ratio': None, 'band_resolve_mas': None}]


@pytest.mark.parametrize('opt, val', [
    ('--dedup-band-resolve-mas', 'abc'),
    ('--dedup-band-resolve-mas', 'f187n='),
    ('--dedup-band-resolve-mas', '0'),
    ('--dedup-winner-emag-ratio', 'nan'),
    ('--dedup-winner-emag-ratio', '-0.8'),
])
def test_merge_partials_rejects_bad_options_before_any_work(tmp_path, dedup_calls,
                                                            capsys, opt, val):
    argv = _tree(tmp_path)
    with pytest.raises(SystemExit) as ei:
        _load().main(argv + [opt, val])
    assert ei.value.code == 2
    assert '--dedup-' in capsys.readouterr().err
    assert dedup_calls == []
    assert not (tmp_path / _M7.replace('_m7', '_m8')).exists()


@pytest.mark.parametrize('opt, val', [
    ('--m8-dedup-band-resolve-mas', 'abc'),
    ('--m8-dedup-band-resolve-mas', 'f212n=60,,f200w=50'),
    ('--m8-dedup-band-resolve-mas', '-5'),
    ('--m8-dedup-winner-emag-ratio', 'nan'),
    ('--m8-dedup-winner-emag-ratio', '0'),
])
def test_cataloging_rejects_bad_dedup_options_at_parse_time(monkeypatch, capsys,
                                                            opt, val):
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as cat
    monkeypatch.setattr(sys, 'argv', ['crowdsource_catalogs_long', opt, val])
    with pytest.raises(SystemExit) as ei:
        cat.main()
    assert ei.value.code == 2
    assert '--m8-dedup-' in capsys.readouterr().err
