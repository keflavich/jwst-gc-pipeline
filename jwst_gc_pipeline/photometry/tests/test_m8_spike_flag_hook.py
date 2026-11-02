"""The m8 dedup hook runs the spike flag (#1035) and honours both switches.

``cataloging._maybe_dedup_m8`` writes the dedup sibling, then calls
``_maybe_flag_m8_spikes``, then makes the brick proposal-scoped copies.  The
flag step must be skippable with ``--no-m8-spike-flag`` (``options.m8_spike_flag``)
and with ``M8_SPIKE_FLAG=0``, and a failure in it must not skip the copies.
"""
import os
from types import SimpleNamespace

import pytest

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry import dedup_catalog
from jwst_gc_pipeline.photometry import spike_flag


@pytest.fixture
def m8(tmp_path, monkeypatch):
    cat = tmp_path / 'catalogs'
    cat.mkdir()
    path = cat / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits'
    path.write_bytes(b'm8')

    def fake_dedup(src, dst, **kw):
        with open(dst, 'wb') as fh:
            fh.write(b'dedup')
    monkeypatch.setattr(dedup_catalog, 'dedup_merged_catalog', fake_dedup)
    monkeypatch.delenv('M8_SPIKE_FLAG', raising=False)
    calls = []

    def fake_flag(path, basepath, bands=None):
        calls.append((path, basepath))
    monkeypatch.setattr(spike_flag, 'flag_m8_spike_artifacts', fake_flag)
    return str(path), calls


def test_flag_runs_by_default(m8):
    path, calls = m8
    out = cataloging._maybe_dedup_m8(path, SimpleNamespace(), label='t')
    assert out.endswith('_m8_dedup.fits') and os.path.exists(out)
    assert calls == [(out, os.path.dirname(os.path.dirname(path)))]


def test_option_disables_flag(m8):
    path, calls = m8
    out = cataloging._maybe_dedup_m8(path, SimpleNamespace(m8_spike_flag=False))
    assert os.path.exists(out)
    assert calls == []


def test_env_disables_flag(m8, monkeypatch):
    path, calls = m8
    monkeypatch.setenv('M8_SPIKE_FLAG', '0')
    out = cataloging._maybe_dedup_m8(path, SimpleNamespace())
    assert os.path.exists(out)
    assert calls == []


def test_expected_flag_error_is_logged_and_dedup_kept(m8, monkeypatch, capsys):
    path, _ = m8

    def boom(path, basepath, bands=None):
        raise ValueError('no rows')
    monkeypatch.setattr(spike_flag, 'flag_m8_spike_artifacts', boom)
    out = cataloging._maybe_dedup_m8(path, SimpleNamespace(), label='t')
    assert out is not None and os.path.exists(out)
    assert 'm8 spike flag FAILED' in capsys.readouterr().out


def test_unexpected_flag_error_still_makes_brick_copies(m8, monkeypatch, capsys):
    path, _ = m8

    def boom(path, basepath, bands=None):
        raise RuntimeError('unexpected')
    monkeypatch.setattr(spike_flag, 'flag_m8_spike_artifacts', boom)
    out = cataloging._maybe_dedup_m8(path, SimpleNamespace(target='brick', field='004'),
                                     label='t')
    assert out is None
    log = capsys.readouterr().out
    assert 'm8 spike flag FAILED' in log
    assert os.path.exists(path.replace('.fits', '_o004.fits'))
    assert os.path.exists(path.replace('_m8.fits', '_m8_dedup_o004.fits'))
