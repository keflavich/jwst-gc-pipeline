"""Optional-dependency wrappers must fail with a clear, actionable error."""
import importlib.util

import pytest

from jwst_gc_pipeline.cmz import coverage_moc, hats_export, hipsgen


def _have(mod):
    return importlib.util.find_spec(mod) is not None


@pytest.mark.skipif(_have('mocpy'), reason='mocpy installed')
def test_coverage_moc_requires_mocpy():
    with pytest.raises(ImportError, match='mocpy'):
        coverage_moc._require_mocpy()


@pytest.mark.skipif(_have('hats_import'), reason='hats-import installed')
def test_hats_export_requires_hats_import():
    with pytest.raises(ImportError, match='hats-import'):
        hats_export._require_hats_import()


def test_hipsgen_requires_jar(monkeypatch):
    # no jar env set -> clear error naming the env var
    monkeypatch.delenv('HIPSGEN_JAR', raising=False)
    monkeypatch.setattr(hipsgen.shutil, 'which', lambda _x: '/usr/bin/java')
    with pytest.raises(RuntimeError, match='HIPSGEN_JAR'):
        hipsgen.build_mono_hips('in', 'out')


def test_hipsgen_requires_java(monkeypatch):
    monkeypatch.setattr(hipsgen.shutil, 'which', lambda _x: None)
    with pytest.raises(RuntimeError, match='Java'):
        hipsgen.build_catalog_hips('cat.fits', 'out')
