"""The ``_satstar_severity_dropped.fits`` sidecar is replaced on every rerun.

A frame with no saturated stars has no ``_satstar_catalog``, so the cache in
``load_or_make_satstar_catalog`` never hits and ``remove_saturated_stars`` runs
again with ``overwrite=False`` over the previous run's (empty) sidecar.  The
sidecar must be rewritten anyway: the hand-off reads it, and a stale one is
worse than none.
"""
import inspect

import pytest

from jwst_gc_pipeline.reduction import saturated_star_finding as ssf


def test_publish_refuses_to_replace_without_overwrite(tmp_path):
    path = tmp_path / 'x_satstar_severity_dropped.fits'
    path.write_text('old')
    with pytest.raises(OSError, match='already exists'):
        ssf._publish_satstar_product(str(path), lambda p: open(p, 'w').write('new'),
                                     False)
    assert path.read_text() == 'old'


def test_publish_replaces_with_overwrite(tmp_path):
    path = tmp_path / 'x_satstar_severity_dropped.fits'
    path.write_text('old')
    ssf._publish_satstar_product(str(path), lambda p: open(p, 'w').write('new'),
                                 True)
    assert path.read_text() == 'new'


def test_severity_sidecar_is_published_with_overwrite_on():
    src = inspect.getsource(ssf.remove_saturated_stars)
    tail = src.split("_satstar_severity_dropped.fits'")[1]
    call = tail.split('_publish_satstar_product(')[1].split(')\n')[0]
    # the last positional argument is the overwrite flag
    assert call.rstrip().endswith('True'), call
