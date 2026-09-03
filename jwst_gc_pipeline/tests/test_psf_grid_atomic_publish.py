"""A PSF grid must not be visible in the cache until it is complete.

``stpsf``'s ``psf_grid(save=True)`` names its own output file and writes it
straight into ``outdir``, so the file EXISTS from ``hdu.writeto``'s create
until its close -- and every reader in this pipeline gates on existence alone
(``saturated_star_finding.get_psf``: ``if os.path.exists(str(psf_fn))``).  A
9438 m12 fan-out shard read a 536 MB ``nircam_nrcb5_f480m_fovp1024_samp2_
npsf16.fits`` thirty seconds before the shard writing it finished (read
2026-09-01T20:17:40, write completed 20:18:10) and died inside astropy with
``TypeError: buffer is too small for requested array`` 16 h 39 m into its own
work; the ``afterok`` finalize went with it (#617, #618).

Two tests, because neither alone is enough:

* :func:`test_a_reader_that_arrives_mid_write_sees_no_partial_grid` shows the
  mechanism, deterministically -- the reader runs at a fixed point INSIDE the
  write rather than racing it, so there is no sleep and no flake -- and shows
  that the same reader against an in-place write gets exactly the error from
  the issue.
* :func:`test_every_saved_psf_grid_is_published_not_written_in_place` pins the
  call sites.  The real failure needs a cold cache and two concurrent shards,
  which CI has neither of, so the structural check is what stops a revert.
"""
import ast
import io
import os
import pathlib
import subprocess

import numpy as np
from astropy.io import fits

from jwst_gc_pipeline.atomic_io import publish_into

ROOT = pathlib.Path(__file__).resolve().parents[2]


# --- the mechanism ----------------------------------------------------------

def _grid_bytes(npsf=4, npix=64):
    """The bytes of a small stand-in for a gridded PSF file."""
    data = np.arange(npsf * npix * npix, dtype='float32').reshape(npsf, npix, npix)
    buf = io.BytesIO()
    fits.PrimaryHDU(data).writeto(buf)
    return buf.getvalue(), (npsf, npix, npix)


def _write_in_halves(path, payload, reader):
    """Write ``payload`` to ``path`` in two goes, running ``reader`` between.

    A stand-in for ``psf_grid(save=True)``, which likewise creates the file and
    then streams half a gigabyte into it.  Calling the reader between the two
    writes is the race with the scheduler taken out of it.
    """
    half = len(payload) // 2
    with open(path, 'wb') as fh:
        fh.write(payload[:half])
        fh.flush()
        reader()
        fh.write(payload[half:])


def test_a_reader_that_arrives_mid_write_sees_no_partial_grid(tmp_path):
    cache = tmp_path / 'psfs'
    cache.mkdir()
    payload, shape = _grid_bytes()
    name = 'nircam_nrcb5_f480m_fovp1024_samp2_npsf16.fits'
    published = cache / name

    seen = []

    def reader():
        """What get_psf does: existence check, then load."""
        if not os.path.exists(str(published)):
            seen.append(('absent', None))
            return
        try:
            seen.append(('loaded', fits.getdata(str(published)).shape))
        except (TypeError, OSError) as ex:
            seen.append(('raised', f'{type(ex).__name__}: {ex}'))

    # 1. In place -- what the code did before this fix.  The reader finds a
    #    file that exists and cannot be loaded.
    _write_in_halves(published, payload, reader)
    assert seen == [('raised', 'TypeError: buffer is too small for requested '
                               'array')], seen
    published.unlink()
    seen.clear()

    # 2. Published -- the writer names its own file inside a private directory
    #    and it is renamed into the cache when it is whole.  The same reader,
    #    at the same point in the write, finds nothing at all.
    with publish_into(str(cache)) as build_dir:
        _write_in_halves(pathlib.Path(build_dir) / name, payload, reader)
        assert seen == [('absent', None)], seen
    seen.clear()
    reader()
    assert seen == [('loaded', shape)], seen
    # and no debris left where a reader could glob it
    assert [p.name for p in cache.iterdir()] == [name]


def test_two_builders_of_the_same_grid_both_succeed(tmp_path):
    """The cold-cache case: N shards all find the grid missing at once.

    Both builds are in flight at the same time (nested, so neither has
    published while the other is writing).  Both finish, one rename wins, and
    what a reader loads is one whole grid rather than a mixture of two.
    """
    cache = tmp_path / 'psfs'
    cache.mkdir()
    name = 'nircam_nrca5_f212n_fovp512_samp2_npsf16.fits'
    published = cache / name

    def build(build_dir, value):
        data = np.full((4, 32, 32), value, dtype='float32')
        fits.PrimaryHDU(data).writeto(os.path.join(build_dir, name))

    with publish_into(str(cache)) as first:
        build(first, 1.0)
        with publish_into(str(cache)) as second:
            build(second, 2.0)
            assert first != second, (
                'two builders must not share a staging directory: they would '
                'interleave inside it and the rename would publish the mixture')
            # both grids written, neither published
            assert not published.exists()
        # the inner build published; it is whole
        assert np.unique(fits.getdata(str(published))).tolist() == [2.0]
    # the outer build published over it; it is whole too
    loaded = fits.getdata(str(published))
    assert loaded.shape == (4, 32, 32)
    assert np.unique(loaded).tolist() == [1.0], (
        'a published grid is one build, not a blend of two')
    assert [p.name for p in cache.iterdir()] == [name]


# --- the call sites ---------------------------------------------------------

def _tracked_py():
    out = subprocess.run(['git', 'ls-files', '*.py'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.split() if '/tests/' not in p]


def _saving_psf_grid_calls(tree):
    """Every ``*.psf_grid(..., save=True, ...)`` Call node in ``tree``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'psf_grid'):
            continue
        for kw in node.keywords:
            if (kw.arg == 'save' and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True):
                yield node


def _publish_into_spans(tree):
    """Line ranges of every ``with publish_into(...)`` body."""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            name = getattr(call, 'func', None)
            if isinstance(call, ast.Call) and (
                    (isinstance(name, ast.Name) and name.id == 'publish_into')
                    or (isinstance(name, ast.Attribute)
                        and name.attr == 'publish_into')):
                spans.append((node.body[0].lineno, node.body[-1].end_lineno))
    return spans


def test_every_saved_psf_grid_is_published_not_written_in_place():
    """An AST check, not a grep: the call must be INSIDE a publish_into body.

    A grep for the name in nearby lines would pass on a call that merely sits
    beside one.
    """
    offenders = []
    checked = 0
    for path in _tracked_py():
        # A SyntaxError is deliberately NOT caught: a tracked .py this
        # interpreter cannot parse is worth failing on, and skipping it would
        # be exactly how a new unguarded site hides.
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        tree = ast.parse(text)
        spans = _publish_into_spans(tree)
        for call in _saving_psf_grid_calls(tree):
            checked += 1
            if not any(lo <= call.lineno <= hi for lo, hi in spans):
                offenders.append(f'{path.relative_to(ROOT)}:{call.lineno}')
    assert checked >= 3, (
        f'the matcher found only {checked} saving psf_grid call(s); it has '
        'stopped seeing the call sites it is supposed to guard')
    assert not offenders, (
        'psf_grid(save=True) writes its file in place, so the file exists '
        'while it is still being written and a reader gating on '
        'os.path.exists loads a truncated grid (#617).  Wrap these in '
        '`with publish_into(<cache dir>) as build_dir:` and point the write '
        'at build_dir:\n  ' + '\n  '.join(offenders))


def test_the_satstar_build_is_one_of_the_guarded_sites():
    """Name the site the issue is about, so the guard cannot go vacuous.

    ``get_psf`` is the reader AND the builder of the shared cache, and it is
    the one that cost 16.6 h.
    """
    path = ROOT / 'jwst_gc_pipeline' / 'reduction' / 'saturated_star_finding.py'
    tree = ast.parse(path.read_text())
    spans = _publish_into_spans(tree)
    calls = list(_saving_psf_grid_calls(tree))
    assert calls, 'saturated_star_finding no longer builds a PSF grid'
    for call in calls:
        assert any(lo <= call.lineno <= hi for lo, hi in spans), (
            f'saturated_star_finding.py:{call.lineno} builds a PSF grid '
            'outside publish_into')
        outdirs = [kw.value for kw in call.keywords if kw.arg == 'outdir']
        assert outdirs, 'the build must name the directory it writes into'
        assert all(isinstance(v, ast.Name) and v.id != 'path_prefix'
                   for v in outdirs), (
            'outdir must be the publish_into build directory, not the shared '
            'cache: writing into path_prefix is the defect in #617')


def test_the_post_write_check_is_not_a_bare_assert():
    """``assert`` is stripped under ``python -O``.

    The check that the published name is the name the reader will look for is
    the only thing standing between a mis-named grid and a cache that stays
    cold for the length of a run, so it has to raise on its own.
    """
    path = ROOT / 'jwst_gc_pipeline' / 'reduction' / 'saturated_star_finding.py'
    src = path.read_text()
    assert 'assert glob.glob' not in src, (
        'the post-write PSF check is a bare assert again; `python -O` removes '
        'it, and a truncated file satisfies the glob in any case')
    raises = [node for node in ast.walk(ast.parse(src))
              if isinstance(node, ast.Raise)
              and isinstance(node.exc, ast.Call)
              and isinstance(node.exc.func, ast.Name)
              and node.exc.func.id == 'FileNotFoundError']
    assert raises, ('nothing checks that the built grid was published under '
                    'the name the reader looks for')
