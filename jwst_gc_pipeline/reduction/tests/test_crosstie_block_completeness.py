"""The ``--remeasure-crosstie`` block is complete, and its unmeasured zeros are marked.

``crosstie_offset`` refuses to report a zero for a catalog it could not resolve
(``test_crosstie_missing_catalog.py``).  Two ways a fabricated zero still reached the
pasted ``CROSSTIE`` block after that refusal:

* the printer streamed one line per filter, so a raise on filter *k* left filters
  1..k-1 on stdout under the "paste into" header -- a block that reads complete while
  being short a filter, and a filter absent from ``shifts`` is ``(0.0, 0.0)`` in
  ``crosstie_constant``;
* a rigid offset larger than ``CROSSTIE_SEED_WIN`` (0.5") leaves no true pairs in the
  window, so the catalogs resolve, no raise fires, and the declined measurement
  printed ``'f115w': (+0.00000, +0.00000),`` byte-identically to a measured line, on
  exit 0.

The brick/1182 constants such a zero would replace are 18-21 mas.
"""
import ast
import os
import subprocess
import sys

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.reduction import build_virac2_offsets as bvo

PASTE_HEADER = 'paste into'


def _write_cat(path, n=400, ra0=266.5, dec0=-28.7, dra_arcsec=0.0, ddec_arcsec=0.0,
               seed=1):
    """A vetted-catalog-shaped file: skycoord + flux, optionally rigidly shifted."""
    rng = np.random.default_rng(seed)
    ra = ra0 + rng.uniform(-0.01, 0.01, n) + dra_arcsec / 3600.0
    dec = dec0 + rng.uniform(-0.01, 0.01, n) + ddec_arcsec / 3600.0
    t = Table({'skycoord': SkyCoord(ra=ra, dec=dec, unit='deg'),
               'flux': np.full(n, 1000.0)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.write(path, overwrite=True)
    return path


def _src(base, filt):
    return os.path.join(base, 'catalogs',
                        f'{filt}_merged_indivexp_merged_x_m2_dao_basic_vetted.fits')


def _field(tmp_path, filts, missing=(), gross=()):
    """Three-filter synthetic region: measurable filters, plus absent / grossly
    misregistered ones.  Returns (rc, region_key, master_path)."""
    base = str(tmp_path / 'field')
    master = str(tmp_path / 'master' / 'f212n_master.fits')
    _write_cat(master)
    for i, f in enumerate(filts):
        if f in missing:
            continue                       # glob resolves to nothing -> raise
        if f in gross:
            _write_cat(_src(base, f), dra_arcsec=5.0)   # 5" rigid: no true pairs in 0.5"
        else:
            _write_cat(_src(base, f), dra_arcsec=0.020 + 0.005 * i, ddec_arcsec=-0.010)
    rc = dict(proposal='1182', field='004', basepath=base,
              filts={f: (f.upper(), 2022.0, 'm2') for f in filts})
    cfg = dict(master_cat=master, master_name='2221 F212N',
               shifts={f: (+0.01868, -0.00080) for f in filts})
    return rc, cfg


@pytest.fixture
def keyed(tmp_path, monkeypatch):
    def _make(filts, missing=(), gross=()):
        rc, cfg = _field(tmp_path, filts, missing=missing, gross=gross)
        monkeypatch.setitem(bvo.REGION, '_tblock', rc)
        monkeypatch.setitem(bvo.CROSSTIE, '_tblock', cfg)
        return rc
    return _make


# ---------------------------------------------------------------- the CLI itself

_DRIVER = '''
import sys, textwrap
import jwst_gc_pipeline.reduction.build_virac2_offsets as bvo

base, master, filts = {base!r}, {master!r}, {filts!r}
rc = dict(proposal='1182', field='004', basepath=base,
          filts={{f: (f.upper(), 2022.0, 'm2') for f in filts}})
bvo.REGION['_tblock'] = rc
bvo.CROSSTIE['_tblock'] = dict(master_cat=master, master_name='2221 F212N',
                               shifts={{f: (0.0, 0.0) for f in filts}})
src = open(bvo.__file__).read()
main_src = textwrap.dedent(src.split("if __name__ == '__main__':", 1)[1])
sys.argv = ['build_virac2_offsets', '--region', '_tblock', '--remeasure-crosstie'] + list(filts)
exec(compile(main_src, bvo.__file__, 'exec'), bvo.__dict__)
'''


def _run_cli(tmp_path, filts, missing=(), gross=()):
    """Run the module's real ``__main__`` remeasure branch in a subprocess."""
    rc, cfg = _field(tmp_path, filts, missing=missing, gross=gross)
    drv = tmp_path / 'drv.py'
    drv.write_text(_DRIVER.format(base=rc['basepath'], master=cfg['master_cat'],
                                  filts=list(filts)))
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(bvo.__file__)))))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    return subprocess.run([sys.executable, str(drv)], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path))


def test_mixed_miss_prints_no_paste_block_at_all(tmp_path):
    """B1: two filters resolve, the third does not.  The raise must leave NOTHING
    paste-ready on stdout -- a two-filter block under the header reads complete."""
    r = _run_cli(tmp_path, ['fa', 'fb', 'fc'], missing=['fc'])
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'CrosstieCatalogMissingError' in r.stderr
    assert PASTE_HEADER not in r.stdout
    paste = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("'")]
    assert paste == [], paste          # not one measured filter escaped as a block line


def test_gross_misregistration_marks_the_line_and_exits_nonzero(tmp_path):
    """B2: a 5" rigid offset resolves both catalogs and leaves no true pairs in the
    0.5" window.  The zero prints, marked, and the CLI does not exit 0."""
    r = _run_cli(tmp_path, ['fa', 'fb', 'fc'], gross=['fc'])
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    fc = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("'fc'")]
    assert len(fc) == 1, r.stdout
    assert 'NOT MEASURED' in fc[0]
    assert 'candidate pairs' in fc[0]
    # the measured filters are NOT marked
    fa = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("'fa'")][0]
    assert 'NOT MEASURED' not in fa


def test_all_measured_cli_exits_zero(tmp_path):
    """The clean case is unchanged: a full block, no marks, exit 0."""
    r = _run_cli(tmp_path, ['fa', 'fb'])
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert PASTE_HEADER in r.stdout
    assert 'NOT MEASURED' not in r.stdout
    assert len([ln for ln in r.stdout.splitlines() if ln.strip().startswith("'")]) == 2


# ------------------------------------------------------------- crosstie_block API

def test_block_raises_before_producing_text(keyed, tmp_path):
    """The all-or-nothing property at its source: the missing filter is LAST, and
    the two that resolved produce no text."""
    rc = keyed(['fa', 'fb', 'fc'], missing=['fc'])
    with pytest.raises(bvo.CrosstieCatalogMissingError):
        bvo.crosstie_block('_tblock', rc, ['fa', 'fb', 'fc'])


def test_block_marks_only_the_unmeasured_filter(keyed, tmp_path):
    rc = keyed(['fa', 'fb', 'fc'], gross=['fc'])
    text, n_unmeasured = bvo.crosstie_block('_tblock', rc, ['fa', 'fb', 'fc'])
    assert n_unmeasured == 1
    marked = [ln for ln in text.splitlines() if 'NOT MEASURED' in ln]
    assert len(marked) == 2                      # the fc line + the trailing summary
    assert marked[0].strip().startswith("'fc'")


def test_marked_block_still_pastes_as_python_and_keeps_the_comment(keyed):
    """The mark rides along into the source file: pasting the block is still valid
    Python, and the comment stays beside the zero it explains."""
    rc = keyed(['fa', 'fc'], gross=['fc'])
    text, _ = bvo.crosstie_block('_tblock', rc, ['fa', 'fc'])
    body = '\n'.join(ln for ln in text.splitlines() if ln.strip().startswith("'"))
    d = ast.literal_eval('{\n' + body + '\n}')   # comments survive, the dict parses
    assert d['fc'] == (0.0, 0.0) and d['fa'] != (0.0, 0.0)
    assert '# NOT MEASURED' in [ln for ln in body.splitlines()
                                if ln.strip().startswith("'fc'")][0]


def test_detail_reports_none_for_a_real_measurement(keyed):
    """A measured shift carries no reason, so nothing marks a real number."""
    rc = keyed(['fa'])
    ra, de, why = bvo.crosstie_offset_detail('fa', rc)
    assert why is None
    assert (ra, de) != (0.0, 0.0)
    assert bvo.crosstie_offset('fa', rc) == (ra, de)   # the 2-tuple API is unchanged


def test_every_zero_carries_a_reason(keyed):
    """No (0,0,None) exists: each refusal says which one it is."""
    rc = keyed(['fc'], gross=['fc'])
    ra, de, why = bvo.crosstie_offset_detail('fc', rc)
    assert (ra, de) == (0.0, 0.0)
    assert why and 'candidate pairs' in why
