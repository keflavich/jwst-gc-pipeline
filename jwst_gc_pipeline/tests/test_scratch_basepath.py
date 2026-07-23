"""Tests for the non-destructive GC_BASEPATH_OVERRIDE redirect (PR #143).

The whole value of this feature is a safety guarantee -- a redirected run must
never write through a symlink into the released tree -- so these tests exercise
BOTH the pure override helper and the staging script's symlink-safety invariant.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from jwst_gc_pipeline.scratch_basepath import apply_basepath_override

SCRIPT = (Path(__file__).resolve().parents[2]
          / 'scripts' / 'reduction' / 'stage_scratch_basepath.sh')


# ---- override helper ------------------------------------------------------
def test_override_unset(monkeypatch):
    monkeypatch.delenv('GC_BASEPATH_OVERRIDE', raising=False)
    assert apply_basepath_override('/orange/adamginsburg/jwst/w51/') \
        == '/orange/adamginsburg/jwst/w51/'


def test_override_blank_is_noop(monkeypatch):
    monkeypatch.setenv('GC_BASEPATH_OVERRIDE', '   ')
    assert apply_basepath_override('/orange/adamginsburg/jwst/w51/') \
        == '/orange/adamginsburg/jwst/w51/'


def test_override_set_normalises_trailing_slash(monkeypatch):
    monkeypatch.setenv('GC_BASEPATH_OVERRIDE', '/scratch/x/w51')
    assert apply_basepath_override('/orange/x/w51/') == '/scratch/x/w51/'
    monkeypatch.setenv('GC_BASEPATH_OVERRIDE', '/scratch/x/w51///')
    assert apply_basepath_override('/anything/') == '/scratch/x/w51/'


# ---- staging script safety ------------------------------------------------
def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b'x')


def _build_real(real: Path):
    pl = real / 'F480M' / 'pipeline'
    # read-only INPUTS
    _touch(pl / 'jw06151001001_03109_00001_nrcblong_align_o001_crf.fits')
    _touch(pl / 'jw06151001001_03109_00001_nrcblong_cal.fits')
    _touch(pl / 'jw06151-o001_t001_nircam_clear-f480m-merged_data_i2d.fits')
    # OUTPUTS the pipeline writes with overwrite=True -- must NEVER be symlinked
    for out in ('jw06151-o001_t001_nircam_clear-f480m-merged_i2d.fits',
                'jw06151-o001_t001_nircam_clear-f480m-merged_residual_infilled_i2d.fits',
                'jw06151-o001_t001_nircam_clear-f480m-merged_satstar_flags_i2d.fits',
                'jw06151-o001_t001_nircam_clear-f480m-merged_m2_daophot_basic_mergedcat_model_i2d.fits',
                'jw06151001001_03109_00001_nrcblong_align_o001_crf_satstar_catalog.fits'):
        _touch(pl / out)
    _touch(real / 'catalogs' / 'f480m_consolidated_satstar_catalog.fits')   # own filter output
    _touch(real / 'catalogs' / 'f405n_consolidated_satstar_catalog.fits')   # partner seed (input)
    _touch(real / 'catalogs' / 'gaia_refcat.fits')
    _touch(real / 'offsets' / 'Offsets_JWST_Brick6151_consensus.csv')       # m2 REWRITES in place
    _touch(real / 'reduction' / 'fwhm_table.ecsv')
    _touch(real / 'regions_' / 'x.reg')
    _touch(real / 'psfs' / 'nircam_nrcb5_f480m_fovp101_samp2_npsf16.fits')


def _underscore_free_root():
    # The staging guard rejects underscores anywhere in the path (they break the
    # downstream filename.split('_') detector parse).  tempfile.mkdtemp draws its
    # random suffix from [a-z0-9_], so the name CAN contain '_' (observed on CI:
    # /tmp/tmpr5j_n6me -> staging exits 2).  Retry until the full path is
    # underscore-free; skip if the temp parent itself is unavoidably underscored.
    for _ in range(50):
        d = tempfile.mkdtemp()
        if '_' not in d:
            return d
        shutil.rmtree(d, ignore_errors=True)
    pytest.skip("no underscore-free temp dir available (TMPDIR path contains '_')")


@pytest.mark.parametrize('mode', ['reduce', 'catalog'])
def test_staging_never_symlinks_outputs_into_real(mode):
    root = _underscore_free_root()
    try:
        real = Path(root) / 'realbase' / 'w51'
        scratch = Path(root) / 'scratchbase' / 'w51'
        _build_real(real)

        r = subprocess.run(
            ['bash', str(SCRIPT), str(real), str(scratch), mode,
             'align_o001_crf', 'F480M'],
            capture_output=True, text=True)
        # The script's own self-check exits 3 on any output-pattern symlink into
        # REAL; a clean exit is itself the primary assertion (catches BLOCKER 1).
        assert r.returncode == 0, f"stage failed:\n{r.stdout}\n{r.stderr}"

        # Independent verification: walk scratch, no symlink resolves into REAL
        # with an output-pattern name.
        for dirpath, _, files in os.walk(scratch):
            for fn in files:
                p = Path(dirpath) / fn
                if p.is_symlink():
                    tgt = os.path.realpath(p)
                    is_output = (fn.endswith('_i2d.fits')
                                 or fn.endswith('_satstar_catalog.fits')
                                 or fn.endswith('.csv')
                                 or 'mergedcat' in fn)
                    assert not (tgt.startswith(str(real)) and is_output), \
                        f"OUTPUT symlinked into REAL: {p} -> {tgt}"

        spl = scratch / 'F480M' / 'pipeline'
        cal = spl / 'jw06151001001_03109_00001_nrcblong_cal.fits'
        crf = spl / 'jw06151001001_03109_00001_nrcblong_align_o001_crf.fits'
        assert cal.is_symlink(), "cal must be staged (input, both modes)"
        if mode == 'catalog':
            assert crf.is_symlink(), "crf is a cataloging input"
            di = spl / 'jw06151-o001_t001_nircam_clear-f480m-merged_data_i2d.fits'
            assert di.exists() and not di.is_symlink(), "data_i2d must be COPIED"
        else:  # reduce: crf is an OUTPUT -> must not be staged at all
            assert not crf.exists(), "crf must not be staged in reduce mode"

        # offsets COPIED (the m2 checkpoint rewrites them in place)
        off = scratch / 'offsets' / 'Offsets_JWST_Brick6151_consensus.csv'
        assert off.exists() and not off.is_symlink(), "offsets must be COPIED, not symlinked"

        # own-filter consolidated NOT staged (it is an output); partner IS, as a COPY
        assert not (scratch / 'catalogs' / 'f480m_consolidated_satstar_catalog.fits').exists()
        partner = scratch / 'catalogs' / 'f405n_consolidated_satstar_catalog.fits'
        assert partner.exists() and not partner.is_symlink(), "partner seed must be COPIED"
    finally:
        shutil.rmtree(root, ignore_errors=True)
