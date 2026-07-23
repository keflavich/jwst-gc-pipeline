"""Unit tests for the per-frame fan-out (option C) state-reconstruction helpers.

The per-frame SLURM split replaces four in-memory cross-phase dicts with on-disk
reconstruction so a fresh process running ONE phase reproduces a monolithic run.
These tests pin the round-trips that make that safe:

* ``_persist_reconciled_satstars`` / ``_reconstruct_reconciled_satstars``
  (the only genuinely-new artifact) -- overrides AND drops survive intact;
* ``_reconstruct_resid_i2d_path`` derives the detection-image name from the
  smoothed-bg name (the residual i2d differs only by the ``_smoothed_bg`` infix);
* ``_reconstruct_prev_merged`` recovers (skycoord, iter_found) from a merged
  catalog on disk;
* the frame-shard predicate (index %% N == I) partitions every frame exactly once
  for any N -> any NSHARDS gives full coverage with no double-fit.

Importing cataloging pulls crowdsource_catalogs_long (webbpsf) -> slow cold.
"""
import inspect
import os
import re

import numpy as np
import pytest

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging as C


class TestReconciledSatstarRoundTrip:
    def test_overrides_and_drops_survive(self, tmp_path):
        path = str(tmp_path / 'satstar_reconciled_m12.fits')
        ovr = [(SkyCoord(266.5 * u.deg, -28.8 * u.deg), 1234.0),
               (SkyCoord(266.6 * u.deg, -28.7 * u.deg), 5.5e4)]
        drp = [SkyCoord(266.7 * u.deg, -28.9 * u.deg)]
        C._persist_reconciled_satstars(path, ovr, drp)
        assert os.path.exists(path)
        ovr2, drp2 = C._reconstruct_reconciled_satstars(path)

        assert len(ovr2) == 2 and len(drp2) == 1
        # overrides: positions + fluxes preserved (order preserved on write)
        for (sc, fl), (sc2, fl2) in zip(ovr, ovr2):
            assert sc.separation(sc2).arcsec < 1e-6
            assert fl == pytest.approx(fl2)
        # drop: position preserved, carried as a SkyCoord with NaN flux internally
        assert drp[0].separation(drp2[0]).arcsec < 1e-6

    def test_empty_roundtrips_to_empty(self, tmp_path):
        path = str(tmp_path / 'empty.fits')
        C._persist_reconciled_satstars(path, [], [])
        ovr2, drp2 = C._reconstruct_reconciled_satstars(path)
        assert ovr2 == [] and drp2 == []

    def test_missing_file_is_empty_not_error(self, tmp_path):
        ovr2, drp2 = C._reconstruct_reconciled_satstars(
            str(tmp_path / 'does_not_exist.fits'))
        assert ovr2 == [] and drp2 == []


class TestResidI2dPathDerivation:
    def test_derives_from_smoothed_bg(self):
        opts = type('O', (), dict(desaturated=False, bgsub=False, group=False))()
        kw = dict(cut_bp='/x', proposal_id='2221', field='001',
                  module='nrcb', filt='F405N', label='m5', options=opts,
                  pupil='clear')
        bg = C._reconstruct_smoothed_bg_path(**kw)
        resid = C._reconstruct_resid_i2d_path(**kw)
        # the residual i2d is the smoothed-bg name minus the _smoothed_bg infix
        assert bg.endswith('_mergedcat_residual_smoothed_bg_i2d.fits')
        assert resid.endswith('_mergedcat_residual_i2d.fits')
        assert resid == bg.replace('_smoothed_bg_i2d.fits', '_i2d.fits')


class TestStartPhasePrevLabel:
    """Regression for the _prev vs _prev_label crash (--manual-start-phase).

    The m12 phase writes every product under merge_label='m2'
    (``merge_label = 'm2' if phase == 'm12' else phase``), so a job starting at
    m3 must reconstruct the previous phase's smoothed-bg / residual-i2d under
    the *m2* token.  Passing raw ``_prev='m12'`` builds a ``*_m12_*`` path that
    no phase ever writes -> FileNotFoundError for every per-frame job that
    starts at m3.
    """

    def _path_for(self, label):
        opts = type('O', (), dict(desaturated=False, bgsub=False, group=False))()
        return C._reconstruct_smoothed_bg_path(
            cut_bp='/x', proposal_id='2221', field='001', module='nrcb',
            filt='F405N', label=label, options=opts, pupil='clear')

    def test_label_token_embedded_verbatim(self):
        # the label lands verbatim in the filename: the m2-labeled path (what
        # m12 writes) and the m12-labeled path (what nothing writes) differ
        assert '_m2_' in self._path_for('m2')
        assert '_m12_' in self._path_for('m12')
        assert self._path_for('m2') != self._path_for('m12')

    def test_start_phase_reconstruction_passes_prev_label(self):
        # call-site guard: inside run_manual_pipeline's start-phase block, the
        # path reconstructors must receive the mapped _prev_label, never raw
        # _prev (note: r'_prev\b' cannot match inside '_prev_label')
        src = inspect.getsource(C.run_manual_pipeline)
        assert "_prev_label = 'm2' if _prev == 'm12' else _prev" in src
        calls = re.findall(
            r'_reconstruct_(?:smoothed_bg|resid_i2d)_path\(([^)]*)\)', src)
        assert calls, "expected reconstruct-path calls in run_manual_pipeline"
        prev_calls = [c for c in calls if '_prev' in c]
        assert prev_calls, "expected _prev_label-based reconstruct-path calls"
        for args in prev_calls:
            assert not re.search(r'_prev\b(?!_label)', args), (
                f"raw _prev passed to a reconstruct-path helper "
                f"(must be _prev_label): ({args})")


class TestPrevMergedReconstruction:
    def test_reads_skycoord_and_iter_found(self, tmp_path):
        path = str(tmp_path / 'merged.fits')
        sc = SkyCoord([266.5, 266.6] * u.deg, [-28.8, -28.7] * u.deg)
        Table({'skycoord': sc, 'iter_found': np.array([2, 5], dtype=int),
               'flux': [1.0, 2.0]}).write(path, overwrite=True)
        out = C._reconstruct_prev_merged(path)
        assert out is not None
        sc2, ifound = out
        assert np.allclose(np.asarray(ifound), [2, 5])
        assert sc[0].separation(sc2[0]).arcsec < 1e-6

    def test_missing_or_columnless_returns_none(self, tmp_path):
        assert C._reconstruct_prev_merged(str(tmp_path / 'nope.fits')) is None
        p = str(tmp_path / 'nocols.fits')
        Table({'flux': [1.0]}).write(p, overwrite=True)
        assert C._reconstruct_prev_merged(p) is None


class TestFrameShardCoverage:
    @pytest.mark.parametrize('n_frames,n_shards', [(48, 16), (5, 8), (10, 1), (50, 50)])
    def test_partition_is_complete_and_disjoint(self, n_frames, n_shards):
        # mirror the in-loop predicate: frame j goes to shard j % n_shards
        seen = []
        for shard_i in range(n_shards):
            seen.extend(j for j in range(n_frames) if j % n_shards == shard_i)
        # every frame fit exactly once, regardless of N vs frame count
        assert sorted(seen) == list(range(n_frames))
        assert len(seen) == len(set(seen))
