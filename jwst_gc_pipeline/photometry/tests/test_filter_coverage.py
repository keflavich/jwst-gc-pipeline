"""Regression tests for the reduced-but-unregistered filter audit (issue #160).

W51 F444W was reduced onto ``align_o001_crf`` alongside the other LW filters but
was missing from ``obs_filters['w51']['6151']``, so no cataloging job ever
requested it and it became the only W51 LW filter with no catalog.  Nothing
failed -- it was simply never asked for -- so only an explicit disk-vs-map audit
catches it.
"""
import pytest

from jwst_gc_pipeline.photometry.filter_coverage import (
    FILTER_DIR_RE, reduced_filters_on_disk, uncataloged_filters, main)
from jwst_gc_pipeline.photometry.merge_catalogs import obs_filters


def _make_field(root, frames_per_filter, suffix='align_o001_crf'):
    """Build a minimal <basepath>/<FILT>/pipeline/<frame>.fits tree."""
    for filt, nframes in frames_per_filter.items():
        pipeline = root / filt / 'pipeline'
        pipeline.mkdir(parents=True)
        for i in range(nframes):
            (pipeline / f'jw06151001001_03101_{i:05d}_nrcalong_{suffix}.fits').write_text('')
    # non-filter siblings that must never be mistaken for a filter directory
    (root / 'catalogs').mkdir()
    (root / 'notes.reg').write_text('')
    return root


def test_filter_dir_regex_accepts_real_filter_names():
    for name in ('F444W', 'F480M', 'F1280W', 'F150W2', 'F322W2', 'F090W'):
        assert FILTER_DIR_RE.match(name), name
    for name in ('catalogs', 'pipeline', 'crds', 'Fsomething', 'audit_plots'):
        assert not FILTER_DIR_RE.match(name), name


def test_reduced_filters_on_disk_counts_frames_and_skips_empty(tmp_path):
    root = _make_field(tmp_path / 'w51', {'F444W': 16, 'F480M': 16})
    (root / 'F200W' / 'pipeline').mkdir(parents=True)  # reduced nothing
    counts = reduced_filters_on_disk(str(root))
    assert counts == {'f444w': 16, 'f480m': 16}
    assert 'f200w' not in counts


def test_uncataloged_filters_flags_the_f444w_gap(tmp_path):
    """The pre-fix W51 map (no f444w) must flag F444W and only F444W."""
    root = _make_field(tmp_path / 'w51', {'F444W': 16, 'F480M': 16, 'F405N': 16})
    prefix_map = {'w51': {'6151': ['f405n', 'f480m']}}
    assert uncataloged_filters(str(root), 'w51', obs_filters_map=prefix_map) == [('f444w', 16)]


def test_uncataloged_filters_empty_once_the_filter_is_registered(tmp_path):
    root = _make_field(tmp_path / 'w51', {'F444W': 16, 'F480M': 16})
    fixed_map = {'w51': {'6151': ['f444w', 'f480m']}}
    assert uncataloged_filters(str(root), 'w51', obs_filters_map=fixed_map) == []


def test_min_frames_and_crf_glob_filters(tmp_path):
    root = _make_field(tmp_path / 'w51', {'F444W': 16, 'F150W': 2})
    prefix_map = {'w51': {'6151': []}}
    assert uncataloged_filters(str(root), 'w51', obs_filters_map=prefix_map,
                               min_frames=8) == [('f444w', 16)]
    # a glob matching no frame reports no gap
    assert uncataloged_filters(str(root), 'w51', obs_filters_map=prefix_map,
                               crf_glob='*destreak_o001_crf.fits') == []


def test_unknown_target_raises_keyerror(tmp_path):
    root = _make_field(tmp_path / 'nosuch', {'F444W': 1})
    with pytest.raises(KeyError, match='not in obs_filters'):
        uncataloged_filters(str(root), 'nosuch', obs_filters_map={'w51': {'6151': []}})


def test_w51_registers_f444w():
    """Issue #160: F444W is reduced for W51, so it must be cataloged too."""
    assert 'f444w' in obs_filters['w51']['6151']


def test_main_exit_code_signals_a_gap(tmp_path, capsys):
    root = _make_field(tmp_path / 'w51', {'F444W': 16})
    # against the real (fixed) map F444W is registered -> no gap, exit 0
    assert main(['--target', 'w51', '--basepath', str(root)]) == 0
    out = capsys.readouterr().out
    assert 'OK' in out
