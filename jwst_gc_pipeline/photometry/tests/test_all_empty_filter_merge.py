"""A filter whose every per-frame catalog is empty must not die in the merge.

gc3571 o003 F770W/F1280W/F1500W and o004 F770W have no detection in any of
their 8 frames (daofind finds nothing in faint, mostly extended MIRI emission).
``combine_singleframe`` skips an empty frame and lets the first non-empty one
seed the base catalog, so with NO non-empty frame the base stayed ``None`` and
the next line, ``len(basecrds)``, raised ``TypeError: object of type 'NoneType'
has no len()``.  The m12 finalize (49062228, 2026-11-03) died on it after a
14 h queue wait.

Policy (user decision, 2026-11-03): the merge refuses an all-empty filter
(``AllFramesEmptyError``) unless ``allow_all_empty=True``, which
``cataloging`` passes only for the pairs in
``FieldAlignment.empty_filters``; the product is then a zero-row catalog and
the m2 checkpoint, which has no zero-source branch, is skipped LOUDLY.
"""
import glob
import inspect
import json
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging, merge_catalogs as MC
from jwst_gc_pipeline.photometry.observation_merge import (
    merge_frames_for_observation)
from jwst_gc_pipeline.reduction import alignment_config as AC

NFRAMES = 3


def _skeleton(nrows=0, ra0=266.4, dec0=-29.0):
    """A dao per-frame catalog with the columns the detector's empty-seed path
    writes (gc3571 o003 F770W m2), holding ``nrows`` sources."""
    t = Table()
    t['x_init'] = np.zeros(nrows)
    t['y_init'] = np.zeros(nrows)
    t['flux_init'] = np.ones(nrows, dtype='f4')
    t['id'] = np.arange(nrows, dtype='i8')
    t['group_id'] = np.arange(nrows, dtype='i8')
    t['group_size'] = np.ones(nrows, dtype='i8')
    t['local_bkg'] = np.zeros(nrows, dtype='f4')
    t['x_fit'] = np.zeros(nrows)
    t['y_fit'] = np.zeros(nrows)
    t['flux_fit'] = np.full(nrows, 10.0, dtype='f4')
    t['x_err'] = np.zeros(nrows)
    t['y_err'] = np.zeros(nrows)
    t['flux_err'] = np.ones(nrows, dtype='f4')
    t['npixfit'] = np.ones(nrows, dtype='i8')
    t['qfit'] = np.full(nrows, 0.1, dtype='f4')
    t['cfit'] = np.full(nrows, 0.1, dtype='f4')
    t['flags'] = np.zeros(nrows, dtype='i8')
    t['iter_detected'] = np.ones(nrows, dtype='i8')
    t['skycoord_centroid'] = SkyCoord(
        (ra0 + np.arange(nrows) * 1e-3) * u.deg, np.full(nrows, dec0) * u.deg)
    t['dra'] = np.zeros(nrows, dtype='f4')
    t['ddec'] = np.zeros(nrows, dtype='f4')
    return t


def _write_frames(basepath, nrows_per_frame):
    d = basepath / 'F770W'
    d.mkdir(parents=True, exist_ok=True)
    (basepath / 'catalogs').mkdir(exist_ok=True)
    for i, n in enumerate(nrows_per_frame, start=1):
        t = _skeleton(n)
        t.meta['FILENAME'] = f'/x/frame{i}_crf.fits'
        t.meta['pixscale'] = 0.11   # the per-frame writer stamps it (#1238)
        t.write(d / f'f770w_mirimage_o003_visit001_vgroup02103_exp{i:05d}'
                    f'_m2_daophot_basic.fits')


def _merge(basepath, **kwargs):
    return MC.merge_individual_frames(
        module='mirimage', filtername='f770w', progid='3571', field='003',
        method='dao', suffix='_basic', target='gc3571', basepath=str(basepath),
        iteration_label='m2', do_replace_saturated=False, **kwargs)


def _product(basepath):
    return basepath / 'catalogs' / 'f770w_mirimage_o003_indivexp_merged_m2_dao_basic.fits'


# ---------------------------------------------------------------------------
# merge_individual_frames
# ---------------------------------------------------------------------------

def test_all_empty_frames_are_refused_by_default(tmp_path):
    _write_frames(tmp_path, [0] * NFRAMES)
    with pytest.raises(MC.AllFramesEmptyError) as exc:
        _merge(tmp_path)
    msg = str(exc.value)
    assert f'all {NFRAMES} per-frame catalogs' in msg
    assert 'F770W' in msg and 'empty_filters' in msg
    assert not _product(tmp_path).exists()


def test_the_refusal_is_a_value_error(tmp_path):
    """The cutout merge tolerates ValueError (observation_merge.
    CUTOUT_MERGE_ERRORS); the new error keeps that contract rather than
    escaping as a different type."""
    assert issubclass(MC.AllFramesEmptyError, ValueError)


def test_allowed_all_empty_writes_a_zero_row_product(tmp_path):
    _write_frames(tmp_path, [0] * NFRAMES)
    out = _merge(tmp_path, allow_all_empty=True)
    assert len(out) == 0
    assert out.meta['EMPTYALL'] is True
    on_disk = Table.read(_product(tmp_path))
    assert len(on_disk) == 0
    assert on_disk.meta['EMPTYALL'] is True
    assert isinstance(on_disk['skycoord'], SkyCoord)
    # the columns downstream selects by name
    for col in ('flux_err', 'skycoord', 'qfit', 'cfit', 'flags', 'local_bkg',
                'dra', 'ddec', 'std_ra', 'std_dec', 'nmatch', 'nmatch_good',
                'clip_fallback', 'flux_err_prop'):
        assert col in on_disk.colnames, col


def test_empty_product_carries_the_renamed_flux_column(tmp_path):
    """replace_saturated renames flux_fit -> flux on every other product; the
    empty one goes through the same step."""
    _write_frames(tmp_path, [0] * NFRAMES)
    out = MC.merge_individual_frames(
        module='mirimage', filtername='f770w', progid='3571', field='003',
        method='dao', suffix='_basic', target='gc3571', basepath=str(tmp_path),
        iteration_label='m2', do_replace_saturated=True, allow_all_empty=True)
    assert 'flux' in out.colnames and 'flux_fit' not in out.colnames
    assert {'is_saturated', 'replaced_saturated'} <= set(out.colnames)
    assert len(out) == 0


def test_some_empty_frames_still_merge_without_the_flag(tmp_path):
    """Empty frames among non-empty ones were already an expected case
    (sickle F1500W); the gate must not fire, with or without the opt-in."""
    _write_frames(tmp_path, [0, 2, 0])
    out = _merge(tmp_path)
    assert len(out) == 2
    assert 'EMPTYALL' not in out.meta
    assert 'EMPTYALL' not in Table.read(_product(tmp_path)).meta
    _product(tmp_path).unlink()
    out = _merge(tmp_path, allow_all_empty=True)
    assert len(out) == 2 and 'EMPTYALL' not in out.meta


# ---------------------------------------------------------------------------
# combine_singleframe
# ---------------------------------------------------------------------------

def _combine(tbls):
    for i, t in enumerate(tbls, start=1):
        t.meta.update(exposure=f'{i:05d}', MODULE='mirimage', filter='f770w')
    return MC.combine_singleframe(
        tbls, nanaverage=MC.nanaverage_numpy, verbose=False)


def test_combine_of_only_empty_frames_returns_zero_rows():
    out = _combine([_skeleton(0) for _ in range(NFRAMES)])
    assert len(out) == 0
    assert isinstance(out['skycoord_avg'], SkyCoord)


def test_the_zero_row_columns_are_the_ones_a_non_empty_combine_gives():
    """Drift guard: the zero-row table is written by hand next to the combine,
    so it must name exactly the columns the same inputs produce with a row in
    each frame -- a column added to one and not the other fails here."""
    empty = _combine([_skeleton(0) for _ in range(NFRAMES)])
    full = _combine([_skeleton(1) for _ in range(NFRAMES)])
    assert len(full) == 1
    assert list(empty.colnames) == list(full.colnames)
    for name in empty.colnames:
        if name == 'skycoord_avg':
            continue
        assert empty[name].dtype == full[name].dtype, name


# ---------------------------------------------------------------------------
# observation_merge pass-through
# ---------------------------------------------------------------------------

def test_the_keyword_reaches_the_merge_and_its_table_comes_back():
    seen = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return 'the-table'

    got = merge_frames_for_observation(
        '3571', '003', merge=_fake, module='mirimage', filtername='f770w',
        allow_all_empty=True)
    assert got == 'the-table'
    assert seen['allow_all_empty'] is True


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('field,filt', [
    ('003', 'F770W'), ('003', 'F1280W'), ('003', 'F1500W'), ('004', 'F770W'),
    ('016', 'F770W'), ('016', 'F1280W'), ('016', 'F1500W'),
    ('003', 'f770w')])
def test_the_gc3571_pairs_are_allowed(field, filt):
    assert AC.empty_filter_allowed('3571', field, filt)


@pytest.mark.parametrize('proposal,field,filt', [
    ('3571', '003', 'F560W'),     # detects sources
    ('3571', '003', 'F1000W'),
    ('3571', '004', 'F1280W'),    # o004 F1280W has rows; only F770W is empty
    ('3571', '001', 'F770W'),     # o001 F770W has one row: not all-empty
    ('3571', '005', 'F770W'),     # o005 F770W has 0-1 rows per frame: not all-empty
    ('3571', '016', 'F560W'),     # o016: ~920-1080 rows per frame
    ('3571', '016', 'F1000W'),    # o016: ~700-740 rows per frame
    ('10678', '003', 'F770W'),    # other program, same field and filter
    ('3571', None, 'F770W'),
    ('3571', '003', None),
])
def test_everything_else_stays_strict(proposal, field, filt):
    assert not AC.empty_filter_allowed(proposal, field, filt)


def test_every_allowlist_entry_names_a_real_field_of_its_entry():
    """A typo in a ``'FIELD:FILTER'`` string would never match and the filter
    would stop the merge again, 14 h into a queue wait."""
    seen = 0
    for cfg in AC.ALIGNMENT_CONFIG:
        for entry in cfg.empty_filters:
            seen += 1
            fld, sep, filt = entry.partition(':')
            assert sep == ':' and fld and filt, entry
            assert filt == filt.upper() and filt.startswith('F'), entry
            if cfg.fields is not None:
                assert fld in cfg.fields, (cfg.proposal, entry)
    assert seen >= 4


# ---------------------------------------------------------------------------
# cataloging: the record and the call site
# ---------------------------------------------------------------------------

def test_the_unverified_record_and_log_line(tmp_path, capsys):
    path = cataloging._record_unverified_empty_filter(
        'm2', 'mirimage', 'F770W', str(tmp_path), '3571', '003',
        '/x/f770w_mirimage_o003_indivexp_merged_m2_dao_basic.fits')
    assert os.path.basename(path) == 'unverified_empty_m2_F770W_o003.json'
    rec = json.load(open(path))
    assert rec['status'] == 'UNVERIFIED' and rec['checkpoint'] == 'skipped'
    assert (rec['stage'], rec['filter'], rec['field'], rec['proposal_id']) == (
        'm2', 'F770W', '003', '3571')
    log = capsys.readouterr().out
    assert 'ASTROM CHECKPOINT [m2] UNVERIFIED' in log and 'F770W' in log
    # no checkpoint reader may mistake it for a verdict
    assert not glob.glob(os.path.join(tmp_path, 'astrometry_checkpoints',
                                      'checkpoint_*'))


def test_the_call_site_skips_the_checkpoint_for_an_empty_product():
    """No test drives the 3000-line ``run_manual_pipeline``, so pin the order
    in its source: the allowlist reaches the merge, the EMPTYALL branch
    records and ``continue``s, and all of that sits BEFORE the checkpoint."""
    src = inspect.getsource(cataloging.run_manual_pipeline)
    merge_at = src.index('merge_frames_for_observation(')
    flag_at = src.index("_merged_tab.meta.get('EMPTYALL')", merge_at)
    record_at = src.index('_record_unverified_empty_filter(', flag_at)
    cont_at = src.index('continue', record_at)
    checkpoint_at = src.index('_run_astrometry_stage_checkpoint(', cont_at)
    assert merge_at < flag_at < record_at < cont_at < checkpoint_at
    assert 'allow_all_empty=_empty_ok' in src[merge_at:flag_at]
    assert '_empty_ok = _empty_filter_allowed(proposal_id, field, filt)' in src
