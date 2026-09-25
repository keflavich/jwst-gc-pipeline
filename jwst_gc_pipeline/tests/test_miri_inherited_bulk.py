"""MIRI F770W on 10678 inherits its per-visit BULK tie from NIRCam F212N plus
a fixed instrument differential (#956), so it registers to the served F212N
frame.

Before this, the 10678 consensus table had no F770W rows, so ``resolve_shift``
gave every MIRI frame (0, 0) and o040/o100/o041 stayed 20.1"/10.5"/5.5" off
in the served MIRI mosaics: tweakreg's 0.4" absolute search cannot reach them.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction import alignment_config as AC
from jwst_gc_pipeline.reduction.unified_alignment import resolve_shift

PROP = '10678'
FN = 'jw10678040001_02201_00001_mirimage_align.fits'
VISIT = 'jw10678040001'


def _row(filt, module, exposure, dra, ddec, visit=VISIT, dec=-28.5):
    return dict(Filter=filt, Module=module, Visit=visit, Exposure=exposure,
                Vgroup='', **{'dra (arcsec)': dra, 'ddec (arcsec)': ddec},
                prov_stage='m2', prov_dec_deg=dec)


def _write(tmp_path, rows):
    (tmp_path / 'offsets').mkdir(exist_ok=True)
    Table(rows=[list(r.values()) for r in rows], names=list(rows[0].keys())).write(
        tmp_path / 'offsets' / f'Offsets_JWST_Brick{PROP}_consensus.csv',
        overwrite=True)
    return str(tmp_path)


def _inh():
    return AC.inherited_bulk(PROP, '040', 'mirimage')


def _expected(dra, ddec, filt, dec):
    diff = _inh().differential[filt]
    return (dra + diff[0] / 1e3 / np.cos(np.deg2rad(dec)), ddec + diff[1] / 1e3)


def test_10678_mirimage_is_configured_to_inherit_from_f212n_only():
    inh = _inh()
    assert inh is not None
    assert inh.donor_filters == ('F212N',)
    assert set(inh.differential) == {'F212N'}
    assert inh.disable_abs_tweakreg


def test_nircam_modules_and_other_fields_do_not_inherit():
    assert AC.inherited_bulk(PROP, '040', 'nrcalong') is None
    assert AC.inherited_bulk('2221', '001', 'mirimage') is None


def test_f212n_bulk_plus_differential_is_applied(tmp_path):
    bp = _write(tmp_path, [_row('F212N', 'all', -1, -4.4865, -19.9146),
                           _row('F480M', 'all', -1, -4.40, -19.80)])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    ra, dec = _expected(-4.4865, -19.9146, 'F212N', -28.5)
    assert sh.inherited_from == 'F212N'
    assert sh.donor_bulk == pytest.approx((-4.4865, -19.9146))
    assert sh.total_ra == pytest.approx(ra)
    assert sh.total_dec == pytest.approx(dec)


def test_f480m_is_not_a_donor(tmp_path):
    """o063: F480M took a tie that F212N refused.  Following F480M would put
    MIRI ~225 mas off the served F212N frame."""
    bp = _write(tmp_path, [_row('F480M', 'all', -1, 0.10, -0.20, dec=-29.0)])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    assert sh.inherited_from == ''
    assert (sh.total_ra, sh.total_dec) == (0.0, 0.0)


def test_bulk_row_dec_converts_the_differential(tmp_path):
    bp = _write(tmp_path, [_row('F212N', 'all', -1, 0.10, -0.20, dec=-29.0)])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    ra, dec = _expected(0.10, -0.20, 'F212N', -29.0)
    assert (sh.total_ra, sh.total_dec) == pytest.approx((ra, dec))


def test_donor_jitter_rows_are_not_inherited(tmp_path):
    """Donor per-exposure rows are per-DETECTOR; one of them must not move MIRI."""
    bp = _write(tmp_path, [_row('F212N', 'all', -1, 0.0, 0.0),
                           _row('F212N', 'nrca2', 1, 5.0, 5.0)])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    ra, dec = _expected(0.0, 0.0, 'F212N', -28.5)
    assert (sh.total_ra, sh.total_dec) == pytest.approx((ra, dec))


def test_own_f770w_rows_sum_on_top_as_residuals(tmp_path):
    bp = _write(tmp_path, [_row('F212N', 'all', -1, 1.0, 2.0),
                           _row('F770W', 'all', -1, 0.010, -0.020),
                           _row('F770W', 'mirimage', 1, 0.003, 0.004)])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    ra, dec = _expected(1.0, 2.0, 'F212N', -28.5)
    assert (sh.bulk_ra, sh.bulk_dec) == pytest.approx((ra, dec))
    assert (sh.jitter_ra, sh.jitter_dec) == pytest.approx((0.013, -0.016))


def test_donor_rows_without_bulk_give_the_differential_only(tmp_path):
    """A refused or under-tolerance F212N tie leaves the served F212N frame
    with no bulk; MIRI follows that frame."""
    bp = _write(tmp_path, [_row('F212N', 'nrcb1', 2, 0.03, 0.01),
                           _row('F212N', 'all', -1, 1.0, 1.0,
                                visit='jw10678041001')])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    ra, dec = _expected(0.0, 0.0, 'F212N', _inh().dec_ref_deg)
    assert sh.inherited_from == 'F212N'
    assert sh.donor_bulk is None
    assert (sh.total_ra, sh.total_dec) == pytest.approx((ra, dec))


def test_no_donor_rows_at_all_inherits_nothing(tmp_path):
    """Donor not checkpointed for this visit: nothing to follow, and the
    reducer keeps MIRI's own absolute tie."""
    bp = _write(tmp_path, [_row('F212N', 'all', -1, 1.0, 1.0,
                                visit='jw10678041001')])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    assert sh.inherited_from == ''
    assert (sh.total_ra, sh.total_dec) == (0.0, 0.0)
    assert sh.table_present


def test_missing_table_is_reported_as_missing(tmp_path):
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', str(tmp_path))
    assert not sh.table_present
    assert sh.inherited_from == ''


def test_nircam_frames_are_unchanged(tmp_path):
    bp = _write(tmp_path, [_row('F212N', 'all', -1, -4.4865, -19.9146)])
    sh = resolve_shift('jw10678040001_02101_00001_nrca1_align.fits', PROP, '040',
                       'F212N', 'nrca1', bp)
    assert sh.inherited_from == ''
    assert (sh.total_ra, sh.total_dec) == pytest.approx((-4.4865, -19.9146))


def test_the_miri_reducer_collects_member_shifts():
    from pathlib import Path
    src = (Path(AC.__file__).resolve().parent / 'PipelineMIRI.py').read_text()
    assert '_member_shifts.append(' in src
    assert 'return _shift' in src


# --- review follow-ups (#964) -------------------------------------------------

from jwst_gc_pipeline.reduction.unified_alignment import (  # noqa: E402
    AlignmentShift, inherited_abs_tie_complete, inherited_header_cards)


def _sh(inh_from):
    return AlignmentShift(inherited_from=inh_from)


@pytest.mark.parametrize('shifts, expected', [
    ([_sh('F212N'), _sh('F212N')], True),
    ([_sh('F212N'), _sh('')], False),      # one visit's donor not checkpointed
    ([_sh('F212N'), None], False),         # fix_alignment skipped a frame
    ([], False),
])
def test_abs_tweakreg_gate(shifts, expected):
    assert inherited_abs_tie_complete(_inh(), shifts) is expected


def test_abs_tweakreg_gate_needs_an_inheriting_entry():
    assert not inherited_abs_tie_complete(None, [_sh('F212N')])
    from dataclasses import replace
    assert not inherited_abs_tie_complete(
        replace(_inh(), disable_abs_tweakreg=False), [_sh('F212N')])


def test_the_miri_reducer_uses_the_tested_gate():
    from pathlib import Path
    src = (Path(AC.__file__).resolve().parent / 'PipelineMIRI.py').read_text()
    assert 'inherited_abs_tie_complete(\n                _inh, _member_shifts)' in src
    assert 'abs_refcat = None' in src
    assert 'header.update(inherited_header_cards(_shift))' in src


def test_header_cards_with_a_donor_bulk():
    cards = inherited_header_cards(AlignmentShift(inherited_from='F212N',
                                                  donor_bulk=(-4.4865, -19.9146)))
    assert cards['ALIGNINH'][0] == 'F212N'
    assert cards['ALIGNBLK'][0] is True
    assert (cards['ALIGNDRA'][0], cards['ALIGNDDE'][0]) == (-4.4865, -19.9146)


def test_header_cards_without_a_donor_bulk():
    cards = inherited_header_cards(AlignmentShift(inherited_from='F212N'))
    assert cards['ALIGNBLK'][0] is False
    assert (cards['ALIGNDRA'][0], cards['ALIGNDDE'][0]) == (0.0, 0.0)
    assert inherited_header_cards(AlignmentShift())['ALIGNINH'][0] == 'none'


@pytest.mark.parametrize('module, exposure', [('all', 1), ('nrca1', -1)])
def test_only_the_all_minus1_row_is_a_donor_bulk(tmp_path, module, exposure):
    """Module='all' with a real exposure, or Exposure=-1 on a detector, is
    not the BULK sentinel: the donor has rows but no bulk."""
    bp = _write(tmp_path, [_row('F212N', module, exposure, 3.0, 3.0)])
    sh = resolve_shift(FN, PROP, '040', 'F770W', 'mirimage', bp)
    assert sh.inherited_from == 'F212N'
    assert sh.donor_bulk is None
    ra, dec = _expected(0.0, 0.0, 'F212N', _inh().dec_ref_deg)
    assert (sh.total_ra, sh.total_dec) == pytest.approx((ra, dec))


def _bulk_corr(visit='jw10678040001'):
    return dict(visit=visit, exposure=None, module=None, filtername='F770W',
                vgroup='', dra_onsky_mas=60.0, ddec_onsky_mas=40.0,
                dec_deg=-28.8, source='m2 tie')


def _exp_corr():
    return dict(visit='jw10678040001', exposure=1, module='mirimage',
                filtername='F770W', vgroup='', dra_onsky_mas=3.0,
                ddec_onsky_mas=-2.0, dec_deg=-28.8, source='m2')


@pytest.mark.parametrize('module', ['mirimage', 'merged', 'nrcb'])
def test_m2_checkpoint_drops_miri_bulk_rows_on_an_inheriting_field(module):
    """A MIRI m2 bulk row vs VIRAC2 would sum on top of the inherited F212N
    bulk and pull refused-tie visits off the F212N frame (#964 review)."""
    from jwst_gc_pipeline.photometry.cataloging import (
        _drop_inherited_bulk_corrections)
    kept = _drop_inherited_bulk_corrections(
        [_bulk_corr(), _exp_corr()], PROP, '040', module, 'miri', 'test')
    assert kept == [_exp_corr()]


def test_m2_checkpoint_keeps_bulk_rows_where_nothing_is_inherited():
    from jwst_gc_pipeline.photometry.cataloging import (
        _drop_inherited_bulk_corrections)
    corrs = [_bulk_corr(), _exp_corr()]
    assert _drop_inherited_bulk_corrections(
        corrs, PROP, '040', 'nrca1', 'nircam', 'test') == corrs
    assert _drop_inherited_bulk_corrections(
        corrs, '2221', '001', 'mirimage', 'miri', 'test') == corrs


def test_m2_checkpoint_calls_the_guard_before_writing():
    from pathlib import Path
    src = (Path(AC.__file__).resolve().parents[1] / 'photometry'
           / 'cataloging.py').read_text()
    i_guard = src.index('corrections = _drop_inherited_bulk_corrections(')
    i_write = src.index('seed_offsets_table_from_consensus(\n')
    assert i_guard < i_write
