"""Unified alignment path: equivalence with the frozen legacy dispatch, the
bulk/jitter split, and the per-component staleness guard.

The equivalence tests are the regression gate for this refactor.  brick (1182,
2221) is released data that has already survived two astrometry failures; if the
unified path moved any of those numbers, this file fails.
"""

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.reduction import alignment_config as ac
from jwst_gc_pipeline.reduction._legacy_alignment import legacy_shift
from jwst_gc_pipeline.reduction import unified_alignment as ua


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _offsets_dir(basepath):
    d = os.path.join(basepath, 'offsets')
    os.makedirs(d, exist_ok=True)
    return d


def _write_locked(basepath, proposal, rows, colnames):
    tbl = Table(rows=rows, names=colnames)
    path = os.path.join(_offsets_dir(basepath),
                        f'Offsets_JWST_Brick{proposal}_VIRAC2locked.csv')
    tbl.write(path, overwrite=True)
    return path


def _write_consensus(basepath, proposal, rows):
    tbl = Table(rows=rows,
                names=('Visit', 'Filter', 'Exposure', 'Module',
                       'dra (arcsec)', 'ddec (arcsec)'))
    path = os.path.join(_offsets_dir(basepath),
                        f'Offsets_JWST_Brick{proposal}_consensus.csv')
    tbl.write(path, overwrite=True)
    return path


def _both(fn, proposal, field, filt, module, basepath, **kw):
    """Return (legacy_total, unified_shift)."""
    leg = legacy_shift(fn, proposal, field, filt, basepath, **kw)
    uni = ua.resolve_shift(fn, proposal, field, filt, module, basepath, **kw)
    return leg, uni


def _assert_same(leg, uni, atol=1e-12):
    assert uni.total_ra == pytest.approx(leg[0], abs=atol)
    assert uni.total_dec == pytest.approx(leg[1], abs=atol)


# ---------------------------------------------------------------------------
# equivalence: recorded-bulk fields (pure constants, no table needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fn,proposal,field,filt', [
    # cloudef 2092 obs002: visit 002 shifted onto visit 001, visit 001 untouched
    ('jw02092002002_02101_00003_nrcb3_destreak.fits', '2092', '002', 'F480M'),
    ('jw02092002001_02101_00003_nrcb3_destreak.fits', '2092', '002', 'F480M'),
    # sickle 3958 lived here (per-filter GNS tie, catch-all filter included)
    # until 2026-08-04.  It is now VIRAC2/TABLE_LOCKED, so it is no longer a
    # "no table needed" field -- see test_sickle_is_locked_on_virac2 below.
    # M4 / M92: arcsecond-scale Gaia ties, per (visit, filter)
    ('jw01979002001_02101_00002_nrcb1_destreak.fits', '1979', '002', 'F150W2'),
    ('jw01979003001_02101_00002_nrcb1_destreak.fits', '1979', '003', 'F322W2'),
    ('jw01334001001_02101_00002_nrcb1_destreak.fits', '1334', '001', 'F090W'),
    ('jw01334001001_02101_00002_nrcb1_destreak.fits', '1334', '001', 'F444W'),
    # a visit/filter with no recorded tie -> stays at the raw frame
    ('jw01334009001_02101_00002_nrcb1_destreak.fits', '1334', '001', 'F212N'),
])
def test_recorded_bulk_matches_legacy(tmp_path, fn, proposal, field, filt):
    leg, uni = _both(fn, proposal, field, filt, 'nrcb', str(tmp_path))
    _assert_same(leg, uni)
    # recorded bulk is all bulk, by definition
    assert uni.jitter_ra == 0.0 and uni.jitter_dec == 0.0
    assert uni.source == ac.RECORDED_BULK


def test_recorded_bulk_onsky_conversion_is_exact():
    """The on-sky mas -> coordinate arcsec conversion must reproduce the legacy
    expression exactly, cos(dec) division included."""
    fn = 'jw01334001001_02101_00002_nrcb1_destreak.fits'
    cfg = ac.resolve('1334', '001')
    dra, ddec, found = ac.lookup_recorded_bulk(cfg, 'jw01334001001', 'F090W')
    assert found
    cosd = np.cos(np.radians(43.139))
    assert dra == pytest.approx(-1832.1 / 1000.0 / cosd, abs=1e-15)
    assert ddec == pytest.approx(-708.2 / 1000.0, abs=1e-15)
    # and the RA term is NOT simply the mas value scaled -- cos(dec) matters
    assert abs(dra - (-1832.1 / 1000.0)) > 0.1


# ---------------------------------------------------------------------------
# equivalence: locked tables (brick / cloudc)
# ---------------------------------------------------------------------------

def test_locked_per_visit_table_matches_legacy(tmp_path):
    """One row per (visit, filter): everything is bulk, jitter is zero."""
    bp = str(tmp_path)
    _write_locked(bp, '1182',
                  rows=[('jw01182004001', 'F200W', -17.5, 0.32),
                        ('jw01182004002', 'F200W', 1.9, -0.44)],
                  colnames=('Visit', 'Filter', 'dra (arcsec)', 'ddec (arcsec)'))
    for visit, exp_ra in (('jw01182004001', -17.5), ('jw01182004002', 1.9)):
        fn = f'{visit}_02101_00005_nrcb3_destreak.fits'
        leg, uni = _both(fn, '1182', '004', 'F200W', 'nrcb', bp)
        _assert_same(leg, uni)
        assert uni.total_ra == pytest.approx(exp_ra)
        assert uni.bulk_ra == pytest.approx(exp_ra)
        assert uni.jitter_ra == pytest.approx(0.0)


def test_locked_per_exposure_table_matches_legacy_and_splits(tmp_path):
    """Per-exposure rows: bulk is the visit-level median, jitter the remainder,
    and the TOTAL is untouched."""
    bp = str(tmp_path)
    _write_locked(
        bp, '2221',
        rows=[('jw02221001001', 'F212N', 1, 'nrcb3', 0.100, -0.200),
              ('jw02221001001', 'F212N', 2, 'nrcb3', 0.110, -0.210),
              ('jw02221001001', 'F212N', 3, 'nrcb3', 0.120, -0.190)],
        colnames=('Visit', 'Filter', 'Exposure', 'Module',
                  'dra (arcsec)', 'ddec (arcsec)'))
    fn = 'jw02221001001_02101_00002_nrcb3_destreak.fits'
    leg, uni = _both(fn, '2221', '001', 'F212N', 'nrcb', bp)
    _assert_same(leg, uni)
    assert uni.total_ra == pytest.approx(0.110)
    # median of (0.100, 0.110, 0.120) = 0.110 -> this exposure is at the bulk
    assert uni.bulk_ra == pytest.approx(0.110)
    assert uni.jitter_ra == pytest.approx(0.0)
    # ... while exposure 1 sits 10 mas below it
    fn1 = 'jw02221001001_02101_00001_nrcb3_destreak.fits'
    leg1, uni1 = _both(fn1, '2221', '001', 'F212N', 'nrcb', bp)
    _assert_same(leg1, uni1)
    assert uni1.jitter_ra == pytest.approx(-0.010)
    assert uni1.bulk_ra + uni1.jitter_ra == pytest.approx(uni1.total_ra)


def test_locked_ambiguous_match_still_raises(tmp_path):
    bp = str(tmp_path)
    _write_locked(bp, '1182',
                  rows=[('jw01182004001', 'F200W', -17.5, 0.32),
                        ('jw01182004001', 'F200W', -17.6, 0.31)],
                  colnames=('Visit', 'Filter', 'dra (arcsec)', 'ddec (arcsec)'))
    fn = 'jw01182004001_02101_00005_nrcb3_destreak.fits'
    with pytest.raises(ValueError, match='match=2'):
        ua.resolve_shift(fn, '1182', '004', 'F200W', 'nrcb', bp)


# ---------------------------------------------------------------------------
# equivalence + split: consensus tables (sgrc / w51)
# ---------------------------------------------------------------------------

def test_consensus_matches_legacy_and_splits_bulk_from_jitter(tmp_path):
    bp = str(tmp_path)
    _write_consensus(bp, '6151', rows=[
        # per-visit BULK sentinel (consensus -> VIRAC2)
        ('jw06151001001', 'F115W', -1, 'all', 0.0300, -0.0120),
        # sparse per-exposure JITTER
        ('jw06151001001', 'F115W', 4, 'nrcb3', 0.0025, -0.0031),
    ])
    fn = 'jw06151001001_06101_00004_nrcb3_destreak.fits'
    leg, uni = _both(fn, '6151', '001', 'F115W', 'nrcb', bp)
    _assert_same(leg, uni)
    assert uni.bulk_ra == pytest.approx(0.0300)
    assert uni.jitter_ra == pytest.approx(0.0025)
    assert uni.total_ra == pytest.approx(0.0325)
    assert uni.source == ac.TABLE_CONSENSUS
    assert uni.reference_frame == ac.GAIA


def test_consensus_exposure_without_jitter_row_gets_bulk_only(tmp_path):
    bp = str(tmp_path)
    _write_consensus(bp, '6151', rows=[
        ('jw06151001001', 'F115W', -1, 'all', 0.0300, -0.0120),
        ('jw06151001001', 'F115W', 4, 'nrcb3', 0.0025, -0.0031),
    ])
    fn = 'jw06151001001_06101_00009_nrcb3_destreak.fits'
    leg, uni = _both(fn, '6151', '001', 'F115W', 'nrcb', bp)
    _assert_same(leg, uni)
    assert uni.jitter_ra == pytest.approx(0.0)
    assert uni.total_ra == pytest.approx(0.0300)


def test_consensus_missing_table_leaves_frame_untied(tmp_path):
    """First reduction pass: no table yet, so the frame stays at the raw frame
    and the checkpoint gets to measure the raw scatter."""
    bp = str(tmp_path)
    fn = 'jw06151001001_06101_00004_nrcb3_destreak.fits'
    leg, uni = _both(fn, '6151', '001', 'F115W', 'nrcb', bp)
    _assert_same(leg, uni)
    assert uni.total_ra == 0.0 and uni.total_dec == 0.0
    # but it IS configured -- distinct from a field with no alignment at all
    assert uni.configured


def test_w51_consensus_reference_frame_is_gaia_not_virac(tmp_path):
    """W51 is outside the VVV/VIRAC2 footprint; the frame must not be hardcoded."""
    bp = str(tmp_path)
    _write_consensus(bp, '6151', rows=[
        ('jw06151001001', 'F480M', -1, 'all', 0.010, 0.020),
    ])
    fn = 'jw06151001001_02101_00001_nrcblong_destreak.fits'
    uni = ua.resolve_shift(fn, '6151', '001', 'F480M', 'nrcb', bp)
    assert uni.reference_frame == ac.GAIA
    assert ac.resolve('4147', '012').reference_frame == ac.VIRAC2


# ---------------------------------------------------------------------------
# the failure this refactor exists to expose
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('proposal,field,target,source,ref_filter', [
    ('2045', '001', 'arches', ac.TABLE_CONSENSUS, 'F212N'),
    ('2045', '003', 'quintuplet', ac.TABLE_LOCKED, 'F212N'),
    ('5365', '001', 'sgrb2', ac.TABLE_LOCKED, 'F212N'),
    ('2092', '005', 'cloudef obs005', ac.TABLE_LOCKED, 'F210M'),
    ('3958', '007', 'sickle', ac.TABLE_LOCKED, 'F210M'),
])
def test_campaign_fields_are_now_tied(proposal, field, target, source, ref_filter):
    """These four fell through the old ``else`` and got (0,0) with no trace, so
    their m2 corrections went into a table nothing read and their re-tie loops
    could never converge.

    Each is declared against the table its own BUILDER writes -- arches has no
    build_virac2_offsets REGION entry and only a checkpoint-written consensus
    table, while quintuplet/sgrb2/cloudef have REGION entries and builder-shaped
    VIRAC2locked tables.  Declaring all four 'consensus' would have pointed the
    reducer at files that do not exist for three of them."""
    cfg = ac.resolve(proposal, field)
    assert cfg is not None, f"{target} still has no alignment source"
    assert cfg.source == source
    assert cfg.reference_frame == ac.VIRAC2
    assert cfg.reference_filter == ref_filter


def test_sickle_is_locked_on_virac2_not_recorded_on_gns():
    """sickle was the last GC field tied in the GNS frame while ``refnames``
    already called it VIRAC2.  GC policy is that GC fields tie to VIRAC2.

    It is TABLE_LOCKED rather than RECORDED_BULK because the route for an
    already-tied field is to BUILD the VIRAC2 table -- step 0 refuses to record a
    fresh tie for a field that already has one ("the field is tied; this (visit,
    band) is not").  The old GNS constants are deliberately NOT carried over:
    re-using them against VIRAC2 would bake the (+71.74, -70.09) mas frame
    difference in as if it were an astrometry correction.
    """
    cfg = ac.resolve('3958', '007')
    assert cfg.reference_frame == ac.VIRAC2
    assert cfg.source == ac.TABLE_LOCKED
    # nothing may survive from the GNS entry -- a leftover recorded_bulk would be
    # applied on top of the table it replaced.
    assert not cfg.recorded_bulk
    assert 'VIRAC2locked' in ac.offsets_table_path('/b', '3958', '007')


def test_two_observations_of_one_proposal_can_use_different_tables():
    """2045 is the case that forces per-observation entries: arches (001) is
    consensus-driven, quintuplet (003) locked."""
    assert ac.resolve('2045', '001').source == ac.TABLE_CONSENSUS
    assert ac.resolve('2045', '003').source == ac.TABLE_LOCKED


def test_gc2211_is_tied():
    """gc2211 had an m2-written VIRAC2locked table with arcsecond-scale ties and
    no dispatch entry, so nothing read it.  One proposal-wide entry covers all
    five observations because they are separated by Visit, not by field."""
    for field in ('023', '028', '046', '049', '050'):
        cfg = ac.resolve('2211', field)
        assert cfg is not None, f"gc2211 o{field} still has no alignment source"
        assert cfg.source == ac.TABLE_LOCKED
        assert cfg.reference_frame == ac.VIRAC2


def test_the_treasury_is_tied_before_its_first_delivery():
    """Program 10678 (gc-treasury, #413) is registered ahead of its first
    delivery: without an entry the first delivery would reduce at the raw
    assign_wcs frame while m2's corrections landed in
    offsets/Offsets_JWST_Brick10678_consensus.csv, which nothing would read --
    the 1939/sgra failure class.  Proposal-wide, because all 139 of its
    observations are one field."""
    for field in ('001', '037', '139'):
        cfg = ac.resolve('10678', field)
        assert cfg is not None, f"gc-treasury o{field} has no alignment source"
        assert cfg.source == ac.TABLE_CONSENSUS
        assert cfg.reference_frame == ac.VIRAC2
        assert cfg.reference_filter == 'F212N'
    assert ac.offsets_channel('10678', '001') == ac.CHANNEL_CONSENSUS
    assert ac.offsets_table_path('/b', '10678', '001') == (
        '/b/offsets/Offsets_JWST_Brick10678_consensus.csv')


def test_corrections_now_reach_the_frames(tmp_path):
    """The shape of the arches failure end to end: a correction sitting in the
    consensus table must now produce a non-zero shift on the frame."""
    bp = str(tmp_path)
    _write_consensus(bp, '2045', rows=[
        ('jw02045001001', 'F212N', -1, 'all', 0.0210, -0.0080),
        ('jw02045001001', 'F212N', 12, 'nrcb3', 0.0043, -0.0019),
    ])
    fn = 'jw02045001001_02101_00012_nrcb3_destreak.fits'
    uni = ua.resolve_shift(fn, '2045', '001', 'F212N', 'nrcb', bp)
    assert uni.configured
    assert uni.bulk_ra == pytest.approx(0.0210)
    assert uni.jitter_ra == pytest.approx(0.0043)
    assert uni.total_ra == pytest.approx(0.0253)
    assert uni.total_ra != 0.0, "this is the RAOFFSET=0.0 regression"


def test_recorded_bulk_field_still_gets_consensus_corrections(tmp_path):
    """A hand-measured bulk must not exclude a field from the re-tie loop.

    The checkpoint records RESIDUALS on top of whatever tie is already applied,
    so its BULK sentinel is the REMAINING consensus->reference offset, not a
    second copy of the recorded constant: the two SUM.  Dropping the sentinel
    would leave the field with no path to VIRAC2 at all and would make the
    checkpoint re-add the same residual forever."""
    bp = str(tmp_path)
    _write_consensus(bp, '2092', rows=[
        ('jw02092002002', 'F480M', -1, 'all', -0.150, 0.075),
        ('jw02092002002', 'F480M', 3, 'nrcb3', 0.0031, -0.0022),
    ])
    fn = 'jw02092002002_02101_00003_nrcb3_destreak.fits'
    uni = ua.resolve_shift(fn, '2092', '002', 'F480M', 'nrcb', bp)
    assert uni.bulk_ra == pytest.approx(0.098 - 0.150)   # constant + residual tie
    assert uni.jitter_ra == pytest.approx(0.0031)
    assert uni.total_ra == pytest.approx(0.098 - 0.150 + 0.0031)
    assert 'consensus' in uni.prov_table and 'alignment_config' in uni.prov_table


def test_recorded_bulk_reader_agrees_with_lookup_consensus_offset(tmp_path):
    """The reduction-side reader and the checkpoint-side reader must not
    disagree about the same frame in the same table."""
    from astropy.table import Table as _T
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        lookup_consensus_offset)
    bp = str(tmp_path)
    path = _write_consensus(bp, '2092', rows=[
        ('jw02092002002', 'F480M', -1, 'all', -0.150, 0.075),
        ('jw02092002002', 'F480M', 3, 'nrcb3', 0.0031, -0.0022),
    ])
    fn = 'jw02092002002_02101_00003_nrcb3_destreak.fits'
    uni = ua.resolve_shift(fn, '2092', '002', 'F480M', 'nrcb', bp)
    chk = lookup_consensus_offset(_T.read(path), 'jw02092002002', 3, 'nrcb3', 'F480M')
    # the reducer applies the recorded constant PLUS everything the checkpoint
    # recorded; the checkpoint's own view is the table part alone
    assert uni.total_ra - 0.098 == pytest.approx(chk[0])
    assert uni.total_dec - (-0.171) == pytest.approx(chk[1])


def test_consensus_correction_is_inert_before_the_checkpoint_runs(tmp_path):
    """No consensus table yet -> the recorded bulk applies unchanged."""
    bp = str(tmp_path)
    fn = 'jw02092002002_02101_00003_nrcb3_destreak.fits'
    uni = ua.resolve_shift(fn, '2092', '002', 'F480M', 'nrcb', bp)
    assert uni.bulk_ra == pytest.approx(0.098)
    assert uni.jitter_ra == 0.0
    assert uni.total_ra == pytest.approx(0.098)


def test_a_genuinely_unknown_field_is_still_reported_not_silent(tmp_path, capsys):
    """The loud-unconfigured path must survive: a proposal nobody has configured
    still gets (0,0), but says so and stays machine-checkable via `configured`."""
    fn = 'jw09999001001_02101_00001_nrcb3_destreak.fits'
    uni = ua.resolve_shift(fn, '9999', '001', 'F212N', 'nrcb', str(tmp_path))
    assert uni.total_ra == 0.0 and uni.total_dec == 0.0
    assert not uni.configured
    out = capsys.readouterr().out
    assert 'NO CONFIGURED ALIGNMENT' in out
    assert 'alignment_config.py' in out


def test_configured_zero_is_distinguishable_from_unconfigured(tmp_path):
    """A genuine zero tie and 'this field is tied to nothing' must not look the
    same -- conflating them is how a field can silently never converge."""
    bp = str(tmp_path)
    _write_consensus(bp, '6151', rows=[
        ('jw06151001001', 'F115W', -1, 'all', 0.0, 0.0),
    ])
    tied = ua.resolve_shift(
        'jw06151001001_06101_00004_nrcb3_destreak.fits',
        '6151', '001', 'F115W', 'nrcb', bp)
    untied = ua.resolve_shift(
        'jw09999001001_02101_00001_nrcb3_destreak.fits',
        '9999', '001', 'F212N', 'nrcb', bp)
    assert tied.total_ra == untied.total_ra == 0.0
    assert tied.configured and not untied.configured


def test_dead_2221_002_branch_is_unreachable():
    """The old chain carried a hardcoded (2221, '002') branch AFTER a branch that
    already matched (2221, '002').  It never ran, so its constants were not
    ported.  If someone re-adds a (2221,'002')-specific rule they must decide
    which one wins, and this pins that the first match is the live one."""
    cfg = ac.resolve('2221', '002')
    assert cfg is not None
    assert cfg.source == ac.TABLE_LOCKED, (
        "cloudc must resolve to the VIRAC2locked table, not the dead hardcoded "
        "per-visit shifts (7.95\"/0.6\") that the old chain could never reach")


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

def test_bulk_plus_jitter_equals_total_everywhere(tmp_path):
    bp = str(tmp_path)
    _write_consensus(bp, '6151', rows=[
        ('jw06151001001', 'F115W', -1, 'all', 0.0300, -0.0120),
        ('jw06151001001', 'F115W', 4, 'nrcb3', 0.0025, -0.0031),
    ])
    _write_locked(bp, '2221',
                  rows=[('jw02221001001', 'F212N', 1, 'nrcb3', 0.10, -0.20),
                        ('jw02221001001', 'F212N', 2, 'nrcb3', 0.13, -0.17)],
                  colnames=('Visit', 'Filter', 'Exposure', 'Module',
                            'dra (arcsec)', 'ddec (arcsec)'))
    # sickle is TABLE_LOCKED as of 2026-08-04, so the invariant needs its table
    # here too -- it used to be a recorded-bulk field that needed none.
    _write_locked(bp, '3958',
                  rows=[('jw03958007001', 'F187N', 1, 'nrcb', -0.0179, -0.1031)],
                  colnames=('Visit', 'Filter', 'Exposure', 'Module',
                            'dra (arcsec)', 'ddec (arcsec)'))
    cases = [
        ('jw06151001001_06101_00004_nrcb3_destreak.fits', '6151', '001', 'F115W'),
        ('jw02221001001_02101_00001_nrcb3_destreak.fits', '2221', '001', 'F212N'),
        ('jw03958007001_02101_00001_nrcalong_destreak.fits', '3958', '007', 'F187N'),
        ('jw02092002002_02101_00003_nrcb3_destreak.fits', '2092', '002', 'F480M'),
    ]
    for fn, proposal, field, filt in cases:
        s = ua.resolve_shift(fn, proposal, field, filt, 'nrcb', bp)
        assert s.bulk_ra + s.jitter_ra == pytest.approx(s.total_ra, abs=1e-12)
        assert s.bulk_dec + s.jitter_dec == pytest.approx(s.total_dec, abs=1e-12)


def test_every_config_entry_is_well_formed():
    for cfg in ac.ALIGNMENT_CONFIG:
        assert cfg.reference_frame in (ac.VIRAC2, ac.GAIA, ac.GNS), cfg
        assert cfg.source in (ac.TABLE_LOCKED, ac.TABLE_CONSENSUS,
                              ac.RECORDED_BULK), cfg
        assert cfg.notes, f"{cfg.proposal}: every entry needs provenance notes"
        if cfg.source == ac.RECORDED_BULK:
            assert cfg.recorded_bulk, cfg
            assert cfg.visit_key in ('full', 'suffix3'), cfg
            if any(e.onsky_mas for e in cfg.recorded_bulk.values()):
                assert cfg.dec_ref_deg is not None, (
                    f"{cfg.proposal}: on-sky entries need dec_ref_deg")
        else:
            assert not cfg.recorded_bulk, cfg


def test_field_specific_entry_wins_over_proposal_wide():
    """2092's two observations resolve independently: obs002 keeps its recorded
    bulk, obs005 is on the consensus path.  Neither inherits the other's tie."""
    assert ac.resolve('2092', '002').source == ac.RECORDED_BULK
    assert ac.resolve('2092', '005').source == ac.TABLE_LOCKED
    assert ac.resolve('2092', '002').recorded_bulk
    assert not ac.resolve('2092', '005').recorded_bulk


# ---------------------------------------------------------------------------
# header write + staleness guard
# ---------------------------------------------------------------------------

def test_header_roundtrip_records_components_and_total():
    hdr = fits.Header()
    shift = ua.AlignmentShift(bulk_ra=0.030, bulk_dec=-0.012,
                              jitter_ra=0.0025, jitter_dec=-0.0031,
                              source=ac.TABLE_CONSENSUS,
                              reference_frame=ac.VIRAC2)
    ua.write_alignment_header(hdr, shift)
    assert hdr[ua.TOTAL_RA_KEY] == pytest.approx(0.0325)
    assert hdr[ua.TOTAL_DEC_KEY] == pytest.approx(-0.0151)
    assert hdr[ua.BULK_RA_KEY] == pytest.approx(0.030)
    assert hdr[ua.JITTER_RA_KEY] == pytest.approx(0.0025)
    assert hdr['ALIGNSRC'] == ac.TABLE_CONSENSUS
    assert hdr['ALIGNREF'] == ac.VIRAC2
    # nothing that reads the historical keyword needs to change
    assert hdr[ua.TOTAL_RA_KEY] == pytest.approx(shift.total_ra)


def test_stale_guard_clean_when_unchanged():
    hdr = fits.Header()
    shift = ua.AlignmentShift(bulk_ra=0.03, bulk_dec=-0.012,
                              jitter_ra=0.0025, jitter_dec=-0.0031)
    ua.write_alignment_header(hdr, shift)
    assert ua.check_alignment_stale(hdr, shift, 'f.fits') is None


def test_stale_guard_catches_the_brick1182_case():
    """Frame baked +1.9" while the table now says -17.5"."""
    hdr = fits.Header()
    ua.write_alignment_header(hdr, ua.AlignmentShift(bulk_ra=1.9, bulk_dec=-0.44))
    now = ua.AlignmentShift(bulk_ra=-17.5, bulk_dec=0.32)
    msg = ua.check_alignment_stale(hdr, now, 'v001.fits')
    assert msg is not None and 'STALE ASTROMETRY' in msg
    assert 'bulk RA' in msg


def test_per_component_guard_catches_offsetting_bulk_and_jitter():
    """The capability the split buys: a re-measured bulk and a re-solved jitter
    that happen to sum to the same total.  A total-only comparison sees nothing;
    per-component sees both."""
    hdr = fits.Header()
    ua.write_alignment_header(
        hdr, ua.AlignmentShift(bulk_ra=0.500, jitter_ra=0.000))
    now = ua.AlignmentShift(bulk_ra=0.300, jitter_ra=0.200)
    assert now.total_ra == pytest.approx(float(hdr[ua.TOTAL_RA_KEY]))
    msg = ua.check_alignment_stale(hdr, now, 'f.fits')
    assert msg is not None
    assert 'bulk RA' in msg and 'jitter RA' in msg
    assert 'per-component' in msg


def test_pre_split_frame_falls_back_to_total_comparison():
    """Frames written before the component keywords existed still get checked."""
    hdr = fits.Header()
    hdr[ua.TOTAL_RA_KEY] = 1.9
    hdr[ua.TOTAL_DEC_KEY] = -0.44
    now = ua.AlignmentShift(bulk_ra=-17.5, bulk_dec=0.32)
    msg = ua.check_alignment_stale(hdr, now, 'legacy.fits')
    assert msg is not None
    assert 'total-only (pre-split frame)' in msg
    # and an unchanged pre-split frame is still clean
    same = ua.AlignmentShift(bulk_ra=1.9, bulk_dec=-0.44)
    assert ua.check_alignment_stale(hdr, same, 'legacy.fits') is None


def test_half_written_offsets_are_stale_not_silently_passed():
    """A frame carrying RAOFFSET but no DEOFFSET reads back as NaN, and
    ``abs(nan) > tol`` is False -- so a plain magnitude comparison would call it
    clean.  It must be reported instead.  (FITS forbids a literal NaN card, so
    the way this actually occurs is a missing keyword.)"""
    hdr = fits.Header()
    hdr[ua.TOTAL_RA_KEY] = 0.0
    assert ua.TOTAL_DEC_KEY not in hdr
    msg = ua.check_alignment_stale(hdr, ua.AlignmentShift(), 'halfwritten.fits')
    assert msg is not None
    assert 'total Dec' in msg


def test_force_realign_env_turns_warning_into_error(monkeypatch):
    hdr = fits.Header()
    ua.write_alignment_header(hdr, ua.AlignmentShift(bulk_ra=1.9))
    now = ua.AlignmentShift(bulk_ra=-17.5)
    with pytest.warns(UserWarning, match='STALE ASTROMETRY'):
        ua.warn_or_raise_if_stale(hdr, now, 'f.fits')
    monkeypatch.setenv('FORCE_REALIGN_ON_DISAGREE', '1')
    with pytest.raises(RuntimeError, match='refusing to silently keep'):
        ua.warn_or_raise_if_stale(hdr, now, 'f.fits')


# ---------------------------------------------------------------------------
# guards that were wired but could not fire
# ---------------------------------------------------------------------------

def test_generation_mismatch_actually_raises(tmp_path):
    """The strong generation layer indexed the stamp with the TABLE's column
    spelling (`calver`) while generation_stamp produces `cal_ver`, so the moment
    a table carried base_* stamps this died on KeyError instead of comparing.
    It never showed because nothing populates the columns yet."""
    from astropy.table import Table as _T
    row = _T(rows=[('1.14.1', 'jwst_1253.pmap', 'True')],
             names=('base_calver', 'base_crds_ctx', 'base_dvacorr'))
    frame_gen = {'cal_ver': '1.21.0', 'crds_ctx': 'jwst_1584.pmap', 'dvacorr': 'True'}
    with pytest.raises(RuntimeError, match='GENERATION MISMATCH'):
        ua._assert_generation_row('f.fits', row, frame_gen, row)


def test_generation_match_passes(tmp_path):
    from astropy.table import Table as _T
    row = _T(rows=[('1.21.0', 'jwst_1584.pmap', 'True')],
             names=('base_calver', 'base_crds_ctx', 'base_dvacorr'))
    frame_gen = {'cal_ver': '1.21.0', 'crds_ctx': 'jwst_1584.pmap', 'dvacorr': 'True'}
    ua._assert_generation_row('f.fits', row, frame_gen, row)  # no raise


def test_partially_keyed_component_header_is_stale_not_a_crash():
    """A frame with RAOFFBLK/RAOFFJIT but no DEOFFBLK used to abort fix_alignment
    with `KeyError: DEOFFBLK`.  The total branch already handled this; the
    component branch must too, and the answer is STALE."""
    hdr = fits.Header()
    hdr[ua.TOTAL_RA_KEY] = 0.0
    hdr[ua.BULK_RA_KEY] = 0.0
    hdr[ua.JITTER_RA_KEY] = 0.0
    msg = ua.check_alignment_stale(hdr, ua.AlignmentShift(), 'partial.fits')
    assert msg is not None
    assert 'bulk Dec' in msg


def test_missing_table_is_distinguishable_from_a_measured_zero(tmp_path):
    """'the table is not there yet' and 'the table says zero' must not produce
    identical objects -- that is the same conflation this module removes one
    level up."""
    bp = str(tmp_path)
    absent = ua.resolve_shift(
        'jw06151001001_02101_00001_nrcb3_destreak.fits',
        '6151', '001', 'F480M', 'nrcb', bp)
    _write_consensus(bp, '6151',
                     rows=[('jw06151001001', 'F480M', -1, 'all', 0.0, 0.0)])
    measured = ua.resolve_shift(
        'jw06151001001_02101_00001_nrcb3_destreak.fits',
        '6151', '001', 'F480M', 'nrcb', bp)
    assert absent.total_ra == measured.total_ra == 0.0
    assert not absent.table_present
    assert measured.table_present
    assert absent.prov_stage == 'NO_TABLE'


def test_header_records_alignment_state_positively():
    """Unconfigured must be a greppable fact in the product, not the absence of
    a card."""
    hdr = fits.Header()
    ua.write_alignment_header(hdr, ua.AlignmentShift(configured=False,
                                                     table_present=False))
    assert hdr['ALIGNSRC'] == 'NONE'
    assert hdr['ALIGNREF'] == 'NONE'
    assert hdr['ALIGNTBL'] == 'ABSENT'


def test_ambiguous_consensus_row_names_the_table_and_frame(tmp_path):
    bp = str(tmp_path)
    _write_consensus(bp, '6151', rows=[
        ('jw06151001001', 'F480M', -1, 'all', 0.01, 0.02),
        ('jw06151001001', 'F480M', 3, 'nrcb3', 0.001, 0.002),
        ('jw06151001001', 'F480M', 3, 'nrcb3', 0.003, 0.004),
    ])
    fn = 'jw06151001001_02101_00003_nrcb3_destreak.fits'
    with pytest.raises(ValueError, match='table=.*consensus.csv'):
        ua.resolve_shift(fn, '6151', '001', 'F480M', 'nrcb', bp)


# reader / writer agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('proposal,field,expect', [
    ('2045', '001', 'consensus'),   # arches: no REGION entry, consensus table
    ('2045', '003', 'locked'),      # quintuplet: REGION entry, locked table
    ('5365', '001', 'locked'),      # sgrb2
    ('4147', '012', 'locked'),      # sgrc -- the field the mismatch broke
    ('2092', '005', 'locked'),      # cloudef obs005
    ('2092', '002', 'consensus'),   # cloudef obs002: constant bulk + table jitter
    ('1182', '004', 'locked'),
    ('6151', '001', 'consensus'),   # w51: no table yet, checkpoint seeds one
    # halo clusters: the BULK stays a hand-measured constant, but the
    # per-exposure jitter is table-driven (`consensus_jitter`), so the
    # checkpoint has somewhere to write.  With 'none' the m2 checkpoint
    # refused its own measured corrections and the m12 finalize stopped.
    ('1334', '001', 'consensus'),   # m92
    ('1979', '002', 'consensus'),   # m4
    ('9999', '001', 'none'),        # unconfigured
])
def test_checkpoint_writes_where_the_reducer_reads(proposal, field, expect):
    """The channel the m2 checkpoint writes to must be the one fix_alignment
    reads from, for every configured field.

    Before this, the checkpoint picked its table by globbing filenames
    (``*locked.csv`` first), independently of what the reducer was configured to
    read.  On sgrc that meant corrections were written to VIRAC2locked while the
    reducer looked for a _consensus.csv that never existed -- so a full reduction
    came out at RAOFFSET=0.0 and the re-tie loop re-measured the same residual
    forever.  Both sides now resolve through alignment_config.
    """
    from jwst_gc_pipeline.photometry.cataloging import _astrom_offsets_channel
    assert _astrom_offsets_channel(proposal, field) == expect


def test_offsets_table_path_follows_the_declared_channel(tmp_path):
    """The path handed to the writer is the one the declared source names -- not
    whatever table happens to sit on disk."""
    from jwst_gc_pipeline.photometry.cataloging import _astrom_find_offsets_table
    bp = str(tmp_path)
    # a locked table on disk for a CONSENSUS-declared field must NOT be returned
    _write_locked(bp, '6151',
                  rows=[('jw06151001001', 'F480M', 0.1, 0.2)],
                  colnames=('Visit', 'Filter', 'dra (arcsec)', 'ddec (arcsec)'))
    assert _astrom_find_offsets_table(bp, '6151', '001') is None
    # ... and once the consensus table exists, that is what comes back
    path = _write_consensus(bp, '6151',
                            rows=[('jw06151001001', 'F480M', -1, 'all', 0.0, 0.0)])
    assert _astrom_find_offsets_table(bp, '6151', '001') == path
