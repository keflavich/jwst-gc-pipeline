"""An EXPOSURE-level obs_id names its observation, so scope it to that one.

``observation_scope_mask`` narrowed on the ``-o`` product spelling
(``jw10678-o135_t135_nircam_f212n``) and kept every row WITHOUT that token as
"unattributed", on the reasoning that such a name attributes the row to no
foreign observation.  That reasoning holds for a candidate association
(``jw10678-c1001_...``) and fails for the exposure spelling
``jw{PPPPP}{OOO}{VVV}_...``, where the three digits after the proposal ARE the
observation number.

Measured on the real 10678 table (1777 rows, 2026-09-12): scoping to observation
134 kept **1774** of them -- every planning placeholder and every other tile --
so each Treasury tile's download called ``get_product_list`` on the whole
program.  With 139 tiles to land that is the #416 failure the mask exists to
prevent.
"""
import numpy as np

from jwst_gc_pipeline.reduction.mast_obs_scope import observation_scope_mask

_ROWS = np.array([
    'jw10678-o135_t135_nircam_clear-f212n',   # 0 o135, product spelling
    'jw10678135001_02101_00001_nrcalong',     # 1 o135, exposure spelling
    'jw10678134001_02101_00001_nrcalong',     # 2 o134, exposure spelling
    'jw10678008001_xx101_00001_nircam',       # 3 o008 planning placeholder
    'jw10678-c1001_t001_nircam_f212n',        # 4 candidate asn, unattributed
])


def test_an_exposure_row_goes_to_its_own_observation():
    keep = observation_scope_mask(_ROWS, '10678', '135')
    assert list(keep) == [True, True, False, False, True]


def test_another_tiles_exposure_row_is_dropped():
    """The bug: row 1 used to be kept here because it carries no `-o`."""
    keep = observation_scope_mask(_ROWS, '10678', '134')
    assert list(keep) == [False, False, True, False, True]


def test_a_planning_placeholder_is_scoped_like_any_exposure_row():
    keep = observation_scope_mask(_ROWS, '10678', '008')
    assert list(keep) == [False, False, False, True, True]


def test_an_observation_with_nothing_released_keeps_only_the_unattributed():
    keep = observation_scope_mask(_ROWS, '10678', '042')
    assert list(keep) == [False, False, False, False, True]


def test_a_candidate_association_stays_unattributed():
    """No `-o`, and no observation digits straight after the proposal -- the
    case the unattributed rule was written for."""
    rows = np.array(['jw10678-c1001_t001_nircam_f212n'])
    for obs in ('001', '042', '135'):
        assert list(observation_scope_mask(rows, '10678', obs)) == [True]


def test_a_joint_field_token_keeps_every_member_in_both_spellings():
    """sgrb2's MIRI is registered `002-998` and sickle's `001-002`."""
    rows = np.array(['jw05365-o002_t001_miri_f770w',
                     'jw05365-o998_t001_miri_f770w',
                     'jw05365002001_02101_00001_mirimage',
                     'jw05365003001_02101_00001_mirimage'])
    keep = observation_scope_mask(rows, '5365', '002-998')
    assert list(keep) == [True, True, True, False]


def test_the_two_field_proposals_still_separate():
    """2221 covers brick (001) and cloudc (002) -- the case the mask was added
    for -- now in the exposure spelling too."""
    rows = np.array(['jw02221-o001_t001_nircam_f405n',
                     'jw02221-o002_t001_nircam_f405n',
                     'jw02221001001_04101_00001_nrcalong',
                     'jw02221002001_04101_00001_nrcalong'])
    assert list(observation_scope_mask(rows, '2221', '001')) == [True, False, True, False]
    assert list(observation_scope_mask(rows, '2221', '002')) == [False, True, False, True]
