"""The checkpoint must not tie one observation against another's reference.

Selection was ``sorted(glob('gaia_virac2_refcat*.fits'))[-1]`` -- the last name
ALPHABETICALLY.  gc2211's catalogs directory holds:

    gaia_virac2_refcat_epoch2023.71.fits          generic
    gaia_virac2_refcat_epoch2023.71_o028.fits     built for o028

and ``_o028`` sorts after the bare name, so every observation got o028's.
Running o023 against it, the m2 checkpoint measured a -9.28" per-exposure
correction; the magnitude limit refused to write it (correctly -- the
measurement was wrong, not the frame) and the o023 m12 finalize died.  gc2211's
five observations are 0.3-17.6 arcmin apart, so a neighbour's refcat is not a
degraded reference, it is the wrong sky.
"""
import pytest

from jwst_gc_pipeline.photometry.cataloging import _pick_refcat

GENERIC = '/x/catalogs/gaia_virac2_refcat_epoch2023.71.fits'
O028 = '/x/catalogs/gaia_virac2_refcat_epoch2023.71_o028.fits'
O046 = '/x/catalogs/gaia_virac2_refcat_epoch2023.71_o046.fits'


def test_the_gc2211_shape_no_longer_picks_o028_for_everyone():
    """The regression, exactly: two candidates, and every observation but o028
    must get the generic one rather than the alphabetically-last."""
    for field in ('023', '046', '049', '050'):
        assert _pick_refcat([GENERIC, O028], field=field) == GENERIC, field


def test_an_observations_own_refcat_wins():
    assert _pick_refcat([GENERIC, O028], field='028') == O028
    assert _pick_refcat([GENERIC, O028, O046], field='046') == O046


def test_a_zero_padded_or_bare_field_both_match():
    """`field` reaches this as '023' from options and as '23' from some
    callers; both name the same observation."""
    assert _pick_refcat([GENERIC, O028], field='28') == O028
    assert _pick_refcat([GENERIC, O028], field=28) == O028


def test_with_no_token_anywhere_the_single_refcat_is_used():
    assert _pick_refcat([GENERIC], field='023') == GENERIC
    assert _pick_refcat([GENERIC]) == GENERIC


def test_no_candidates_is_None_not_an_error():
    """A field with no seed refcat runs consensus-only checks; that is a
    supported state, not a failure."""
    assert _pick_refcat([], field='023') is None
    assert _pick_refcat([]) is None


def test_only_OTHER_observations_refcats_REFUSES():
    """The case that must never be silent.  With nothing but foreign refcats,
    picking one is choosing the wrong sky -- so it raises instead."""
    with pytest.raises(ValueError, match='wrong sky'):
        _pick_refcat([O028, O046], field='023')


def test_the_refusal_names_what_is_available_and_what_was_wanted():
    with pytest.raises(ValueError) as exc:
        _pick_refcat([O028, O046], field='023')
    msg = str(exc.value)
    assert "'028'" in msg and "'046'" in msg
    assert 'o023' in msg
    assert 'build_gaia_virac2_refcat_byquery' in msg or 'ASTROM_REFCAT' in msg


def test_no_field_falls_back_to_the_generic_rather_than_a_token():
    """A caller that cannot say which observation it is must still not be
    handed one observation's catalog by alphabetical accident."""
    assert _pick_refcat([GENERIC, O028], field=None) == GENERIC


def test_no_field_and_only_tokened_candidates_REFUSES():
    with pytest.raises(ValueError, match='wrong sky'):
        _pick_refcat([O028, O046], field=None)


def test_several_untokened_candidates_take_the_last():
    """Unchanged behaviour where the old rule was harmless: among refcats that
    make no observation claim, the newest-sorting name still wins."""
    older = '/x/catalogs/gaia_virac2_refcat_epoch2022.10.fits'
    assert _pick_refcat([older, GENERIC], field='023') == GENERIC
