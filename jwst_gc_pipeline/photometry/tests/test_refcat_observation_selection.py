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


# ---------------------------------------------------------------------------
# The THREADING, not just the selector.  Every test above calls _pick_refcat
# with an explicit field, so dropping `field=` at either call site left them all
# green while gc2211 went straight back to o028's catalogue.  These drive the
# real entry points against a tmp catalogs/ directory holding both names.
# ---------------------------------------------------------------------------

def _catalogs(tmp_path, names):
    d = tmp_path / 'catalogs'
    d.mkdir(parents=True, exist_ok=True)
    from astropy.table import Table
    for n in names:
        Table({'RA': [266.4], 'DEC': [-28.9],
               'source': ['GaiaDR3']}).write(str(d / n), overwrite=True)
    return str(tmp_path)


BOTH = ['gaia_virac2_refcat_epoch2023.71.fits',
        'gaia_virac2_refcat_epoch2023.71_o028.fits']


@pytest.mark.parametrize('field,want', [
    ('023', 'gaia_virac2_refcat_epoch2023.71.fits'),
    ('046', 'gaia_virac2_refcat_epoch2023.71.fits'),
    ('028', 'gaia_virac2_refcat_epoch2023.71_o028.fits'),
])
def test_the_STAGE_checkpoint_passes_its_field_through(tmp_path, field, want,
                                                       capsys, monkeypatch):
    """Drop `field=` at cataloging.py's stage-checkpoint call and this fails."""
    import os as _os
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    bp = _catalogs(tmp_path, BOTH)
    _astrom_checkpoint_refcat(bp, field=field)
    chosen = [ln for ln in capsys.readouterr().out.splitlines()
              if 'reference catalog' in ln]
    assert chosen and _os.path.basename(chosen[-1].split()[-1]) == want, chosen


def test_ASTROM_REFCAT_still_overrides_the_observation(tmp_path, monkeypatch):
    """The env escape hatch must keep winning -- it is how an operator points a
    run at a catalogue the rule would not choose."""
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    bp = _catalogs(tmp_path, BOTH)
    forced = f'{bp}/catalogs/gaia_virac2_refcat_epoch2023.71_o028.fits'
    monkeypatch.setenv('ASTROM_REFCAT', forced)
    out = _astrom_checkpoint_refcat(bp, field='023')
    assert out is not None


@pytest.mark.parametrize('field,want', [
    ('023', 'gaia_virac2_refcat_epoch2023.71.fits'),
    ('028', 'gaia_virac2_refcat_epoch2023.71_o028.fits'),
])
def test_the_REDUCER_side_bulk_path_agrees_with_the_checkpoint(tmp_path, field,
                                                               want):
    """`refcat_for_frame` is what step0_bulk_offset uses, and #338's error
    message sends operators to step0 for exactly the two multi-observation
    fields.  The two halves of a field's remediation must not disagree about
    which sky they are tying to."""
    import os as _os
    from jwst_gc_pipeline.reduction.bulk_offset_step0 import refcat_for_frame
    bp = _catalogs(tmp_path, BOTH)
    path, cands = refcat_for_frame(bp, 'VIRAC2', field=field)
    assert len(cands) == 2
    assert _os.path.basename(path) == want


def test_both_entry_points_choose_the_SAME_file(tmp_path, monkeypatch):
    """The property that matters more than either one individually."""
    import os as _os
    from jwst_gc_pipeline.photometry.cataloging import _pick_refcat
    from jwst_gc_pipeline.reduction.bulk_offset_step0 import refcat_for_frame
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    bp = _catalogs(tmp_path, BOTH)
    import glob as _glob
    cands = sorted(_glob.glob(f'{bp}/catalogs/gaia_virac2_refcat*.fits'))
    for field in ('023', '028', '046', '049', '050'):
        a = _pick_refcat(cands, field=field)
        b, _ = refcat_for_frame(bp, 'VIRAC2', field=field)
        assert _os.path.basename(a) == _os.path.basename(b), field


def test_field_has_NO_default_so_a_call_site_cannot_drop_it():
    """The threading is what actually broke, and a suite that exercises the
    selector cannot see a call site stop passing `field=`.  Making the
    parameter required turns that omission into a TypeError instead of a silent
    return to o028's catalogue for every observation."""
    import inspect
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    sig = inspect.signature(_astrom_checkpoint_refcat)
    assert sig.parameters['field'].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        _astrom_checkpoint_refcat('/nonexistent')


@pytest.mark.parametrize('func_name', [
    '_run_astrometry_stage_checkpoint',
    '_run_crossfilter_astrom_checkpoint',
])
def test_every_call_site_actually_passes_field(func_name):
    """Source-level, because the runtime TypeError above only fires when the
    enclosing checkpoint RUNS -- which needs a full merged catalog and is not
    something the suite drives.  Dropping `field=` at either call site is the
    exact regression, and it left all 17 behavioural tests green.

    Same shape as the repo's other threading guards (test_no_sip_frame_astrometry,
    test_no_adhoc_nn_median_astrometry): pin the call, not just the callee.
    """
    import inspect
    import re as _re
    from jwst_gc_pipeline.photometry import cataloging
    src = inspect.getsource(getattr(cataloging, func_name))
    calls = _re.findall(r'_astrom_checkpoint_refcat\s*\((?:[^()]|\([^()]*\))*\)',
                        src)
    assert calls, f'{func_name} no longer calls _astrom_checkpoint_refcat'
    for c in calls:
        assert 'field' in c, (
            f'{func_name} calls _astrom_checkpoint_refcat without a field -- '
            f'gc2211 goes back to o028 for every observation: {c}')


# --- Gaia-only fields (#1167) ----------------------------------------------

def _chosen(capsys):
    import os as _os
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if 'reference catalog' in ln]
    return _os.path.basename(lines[-1].split()[-1]) if lines else None


def test_gaia_only_field_falls_back_to_gaia_refcat(tmp_path, capsys, monkeypatch):
    """wd1 (1905 o001, GAIA frame) has only catalogs/gaia_refcat.fits."""
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    bp = _catalogs(tmp_path, ['gaia_refcat.fits'])
    out = _astrom_checkpoint_refcat(bp, field='001', proposal_id='1905',
                                    gaia_fallback=True)
    assert out is not None
    assert _chosen(capsys) == 'gaia_refcat.fits'


def test_gaia_fallback_is_off_by_default(tmp_path, monkeypatch):
    """Only the m2 stage checkpoint asks for the fallback (#1201 review)."""
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    bp = _catalogs(tmp_path, ['gaia_refcat.fits'])
    assert _astrom_checkpoint_refcat(bp, field='001', proposal_id='1905') is None


def test_gaia_fallback_matches_exact_names_only(tmp_path, capsys, monkeypatch):
    """A backup or test build must not be preferred over gaia_refcat.fits."""
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    bp = _catalogs(tmp_path, ['gaia_refcat.fits', 'gaia_refcat_old.fits',
                              'gaia_refcat_test.fits'])
    _astrom_checkpoint_refcat(bp, field='001', proposal_id='1905',
                              gaia_fallback=True)
    assert _chosen(capsys) == 'gaia_refcat.fits'
    bp2 = _catalogs(tmp_path / 'b', ['gaia_refcat_old.fits'])
    assert _astrom_checkpoint_refcat(bp2, field='001', proposal_id='1905',
                                     gaia_fallback=True) is None


def test_gaia_fallback_takes_this_observations_token(tmp_path, capsys,
                                                     monkeypatch):
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    bp = _catalogs(tmp_path, ['gaia_refcat.fits', 'gaia_refcat_o001.fits'])
    _astrom_checkpoint_refcat(bp, field='001', proposal_id='1905',
                              gaia_fallback=True)
    assert _chosen(capsys) == 'gaia_refcat_o001.fits'


def test_virac2_refcat_still_preferred_over_gaia_refcat(tmp_path, capsys,
                                                        monkeypatch):
    from jwst_gc_pipeline.photometry.cataloging import _astrom_checkpoint_refcat
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    bp = _catalogs(tmp_path, ['gaia_refcat.fits',
                              'gaia_virac2_refcat_epoch2023.71.fits'])
    _astrom_checkpoint_refcat(bp, field='001', proposal_id='1905',
                              gaia_fallback=True)
    assert _chosen(capsys) == 'gaia_virac2_refcat_epoch2023.71.fits'


def test_virac2_framed_field_does_not_fall_back_to_gaia(tmp_path, monkeypatch):
    """A VIRAC2-framed field with only a Gaia catalog still refuses (#1167)."""
    from jwst_gc_pipeline.photometry.cataloging import (
        MissingReferenceCatalogError, _astrom_checkpoint_refcat)
    from jwst_gc_pipeline.reduction import alignment_config as ac
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    monkeypatch.delenv('ALLOW_CONSENSUS_ONLY_ASTROMETRY', raising=False)
    monkeypatch.setattr(ac, 'reference_catalog_required', lambda p, f: True)
    bp = _catalogs(tmp_path, ['gaia_refcat.fits'])
    with pytest.raises(MissingReferenceCatalogError):
        _astrom_checkpoint_refcat(bp, field='001', proposal_id='2221',
                                  gaia_fallback=True)


# --- which checkpoints take the fallback (#1201 review) ---------------------

class _WdOpt:
    target = 'wd1'
    proposal_id = '1905'
    field = '001'
    modules = 'nrca'
    each_exposure = True
    cutout_region = ''
    epsf = False


def _wd1_tree(tmp_path, label):
    from astropy.table import Table
    bp = _catalogs(tmp_path, ['gaia_refcat.fits'])
    cut_bp = tmp_path / 'cut'
    d = cut_bp / 'F150W'
    d.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        Table({'x': [1.0]}).write(str(
            d / f'f150w_nrca1_visit001_vgroup02101_exp{i:05d}_{label}'
                f'_daophot_basic.fits'))
    return bp, str(cut_bp)


def _stub_stage_checkpoint(monkeypatch):
    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    from jwst_gc_pipeline.photometry import cataloging as _cat
    for k in ('ASTROM_CHECKPOINT', 'ASTROM_CHECKPOINT_WARN_ONLY', 'ASTROM_REFCAT'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('ASTROM_SATSTAR_CONSENSUS', '0')
    monkeypatch.setattr(_cat, '_drop_foreign_obs_duplicates',
                        lambda fns, *a, **k: fns)
    monkeypatch.setattr(ac, 'resolve_tie_reference', lambda refcat, *a, **k: refcat)
    seen = {}

    def _fake(tables, merge_label, refcat=None, **kw):
        seen[merge_label] = refcat
        return dict(passed=True, failures=[], corrections=[],
                    unverified_blocking=[], record_path='x')
    monkeypatch.setattr(ac, 'run_visit_checkpoint', _fake)
    return _cat, seen


@pytest.mark.parametrize('label,expect_gaia', [('m2', True), ('m1', False),
                                               ('m3', False), ('m7', False)])
def test_only_the_m2_stage_checkpoint_uses_gaia(tmp_path, monkeypatch, label,
                                                expect_gaia):
    _cat, seen = _stub_stage_checkpoint(monkeypatch)
    bp, cut_bp = _wd1_tree(tmp_path, label)
    cache = {}
    _cat._run_astrometry_stage_checkpoint(label, 'nrca', 'F150W', cut_bp, bp,
                                          '1905', _WdOpt(), cache, context='t')
    if label in seen:
        assert (seen[label] is not None) == expect_gaia, seen
    else:
        assert not expect_gaia
    # the shared key the m7 cross-filter checkpoint reads stays consensus-only
    assert cache.get('refcat') is None


def test_m2_gaia_does_not_leak_into_later_stages_or_crossfilter(tmp_path,
                                                                monkeypatch):
    """m2 then m3 in one process share the cache; m3 and the m7 cross-filter
    checkpoint must still see no reference."""
    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    _cat, seen = _stub_stage_checkpoint(monkeypatch)
    bp, cut_bp = _wd1_tree(tmp_path, 'm2')
    _wd1_tree(tmp_path, 'm3')
    cache = {}
    _cat._run_astrometry_stage_checkpoint('m2', 'nrca', 'F150W', cut_bp, bp,
                                          '1905', _WdOpt(), cache, context='t')
    assert seen['m2'] is not None
    assert cache.get('refcat') is None
    xf = {}
    monkeypatch.setattr(_cat, 'run_crossfilter_checkpoint',
                        lambda catalogs, refcat=None, **kw: xf.setdefault('r', refcat),
                        raising=False)
    monkeypatch.setattr(ac, 'run_crossfilter_checkpoint',
                        lambda catalogs, refcat=None, **kw: xf.setdefault('r', refcat))
    from astropy.table import Table
    vp = {}
    for f in ('F150W', 'F200W'):
        p = tmp_path / f'{f}_vetted.fits'
        Table({'x': [1.0]}).write(str(p))
        vp[f] = str(p)
    _cat._run_crossfilter_astrom_checkpoint(vp, cut_bp, bp, cache,
                                            obs_token='_o001',
                                            proposal_id='1905')
    assert 'r' in xf and xf['r'] is None
