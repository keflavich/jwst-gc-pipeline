"""A VIRAC2-framed field must not catalog without its reference catalog (#415).

Without the refcat the astrometry checkpoints fall back to consensus-only
checks: they verify that the exposures agree WITH EACH OTHER and say nothing
about where that agreed frame sits on the sky.  That is how proposal 1939's
sgra mosaics shipped ~14.8" from VIRAC2 with every internal check green, and
it is the state a freshly-registered field starts in -- ``gc-treasury``'s
registry entry names a catalog that has yet to be built.

The gate is scoped by the field's declared frame rather than applied to
everything, because the two are not interchangeable.  Measured across the
registry on 2026-08-20:

    VIRAC2-framed   arches brick cloudc cloudef gc2211 quintuplet sgra
                    sgrb2 sgrc sickle          -- all carry a refcat
                    gc-treasury                -- carries none
    Gaia-framed     m4 m92 ngc6397 w51         -- none carries one

so requiring the file wherever the frame is VIRAC2 stops the one field that
needs stopping and leaves every currently-running field alone.
"""
import os

import pytest

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.reduction import alignment_config as ac


@pytest.fixture
def catdir(tmp_path):
    (tmp_path / 'catalogs').mkdir()
    return tmp_path


def _refcat(catdir, name='gaia_virac2_refcat_epoch2026.65_o037.fits'):
    path = catdir / 'catalogs' / name
    path.touch()
    return path


def test_the_frames_in_the_registry_are_what_this_gate_assumes():
    """The scoping claim, asserted against the registry rather than restated.

    If a Gaia-framed field is ever re-declared on VIRAC2, this gate starts
    requiring a refcat for it; that should be a visible decision, not a
    surprise on the next catalog run.
    """
    assert cataloging._refcat_is_required('10678', '037')     # gc-treasury
    assert cataloging._refcat_is_required('2221', '001')      # brick
    assert cataloging._refcat_is_required('1939', '001')      # sgra
    assert not cataloging._refcat_is_required('6151', '001')  # w51, Gaia
    assert not cataloging._refcat_is_required('1979', '002')  # m4, Gaia
    assert not cataloging._refcat_is_required('1334', '001')  # m92, Gaia
    # a proposal with no entry at all keeps the old behaviour
    assert not cataloging._refcat_is_required('99999', '001')
    # ... and the gate cannot fire when the caller does not know its proposal
    assert not cataloging._refcat_is_required(None, '037')


def test_a_virac_field_without_a_refcat_stops_the_run(catdir, monkeypatch):
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    monkeypatch.delenv(cataloging.ALLOW_CONSENSUS_ONLY_ENV, raising=False)
    with pytest.raises(cataloging.MissingReferenceCatalogError) as excinfo:
        cataloging._astrom_checkpoint_refcat(str(catdir), field='037',
                                             proposal_id='10678')
    msg = str(excinfo.value)
    # the message has to carry the way out, or it just relocates the puzzle
    assert 'build_gaia_virac2_refcat_byquery' in msg
    assert '--obs-token' in msg
    assert cataloging.ALLOW_CONSENSUS_ONLY_ENV in msg
    assert 'ASTROM_REFCAT' in msg


def test_a_gaia_framed_field_still_runs_without_one(catdir, monkeypatch, capsys):
    """w51 and the globulars carry no refcat and catalog fine today."""
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    got = cataloging._astrom_checkpoint_refcat(str(catdir), field='001',
                                               proposal_id='6151')
    assert got is None
    assert 'consensus-only' in capsys.readouterr().out


def test_the_override_runs_but_says_the_frame_is_unverified(catdir, monkeypatch,
                                                            capsys):
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    monkeypatch.setenv(cataloging.ALLOW_CONSENSUS_ONLY_ENV, '1')
    assert cataloging._astrom_checkpoint_refcat(str(catdir), field='037',
                                                proposal_id='10678') is None
    out = capsys.readouterr().out
    assert 'UNVERIFIED' in out, out


def test_a_present_refcat_is_loaded_and_the_gate_stays_quiet(catdir, monkeypatch):
    """The gate must not fire when the file the registry names is on disk."""
    path = _refcat(catdir)
    seen = {}

    def _load(p):
        seen['path'] = p
        return 'REFCAT'

    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    monkeypatch.setattr(
        'jwst_gc_pipeline.photometry.visit_consensus.load_reference_catalog',
        _load)
    got = cataloging._astrom_checkpoint_refcat(str(catdir), field='037',
                                               proposal_id='10678')
    assert got == 'REFCAT'
    assert seen['path'] == str(path)


@pytest.mark.parametrize('override', ['unset', '1'])
def test_another_observations_catalog_is_never_substituted(catdir, monkeypatch,
                                                           override):
    """o042's catalog must not satisfy o037, override or not.

    `pick_refcat` already refuses this and raises before the new gate is
    reached -- "not a degraded measurement, it is the wrong sky".  Asserted
    here because the two guards answer different questions (is there a
    catalog for THIS observation, versus may this field run without one) and
    the answer to the second must never be allowed to weaken the first:
    ALLOW_CONSENSUS_ONLY_ASTROMETRY buys a run with no absolute tie, never a
    run tied to the wrong pointing.
    """
    _refcat(catdir, 'gaia_virac2_refcat_epoch2026.65_o042.fits')
    monkeypatch.delenv('ASTROM_REFCAT', raising=False)
    if override == 'unset':
        monkeypatch.delenv(cataloging.ALLOW_CONSENSUS_ONLY_ENV, raising=False)
    else:
        monkeypatch.setenv(cataloging.ALLOW_CONSENSUS_ONLY_ENV, override)
    with pytest.raises(ValueError, match='wrong sky'):
        cataloging._astrom_checkpoint_refcat(str(catdir), field='037',
                                             proposal_id='10678')


def test_astrom_refcat_env_still_wins(catdir, monkeypatch):
    """The documented escape hatch keeps working ahead of the gate."""
    path = _refcat(catdir, 'somewhere_else.fits')
    monkeypatch.setenv('ASTROM_REFCAT', str(path))
    monkeypatch.setattr(
        'jwst_gc_pipeline.photometry.visit_consensus.load_reference_catalog',
        lambda p: 'REFCAT')
    assert cataloging._astrom_checkpoint_refcat(str(catdir), field='037',
                                                proposal_id='10678') == 'REFCAT'


def test_the_builder_writes_the_name_the_selector_reads():
    """Round-trip: what the builder names, `pick_refcat` must select.

    The two halves were written apart -- the builder wrote untokened names
    while the selector matched on `_o<NNN>` -- so the token is asserted through
    both rather than in either alone.
    """
    from jwst_gc_pipeline.astrometry_utils import pick_refcat
    from jwst_gc_pipeline.reduction.build_gaia_virac2_refcat_byquery import (
        refcat_filename)

    cands = [f'/x/catalogs/{refcat_filename("2026.65", t)}'
             for t in ('037', '042')]
    assert pick_refcat(cands, field='037').endswith('_o037.fits')
    assert pick_refcat(cands, field='042').endswith('_o042.fits')
    # 'o37', 37 and '037' all name one file, so a runner cannot half-miss
    for spelling in ('o037', 'o37', 37, '37'):
        assert refcat_filename('2026.65', spelling).endswith('_o037.fits')
    assert refcat_filename('2026.65') == 'gaia_virac2_refcat_epoch2026.65.fits'


def test_the_checkpoint_call_site_passes_the_proposal():
    """The gate is dead unless the caller hands it the proposal.

    `_refcat_is_required` returns False for `proposal_id=None`, deliberately --
    a caller that does not know its proposal cannot be judged.  So dropping the
    argument at the one call site silently restores the old silent-degradation
    behaviour while every test above still passes, which is how a fix gets
    pinned at the API and not where it runs.  Asserted over the source: the
    call must take its proposal from `options`.
    """
    import ast
    import inspect

    src = inspect.getsource(cataloging)
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == '_astrom_checkpoint_refcat']
    assert calls, 'no call to _astrom_checkpoint_refcat found'
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert 'proposal_id' in kw, (
            'the checkpoint call must pass proposal_id, or the refcat gate '
            'never fires in production')
        value = kw['proposal_id']
        # either getattr(options, 'proposal_id', None) at the outer call sites,
        # or the parameter of a helper that its own caller fills the same way
        if isinstance(value, ast.Call):
            assert getattr(value.func, 'id', None) == 'getattr', ast.dump(value)
            assert value.args[1].value == 'proposal_id', ast.dump(value)
        else:
            assert isinstance(value, ast.Name), ast.dump(value)
            assert value.id == 'proposal_id', ast.dump(value)
