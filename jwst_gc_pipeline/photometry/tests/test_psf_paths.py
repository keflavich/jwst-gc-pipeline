"""Tests for the centralized PSF-grid path resolver (psf_paths.py).

Pins the central-first / legacy-fallback read order and the central naming
(keyed only by instrument+module+filter+oversample+blur, not proposal/field)
so the shared store can't silently regress to per-field duplication.
"""
from jwst_gc_pipeline.photometry import psf_paths as PP

ROOT = '/jwst'


def test_central_name_keyed_by_physics_only():
    p = PP.central_merged_psf_grid_path(ROOT, 'NIRCam', 'nrcb5', 'f405n',
                                        oversample=1, blur=False)
    assert p == '/jwst/psfs_shared/nircam_nrcb5_F405N_PSFgrid_oversample1.fits'


def test_central_name_blur_and_oversample():
    p = PP.central_merged_psf_grid_path(ROOT, 'MIRI', 'mirimage', 'f770w',
                                        oversample=2, blur=True)
    assert p == '/jwst/psfs_shared/miri_mirimage_F770W_PSFgrid_oversample2_blur.fits'


def test_legacy_path_keeps_proposal_and_field():
    p = PP.legacy_merged_psf_grid_path(ROOT, 'brick', 'f405n', '1182', '001',
                                       oversample=1, blur=False)
    assert p == '/jwst/brick/psfs/F405N_1182_001_merged_PSFgrid_oversample1.fits'


def test_the_merged_grid_name_carries_the_oversample_token():
    """Every merged grid on disk is named ``..._oversample{N}[_blur].fits``.

    ``reduction.saturated_star_finding.get_psf`` used to build this name by hand
    without the token, so it matched none of the 237 grids that exist and fell
    back to a detector-specific grid without saying it had looked.  Both callers
    now go through this one function, so they cannot drift apart again.
    """
    name = PP.legacy_merged_psf_grid_name('f405n', '1182', '001', oversample=2)
    assert name == 'F405N_1182_001_merged_PSFgrid_oversample2.fits'
    assert PP.legacy_merged_psf_grid_path(ROOT, 'brick', 'f405n', '1182', '001',
                                          oversample=2).endswith(name)


def test_the_reader_of_merged_grids_uses_the_shared_name():
    """`saturated_star_finding` must not re-spell the name it looks for.

    Its copy is what drifted; a second copy anywhere is the same defect again.
    """
    import inspect
    from jwst_gc_pipeline.reduction import saturated_star_finding as SSF

    src = inspect.getsource(SSF.get_psf)
    assert 'legacy_merged_psf_grid_name' in src, (
        'get_psf builds the merged-grid filename itself again')
    assert 'merged_PSFgrid' not in src, (
        'get_psf has a hand-written copy of the merged-grid filename')
    assert "PROGRAM'][1:5]" not in src and 'PROGRAM"][1:5]' not in src, (
        'get_psf reads the proposal with the 4-digit-only PROGRAM slice again')


# ---------------------------------------------------------------------------
# the header-keyed name (issue #414): PROGRAM is MAST's padded form
# ---------------------------------------------------------------------------

def _header(program, obs='001'):
    return {'PROGRAM': program, 'OBSERVTN': obs}


def test_the_header_keyed_name_matches_the_slice_on_a_four_digit_frame():
    """A real brick frame carries PROGRAM='01182'.  The name is unchanged from
    what the old ``PROGRAM[1:5]`` slice built, so no grid on disk is renamed."""
    name = PP.legacy_merged_psf_grid_name_from_header(
        _header('01182'), 'f405n', oversample=2)
    assert name == 'F405N_1182_001_merged_PSFgrid_oversample2.fits'
    assert name == PP.legacy_merged_psf_grid_name('f405n', '01182'[1:5], '001',
                                                  oversample=2)


def test_the_header_keyed_name_keeps_all_five_digits_of_a_treasury_frame():
    """The writer (`reduction.make_merged_psf`) names the grid with the
    registry proposal '10678'.  The old slice asked for ``F212N_0678_...``,
    which nothing writes, and the caller fell through to a detector-specific
    grid with no report that it had looked."""
    name = PP.legacy_merged_psf_grid_name_from_header(
        _header('10678'), 'f212n', oversample=2)
    assert name == 'F212N_10678_001_merged_PSFgrid_oversample2.fits'
    assert name != PP.legacy_merged_psf_grid_name('f212n', '10678'[1:5], '001',
                                                  oversample=2)


def test_the_header_keyed_name_agrees_with_what_the_writer_names():
    """Reader and writer both start from the registry proposal, for both
    digit widths."""
    for proposal, program in (('1182', '01182'), ('10678', '10678'),
                              ('12587', '12587')):
        written = PP.legacy_merged_psf_grid_name('f212n', proposal, '001',
                                                 oversample=2)
        read = PP.legacy_merged_psf_grid_name_from_header(
            _header(program), 'f212n', oversample=2)
        assert read == written, proposal


def _resolve(present):
    return PP.resolve_merged_psf_grid_path(
        ROOT, 'brick', 'NIRCam', 'nrcb5', 'f405n', '1182', '001',
        exists=lambda path: path in present)


def test_resolve_prefers_central_when_present():
    central = PP.central_merged_psf_grid_path(ROOT, 'NIRCam', 'nrcb5', 'f405n')
    legacy = PP.legacy_merged_psf_grid_path(ROOT, 'brick', 'f405n', '1182', '001')
    assert _resolve({central, legacy}) == central


def test_resolve_falls_back_to_legacy():
    legacy = PP.legacy_merged_psf_grid_path(ROOT, 'brick', 'f405n', '1182', '001')
    assert _resolve({legacy}) == legacy


def test_resolve_returns_central_when_neither_exists():
    central = PP.central_merged_psf_grid_path(ROOT, 'NIRCam', 'nrcb5', 'f405n')
    assert _resolve(set()) == central
