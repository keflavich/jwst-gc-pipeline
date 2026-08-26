"""Seeding a name the reader does not probe for is silent (#420).

The whole value of ``scripts/reduction/seed_psf_cache.py`` is that
``get_psf_model`` finds a grid in the field's ``psfs/`` and skips a MAST login
plus a ~17-20 min Poppy build (~7-8 h for the merged/all-detectors path).  A
seeded file under a name the reader never looks up costs exactly that build and
reports nothing, so the first test pins the filename shape to the two f-strings
in the reader itself.

The rest are about the two ways this can quietly do the wrong thing: overwrite
a grid that is already there, and hard-link across the /blue--/orange boundary,
which is ``EXDEV``.
"""
import importlib.util
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).parents[1]
SCRIPT = REPO / 'scripts' / 'reduction' / 'seed_psf_cache.py'
READER = (REPO / 'jwst_gc_pipeline' / 'photometry'
          / 'crowdsource_catalogs_long.py')


def _load():
    spec = importlib.util.spec_from_file_location('seed_psf_cache', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEED = _load()


def test_the_glob_matches_the_name_the_reader_probes_for():
    """Both cache lookups in ``get_psf_model`` -- the per-detector one and the
    merged/all-detectors one -- build the same name.  Pin ours to theirs."""
    src = READER.read_text()
    literal = ("_fovp101_samp{_samp}_npsf16.fits'")
    assert src.count(literal) == 2, (
        'the reader no longer builds two _fovp101_samp{N}_npsf16.fits names; '
        'seed_psf_cache.GRID_GLOB has to follow it')
    ours = SEED.GRID_GLOB.format(filt='f212n')
    assert ours == '*_f212n_fovp101_samp*_npsf16.fits'
    for name in ('nircam_nrca1_f212n_fovp101_samp2_npsf16.fits',
                 'nircam_nrcb5_f480m_fovp101_samp4_npsf16.fits',
                 'miri_mirim_f770w_fovp101_samp4_npsf16.fits'):
        filt = name.split('_')[2]
        import fnmatch
        assert fnmatch.fnmatch(name, SEED.GRID_GLOB.format(filt=filt)), name


def _grid(directory, name, content=b'grid'):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(content)
    return directory / name


F212N_A1 = 'nircam_nrca1_f212n_fovp101_samp2_npsf16.fits'
F480M_A5 = 'nircam_nrca5_f480m_fovp101_samp2_npsf16.fits'


def test_donor_dirs_deduplicates_a_symlinked_target(tmp_path):
    """`/orange/adamginsburg/jwst/brick` is a symlink to `/blue/.../jwst/brick`,
    so listing both roots offers the same directory twice and the second copy
    reads as an independent donor."""
    blue, orange = tmp_path / 'blue', tmp_path / 'orange'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    orange.mkdir()
    (orange / 'brick').symlink_to(blue / 'brick')
    dirs = SEED.donor_dirs(roots=[str(blue), str(orange)])
    assert len(dirs) == 1, dirs


def test_donor_dirs_skips_the_destination_field(tmp_path):
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    _grid(blue / 'gc-treasury' / 'psfs', F212N_A1)
    dirs = SEED.donor_dirs(roots=[str(blue)], skip=('gc-treasury',))
    assert [os.path.basename(os.path.dirname(d)) for d in dirs] == ['brick']


def test_donor_grids_keeps_one_entry_per_filename(tmp_path):
    """brick and cloudc both hold the eight F212N SW grids; that is one grid to
    seed, not two."""
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    _grid(blue / 'cloudc' / 'psfs', F212N_A1)
    grids = SEED.donor_grids('F212N', SEED.donor_dirs(roots=[str(blue)]))
    assert list(grids) == [F212N_A1]


def test_donor_grids_matches_the_filter_case_insensitively(tmp_path):
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    assert SEED.donor_grids('F212N', SEED.donor_dirs(roots=[str(blue)]))
    assert not SEED.donor_grids('F480M', SEED.donor_dirs(roots=[str(blue)]))


def test_plan_reports_a_filter_with_no_donor_as_missing(tmp_path):
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    rows = SEED.plan('gc-treasury', ['F212N', 'F444W'], roots=[str(blue)],
                     dest=str(tmp_path / 'dest'))
    actions = {r[4] for r in rows}
    assert actions == {'seed', 'missing'}
    assert any('f444w' in r[0] for r in rows if r[4] == 'missing')


def test_plan_marks_an_already_seeded_grid_present(tmp_path):
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    dest = tmp_path / 'dest'
    _grid(dest, F212N_A1)
    rows = SEED.plan('gc-treasury', ['F212N'], roots=[str(blue)],
                     dest=str(dest))
    assert [r[4] for r in rows] == ['present']


def test_place_refuses_to_overwrite(tmp_path):
    src = _grid(tmp_path / 'src', F212N_A1)
    dst = _grid(tmp_path / 'dst', F212N_A1, content=b'mine')
    with pytest.raises(FileExistsError):
        SEED.place(str(src), str(dst), 'hardlink')
    assert dst.read_bytes() == b'mine'


def test_hardlink_shares_the_inode(tmp_path):
    src = _grid(tmp_path / 'src', F212N_A1)
    dst = tmp_path / 'dst'
    dst.mkdir()
    SEED.place(str(src), str(dst / F212N_A1), 'hardlink')
    assert os.stat(src).st_ino == os.stat(dst / F212N_A1).st_ino


def test_symlink_resolves_to_the_donor(tmp_path):
    src = _grid(tmp_path / 'src', F480M_A5)
    dst = tmp_path / 'dst'
    dst.mkdir()
    SEED.place(str(src), str(dst / F480M_A5), 'symlink')
    assert (dst / F480M_A5).is_symlink()
    assert os.path.realpath(dst / F480M_A5) == str(src)
    # os.path.exists follows the link, which is what the reader's cache check
    # calls before to_griddedpsfmodel.
    assert os.path.exists(dst / F480M_A5)


def test_copy_leaves_no_partial_behind(tmp_path):
    """`os.path.exists` on a half-written grid is a cache hit; the reader loads
    it and the run dies on a truncated FITS instead of rebuilding."""
    src = _grid(tmp_path / 'src', F212N_A1, content=b'x' * 4096)
    dst = tmp_path / 'dst'
    dst.mkdir()
    SEED.place(str(src), str(dst / F212N_A1), 'copy')
    assert (dst / F212N_A1).read_bytes() == b'x' * 4096
    assert list(dst.glob('*.partial')) == []


def test_link_kind_picks_hardlink_on_one_filesystem(tmp_path):
    src = _grid(tmp_path / 'src', F212N_A1)
    assert SEED.link_kind(str(src), str(tmp_path / 'dst')) == 'hardlink'


def test_link_kind_walks_up_to_an_existing_ancestor(tmp_path):
    """The destination psfs/ does not exist before the first seed -- that is the
    state this script is for."""
    src = _grid(tmp_path / 'src', F212N_A1)
    deep = tmp_path / 'gc-treasury' / 'psfs'
    assert SEED.link_kind(str(src), str(deep)) == 'hardlink'


def test_link_kind_picks_symlink_across_filesystems(tmp_path):
    """/blue/adamginsburg is /blue2/hpg and /orange/adamginsburg is /orange/hpg;
    os.link between them is EXDEV, so the choice is made from st_dev rather
    than attempted and rescued."""
    src = _grid(tmp_path / 'src', F480M_A5)

    def fake_stat(path):
        return type('S', (), {'st_dev': 1 if 'src' in str(path) else 2})()

    assert SEED.link_kind(str(src), str(tmp_path / 'dst'),
                          stat=fake_stat) == 'symlink'


def test_copy_flag_overrides_the_link_kind(tmp_path):
    src = _grid(tmp_path / 'src', F212N_A1)
    assert SEED.link_kind(str(src), str(tmp_path / 'dst'), copy=True) == 'copy'


def test_main_is_a_dry_run_by_default(tmp_path, capsys, monkeypatch):
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    monkeypatch.setattr(SEED.fields, 'ROOTS', {'blue': str(blue)})
    monkeypatch.setattr(SEED.fields, 'fields_basepath',
                        lambda field: str(tmp_path / 'dest' / field) + '/')
    SEED.main(['--field', 'gc-treasury', '--filters', 'F212N'])
    assert 'dry run' in capsys.readouterr().out
    assert not (tmp_path / 'dest').exists()


def test_main_apply_seeds_the_fields_own_psfs_directory(tmp_path, capsys,
                                                        monkeypatch):
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    monkeypatch.setattr(SEED.fields, 'ROOTS', {'blue': str(blue)})
    monkeypatch.setattr(SEED.fields, 'fields_basepath',
                        lambda field: str(tmp_path / 'dest' / field) + '/')
    rc = SEED.main(['--field', 'gc-treasury', '--filters', 'F212N', '--apply'])
    assert rc == 0, capsys.readouterr().out
    seeded = tmp_path / 'dest' / 'gc-treasury' / 'psfs' / F212N_A1
    assert seeded.exists()


def test_main_reports_nonzero_when_a_filter_has_no_donor(tmp_path, monkeypatch):
    """A silent zero would say "seeded" on a run that left the expensive
    rebuild in place."""
    blue = tmp_path / 'blue'
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    monkeypatch.setattr(SEED.fields, 'ROOTS', {'blue': str(blue)})
    monkeypatch.setattr(SEED.fields, 'fields_basepath',
                        lambda field: str(tmp_path / 'dest' / field) + '/')
    rc = SEED.main(['--field', 'gc-treasury', '--filters', 'F212N', 'F444W',
                    '--apply'])
    assert rc == 1


def test_plan_prefers_a_donor_on_the_destinations_filesystem(tmp_path):
    """The eight F212N SW grids exist in brick/cloudc (/blue) and in
    arches/quintuplet/sgra/sgrb2 (/orange).  A /blue-rooted field should hard-
    link the /blue copy rather than symlink into another filesystem."""
    blue, orange = tmp_path / 'blue', tmp_path / 'orange'
    _grid(orange / 'arches' / 'psfs', F212N_A1)
    _grid(blue / 'brick' / 'psfs', F212N_A1)
    dest = tmp_path / 'blue' / 'gc-treasury' / 'psfs'

    def fake_stat(path):
        dev = 2 if str(path).startswith(str(orange)) else 1
        return type('S', (), {'st_dev': dev})()

    rows = SEED.plan('gc-treasury', ['F212N'], roots=[str(blue), str(orange)],
                     dest=str(dest), stat=fake_stat)
    assert [r[4] for r in rows] == ['seed']
    assert 'brick' in rows[0][1], rows
    assert rows[0][3] == 'hardlink'
