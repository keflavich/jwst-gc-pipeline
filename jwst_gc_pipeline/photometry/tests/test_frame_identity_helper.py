"""Every site that parses a frame's identity uses the shared basename helper.

``#472`` fixed the underscore-index parse in ``run_manual_pipeline`` and left
four copies of the same expression standing on the same input -- ``get_filenames``
returns FULL PATHS, and the indices count underscores from the left, so a field
directory with an underscore shifts every one of them (``gc2211_o023``,
``gc2211_o028/046/049/050``, ``cloudef_controlfield``).

The four:

    photometry/forced_fill.py                 _frame_args_from_filename
    photometry/crowdsource_catalogs_long.py   list-missing-tasks bookkeeping
    photometry/crowdsource_catalogs_long.py   per-frame loop -> _expected_output_exists
    photometry/legacy/crowdsource_step.py     per-frame overlap loop

Two further copies in ``crowdsource_catalogs_long.py`` (the mergedcat per-frame
builder and the iter2 overlap loop) already took a basename and were correct;
they go through the same helper so the expression has one home rather than
three correct spellings and four broken ones.

`test_frame_identity_from_basename.py` states the property; these tests pin the
CALL SITES, which is what the copies escaped.
"""
import inspect
import os

import pytest

from jwst_gc_pipeline.photometry.naming import frame_identity

#: (path, visit, vgroup, exposure, detector) -- the #472 cases, run through the
#: shared helper this time.
CASES = [
    ('/orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline/'
     'jw02211023001_02201_00001_nrca1_destreak_o023_crf.fits',
     '001', '02201', '00001', 'nrca1'),
    ('/orange/adamginsburg/jwst/cloudef_controlfield/F162M/pipeline/'
     'jw02092005001_02101_00003_nrcb2_align_o005_crf.fits',
     '001', '02101', '00003', 'nrcb2'),
    ('/orange/adamginsburg/jwst/gc2211_o050//F277W/pipeline/'
     'jw02211050001_06201_00012_nrcblong_destreak_o050_crf.fits',
     '001', '06201', '00012', 'nrcblong'),
    ('/orange/adamginsburg/jwst/brick/F200W/pipeline/'
     'jw01182004001_04101_00007_nrca3_destreak_o004_crf.fits',
     '001', '04101', '00007', 'nrca3'),
]


@pytest.mark.parametrize('path,visit,vgroup,exposure,detector', CASES)
def test_the_helper_reads_the_basename(path, visit, vgroup, exposure, detector):
    assert frame_identity(path) == (visit, vgroup, exposure, detector)
    # and the basename alone gives the same answer as the full path
    assert frame_identity(os.path.basename(path)) == frame_identity(path)


def test_a_joint_multiobs_field_folds_the_observation_into_the_vgroup():
    """sgrb2 obs998 ("redo") reused obs002's mosaic tile numbers, so both map to
    vgroup 02101 visit 001 and would write one another's per-frame outputs.  A
    field token with a '-' is what says the run is joint."""
    a = ('/orange/adamginsburg/jwst/sgrb2/F187N/pipeline/'
         'jw05365002001_02101_00001_nrca1_destreak_o002_crf.fits')
    b = ('/orange/adamginsburg/jwst/sgrb2/F187N/pipeline/'
         'jw05365998001_02101_00001_nrca1_destreak_o998_crf.fits')
    assert frame_identity(a)[1] == frame_identity(b)[1] == '02101'
    assert frame_identity(a, field='002-998')[1] == '00202101'
    assert frame_identity(b, field='002-998')[1] == '99802101'
    # a single-obs field is untouched
    assert frame_identity(a, field='002')[1] == '02101'


def test_sixteen_frames_of_a_filter_get_sixteen_identities():
    """The property the per-frame collision guard enforces, on the path form
    that broke it."""
    root = '/orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline'
    names = [f'{root}/jw02211023001_02201_{e:05d}_{d}_destreak_o023_crf.fits'
             for e in (1, 2, 3, 4)
             for d in ('nrca1', 'nrca2', 'nrca3', 'nrca4')]
    assert len({frame_identity(n) for n in names}) == len(names) == 16


def test_forced_fill_builds_its_frame_args_from_the_basename():
    """The m8 / forced-fill path.  Before this, a `gc2211_o023` frame produced
    visit '211', vgroup 'o023/F200W/pipeline/jw02211023001' (a vgroup with
    slashes in it, which then goes into an output filename) and detector
    '00001', so `module=` was an exposure number and every frame of the filter
    shared one identity."""
    from jwst_gc_pipeline.photometry import forced_fill

    path = ('/orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline/'
            'jw02211023001_02201_00001_nrca1_destreak_o023_crf.fits')
    args = forced_fill._frame_args_from_filename(
        path, options=None, filt='f200w', field='023',
        basepath='/orange/adamginsburg/jwst/gc2211_o023/',
        proposal_id='2211', bg_boxsizes=None, pupil='clear', resbg_path=None,
        satstar_overrides=None, satstar_drops=None, module='merged')

    assert args['visit_id'] == '001'
    assert args['vgroup_id'] == '02201'
    assert args['exposurenumber'] == 1
    assert args['module'] == 'nrca1'
    assert '/' not in args['vgroup_id']


def test_forced_fill_still_folds_the_observation_for_a_joint_field():
    from jwst_gc_pipeline.photometry import forced_fill

    path = ('/orange/adamginsburg/jwst/sgrb2/F187N/pipeline/'
            'jw05365998001_02101_00001_nrca1_destreak_o998_crf.fits')
    args = forced_fill._frame_args_from_filename(
        path, options=None, filt='f187n', field='002-998',
        basepath='/orange/adamginsburg/jwst/sgrb2/',
        proposal_id='5365', bg_boxsizes=None, pupil='clear', resbg_path=None,
        satstar_overrides=None, satstar_drops=None, module='merged')
    assert args['vgroup_id'] == '99802101'


#: (module path, attribute, how many times the helper must appear)
CALL_SITES = [
    ('jwst_gc_pipeline.photometry.forced_fill', '_frame_args_from_filename', 1),
    ('jwst_gc_pipeline.photometry.legacy.crowdsource_step', None, 1),
    ('jwst_gc_pipeline.photometry.crowdsource_catalogs_long', None, 4),
]


@pytest.mark.parametrize('modname,attr,count', CALL_SITES)
def test_no_call_site_parses_the_path(modname, attr, count):
    """A future edit that drops the helper and re-inlines the split
    reintroduces the bug, and the cases above would not see it because they
    exercise the helper directly."""
    import importlib

    mod = importlib.import_module(modname)
    src = (inspect.getsource(getattr(mod, attr)) if attr
           else inspect.getsource(mod))
    assert src.count('frame_identity(') >= count, (
        f'{modname} should reach frame_identity() {count}x')
    assert 'split("_")[0][-3:]' not in src, (
        f'{modname} parses a frame identity by hand again; use frame_identity()')
    assert "split('_')[0][-3:]" not in src
