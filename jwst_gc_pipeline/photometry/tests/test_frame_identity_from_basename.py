"""A frame's identity comes from its BASENAME, never from its path.

``run_manual_pipeline`` builds each frame's per-frame output identity by counting
underscores from the left::

    exposure_id   = base.split("_")[2]
    visit_id      = base.split("_")[0][-3:]
    vgroup_id     = base.split("_")[1]
    file_detector = base.split("_")[3]

Run against the full path, every one of those indices shifts as soon as the
DIRECTORY contains an underscore.  Measured on 2026-08-22 with a basepath of
``/orange/adamginsburg/jwst/gc2211_o023/``::

    visit_id  '211'                               (from '.../jwst/gc2211')
    vgroup_id 'o023//F200W/pipeline/jw02211023001'
    detector  '00001'

so all four exposures of a filter shared one identity and the collision guard
aborted the run -- correctly, since the alternative is four frames writing one
file.  Three fields lost every m12 shard to this: gc2211_o023, gc2211_o028 and
cloudef_controlfield.

Underscored field directories are ordinary now (the gc2211 per-observation split,
the cloudef control field), so this is pinned rather than left to the next person
who adds one.
"""
import os

import pytest


#: (path, visit, vgroup, exposure, detector)
CASES = [
    # underscored field directories -- the case that broke
    ('/orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline/'
     'jw02211023001_02201_00001_nrca1_destreak_o023_crf.fits',
     '001', '02201', '00001', 'nrca1'),
    ('/orange/adamginsburg/jwst/cloudef_controlfield/F162M/pipeline/'
     'jw02092005001_02101_00003_nrcb2_align_o005_crf.fits',
     '001', '02101', '00003', 'nrcb2'),
    # a trailing slash on the basepath gives the doubled separator seen in the
    # real failure ('o023//F200W'), so keep one case with it
    ('/orange/adamginsburg/jwst/gc2211_o050//F277W/pipeline/'
     'jw02211050001_06201_00012_nrcblong_destreak_o050_crf.fits',
     '001', '06201', '00012', 'nrcblong'),
    # ordinary directories, unchanged
    ('/orange/adamginsburg/jwst/brick/F200W/pipeline/'
     'jw01182004001_04101_00007_nrca3_destreak_o004_crf.fits',
     '001', '04101', '00007', 'nrca3'),
    ('/orange/adamginsburg/jwst/sickle/F187N/pipeline/'
     'jw03958007001_0310c_00004_nrcb1_destreak_o007_crf.fits',
     '001', '0310c', '00004', 'nrcb1'),
]


def _identity(path):
    """Exactly what `run_manual_pipeline` does, via the basename."""
    base = os.path.basename(path)
    return (base.split('_')[0][-3:], base.split('_')[1],
            base.split('_')[2], base.split('_')[3])


@pytest.mark.parametrize('path,visit,vgroup,exposure,detector', CASES)
def test_identity_is_read_from_the_basename(path, visit, vgroup, exposure,
                                            detector):
    assert _identity(path) == (visit, vgroup, exposure, detector)


@pytest.mark.parametrize('path,visit,vgroup,exposure,detector', CASES)
def test_the_path_split_is_what_went_wrong(path, visit, vgroup, exposure,
                                           detector):
    """The old expression, shown failing where the directory has an underscore
    and agreeing where it does not -- so the fix is not mistaken for a
    no-op."""
    old = (path.split('_')[0][-3:], path.split('_')[1],
           path.split('_')[2], path.split('_')[3])
    underscored = '_' in os.path.basename(os.path.dirname(
        os.path.dirname(os.path.dirname(path.rstrip('/')))))
    if underscored:
        assert old != (visit, vgroup, exposure, detector)
    else:
        assert old == (visit, vgroup, exposure, detector)


def test_the_production_code_splits_a_basename():
    """Pins the call site itself: a future edit that drops `os.path.basename`
    reintroduces the bug, and the parametrised cases above would not see it
    because they reimplement the expression."""
    import inspect

    from jwst_gc_pipeline.photometry import cataloging

    src = inspect.getsource(cataloging.run_manual_pipeline)
    _, _, tail = src.partition('for filename in candidate_frames:')
    head = tail[:1200]
    assert '_base = os.path.basename(filename)' in head, (
        'frame identity is being parsed from something other than the basename')
    for slot in ('exposure_id = _base.split', 'visit_id = _base.split',
                 'vgroup_id = _base.split', 'file_detector = _base.split'):
        assert slot in head, slot


def test_every_frame_of_a_filter_gets_a_distinct_identity():
    """The property the collision guard enforces, stated directly: four
    detectors of one exposure differ, and four exposures of one detector
    differ."""
    root = '/orange/adamginsburg/jwst/gc2211_o023/F200W/pipeline'
    names = [f'{root}/jw02211023001_02201_{e:05d}_{d}_destreak_o023_crf.fits'
             for e in (1, 2, 3, 4)
             for d in ('nrca1', 'nrca2', 'nrca3', 'nrca4')]
    ids = [_identity(n) for n in names]
    assert len(set(ids)) == len(names) == 16
