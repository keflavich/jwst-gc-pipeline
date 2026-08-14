"""A field spec that names data which is not there must be caught before the queue.

The case this exists for is `run_sgra_4147_o001.sh`, which drove Sgr A* as
proposal 4147 for a whole campaign.  There is no 4147 data in the sgra tree --
every `_cal` frame is `jw01939001001_...` -- so the reduce array would have
failed every task, ~20 h after submission, and the retie loop would then have
refused to catalog the rest.

What makes it worth a test rather than a habit: every check the loop already
makes PASSES on that spec.  `alignment_config` resolves 4147/001 to a real
offsets table and the table is on disk.  It is the wrong field's table, and
nothing downstream can tell.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'scripts', 'reduction'))

import preflight_reduce_inputs as pf              # noqa: E402


def _field(tmp_path, target, filt, asn=(), cal=()):
    d = tmp_path / target / filt / 'pipeline'
    d.mkdir(parents=True, exist_ok=True)
    for name in asn:
        (d / name).write_text('{}')
    for name in cal:
        (d / name).write_text('x')
    return d


def test_a_complete_field_passes(tmp_path):
    _field(tmp_path, 'sgra', 'F115W',
           asn=['jw01939-o001_x_asn.json'],
           cal=['jw01939001001_02101_00001_nrca1_cal.fits',
                'jw01939001001_02101_00001_nrcb1_cal.fits'])
    rows = pf.check(str(tmp_path), 'sgra', '1939', '001', ['F115W'],
                    ['nrca', 'nrcb', 'merged'])
    assert rows[0][5] == ''


def test_the_sgra_case_the_wrong_proposal_entirely(tmp_path):
    """The data is 1939; the runner said 4147.  Everything downstream of this
    check accepts 4147 because a Brick4147 offsets table exists."""
    _field(tmp_path, 'sgra', 'F115W',
           asn=['jw01939-o001_x_asn.json'],
           cal=['jw01939001001_02101_00001_nrca1_cal.fits'])
    rows = pf.check(str(tmp_path), 'sgra', '4147', '001', ['F115W'], ['nrca'])
    assert 'no asn' in rows[0][5]


def test_a_present_proposal_with_the_wrong_OBSERVATION(tmp_path):
    _field(tmp_path, 'gc2211', 'F200W',
           asn=['jw02211-o046_x_asn.json'],
           cal=['jw02211046001_02101_00001_nrca1_cal.fits'])
    rows = pf.check(str(tmp_path), 'gc2211', '2211', '049', ['F200W'], ['nrca'])
    assert 'no asn' in rows[0][5]


def test_a_module_the_observation_does_not_have(tmp_path):
    """gc2211 o050 is nrcb-only in both filters.  Asking for nrca fails that
    task and takes `merged` with it -- the reason its runner overrides
    MODULES."""
    _field(tmp_path, 'gc2211', 'F200W',
           asn=['jw02211-o050_x_asn.json'],
           cal=['jw02211050001_02101_00001_nrcb1_cal.fits'])
    rows = pf.check(str(tmp_path), 'gc2211', '2211', '050', ['F200W'],
                    ['nrca', 'nrcb', 'merged'])
    assert rows[0][4] == ['nrca']
    assert "no _cal frames for module(s) ['nrca']" in rows[0][5]


def test_merged_is_not_looked_for_as_an_input(tmp_path):
    """`merged` is a product of the two module reductions.  Treating it as an
    input would fail every field."""
    _field(tmp_path, 'sickle', 'F187N',
           asn=['jw03958-o007_x_asn.json'],
           cal=['jw03958007001_03106_00001_nrcb1_cal.fits'])
    rows = pf.check(str(tmp_path), 'sickle', '3958', '007', ['F187N'],
                    ['nrcb', 'merged'])
    assert rows[0][5] == ''


def test_a_long_wavelength_detector_counts_for_its_family(tmp_path):
    """The `_cal` frames of an LW filter are `nrcalong`/`nrcblong`; a module
    spec names `nrca`/`nrcb`.  Matching the token literally would report every
    LW filter as missing both modules."""
    _field(tmp_path, 'sgrb2', 'F360M',
           asn=['jw05365-o001_x_asn.json'],
           cal=['jw05365001001_07101_00001_nrcalong_cal.fits',
                'jw05365001001_07101_00001_nrcblong_cal.fits'])
    rows = pf.check(str(tmp_path), 'sgrb2', '5365', '001', ['F360M'],
                    ['nrca', 'nrcb', 'merged'])
    assert rows[0][3] == ['nrca', 'nrcb']
    assert rows[0][5] == ''


def test_a_filter_the_field_does_not_have_at_all(tmp_path):
    """F150W exists only in gc2211 o028; F200W in every observation except it.
    Passing one filter list for all five asked each run for a band it lacks."""
    rows = pf.check(str(tmp_path), 'gc2211', '2211', '023', ['F150W'], ['nrca'])
    assert 'no directory' in rows[0][5]


def test_the_exit_status_is_nonzero_when_something_is_missing(tmp_path, capsys):
    _field(tmp_path, 'sgra', 'F115W', asn=[], cal=[])
    rc = pf.main(['--target', 'sgra', '--proposal', '4147', '--obsid', '001',
                  '--filters', 'F115W', '--root', str(tmp_path)])
    assert rc == 1
    assert 'check the PROPOSAL and OBSID' in capsys.readouterr().out


def test_the_exit_status_is_zero_when_everything_is_there(tmp_path):
    _field(tmp_path, 'sgra', 'F115W',
           asn=['jw01939-o001_x_asn.json'],
           cal=['jw01939001001_02101_00001_nrca1_cal.fits',
                'jw01939001001_02101_00001_nrcb1_cal.fits'])
    assert pf.main(['--target', 'sgra', '--proposal', '1939', '--obsid', '001',
                    '--filters', 'F115W', '--root', str(tmp_path)]) == 0
