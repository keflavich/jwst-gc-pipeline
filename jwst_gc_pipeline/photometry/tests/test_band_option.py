"""Per-band values of the manual-pipeline options (``band_option``).

An option takes a plain number (every band) or ``BAND=VALUE`` pairs
(``F212N=50,F480M=0``); a band that no pair names uses the default of the
option.  The tests cover the parser, the resolution for one band, the optparse
callback, and every option and call site of the pipeline that uses them.
"""
import ast
import inspect
import optparse
import types

import pytest

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as CL
from jwst_gc_pipeline.photometry.band_option import (
    band_mopt, band_option_callback, band_value, parse_band_values)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

OPT = '--some-option'


def test_plain_number_is_a_float():
    for raw in ('50', ' 50 ', 50, 50.0, '5e1'):
        value = parse_band_values(raw, OPT)
        assert value == 50.0 and type(value) is float
    assert parse_band_values('0', OPT) == 0.0
    assert parse_band_values(0, OPT) == 0.0


def test_pairs_give_lower_case_bands():
    assert parse_band_values('F212N=50,F480M=0', OPT) == {'f212n': 50.0, 'f480m': 0.0}
    assert parse_band_values(' f212n = 50 , F480m=0.5 ', OPT) == {'f212n': 50.0, 'f480m': 0.5}
    assert parse_band_values('F212N=30', OPT) == {'f212n': 30.0}


def test_a_parsed_mapping_is_validated_and_copied():
    parsed = parse_band_values({'F212N': 50, 'f480m': '0'}, OPT)
    assert parsed == {'f212n': 50.0, 'f480m': 0.0}
    assert parse_band_values(parsed, OPT) == parsed
    assert parse_band_values(parsed, OPT) is not parsed
    assert parse_band_values(30.0, OPT) == 30.0


@pytest.mark.parametrize('raw, fragment', [
    ('50,F480M=0', 'mixes a plain number'),
    ('F212N=50,30', 'mixes a plain number'),
    ('F212N=50,F212N=40', 'appears twice'),
    ('F212N=50,f212n=40', 'appears twice'),
    ('F212N=abc', "'abc'"),
    ('F212N=-1', '>= 0'),
    ('-1', '>= 0'),
    ('nan', 'finite'),
    ('inf', 'finite'),
    ('F212N=nan', 'finite'),
    ('F480M=inf,F212N=1', 'finite'),
    ('abc', 'not a number or a list'),
    ('50,30', 'not a number or a list'),
    ('F212N=50,', 'cannot read'),
    ('=50', 'cannot read'),
    ('F212N=', 'cannot read'),
    ('F212N=50=60', 'cannot read'),
    ('F212N:50', 'not a number or a list'),
    ('F 212N=50', 'cannot read'),
    ('', 'is empty'),
    ('  ', 'is empty'),
])
def test_a_malformed_string_names_the_option_and_the_text(raw, fragment):
    with pytest.raises(ValueError) as err:
        parse_band_values(raw, OPT)
    message = str(err.value)
    assert message.startswith(OPT + ':')
    assert fragment in message
    if raw.strip():
        assert repr(raw.strip()) in message


@pytest.mark.parametrize('raw', [None, True, {}, {'f212n': -1}, {'f212n': 'x'}, {'F212N': 1, 'f212n': 2},
                                 {'': 1}, [50], (1, 2)])
def test_other_malformed_values_raise_value_error(raw):
    with pytest.raises(ValueError, match=OPT):
        parse_band_values(raw, OPT)


def test_band_value_resolves_one_band():
    assert band_value('50', 'F212N', 7.0, OPT) == 50.0
    assert band_value(50, 'F480M', 7.0, OPT) == 50.0
    assert band_value('F212N=50,F480M=0', 'F212N', 7.0, OPT) == 50.0
    assert band_value('F212N=50,F480M=0', 'f212n', 7.0, OPT) == 50.0
    assert band_value('F212N=50,F480M=0', 'F480M', 7.0, OPT) == 0.0
    assert band_value({'f212n': 50.0}, ' F212N ', 7.0, OPT) == 50.0
    # a band that no pair names uses the default
    assert band_value('F212N=50', 'F480M', 7.0, OPT) == 7.0
    assert type(band_value('F212N=50', 'F480M', 7, OPT)) is float
    assert band_value('F212N=50', 'F182M', 0.0, OPT) == 0.0
    with pytest.raises(ValueError, match=OPT):
        band_value('F212N=-3', 'F212N', 7.0, OPT)


def test_band_mopt_reads_the_option_with_its_default(capsys):
    name = 'manual_seed_round_loose_prom_min'
    default = MANUAL_DEFAULTS[name]
    assert band_mopt(types.SimpleNamespace(), name, 'F212N') == default
    assert band_mopt(types.SimpleNamespace(**{name: 30.0}), name, 'F480M') == 30.0
    pairs = types.SimpleNamespace(**{name: 'F212N=30,F480M=0'})
    assert band_mopt(pairs, name, 'F212N') == 30.0
    assert band_mopt(pairs, name, 'F480M') == 0.0
    assert band_mopt(pairs, name, 'F182M') == default
    assert capsys.readouterr().out == ''
    with pytest.raises(KeyError):
        band_mopt(types.SimpleNamespace(), 'definitely_not_a_knob', 'F212N')
    with pytest.raises(ValueError, match='--manual-seed-round-loose-prom-min'):
        band_mopt(types.SimpleNamespace(**{name: '30,F212N=1'}), name, 'F212N')


def test_band_mopt_log_names_the_value_the_band_uses(capsys):
    name = 'manual_seed_round_loose_prom_min'
    flag = '--manual-seed-round-loose-prom-min'
    pairs = types.SimpleNamespace(**{name: 'F212N=30,F480M=0'})
    band_mopt(pairs, name, 'F212N', label='m7:F212N')
    band_mopt(pairs, name, 'F182M', label='m7:F182M')
    first, second = capsys.readouterr().out.splitlines()
    assert first == f'[m7:F212N] {flag} F212N=30,F480M=0: F212N uses 30'
    assert second == (f'[m7:F182M] {flag} F212N=30,F480M=0: F182M uses {MANUAL_DEFAULTS[name]:g}'
                      ' (not listed: the default)')
    # a plain number, and no label, print nothing
    band_mopt(types.SimpleNamespace(**{name: 30.0}), name, 'F212N', label='m7:F212N')
    band_mopt(pairs, name, 'F212N')
    assert capsys.readouterr().out == ''


def _parser():
    parser = optparse.OptionParser()
    parser.add_option('--per-band', dest='per_band', type='string', action='callback',
                      callback=band_option_callback, default=1.5)
    return parser


def test_callback_stores_the_normalized_value():
    parser = _parser()
    assert parser.parse_args([])[0].per_band == 1.5
    assert parser.parse_args(['--per-band', '30'])[0].per_band == 30.0
    assert type(parser.parse_args(['--per-band=30'])[0].per_band) is float
    assert parser.parse_args(['--per-band', 'F212N=50,F480M=0'])[0].per_band == {'f212n': 50.0, 'f480m': 0.0}
    assert parser.parse_args(['--per-band=F212N=50'])[0].per_band == {'f212n': 50.0}
    # the last occurrence wins, as for any option
    assert parser.parse_args(['--per-band', '30', '--per-band', 'F480M=2'])[0].per_band == {'f480m': 2.0}


@pytest.mark.parametrize('raw', ['50,F480M=0', 'F212N=50,F212N=40', 'F212N=-1', 'abc', ''])
def test_callback_error_stops_the_command_line(raw, capsys):
    with pytest.raises(SystemExit) as stop:
        _parser().parse_args(['--per-band', raw])
    assert stop.value.code == 2
    assert '--per-band:' in capsys.readouterr().err


def _per_band_add_options():
    """The ``add_option`` calls of the command line that take ``band_option_callback``."""
    tree = ast.parse(inspect.getsource(CL))
    return [call for call in ast.walk(tree)
            if isinstance(call, ast.Call) and getattr(call.func, 'attr', None) == 'add_option'
            and any(k.arg == 'callback' and ast.unparse(k.value) == 'band_option_callback'
                    for k in call.keywords)]


def test_the_options_of_the_command_line_take_per_band_values():
    calls = _per_band_add_options()
    assert calls
    for call in calls:
        parser = optparse.OptionParser()
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        flag = ast.literal_eval(call.args[0])
        dest = ast.literal_eval(kw['dest'])
        assert flag == '--' + dest.replace('_', '-')
        assert dest in MANUAL_DEFAULTS
        assert kw['type'] in ("'string'", '"string"') and kw['action'] in ("'callback'", '"callback"')
        assert kw['default'] in (f"MANUAL_DEFAULTS['{dest}']", f'MANUAL_DEFAULTS["{dest}"]')
        # the option as the command line defines it
        env = {'parser': parser, 'MANUAL_DEFAULTS': MANUAL_DEFAULTS,
               'band_option_callback': band_option_callback}
        expr = ast.fix_missing_locations(ast.Expression(call))
        eval(compile(expr, '<add_option>', 'eval'), env)
        assert getattr(parser.parse_args([])[0], dest) == MANUAL_DEFAULTS[dest]
        assert getattr(parser.parse_args([flag, '12'])[0], dest) == 12.0
        assert getattr(parser.parse_args([f'{flag}=F212N=3,F480M=0'])[0], dest) == {'f212n': 3.0, 'f480m': 0.0}
        with pytest.raises(SystemExit):
            parser.parse_args([flag, 'F212N=3,4'])


def test_every_band_mopt_call_of_the_pipeline_names_a_default():
    tree = ast.parse(inspect.getsource(cataloging))
    calls = [call for call in ast.walk(tree)
             if isinstance(call, ast.Call) and getattr(call.func, 'id', None) == 'band_mopt']
    assert calls
    for call in calls:
        name = ast.literal_eval(call.args[1])
        assert isinstance(MANUAL_DEFAULTS[name], float)
        assert ast.unparse(call.args[2]) == 'filt'
