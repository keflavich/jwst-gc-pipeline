"""Where the pipeline runs, and with what.

``config.yaml`` ships with HiPerGator's settings. To run elsewhere, copy it,
edit it, and point ``GC_PIPELINE_CONFIG`` at the copy. A copy need only contain
what differs: missing keys fall back to the shipped file.
"""
import copy
import os

import yaml

ENV_VAR = 'GC_PIPELINE_CONFIG'

PACKAGED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'config.yaml')


class ConfigError(ValueError):
    """The config file says something the runner cannot act on."""


def _merge(base, override):
    """``override`` wins, one key at a time, recursing into dicts.

    So a copy that sets only ``slurm.qos`` keeps every other slurm setting.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path=None):
    """The active configuration.

    ``path`` overrides ``GC_PIPELINE_CONFIG``, which overrides the packaged
    file.
    """
    with open(PACKAGED) as fh:
        config = yaml.safe_load(fh)
    chosen = path or os.environ.get(ENV_VAR, '').strip()
    if chosen:
        if not os.path.exists(chosen):
            raise ConfigError(
                f'{ENV_VAR}={chosen!r} names a file that is not there')
        with open(chosen) as fh:
            config = _merge(config, yaml.safe_load(fh) or {})
        config['source'] = chosen
    else:
        config['source'] = PACKAGED
    _validate(config)
    return config


def _validate(config):
    if config.get('scheduler') not in ('slurm', 'local'):
        raise ConfigError(
            f"scheduler is {config.get('scheduler')!r}; it must be 'slurm' or "
            f"'local'")
    for name, stage in (config.get('stages') or {}).items():
        fan = stage.get('fan_out')
        if fan not in (None, 'filter', 'program-filter', 'none'):
            raise ConfigError(
                f"stage {name}: fan_out is {fan!r}; it must be 'filter', "
                f"'program-filter' or 'none'")


def stage(config, name):
    """One stage's settings."""
    stages = config.get('stages') or {}
    if name not in stages:
        raise ConfigError(
            f'stage {name!r} is missing from {config.get("source")}; '
            f'it defines {sorted(stages)}')
    return stages[name]


def environment(config):
    """The environment jobs run with.

    A variable already set in the caller's environment wins, so an interactive
    override needs no edit to the config.
    """
    out = {}
    for key, value in (config.get('environment') or {}).items():
        out[key] = os.environ.get(key) or str(value)
    return out
