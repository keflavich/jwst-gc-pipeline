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


#: Top-level keys the runner reads.  A file with anything else in it is more
#: likely a typo than an extension, and a typo here changes nothing silently.
KNOWN_KEYS = {'scheduler', 'slurm', 'environment', 'python', 'stages',
              'cutout', 'source'}

KNOWN_STAGE_KEYS = {'submit_script', 'cpus', 'memory', 'walltime', 'fan_out',
                    'skip_step1and2', 'modules'}

#: Keys the runner reads out of the ``slurm`` block.  A misspelled one is
#: ignored rather than obeyed -- ``logdir:`` would leave logs at whatever the
#: submit scripts' own #SBATCH directive says -- so reject it here.
KNOWN_SLURM_KEYS = {'account', 'qos', 'partition', 'log_dir'}


def _validate(config):
    unknown = set(config) - KNOWN_KEYS
    if unknown:
        raise ConfigError(
            f'unknown top-level key(s) {sorted(unknown)}; this file sets '
            f'{sorted(KNOWN_KEYS - {"source"})}')
    odd_slurm = set(config.get('slurm') or {}) - KNOWN_SLURM_KEYS
    if odd_slurm:
        raise ConfigError(
            f'slurm: unknown key(s) {sorted(odd_slurm)}; it sets '
            f'{sorted(KNOWN_SLURM_KEYS)}')
    if config.get('scheduler') not in ('slurm', 'local'):
        raise ConfigError(
            f"scheduler is {config.get('scheduler')!r}; it must be 'slurm' or "
            f"'local'")
    for name, stage in (config.get('stages') or {}).items():
        odd = set(stage) - KNOWN_STAGE_KEYS
        if odd:
            raise ConfigError(
                f'stage {name}: unknown key(s) {sorted(odd)}; a stage sets '
                f'{sorted(KNOWN_STAGE_KEYS)}')
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


def submit_script(config, stage_name, instrument='nircam'):
    """The submit script for one stage and instrument, as an absolute path.

    Raises when the instrument has none, rather than falling back to another
    instrument's script: they call different stage-1 drivers.
    """
    scripts = stage(config, stage_name).get('submit_script')
    if isinstance(scripts, str):
        scripts = {'nircam': scripts}
    chosen = (scripts or {}).get(instrument)
    if not chosen:
        raise ConfigError(
            f'stage {stage_name} has no submit script for {instrument} in '
            f'{config.get("source")}; it has {sorted(scripts or {})}.  Run it '
            f'with scheduler: local, or add one.')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute = chosen if os.path.isabs(chosen) else os.path.join(root, chosen)
    if not os.path.exists(absolute):
        raise ConfigError(
            f'stage {stage_name}: submit_script {chosen!r} is not there '
            f'(looked in {absolute})')
    return absolute


def environment(config):
    """The environment jobs run with.

    A variable already set in the caller's environment wins, so an interactive
    override needs no edit to the config.
    """
    out = {}
    for key, value in (config.get('environment') or {}).items():
        out[key] = os.environ.get(key) or str(value)
    return out


def apply_crds_environment(config=None):
    """Point CRDS at a cache, before ``jwst`` is imported.

    The reduce drivers call this at import time: ``jwst`` reads ``CRDS_PATH``
    when it loads, so setting it afterwards is too late.  A value already
    exported -- by ``run_pipeline``, by a submit script, or by hand -- wins over
    the configured one, so a machine with its own cache needs no edit here.

    Returns the cache and server it settled on, for logging.  Only those two:
    ``CRDS_USERNAME`` and ``CRDS_PASSWORD`` are also real CRDS variables, and a
    SLURM log is a wider audience than the process.
    """
    for key, value in environment(config if config is not None else load()).items():
        # `not os.environ.get`, rather than setdefault: an exported-but-empty
        # CRDS_PATH counts as set, and jwst would fall back to ~/crds_cache and
        # download tens of GB into a quota-limited home directory.
        if key.startswith('CRDS_') and not os.environ.get(key):
            os.environ[key] = value
    return {key: os.environ[key] for key in ('CRDS_PATH', 'CRDS_SERVER_URL')
            if key in os.environ}
