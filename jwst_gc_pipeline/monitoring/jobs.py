"""Live SLURM state and log-error scraping for the pipeline monitor.

Two independent sources of "what is happening right now":

* the **queue** (``squeue``/``sacct``) -- what SLURM thinks is running, and
* the **logs** -- what the job actually printed.

Both are keyed back to a field by the submit-time job name, whose standing
format is ``<target><proposal>-o<obsid>-<stage>[-<FILTER>]`` (see
``scripts/reduction/submit_cataloging.sbatch``); older jobs used looser names
(``brick-catalog``, ``pf_sgrb2_m12_s3``), so the parser accepts several shapes
and reports which one matched rather than guessing silently.

Everything here degrades to an empty result when SLURM is absent, so the
monitor runs on a machine with no scheduler (and in tests).
"""
import os
import re
import shutil
import subprocess
from collections import Counter

#: Where the sbatch templates send stdout (``--output=`` in every submit script).
LOG_DIR = os.environ.get('GC_MONITOR_LOG_DIR', '/orange/adamginsburg/jwst/logs')

#: Job states that mean "this field has work in flight".
ACTIVE_STATES = ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED')

#: Terminal states that are failures (as opposed to COMPLETED).
FAILED_STATES = ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                 'CANCELLED', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED')

# --------------------------------------------------------------------------
# Job-name parsing
# --------------------------------------------------------------------------

# ``<target><proposal>`` is CONCATENATED with no separator, and several targets
# themselves end in digits (sgrb2, gc2211, wd1, wd2, m4, m92).  No regex can
# split ``sgrb25365`` correctly on its own -- ``sgrb``+``25365`` is just as valid
# a reading as ``sgrb2``+``5365`` -- so the head is captured whole and split
# against the field registry (longest registered name first).
#
# brick2221-o001-m12-fanout / arches2045-o001-cat-F182M / sgrb25365-o001-m12-finalize
_NAME_FULL = re.compile(
    r'^(?P<head>[a-z][a-z0-9]*)-o(?P<obsid>\d+)'
    r'-(?P<stage>[a-z0-9]+)(?:-(?P<filter>[Ff]\d{3,4}[A-Za-z0-9]*))?')
# arches-001-m12-fanout  (older: no proposal, obsid separated by a dash)
_NAME_NOPROP = re.compile(
    r'^(?P<head>[a-z][a-z0-9]*)-(?P<obsid>\d{3})'
    r'-(?P<stage>[a-z0-9]+)(?:-(?P<filter>[Ff]\d{3,4}[A-Za-z0-9]*))?')
# pf_sgrb2_m12_s3 / pf_arches_m12_fin -- the trailing _s<N>/_fin is the shard,
# not part of the stage, so it is not captured.
_NAME_PF = re.compile(r'^pf_(?:gc_)?(?P<head>[a-z][a-z0-9]*)_(?P<stage>m\d+)')
# brick-catalog / w51-catalog / w51-9filt-reseed-gaiafix / w51-miri-rereduce
_NAME_LOOSE = re.compile(r'^(?P<head>[a-z][a-z0-9]*)-(?P<stage>.+)$')

_PROPOSAL_RE = re.compile(r'^\d{4,5}$')


def known_targets():
    """The field names the registry knows, longest first.

    Longest-first matters: ``sgrb2`` must be tried before ``sgrb`` would be, and
    ``gc2211`` before any shorter prefix.  Falls back to an empty tuple if the
    registry cannot be read -- the monitor still parses names, it just cannot
    attribute them, which is reported rather than guessed.
    """
    try:
        from .. import fields as _fields
        names = set(_fields.BY_NAME)
    except (ImportError, AttributeError, OSError):
        return ()
    return tuple(sorted(names, key=len, reverse=True))


_TARGETS_CACHE = None


def _resolve_head(head, targets=None):
    """``'sgrb25365' -> ('sgrb2', '5365')``; ``'brick' -> ('brick', '')``.

    Returns ``(None, head)`` when nothing registered prefixes ``head``, so an
    unrelated job (``data-qa-mast-download``, ``interactive``) is reported as
    unattributed rather than filed under a fabricated field.  The remainder is
    only accepted as a proposal id if it looks like one.
    """
    global _TARGETS_CACHE
    if targets is None:
        if _TARGETS_CACHE is None:
            _TARGETS_CACHE = known_targets()
        targets = _TARGETS_CACHE
    for name in targets:                      # longest-first: sgrb2 before sgrb
        if head == name:
            return name, ''
        if head.startswith(name):
            rest = head[len(name):]
            if _PROPOSAL_RE.match(rest):
                return name, rest
    return None, head


def parse_job_name(name, targets=None):
    """``'brick2221-o001-m12-fanout' -> {'target': 'brick', 'proposal': '2221',
    'obsid': '001', 'stage': 'm12', 'filter': None, 'name_kind': 'full'}``.

    Returns ``None`` when the name does not resolve to a registered field, so
    the caller buckets the job as unattributed rather than mis-assigning it.
    ``name_kind`` records which shape matched: a ``loose`` or ``pf`` match
    carries no observation id, so it must never be used to claim that a specific
    observation is running.
    """
    text = str(name)
    for kind, rx in (('full', _NAME_FULL), ('noprop', _NAME_NOPROP),
                     ('pf', _NAME_PF), ('loose', _NAME_LOOSE)):
        m = rx.match(text)
        if not m:
            continue
        got = m.groupdict()
        target, proposal = _resolve_head(got['head'], targets)
        if target is None:
            continue                          # try a looser shape
        return {'target': target,
                'proposal': proposal or None,
                'obsid': got.get('obsid'),
                'stage': got.get('stage'),
                'filter': (got.get('filter') or '').upper() or None,
                'name_kind': kind}
    # a bare target with no suffix at all ('brick')
    target, proposal = _resolve_head(text, targets)
    if target is not None:
        return {'target': target, 'proposal': proposal or None, 'obsid': None,
                'stage': None, 'filter': None, 'name_kind': 'bare'}
    return None


# --------------------------------------------------------------------------
# squeue
# --------------------------------------------------------------------------

_SQUEUE_FMT = '%i|%j|%T|%M|%L|%D|%R|%P'
_SQUEUE_KEYS = ('jobid', 'name', 'state', 'elapsed', 'timeleft', 'nodes',
                'reason', 'partition')


def _run(cmd, timeout=30):
    """Run ``cmd``; return stdout, or '' if the tool is missing/slow/errors.

    A monitor must never fail because the scheduler is unreachable -- an absent
    queue is reported as "no jobs", which the page labels as such.
    """
    if not shutil.which(cmd[0]):
        return ''
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return ''
    return proc.stdout if proc.returncode == 0 else ''


def squeue_jobs(user=None):
    """Every queued/running job for ``user`` as a list of dicts.

    Each dict carries the raw squeue columns plus the parsed job-name fields
    (``target``/``proposal``/``obsid``/``stage``/``filter``/``name_kind``);
    unparseable names get ``target=None``.
    """
    user = user or os.environ.get('USER') or ''
    out = _run(['squeue', '-u', user, '-h', '-o', _SQUEUE_FMT])
    jobs = []
    for line in out.splitlines():
        parts = line.split('|')
        if len(parts) < len(_SQUEUE_KEYS):
            continue
        job = dict(zip(_SQUEUE_KEYS, (p.strip() for p in parts)))
        parsed = parse_job_name(job['name']) or {}
        job.update({k: parsed.get(k) for k in
                    ('target', 'proposal', 'obsid', 'stage', 'filter', 'name_kind')})
        jobs.append(job)
    return jobs


def jobs_by_target(jobs):
    """``{target: [job, ...]}`` for jobs whose name identified a target."""
    out = {}
    for job in jobs:
        if job.get('target'):
            out.setdefault(job['target'], []).append(job)
    return out


def sacct_recent(user=None, since='now-2days'):
    """Recently-finished jobs (``sacct``), for the "what just failed" panel.

    Only the top-level job row is kept (``.batch``/``.extern`` steps repeat the
    same name and would double-count a failure).
    """
    user = user or os.environ.get('USER') or ''
    out = _run(['sacct', '-u', user, '-S', since, '-X', '-n', '-P',
                '-o', 'JobID,JobName,State,Elapsed,ExitCode,End'])
    rows = []
    for line in out.splitlines():
        parts = line.split('|')
        if len(parts) < 6:
            continue
        jobid, name, state, elapsed, exitcode, end = (p.strip() for p in parts[:6])
        parsed = parse_job_name(name) or {}
        rows.append({'jobid': jobid, 'name': name, 'state': state.split()[0],
                     'elapsed': elapsed, 'exitcode': exitcode, 'end': end,
                     **{k: parsed.get(k) for k in
                        ('target', 'proposal', 'obsid', 'stage', 'filter', 'name_kind')}})
    return rows


# --------------------------------------------------------------------------
# Log error signatures
# --------------------------------------------------------------------------

#: ``(severity, label, compiled regex)``.  Severity drives the page's colour and
#: the field's overall verdict: ``error`` means the run is broken, ``warn`` means
#: it produced output that needs a human look, ``info`` is progress.
#:
#: Every pattern below is a string the pipeline actually emits -- an exception
#: class it raises, a guard message it prints, or a scheduler/kernel message.
#: Adding a pattern that nothing emits makes the panel look clean when it is not,
#: so keep them tied to a raise/print site.
LOG_SIGNATURES = [
    # --- hard failures -----------------------------------------------------
    ('error', 'traceback', re.compile(r'^Traceback \(most recent call last\)')),
    ('error', 'oom-kill', re.compile(r'oom-kill|Out Of Memory|OUT_OF_MEMORY|'
                                     r'Killed process|slurmstepd:.*Killed')),
    ('error', 'walltime', re.compile(r'DUE TO TIME LIMIT|CANCELLED AT .* DUE TO')),
    ('error', 'dense-nn-median',
     re.compile(r'DenseNNMedianAstrometryError')),
    ('error', 'offsets-table',
     re.compile(r'OffsetsTableUpdateError')),
    ('error', 'astrom-checkpoint',
     re.compile(r'AstrometryCheckpointError|astrometry checkpoint|'
                r'ASTROMETRY CHECKPOINT')),
    ('error', 'wcs-no-convergence', re.compile(r'NoConvergence')),
    # The PSF-grid build is a real, recurring failure: an unsupported filter
    # (F150W2 -> "wavelengths are too long for NIRCam short wave channel") kills
    # the whole filter, and a storage fault reading a cached grid shows up as
    # Errno 5 behind the same "Failed to download PSF" wrapper.
    ('error', 'psf-build', re.compile(r'Failed to download PSF|'
                                      r'wavelengths are too (long|short) for')),
    ('error', 'io-error', re.compile(r'Errno 5|Input/output error')),
    ('error', 'field-registry', re.compile(r'FieldRegistryError')),
    ('error', 'missing-input',
     re.compile(r'FileNotFoundError|No such file or directory')),
    ('error', 'fatal-guard', re.compile(r'^FATAL:')),
    # --- warnings ----------------------------------------------------------
    ('warn', 'cutout-no-overlap', re.compile(r'CutoutNoOverlap|'
                                             r'cutout region does not overlap')),
    ('warn', 'raoffset-disagree',
     re.compile(r'RAOFFSET.*differs|disagreement guard|FORCE_REALIGN_ON_DISAGREE')),
    ('warn', 'safety-override',
     re.compile(r'ASTROM_CHECKPOINT=0|ALLOW_CROSSFILTER_ASTROM_FAIL|'
                r'ALLOW_LATE_STAGE_ASTROM_SHIFT|ALLOW_MISSING_MERGEDCAT_MOSAIC')),
    ('warn', 'window-edge',
     re.compile(r'window_edge_fraction|swept=True|near the window edge')),
    # NOTE: there is deliberately no generic "0 sources" pattern.  Lines like
    # "Satstar summary: 0/0 sources accepted" are the normal, correct output of a
    # frame with no saturated stars (every 5-arcsec probe prints hundreds of
    # them), so matching them turns every healthy run amber and trains the reader
    # to ignore the colour.
    # --- progress ----------------------------------------------------------
    ('info', 'start', re.compile(r'^CATALOG start|^REDUCE start|^CUTOUT PIPELINE:')),
    ('info', 'done', re.compile(r'^CATALOG done|^CUTOUT PIPELINE DONE')),
]

#: Only the tail of a log is scanned by default: these files reach gigabytes and
#: the interesting failure is at the end.  ``head_bytes`` also grabs the opening
#: banner so the monitor can show what the job was told to do.
TAIL_BYTES = 400_000
HEAD_BYTES = 8_000


def scan_log(path, tail_bytes=TAIL_BYTES, head_bytes=HEAD_BYTES):
    """``{'path', 'size', 'mtime', 'hits': {label: count}, 'worst': severity,
    'lines': [(severity, label, text), ...]}`` for one log file.

    ``lines`` keeps at most a handful of examples per label so the page stays
    small; ``hits`` keeps the full counts.
    """
    try:
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    chunks = []
    try:
        with open(path, 'rb') as fh:
            chunks.append(fh.read(min(head_bytes, size)))
            if size > head_bytes + tail_bytes:
                fh.seek(-tail_bytes, os.SEEK_END)
                chunks.append(fh.read())
    except OSError:
        return None
    text = b'\n'.join(chunks).decode('utf-8', 'replace')

    hits = Counter()
    examples = []
    seen = Counter()
    for line in text.splitlines():
        for severity, label, rx in LOG_SIGNATURES:
            if rx.search(line):
                hits[label] += 1
                if seen[label] < 3:
                    seen[label] += 1
                    examples.append((severity, label, line.strip()[:300]))
                break
    order = {'error': 0, 'warn': 1, 'info': 2}
    worst = None
    for severity, label, _ in LOG_SIGNATURES:
        if hits.get(label) and (worst is None or order[severity] < order[worst]):
            worst = severity
    return {'path': path, 'size': size, 'mtime': mtime, 'hits': dict(hits),
            'worst': worst, 'lines': examples}


#: ``catalog_brick2221-o001-cut5-F212N_38511678_4294967294.out`` ->
#: the embedded job name.  The submit scripts use ``--output=..._%x_%A_%a.out``,
#: so the job name sits between the stage prefix and the two numeric ids.
_LOGNAME_RE = re.compile(
    r'^(?P<stage>[a-z0-9]+)_(?P<jobname>.+?)_(?P<jobid>\d+)(?:_(?P<arrayidx>\d+))?'
    r'\.(?:out|log)$')


def log_job_name(filename):
    """The job name embedded in a log filename, or ``None``."""
    m = _LOGNAME_RE.match(os.path.basename(str(filename)))
    return m.group('jobname') if m else None


def log_belongs_to(filename, target, obsid=None):
    """Does this log belong to ``target`` (and ``obsid``, when it names one)?

    A substring match on the field name is not enough: every gc2211 observation
    shares the string ``gc2211``, so an ``o050`` failure would be reported
    against ``o023`` as well.  When the log's job name carries an observation the
    match must agree; when it does not, the log is field-level and shown for
    every observation, which is the truthful reading of a name that omits it.
    """
    name = log_job_name(filename)
    if name is None:
        return target in os.path.basename(str(filename))
    parsed = parse_job_name(name)
    if not parsed or parsed.get('target') != target:
        return False
    if obsid is None or parsed.get('obsid') is None:
        return True
    return str(parsed['obsid']).lstrip('0') == str(obsid).lstrip('0')


def logs_for_target(target, log_dir=None, limit=12, max_age_days=30, obsid=None):
    """The newest log files whose job name resolves to ``target`` (and ``obsid``).

    Logs are named ``<stage>_<jobname>_<jobid>[_<arrayidx>].out`` by the submit
    scripts, so the job name -- and therefore the field and observation -- is
    recoverable from the filename alone; no scheduler needed.
    """
    log_dir = log_dir or LOG_DIR
    if not os.path.isdir(log_dir):
        return []
    cutoff = max_age_days * 86400
    import time
    now = time.time()
    # Filter by NAME first and stat only the survivors: this directory holds
    # ~7,000 logs and a stat per entry on NFS is the slow part.
    try:
        names = [n for n in os.listdir(log_dir)
                 if n.endswith(('.out', '.log')) and target in n
                 and log_belongs_to(n, target, obsid)]
    except OSError:
        return []
    found = []
    for name in names:
        path = os.path.join(log_dir, name)
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            continue
        if now - mtime > cutoff:
            continue
        found.append((mtime, path))
    found.sort(reverse=True)
    return [p for _, p in found[:limit]]
