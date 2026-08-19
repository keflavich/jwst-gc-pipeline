"""Derive and submit the tiny "probe" cutout runs the monitor is exercised on.

A probe is a ~5-arcsec ``--cutout-region`` cataloging run: the full per-exposure
cataloging chain (m12 -> m6) over a handful of cropped frames, which finishes in
minutes instead of a day.  It exists so the monitor can be validated against a
REAL, freshly-produced run for every field -- not only against whatever happens
to be on disk -- and so a field whose inputs are broken says so cheaply.

What a probe can and cannot produce
-----------------------------------
A one-filter cutout runs phases m12, m3, m4, m5, m6.  It stops there: m7 is the
cross-band merge and m8 the cross-band dedup, and both need at least two
filters; the all-filter merge additionally reads ``<basepath>/catalogs/``, which
a cutout never writes (it writes ``<basepath>/cutouts/<label>/catalogs/``).  So
a green probe means "reduction inputs, PSF grids, astrometry checkpoint and the
single-filter cataloging chain all work for this field" -- not that the field's
cross-band products are healthy.

Choosing the parameters
-----------------------
``each_suffix`` comes from ``run_pipeline.resolve`` -- the same value the real
submission uses -- and the probe filter is then chosen as the first preferred
filter whose frames on disk actually CARRY that suffix.  Picking the filter
first and the suffix second is what produced the wd1 F150W failure (a filter
that was never destreaked was catalogued against the destreaked suffix and
matched nothing), so the order here is deliberate.
"""
import glob
import os
import re
import shlex
import subprocess

import numpy as np

from ..mast_names import jw_prefix

#: Preferred probe filters, narrow/medium SW first: they are the fastest to fit
#: and the best-exercised.  ``F150W2``/``F322W2`` are last because only the
#: globular-cluster fields (m4, ngc6397) carry them.
PROBE_FILTER_PREFERENCE = (
    'F212N', 'F182M', 'F187N', 'F210M', 'F200W', 'F150W', 'F115W', 'F162M',
    'F405N', 'F410M', 'F466N', 'F480M', 'F444W', 'F356W', 'F360M', 'F323N',
    'F150W2', 'F322W2')

#: Default probe size in arcsec.  5" is ~160 SW / ~80 LW detector pixels, well
#: above the cataloging floor (``_MIN_CUTOUT_PIX`` = 16) while still cheap.
DEFAULT_PROBE_ARCSEC = 5.0

#: Label every probe shares, so the monitor can find them all with one glob and
#: a rerun overwrites rather than accumulating one directory per attempt.
PROBE_LABEL = 'monitor5as'

_SUFFIX_RE = re.compile(r'_((?:destreak_|align_)?o\d+_crf)\.fits$')


class ProbeError(ValueError):
    """A field cannot be probed (no frames, no resolvable suffix)."""


def _frame_prefix(proposal=None, obsid=None):
    """``'jw06778001'`` -- the exposure-name prefix pinning proposal+observation."""
    if proposal and obsid:
        return f'{jw_prefix(proposal)}{int(obsid):03d}'
    return 'jw'


def suffix_census(base, filt, proposal=None, obsid=None):
    """``{suffix: count}`` over the NIRCam CRF frames of one filter directory.

    Pinned to one proposal+observation when given, for the same reason
    ``choose_center`` is: one directory can hold several proposals' frames.
    """
    counts = {}
    pattern = os.path.join(base, filt, 'pipeline',
                           f'{_frame_prefix(proposal, obsid)}*nrc*_crf.fits')
    for path in glob.glob(pattern):
        m = _SUFFIX_RE.search(os.path.basename(path))
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def choose_filter(base, each_suffix, preference=PROBE_FILTER_PREFERENCE,
                  proposal=None, obsid=None):
    """First preferred filter whose frames carry ``each_suffix``.

    Returns ``(filter, n_frames)``.  Raises ``ProbeError`` when NO filter carries
    the resolved suffix -- which is a real finding about the field (its reduction
    never produced the product cataloging is configured to read), not a reason to
    silently substitute a different suffix.
    """
    for filt in preference:
        counts = suffix_census(base, filt, proposal, obsid)
        if each_suffix in counts:
            return filt, counts[each_suffix]
    seen = {f: suffix_census(base, f, proposal, obsid) for f in preference
            if os.path.isdir(os.path.join(base, f))}
    raise ProbeError(
        f'no filter under {base} carries the resolved each-suffix '
        f'{each_suffix!r} for {proposal}/o{obsid}; '
        f'found {  {k: sorted(v) for k, v in seen.items() if v} }')


def choose_center(base, filt, each_suffix, proposal=None, obsid=None, n_sample=16):
    """``(ra_deg, dec_deg, source_file, n_overlapping)`` for the probe centre.

    The centre is picked from the EXPOSURE FRAMES cataloging will actually read,
    not from the drizzled mosaic.  A mosaic's covered area is the union of every
    visit, module and dither, so its interior contains points that no frame of
    the filter/observation being probed covers -- pointing a cutout there makes
    the run die with "``--cutout-region`` overlapped none of the frames" while
    the field is perfectly healthy.  (Measured: the sgra F212N mosaic's median
    covered pixel sits ~100" from its own ``destreak_o001_crf`` frames.)

    So: sample frames, take each one's array centre as a candidate, and keep the
    candidate covered by the MOST frames.  A candidate is a real pixel of at
    least its own frame, so the result can never overlap nothing, and maximising
    coverage puts the probe in the dither stack where the run is most
    representative.

    The GWCS is read via ``frame_wcs`` -- a detector-frame SCI header's SIP WCS
    is only an approximation of it, and on a field whose alignment is being
    corrected the two can differ by far more than a 5-arcsec box.
    """
    from ..frame_wcs import frame_wcs

    pipe = os.path.join(base, filt, 'pipeline')
    # Pin the glob to this proposal+observation.  One filter directory can hold
    # frames from SEVERAL proposals under the same suffix -- ngc6334's F182M
    # carries both 6778 and 7213 ``align_o001_crf`` frames, pointing at different
    # sky -- so an unpinned glob picks a frame the run will never read, and the
    # cutout lands where none of its own frames are.
    prefix = f'{jw_prefix(proposal)}{int(obsid):03d}' if proposal and obsid else 'jw'
    frames = sorted(glob.glob(os.path.join(pipe, f'{prefix}*_{each_suffix}.fits')))
    if not frames:
        raise ProbeError(
            f'no {prefix}*_{each_suffix}.fits frames under {pipe}')
    step = max(1, len(frames) // n_sample)
    sample = frames[::step][:n_sample]

    wcses, kept = [], []
    for path in sample:
        try:
            wcses.append(frame_wcs(path))
            kept.append(path)
        except (OSError, ValueError, KeyError):
            continue
    if not wcses:
        raise ProbeError(f'no readable GWCS among {len(sample)} {filt} frames')

    candidates = []
    for path, ww in zip(kept, wcses):
        shape = getattr(ww, 'pixel_shape', None) or (2048, 2048)
        try:
            sky = ww.pixel_to_world(shape[0] / 2.0, shape[1] / 2.0)
        except (ValueError, RuntimeError):
            continue
        candidates.append((path, sky))
    if not candidates:
        raise ProbeError(f'no frame centre could be projected for {filt}')

    best = None
    for path, sky in candidates:
        n_cover = 0
        for ww in wcses:
            shape = getattr(ww, 'pixel_shape', None) or (2048, 2048)
            try:
                x, y = ww.world_to_pixel(sky)
            except (ValueError, RuntimeError):
                continue
            if 0 <= float(x) < shape[0] and 0 <= float(y) < shape[1]:
                n_cover += 1
        if best is None or n_cover > best[3]:
            best = (float(sky.ra.deg), float(sky.dec.deg),
                    os.path.basename(path), n_cover)
    return best


def plan_probe(target, instrument='nircam', size_arcsec=DEFAULT_PROBE_ARCSEC,
               label=PROBE_LABEL, obsid=None):
    """The full probe recipe for one field, or a dict carrying ``error``.

    Never raises for an unprobeable field: an error is part of the monitor's
    output (``omegacen`` has no delivered data at all), so it is returned as a
    row rather than aborting the whole matrix.

    ``obsid`` names the observation to probe.  It is what a field claiming
    every observation of its proposal (gc-treasury/10678) needs, since the
    registry hands back only the wildcard for it and the wildcard names no
    cutout.  Omitted, the registered observations are tried in order, which is
    what every enumerated field wants.
    """
    from .. import fields as _fields
    from ..run_pipeline import resolve
    from . import scan

    try:
        obs = scan.observations(target, instrument)
    except scan.ScanError as ex:
        return {'target': target, 'error': str(ex)}
    if not obs:
        return {'target': target, 'error': f'no {instrument} observations registered'}

    base = scan.basepath(target)
    errors = []
    wanted = None if obsid is None else str(obsid)
    for proposal, registered in obs:
        if registered == _fields.WILDCARD_OBSID and wanted is None:
            # A wildcard registration names no observation number, and
            # `resolve` zero-pads what it is given -- '*' becomes '00*',
            # which no registry lookup answers.  The message that came back
            # from there told the operator to register `nircam: ['00*']`,
            # which is not a thing fields.yaml accepts.
            errors.append(
                f"{proposal}: {target} claims every observation of the "
                f"proposal (fields.yaml obsids: '*'), so there is no "
                f"observation number to build a probe cutout from; pass "
                f"obsid=<NNN> to name one, e.g. plan_probe({target!r}, "
                f"obsid='042').")
            continue
        obsid_here = registered
        if wanted is not None:
            if registered == _fields.WILDCARD_OBSID:
                obsid_here = wanted
            elif str(registered) != wanted:
                errors.append(f'{proposal}/o{registered}: skipped, the caller '
                              f'asked for o{wanted}')
                continue
        try:
            plan = resolve(proposal, obsid_here, instrument)
            each_suffix = plan['each_suffix']
            filt, nframes = choose_filter(base, each_suffix,
                                          proposal=proposal, obsid=obsid_here)
            ra, dec, frame, n_cover = choose_center(base, filt, each_suffix,
                                                    proposal, obsid_here)
        except (ProbeError, KeyError, ValueError) as ex:
            errors.append(f'{proposal}/o{obsid_here}: {ex}')
            continue
        return {
            'target': target, 'proposal': str(proposal), 'obsid': str(obsid_here),
            'instrument': instrument, 'basepath': base, 'filter': filt,
            'each_suffix': each_suffix, 'n_frames': nframes,
            'ra': ra, 'dec': dec, 'center_from': frame, 'n_overlapping': n_cover,
            'size_arcsec': float(size_arcsec), 'label': label,
            'cutout_region': f'{ra:.6f},{dec:.6f},{size_arcsec:g}',
            'job_name': f'{target}{proposal}-o{obsid_here}-cut{size_arcsec:g}-{filt}',
        }
    return {'target': target, 'error': '; '.join(errors) or 'no probeable observation'}


def plan_all(targets=None, **kwargs):
    """``[plan_probe(t), ...]`` for every registered field (or ``targets``)."""
    from . import scan
    return [plan_probe(t, **kwargs) for t in (targets or scan.all_targets())]


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------

#: The cataloging submitter already handles environment, CRDS, TMPDIR, the
#: run-guard and the in-queue rename; the probe only adds --cutout-region via its
#: EXTRA_ARGS passthrough.  Passing the region through --export would break: the
#: region contains commas, which sbatch treats as variable separators, so the
#: environment is exported from this process instead (--export=ALL).
SUBMIT_SCRIPT = 'scripts/reduction/submit_cataloging.sbatch'

#: A probe is small: 8 cores and 32 GB finish a 5-arcsec cutout in minutes, and
#: asking for the production 32/128 would queue behind real work for no gain.
PROBE_RESOURCES = {'cpus': 8, 'memory': '32gb', 'walltime': '2:00:00'}


def submit_command(plan, repo_root, pipe_root=None, resources=None,
                   extra_args=(), qos='astronomy-dept-b', account='astronomy-dept'):
    """``(env, argv)`` for one probe submission.

    ``pipe_root`` pins the worktree whose code should run (prepended to
    PYTHONPATH by the submit script), so a probe validates the branch under
    test rather than whatever is pip-installed.
    """
    res = dict(PROBE_RESOURCES, **(resources or {}))
    env = {
        'PROPOSAL': plan['proposal'], 'FIELD': plan['obsid'],
        'TARGET': plan['target'], 'MODULES': 'merged',
        'EACH_SUFFIX': plan['each_suffix'], 'FILTERS': plan['filter'],
        'EXTRA_ARGS': ' '.join([
            f"--cutout-region={plan['cutout_region']}",
            f"--cutout-label={plan['label']}", *extra_args]),
    }
    if pipe_root:
        env['PIPE_ROOT'] = pipe_root
    argv = ['sbatch', '--export=ALL',
            f"--job-name={plan['job_name']}",
            f"--cpus-per-task={res['cpus']}", f"--mem={res['memory']}",
            f"--time={res['walltime']}", f'--qos={qos}', f'--account={account}',
            os.path.join(repo_root, SUBMIT_SCRIPT)]
    return env, argv


def submit(plan, repo_root, execute=False, **kwargs):
    """Submit one probe.  Dry-run unless ``execute`` -- returns the job id."""
    env, argv = submit_command(plan, repo_root, **kwargs)
    shown = ' '.join(f'{k}={shlex.quote(v)}' for k, v in sorted(env.items()))
    line = f'{shown} {" ".join(shlex.quote(a) for a in argv)}'
    if not execute:
        return {'submitted': False, 'command': line}
    proc = subprocess.run(argv, capture_output=True, text=True,
                          env={**os.environ, **env})
    if proc.returncode != 0:
        return {'submitted': False, 'command': line,
                'error': (proc.stderr or proc.stdout).strip()}
    jobid = ''.join(ch for ch in proc.stdout if ch.isdigit())
    return {'submitted': True, 'command': line, 'jobid': jobid}
