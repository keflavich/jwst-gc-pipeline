"""Unit tests for the operational scripts in scripts/reduction/ (not part of
the package; imported by path)."""
import importlib.util
import json
import os
import subprocess
import time

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reduction')


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPTS, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rename_stale_band_token():
    m = _load('rename_stale_mosaics')
    assert m.band_of('jw02221-o001_t001_nircam_clear-f182m-merged-reproject-vvv_i2d.fits') == 'f182m'
    assert m.band_of('jw02221-o002_t001_miri_f2550w_realigned-to-vvv.fits') == 'f2550w'
    assert m.band_of('jw02221-o001_t001_nircam_clear-F405N-merged_realigned-to-vvv.fits') == 'f405n'
    assert m.band_of('no_band_here.fits') is None


def test_rename_stale_staleness_logic(tmp_path):
    """A pre-campaign realigned mosaic is renamed; a same-campaign one is kept."""
    m = _load('rename_stale_mosaics')
    m.BASE = str(tmp_path)
    pipe = tmp_path / 'myfield' / 'F182M' / 'pipeline'
    pipe.mkdir(parents=True)
    ref = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged_data_i2d.fits'
    stale = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged-reproject-vvv_i2d.fits'
    fresh = pipe / 'jw1-o001_t001_nircam_clear-f182m-merged_realigned-to-refcat.fits'
    now = time.time()
    for p, age_days in ((ref, 0), (stale, 400), (fresh, 0.5)):
        p.write_bytes(b'x')
        os.utime(p, (now - age_days * 86400,) * 2)
    plan = m.rename_stale_for_field('myfield', execute=True)
    assert len(plan) == 1
    assert not stale.exists()
    assert (str(stale) + m.SUFFIX) == str(stale) + '_badastrometry_stale'
    assert os.path.exists(str(stale) + m.SUFFIX)
    assert fresh.exists()


def test_purge_satstar_caches(tmp_path):
    m = _load('purge_satstar_caches')
    pipe = tmp_path / 'brick' / 'F182M' / 'pipeline'
    cats = tmp_path / 'brick' / 'catalogs'
    pipe.mkdir(parents=True)
    cats.mkdir(parents=True)
    a = pipe / 'exp1_m12_satstar_catalog.fits'
    b = cats / 'f182m_consolidated_satstar_catalog.fits'
    other = pipe / 'exp1_m12_daophot_basic.fits'
    for p in (a, b, other):
        p.write_bytes(b'x')
    # dry run: nothing moves
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=False)
    assert n == 2 and a.exists() and b.exists()
    # execute: both cache levels sidelined, unrelated file untouched
    n = m.purge(str(tmp_path), 'brick', ['F182M'], execute=True)
    assert n == 2
    assert not a.exists() and not b.exists()
    assert os.path.exists(str(a) + m.SUFFIX) and os.path.exists(str(b) + m.SUFFIX)
    assert other.exists()
    # idempotent: second execute finds nothing
    assert m.purge(str(tmp_path), 'brick', ['F182M'], execute=True) == 0


# --- apply_m2_checkpoint_corrections: per-exposure extension ---------------

def _write_m2_record(records_dir, filt, visit_exposures):
    """visit_exposures: {visit_int: [exposure ints]} -> a minimal m2 record with
    the full exposure enumeration (one detector), no corrections needed."""
    visits = []
    for vnum, exps in visit_exposures.items():
        visits.append(dict(visit=str(vnum), filtername=filt, exposures=[
            dict(key=[str(vnum), e, 'nrca1', filt]) for e in exps]))
    rec = dict(stage='m2', filtername=filt, visits=visits, corrections=[])
    with open(os.path.join(records_dir, f'checkpoint_m2_{filt}_latest.json'), 'w') as fh:
        json.dump(rec, fh)


def test_exposure_universe_keyed_by_visit_and_filter(tmp_path):
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    # 1182-like: two visits sharing exposure numbers 1..12; plus a 2221-like
    # single-visit filter with 1..3
    _write_m2_record(str(rd), 'F200W', {1: list(range(1, 13)), 2: list(range(1, 13))})
    _write_m2_record(str(rd), 'F182M', {1: [1, 2, 3]})
    u = m.load_exposure_universe(str(rd))
    assert u[(1, 'F200W')] == list(range(1, 13))
    assert u[(2, 'F200W')] == list(range(1, 13))
    assert u[(1, 'F182M')] == [1, 2, 3]
    assert (2, 'F182M') not in u


def test_extend_covers_subfloor_exposure_no_phantom_rows(tmp_path):
    """The reviewer's gap: an exposure sub-floor in EVERY filter carries no
    correction but is a real frame -- it must still get a row (from the record
    universe), and a visit must never receive another visit's exposure numbers."""
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    # F200W tiles two visits, exposures 1..3 each; NONE carry a correction here
    _write_m2_record(str(rd), 'F200W', {1: [1, 2, 3], 2: [1, 2, 3]})
    universe = m.load_exposure_universe(str(rd))

    # pristine per-visit table (no Exposure column): one row per (visit, filter)
    tbl = Table(dict(
        Visit=['jw01182004001', 'jw01182004002'],
        Filter=['F200W', 'F200W'],
        **{'dra (arcsec)': [0.0, 0.0], 'ddec (arcsec)': [0.0, 0.0]}))
    tp = tmp_path / 'offsets.csv'
    tbl.write(str(tp), overwrite=True)

    out, extended = m.extend_table_to_per_exposure(
        str(tp), universe, extend_filters={'F200W'})
    assert extended
    assert 'Exposure' in out.colnames
    # visit 1 gets exposures 1,2,3; visit 2 gets exposures 1,2,3 -- 6 rows total
    v1 = sorted(int(r['Exposure']) for r in out if m._table_visit_number(r['Visit']) == 1)
    v2 = sorted(int(r['Exposure']) for r in out if m._table_visit_number(r['Visit']) == 2)
    assert v1 == [1, 2, 3]      # incl. exp 3, which carried NO correction
    assert v2 == [1, 2, 3]      # no phantom exposure numbers from the other visit
    assert len(out) == 6


def test_extend_leaves_unextended_filter_as_single_visit_row(tmp_path):
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    rd = tmp_path / 'astrometry_checkpoints'
    rd.mkdir()
    _write_m2_record(str(rd), 'F410M', {1: [1, 2, 3, 4]})
    universe = m.load_exposure_universe(str(rd))
    tbl = Table(dict(Visit=['jw02221001001'], Filter=['F410M'],
                     **{'dra (arcsec)': [0.0], 'ddec (arcsec)': [0.0]}))
    tp = tmp_path / 'o.csv'
    tbl.write(str(tp), overwrite=True)
    # filter NOT in extend_filters -> keeps one per-visit row, Exposure = -1
    out, extended = m.extend_table_to_per_exposure(
        str(tp), universe, extend_filters=set())
    assert extended is False


def test_extend_idempotent_when_exposure_column_present(tmp_path):
    from astropy.table import Table
    m = _load('apply_m2_checkpoint_corrections')
    tbl = Table(dict(Visit=['jw02221001001'], Filter=['F410M'], Exposure=[1],
                     **{'dra (arcsec)': [0.0], 'ddec (arcsec)': [0.0]}))
    tp = tmp_path / 'o.csv'
    tbl.write(str(tp), overwrite=True)
    out, extended = m.extend_table_to_per_exposure(str(tp), {}, {'F410M'})
    assert extended is False
    assert len(out) == 1


PERFRAME_SBATCH = os.path.join(
    SCRIPTS, 'submit_cataloging_perframe_phase.sbatch')


def _perframe_sbatch_text():
    """The script with line continuations folded away.

    A rename split across a backslash-continuation would otherwise be collected
    by nothing and pass the gate test vacuously.
    """
    with open(PERFRAME_SBATCH) as fh:
        return fh.read().replace('\\\n', ' ')


def test_perframe_runtime_rename_never_clobbers_a_submitted_name():
    """A submit-time job name must survive the per-frame phase script.

    The standing naming rule wants target+program+obsid+stage on the job at
    SUBMIT time, because a queued job shows only that.  The phase script used to
    rename itself unconditionally to "<target>-pf-<phase>-<mode>", which drops
    the program and the obsid -- brick and cloudc are both 2221, and gc2211 has
    five observations, so the degraded name is genuinely ambiguous.  Every
    `scontrol update ... JobName` here must therefore be gated on the rename
    guard, which only fires for a bare submission.
    """
    text = _perframe_sbatch_text()
    renames = [ln.strip() for ln in text.splitlines()
               if not ln.lstrip().startswith('#')
               and 'scontrol update' in ln and 'JobId=' in ln]
    assert renames, 'expected the phase script to still contain renames'
    for line in renames:
        assert line.startswith('_pf_rename_wanted &&'), (
            f'ungated runtime rename would clobber the submit-time name: {line}')


def test_perframe_shard_name_does_not_carry_the_array_index():
    """The shard index must not be baked into the job NAME.

    `scontrol update JobId=<task>` does not reliably address one element of an
    array: sgrb2 m12 fanout 38867646 came out `pf_sgrb2_m12_s15` on all 16
    tasks, and sgrc 38851171 had tasks 13 and 15 both reading
    `pf_sgrc_m12_s15` (2026-08-07).  The shard is already unambiguous in the
    array-task id, so the name must not try to carry it.
    """
    text = _perframe_sbatch_text()
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            continue
        if 'scontrol update' in line or 'JobName=' in line:
            assert 'SLURM_ARRAY_TASK_ID' not in line, (
                f'array index must not go into the job name: {line.strip()}')


RETIE_LOOP = os.path.join(SCRIPTS, 'run_field_retie_loop.sh')


def _run_reduce_gate(tmp_path, states, ntasks, jobid='9999', keep_errexit=False):
    """Drive reduce_fully_succeeded() for real, with a stub `sacct` on PATH.

    Returns (returncode, output).  Sourcing the loop needs its four required
    vars; RETIE_LOOP_SOURCE_ONLY makes it stop before the iteration loop.
    """
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'sacct').write_text('#!/bin/bash\nprintf "%s\\n" $SACCT_STATES\n')
    (stub / 'sacct').chmod(0o755)
    relax = '' if keep_errexit else 'set +e +u +o pipefail'
    script = f"""
        export PATH="{stub}:$PATH"
        export PROPOSAL=4147 FIELD=012 TARGET=sgrc FILTERS="a b"
        export RETIE_LOOP_SOURCE_ONLY=1
        source "{RETIE_LOOP}" >/dev/null 2>&1
        {relax}
        export SACCT_STATES="{states}"
        reduce_fully_succeeded "{jobid}" {ntasks}
    """
    proc = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_reduce_gate_stops_on_the_sgrc_partial_failure(tmp_path):
    """The case this PR exists for: 4 COMPLETED + 4 FAILED must not be cataloged.

    A filter whose reduce failed keeps the PREVIOUS iteration's WCS, so the m12
    merge compares this iteration's frames for some bands against last
    iteration's for others, and the m2 checkpoint writes that mixture into the
    consensus table as a correction.  sgrc iteration 3 (38870453, 2026-08-07).
    """
    rc, out = _run_reduce_gate(
        tmp_path, 'COMPLETED COMPLETED COMPLETED COMPLETED FAILED FAILED FAILED FAILED', 8)
    assert rc == 1, f'a partially-failed reduce must stop the loop:\n{out}'
    assert 'STOPPING before cataloging' in out
    assert '4/8 completed' in out


def test_reduce_gate_proceeds_when_every_task_completed(tmp_path):
    """And the happy path must NOT stop -- including under `set -e`.

    `grep -c` exits 1 when the count is 0, so an unguarded count of the
    non-COMPLETED tasks would kill the loop on exactly the all-succeeded case.
    """
    rc, out = _run_reduce_gate(tmp_path, 'COMPLETED ' * 8, 8)
    assert rc == 0, f'a fully successful reduce must proceed:\n{out}'
    assert 'STOPPING' not in out
    assert '8/8 completed, 0 not' in out


def test_reduce_gate_stops_when_nothing_completed(tmp_path):
    """Zero COMPLETED is the other `grep -c` zero-count case."""
    rc, out = _run_reduce_gate(tmp_path, 'FAILED ' * 8, 8)
    assert rc == 1, f'a wholly failed reduce must stop the loop:\n{out}'
    assert '0/8 completed' in out


def test_reduce_gate_stops_when_a_requeued_task_double_counts(tmp_path):
    """n_done > ntasks fails in the safe direction: stop, never catalog."""
    rc, out = _run_reduce_gate(tmp_path, 'COMPLETED ' * 9, 8)
    assert rc == 1, f'an unexpected task count must stop the loop:\n{out}'


def test_reduce_gate_stops_when_no_job_id_was_parsed(tmp_path):
    """An unparseable sbatch must stop, not silently catalog."""
    rc, out = _run_reduce_gate(tmp_path, '', 8, jobid='')
    assert rc == 1, f'a missing job id must stop the loop:\n{out}'
    assert 'could not parse a job id' in out


def test_reduce_gate_survives_errexit_on_the_happy_path(tmp_path):
    """`grep -c` exits 1 on a zero count, and the loop runs under `set -euo`.

    Called OUTSIDE an if-condition (where bash would suspend errexit), an
    unguarded count of the non-COMPLETED tasks aborts the script on exactly the
    all-succeeded case -- so a fully successful reduce would kill the loop
    silently, which is worse than the bug the guard fixes.  Verified against the
    unguarded form: rc=1 with no output at all.
    """
    rc, out = _run_reduce_gate(tmp_path, 'COMPLETED ' * 8, 8, keep_errexit=True)
    assert rc == 0, f'errexit must not abort the all-completed case:\n{out}'
    assert '8/8 completed, 0 not' in out


def test_the_loop_actually_calls_the_gate_between_reduce_and_catalog():
    """The gate has to be WIRED, not just correct.

    Every other test here sources the script and calls
    `reduce_fully_succeeded` directly, so deleting the call site leaves the
    suite green while restoring #327 in full: catalog a mixed reduce, m12
    merges this iteration's frames for the filters that succeeded with last
    iteration's for the ones that did not, and m2 writes the mixture into the
    consensus table as a correction.
    """
    with open(RETIE_LOOP) as fh:
        text = fh.read()
    between = text[text.index('--- 1. reduce'):text.index('--- 2. catalog')]
    assert 'reduce_fully_succeeded' in between, (
        'the loop must call the gate between reduce and catalog')
    assert 'exit 1' in between, (
        'the loop must stop, not continue, when the reduce is incomplete')


# ---------------------------------------------------------------------------
# Runtime job renames, generalised over EVERY submitter that does one (#330).
#
# submit_cataloging_m7.sbatch had the same defect #326 fixed one script over,
# twice and neither guarded, so the second write won over the submit-time name.
# Parameterising rather than duplicating means the next script to grow a rename
# is covered without anyone remembering to add a test.
# ---------------------------------------------------------------------------

#: Every submitter that renames itself at runtime, with the placeholder its
#: `#SBATCH --job-name` carries.  The guard idiom is deliberately NOT pinned:
#: the repo uses two spellings (a `_*_rename_wanted` helper and a bare
#: `if [ "${SLURM_JOB_NAME:-x}" = "x" ]`), and an earlier version of this test
#: enforced one of them -- which excluded the correctly-guarded scripts written
#: in the other, and let a one-character `=` -> `!=` inversion through.  These
#: tests EXECUTE the guard instead.
RENAMING_SBATCH = {
    'submit_cataloging_perframe_phase.sbatch': 'pf',
    'submit_cataloging_m7.sbatch': 'catalog_m7',
    'submit_cataloging.sbatch': 'catalog',
}


def _folded(basename):
    """The script with line continuations folded away, so a rename split across
    a backslash is still one line to match against."""
    with open(os.path.join(SCRIPTS, basename)) as fh:
        return fh.read().replace('\\\n', ' ')


def _rename_lines(text):
    return [ln.strip() for ln in text.splitlines()
            if not ln.lstrip().startswith('#')
            and 'scontrol update' in ln and 'JobId=' in ln]


def _rename_attempts(script, placeholder, job_name):
    """Run the script's rename logic with a stub `scontrol` and report what it
    tried to set the name to.

    Behavioural, not textual: the previous version string-matched the guard's
    definition line, so inverting it (`=` -> `!=`) -- which reproduces the exact
    #330 defect -- left every test green.  It also only recognised the literal
    `JobId=`, so re-adding a rename with slurm's equally-valid lowercase
    `jobid=` was invisible.
    """
    import subprocess
    import tempfile
    text = _folded(script)
    lines = text.splitlines()
    keep, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith('#'):
            i += 1
            continue
        low = line.lower()
        st = line.lstrip()
        if st.startswith('echo ') or st.startswith('printf '):
            # a message that merely MENTIONS scontrol (perframe_phase:209
            # suggests re-pointing a dependency) is not a rename
            i += 1
            continue
        if 'rename_wanted()' in line:            # guard helper definition
            keep.append(line)
        elif line.lstrip().startswith('if ') and 'SLURM_JOB_NAME' in line:
            # a guard written as an if-block: keep the WHOLE block, or the
            # rename inside it runs unconditionally here and the test reports a
            # clobber that does not happen.
            block, depth = [line], 1
            j = i + 1
            while j < len(lines) and depth:
                block.append(lines[j])
                st = lines[j].strip()
                if st.startswith('if '):
                    depth += 1
                elif st == 'fi':
                    depth -= 1
                j += 1
            if any('scontrol update' in b.lower() for b in block):
                keep.extend(block)
            i = j
            continue
        elif 'scontrol update' in low and 'jobid=' in low:
            keep.append(line)
        i += 1
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'scontrol'), 'w') as fh:
            fh.write('#!/bin/bash\necho "$@"\n')
        os.chmod(os.path.join(td, 'scontrol'), 0o755)
        env = dict(os.environ)
        env.update(PATH=td + os.pathsep + env['PATH'],
                   SLURM_JOB_ID='12345', SLURM_JOB_NAME=job_name,
                   TARGET='brick', PROPOSAL='2221', FIELD='001',
                   FILT='F182M', PHASE='m12', MODE='fanout')
        out = subprocess.run(['bash', '-c', '\n'.join(keep)], env=env,
                             capture_output=True, text=True, timeout=60)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


@pytest.mark.parametrize('script,placeholder', sorted(RENAMING_SBATCH.items()))
def test_a_submit_time_name_SURVIVES(script, placeholder):
    """The whole point of #330/#326: a name given at submit time must not be
    overwritten at runtime."""
    attempts = _rename_attempts(script, placeholder, 'brick2221-o001-cat')
    assert not attempts, (
        f'{script}: renamed itself over the submit-time name -> {attempts}')


@pytest.mark.parametrize('script,placeholder', sorted(RENAMING_SBATCH.items()))
def test_a_BARE_submission_still_gets_renamed(script, placeholder):
    """And the guard must actually fire when it should -- an inverted guard
    passes the test above vacuously."""
    attempts = _rename_attempts(script, placeholder, placeholder)
    assert attempts, (
        f'{script}: a bare submission (SLURM_JOB_NAME={placeholder!r}) was '
        f'never renamed; the guard cannot fire')


@pytest.mark.parametrize('script,placeholder', sorted(RENAMING_SBATCH.items()))
def test_the_bare_path_name_is_readable_by_the_monitor(script, placeholder):
    """`cat_brick_m7` returned None from parse_job_name -- invisible to the
    monitor, not merely ambiguous.  Whatever the bare path picks must resolve to
    a registered field AND name a stage.

    The obsid is NOT required here: #326 settled that the bare fallback keeps
    the `pf_<target>_<phase>` shape because that is what `_NAME_PF` reads, and
    it carries no obsid by construction.  The obsid requirement belongs on the
    SUBMIT-time names, which is where
    test_every_emitted_job_name_resolves_to_a_field_and_an_observation puts it.
    """
    from jwst_gc_pipeline.monitoring.jobs import parse_job_name
    checked = 0
    for line in _rename_attempts(script, placeholder, placeholder):
        for tok in line.split():
            low = tok.lower()
            if low.startswith('jobname=') or low.startswith('name='):
                name = tok.split('=', 1)[1]
                parsed = parse_job_name(name)
                assert parsed is not None, (
                    f'{script}: bare-path name {name!r} is unattributable -- '
                    f'the monitor cannot file it under any field')
                assert parsed['stage'], (
                    f'{script}: bare-path name {name!r} names no stage')
                checked += 1
    assert checked, f'{script}: no bare-path name was emitted to check'


def test_m7_renames_itself_only_once():
    """It used to do it twice, unconditionally, and the second write won."""
    assert len(_rename_lines(_folded('submit_cataloging_m7.sbatch'))) == 1


# ---------------------------------------------------------------------------
# The names themselves have to be readable by the monitor.  These go through
# parse_job_name rather than asserting a string, so a shape that looks fine but
# does not resolve to a field cannot pass.
# ---------------------------------------------------------------------------

def _emitted_job_names(text, env):
    """Every `--job-name=` / `JobName=` value in a script, with env expanded."""
    import re as _re
    out = []
    for m in _re.finditer(r'(?:--job-name=|JobName=)"([^"]+)"', text):
        name = m.group(1)
        for k, v in env.items():
            name = name.replace('${' + k + '}', v).replace('$' + k, v)
        out.append(name)
    return out


CHAIN_ENV = {'TARGET': 'brick', 'PROPOSAL': '2221', 'FIELD': '001',
             'ph': 'm12'}


@pytest.mark.parametrize('script', [
    'submit_cataloging_chain.sh',
    'submit_cataloging_m7.sbatch',
    'submit_cataloging_perframe.sh',
])
def test_every_emitted_job_name_resolves_to_a_field_and_an_observation(script):
    """`cat_brick_m7` -- the name m7 used to give itself -- returns None from
    parse_job_name: _NAME_PF only accepts a `pf_` head, so it matches no shape
    and _resolve_head finds no registered field inside it.  It was invisible to
    the monitor, not merely ambiguous.  A name that carries the obsid parses as
    `full` and is the only kind that can say WHICH observation is running --
    brick and cloudc are both 2221 and gc2211 has five observations.
    """
    from jwst_gc_pipeline.monitoring.jobs import parse_job_name
    names = _emitted_job_names(_folded(script), CHAIN_ENV)
    assert names, f'{script}: no job names found'
    for name in names:
        parsed = parse_job_name(name)
        assert parsed is not None, (
            f'{script}: job name {name!r} is unattributable -- the monitor '
            f'cannot file it under any field')
        assert parsed['obsid'], (
            f'{script}: job name {name!r} parsed as {parsed["name_kind"]} with '
            f'no obsid; it cannot identify which observation is running')


def test_the_underscore_form_this_replaced_really_was_unreadable():
    """Pins the measurement the fix rests on, so nobody reintroduces it."""
    from jwst_gc_pipeline.monitoring.jobs import parse_job_name
    assert parse_job_name('cat_brick_m7') is None
    assert parse_job_name('cat_gc2211_m7') is None
    # and the dashed form it alternated with parsed, but only loosely
    loose = parse_job_name('brick-catalog-m7')
    assert loose['name_kind'] == 'loose' and loose['obsid'] is None
