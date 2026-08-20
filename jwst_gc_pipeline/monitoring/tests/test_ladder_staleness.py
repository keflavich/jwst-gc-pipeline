"""A stage built before an earlier stage is describing products that are gone.

The ladder showed PRESENCE only -- done / part / ambiguous -- so a stage from a
superseded generation rendered identically to one built this morning.  Two live
cases on 2026-08-19:

    sgrb2    m12 2026-08-18 · m3 m4 m5 m6 all 2026-07-04   (six weeks, several
             re-ties and an F210M frame fix in between)
    cloudef  m3 2026-08-19 · m7 cross-band merge 2026-07-01

Within one chain the stages run in order, so a later stage is always newer.  A
later stage that is OLDER was built from inputs that have since been
regenerated, and reading it as "done" is how a field looks finished while its
top half describes files that no longer exist.
"""
from jwst_gc_pipeline.monitoring import render


def _run(per_filter, crossband=None):
    return {'per_filter': per_filter, 'crossband': crossband or {}}


def _step(n=8, mtime=None, scope='ok'):
    return {'n': n, 'mtime': mtime, 'scope': scope}


T0 = 1_750_000_000.0
HOUR = 3600.0


def test_stages_in_order_are_not_stale():
    run = _run({'F212N': {'m12': _step(mtime=T0), 'm3': _step(mtime=T0 + HOUR),
                          'm4': _step(mtime=T0 + 2 * HOUR)}})
    st = render._ladder_state(run)
    assert st['m12'] == 'done' and st['m3'] == 'done' and st['m4'] == 'done'


def test_a_later_stage_older_than_an_earlier_one_is_stale():
    """The sgrb2 shape: m12 re-run this week, m3..m6 left from six weeks ago."""
    run = _run({'F212N': {'m12': _step(mtime=T0 + 100 * HOUR),
                          'm3': _step(mtime=T0), 'm4': _step(mtime=T0),
                          'm5': _step(mtime=T0), 'm6': _step(mtime=T0)}})
    st = render._ladder_state(run)
    assert st['m12'] == 'done'
    for step in ('m3', 'm4', 'm5', 'm6'):
        assert st[step] == 'stale', (step, st[step])


def test_a_stale_crossband_merge_is_flagged():
    """The cloudef shape: an m7 cross-band product older than the m3 under it."""
    run = _run({'F162M': {'m3': _step(mtime=T0 + 100 * HOUR)}},
               crossband={'m7': _step(mtime=T0)})
    assert render._ladder_state(run)['m7'] == 'stale'


def test_stale_outranks_part():
    """A stage that is both incomplete AND older than its inputs is reported as
    the more actionable of the two: re-running it fixes both."""
    run = _run({'A': {'m12': _step(mtime=T0 + 100 * HOUR), 'm4': _step(mtime=T0)},
                'B': {'m12': _step(mtime=T0 + 100 * HOUR), 'm4': _step(n=0)}})
    assert render._ladder_state(run)['m4'] == 'stale'


def test_ambiguous_still_outranks_stale():
    """`ambig` means the product cannot be attributed to this observation at
    all, which has to be resolved before its age means anything."""
    run = _run({'A': {'m12': _step(mtime=T0 + 100 * HOUR),
                      'm4': _step(mtime=T0, scope='ambiguous')}})
    assert render._ladder_state(run)['m4'] == 'ambig'


def test_a_missing_stage_stays_missing():
    run = _run({'A': {'m12': _step(mtime=T0)}})
    st = render._ladder_state(run)
    assert st['m4'] == ''


def test_a_stage_with_no_mtime_is_not_called_stale():
    """`mtime` is sampled and can be absent; absence is not evidence of age."""
    run = _run({'A': {'m12': _step(mtime=T0 + 100 * HOUR),
                      'm4': _step(mtime=None)}})
    assert render._ladder_state(run)['m4'] == 'done'


def test_the_legend_describes_the_new_state():
    """A colour nobody can look up is a colour that gets guessed at -- which is
    what happened with `part` being read as "queued"."""
    import inspect
    src = inspect.getsource(render)
    assert 'gcm-step.stale' in src, 'the state needs a style of its own'
    assert 'predates an earlier stage' in src, 'the legend must explain it'
