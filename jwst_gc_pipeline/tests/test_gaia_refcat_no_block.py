"""The Gaia leg of the refcat builder cannot cap, hang, or block a tile.

Three defects, all measured, all in ``query_gaia``:

1. THE CAP GUARD WAS DEAD FOR BOTH LAUNCHERS.  The test read
   ``if launcher is Gaia.launch_job and len(res) == 2000``, and
   ``astroquery.gaia.Gaia`` is an INSTANCE of ``GaiaClass``, so each attribute
   access builds a fresh bound method: on astroquery 0.4.11
   ``Gaia.launch_job is Gaia.launch_job`` is ``False`` (``==`` is ``True``).
   ``len(res)`` was therefore never consulted for EITHER launcher -- broader
   than issue #856, which read it as guarding the sync path only.  The raise
   also sat inside the ``try`` whose ``except Exception`` was two lines below,
   so a live guard would have been swallowed and retried as a transient.  The
   cap reached disk three times: gc-treasury o134 (2000 Gaia where the uncapped
   query reads 8794), o135 (2000 vs 9937), and sgrb2's production refcat of
   2026-06-18 (2000 vs a live 9488).  The truncation is AZIMUTHAL, not a random
   thinning -- o135's capped Gaia read octant max/min 25.9 against the
   rebuild's 1.62, and left exactly ONE Gaia star inside the MIRI F770W
   footprint against 225 in the rebuild.

2. NOTHING BOUNDED THE ESA TAP IN WALL CLOCK.  No layer had one: the builder
   passed no timeout, the launchers accept none, the string "timeout" does not
   occur in astroquery 0.4.11's ``utils/tap/`` or ``gaia/`` outside their tests,
   the connections are bare ``HTTPSConnection(host, port)``, and the async wait
   is ``while True: ... time.sleep(0.5)``.  Tile o131 hung 25+ minutes and was
   killed.  Program 10678 has 139 tiles and ``reference_catalog_required`` is
   True for it, so a hung Gaia query blocks that tile's reduction.

3. A GAIA OUTAGE KILLED THE BUILD.  ``query_gaia`` ended in
   ``return _query_gaia_vizier(...)``, so a VizieR failure propagated and the
   tile got no reference catalog at all.  That is backwards: VIRAC2 is the GC
   reference catalog (152k-185k rows per 9' cone against 5.6k-10.5k Gaia), and
   the Gaia component feeds exactly two consumers, both non-gating -- the
   ``sparse`` leg of ``visit_consensus.load_reference_catalog`` behind
   ``measure_reference_tie``'s ~100 mas gross backstop, and
   ``check_interframe_overlap``'s explicitly diagnostic Gaia leg.  Verified in
   this environment against the live code: ``measure_offset`` returns ``None``
   for a zero-length reference, which sets ``sparse_untrustworthy`` and hence
   ``cross_gross_ok=True``; ``agree_across_references`` returns
   ``sep_mas=nan``; and the overlap gate's ``gaia = rc[gm] if gm.any() else
   None`` makes it ``continue``.  A repo-wide grep finds no ``n_gaia``/``NGAIA``
   floor anywhere, so no Gaia count is REQUIRED and losing the leg costs a
   cross-check and nothing else.

No test here touches the network: the backends are monkeypatched.
"""
import os
import time

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction import build_gaia_virac2_refcat_byquery as B


# --------------------------------------------------------------------------
# fixtures: fake query results, no network
# --------------------------------------------------------------------------

def _gaia_table(n, seed=0):
    """An ESA-TAP-shaped Gaia result: the column names ``main`` actually reads."""
    rng = np.random.default_rng(seed)
    t = Table()
    t['ra'] = 266.84 + rng.normal(0, 0.05, n)
    t['dec'] = -28.33 + rng.normal(0, 0.05, n)
    t['pmra'] = rng.normal(0, 3.0, n)
    t['pmdec'] = rng.normal(0, 3.0, n)
    t['phot_g_mean_mag'] = rng.uniform(12, 21, n)
    t['ref_epoch'] = np.full(n, 2016.0)
    return t


def _virac_table(n, seed=1):
    """A VizieR II/387-shaped VIRAC2 result."""
    rng = np.random.default_rng(seed)
    t = Table()
    t['RAJ2000'] = 266.84 + rng.normal(0, 0.05, n)
    t['DEJ2000'] = -28.33 + rng.normal(0, 0.05, n)
    t['pmRA'] = rng.normal(0, 4.0, n)
    t['pmDE'] = rng.normal(0, 4.0, n)
    t['Jmag'] = rng.uniform(13, 20, n)
    t['Hmag'] = rng.uniform(12, 19, n)
    t['Ksmag'] = rng.uniform(11, 18, n)
    return t


class _FakeJob:
    def __init__(self, table):
        self._table = table

    def get_results(self):
        return self._table


class _FakeGaia:
    """Stands in for ``astroquery.gaia.Gaia`` -- an INSTANCE, like the real one,
    so ``getattr(Gaia, name)`` builds a fresh bound method exactly as it does in
    production.  That is the property the dead guard tripped over."""

    def __init__(self, async_result=None, sync_result=None,
                 async_error=None, sync_error=None, hang_s=0.0):
        self.async_result, self.sync_result = async_result, sync_result
        self.async_error, self.sync_error = async_error, sync_error
        self.hang_s = hang_s
        self.calls = []

    def launch_job_async(self, query, **kw):
        self.calls.append('launch_job_async')
        if self.hang_s:
            time.sleep(self.hang_s)
        if self.async_error is not None:
            raise self.async_error
        return _FakeJob(self.async_result)

    def launch_job(self, query, **kw):
        self.calls.append('launch_job')
        if self.hang_s:
            time.sleep(self.hang_s)
        if self.sync_error is not None:
            raise self.sync_error
        return _FakeJob(self.sync_result)


@pytest.fixture
def fake_gaia_module(monkeypatch):
    """Install a fake ``astroquery.gaia`` so ``_query_gaia_esa_tap``'s
    ``from astroquery.gaia import Gaia`` resolves to our stand-in."""
    import sys
    import types

    def install(fake):
        mod = types.ModuleType('astroquery.gaia')
        mod.Gaia = fake
        monkeypatch.setitem(sys.modules, 'astroquery.gaia', mod)
        return fake
    return install


# --------------------------------------------------------------------------
# 1. the cap test fires on BOTH launchers
# --------------------------------------------------------------------------

def test_the_bound_method_identity_that_killed_the_old_guard_still_holds():
    """The premise of the whole fix, pinned against the real astroquery so a
    version bump that changed it would be noticed: ``Gaia.launch_job`` is not
    identical to itself, so ``launcher is Gaia.launch_job`` could never be True.
    ``==`` is, which is why the bug read as correct."""
    from astroquery.gaia import Gaia
    assert Gaia.launch_job is not Gaia.launch_job
    assert Gaia.launch_job == Gaia.launch_job
    for launcher in (Gaia.launch_job_async, Gaia.launch_job):
        assert (launcher is Gaia.launch_job) is False


@pytest.mark.parametrize("which", ["async", "sync"])
def test_exactly_2000_rows_is_refused_from_either_launcher(fake_gaia_module, which):
    """The defect: the old test exempted ``launch_job_async``, which is tried
    FIRST, and (through the identity bug) the sync path as well.  o134 and o135
    were written capped on 2026-09-12, three months after the guard landed."""
    capped = _gaia_table(B.GAIA_SYNC_ROW_CAP)
    if which == "async":
        fake, expect = _FakeGaia(async_result=capped), "launch_job_async"
    else:
        fake = _FakeGaia(async_error=RuntimeError("500"), sync_result=capped)
        expect = "launch_job "      # the trailing space excludes _async
    fake_gaia_module(fake)
    with pytest.raises(B.GaiaRowCapError) as ex:
        B._query_gaia_esa_tap(266.84, -28.33, 0.17, retries=1, budget_s=30)
    assert str(B.GAIA_SYNC_ROW_CAP) in str(ex.value)
    assert expect in str(ex.value), (
        f"the cap was caught but the message does not name the launcher "
        f"({which}): {ex.value}")


def test_a_capped_response_is_not_retried_as_if_it_were_transient(fake_gaia_module):
    """The old raise sat inside the try whose ``except Exception`` followed two
    lines later, so even a live guard would have been printed as a failed
    attempt and retried.  Retrying is pointless -- the same query returns the
    same cap -- so the cap must escape the loop on the first occurrence."""
    fake = _FakeGaia(async_result=_gaia_table(B.GAIA_SYNC_ROW_CAP))
    fake_gaia_module(fake)
    with pytest.raises(B.GaiaRowCapError):
        B._query_gaia_esa_tap(266.84, -28.33, 0.17, retries=6, budget_s=60)
    assert fake.calls == ['launch_job_async'], (
        f"a capped response was retried {len(fake.calls)} times: {fake.calls}")


def test_a_capped_esa_response_is_never_written(fake_gaia_module, monkeypatch):
    """End to end through ``query_gaia``: a cap must not become a refcat.  With
    VizieR also down the build goes VIRAC-only, which is strictly better than
    the 22%-complete, azimuthally clustered Gaia subset the capped files hold."""
    fake_gaia_module(_FakeGaia(async_result=_gaia_table(B.GAIA_SYNC_ROW_CAP)))
    monkeypatch.setattr(B, 'GAIA_BACKENDS', {
        'vizier': lambda *a, **k: (_ for _ in ()).throw(RuntimeError("VizieR down")),
        'esa-tap': B._query_gaia_esa_tap})
    t, src = B.query_gaia(266.84, -28.33, 0.17, retries=1, budget_s=30)
    assert t is None and src == 'none'


def test_a_row_count_near_but_not_at_the_cap_is_accepted(fake_gaia_module):
    """The cap is an EXACT signature, not a floor: the ten healthy gc-treasury
    tiles read 5627-10480 Gaia and must not be refused, and neither must a cone
    that genuinely holds 1999 or 2001."""
    for n in (1999, 2001, 8794, 9937):
        fake_gaia_module(_FakeGaia(async_result=_gaia_table(n)))
        got = B._query_gaia_esa_tap(266.84, -28.33, 0.17, retries=1, budget_s=30)
        assert len(got) == n


def test_the_adql_carries_its_own_top_so_astroquery_cannot_inject_2000():
    """Belt as well as braces: ``taputils.set_top_in_query`` only injects a TOP
    when the query has none, so an explicit TOP PREVENTS the sync truncation
    instead of only detecting it.  Verified against the real astroquery."""
    from astroquery.utils.tap import taputils
    q = B.gaia_adql(266.84, -28.33, 0.17)
    assert f"TOP {B.GAIA_ADQL_TOP}" in q
    assert taputils.set_top_in_query(q, B.GAIA_SYNC_ROW_CAP) == q
    # ...and the old, TOP-less form is exactly what astroquery truncates.
    bare = ("SELECT ra,dec FROM gaiadr3.gaia_source WHERE 1=1")
    assert f"TOP {B.GAIA_SYNC_ROW_CAP}" in taputils.set_top_in_query(
        bare, B.GAIA_SYNC_ROW_CAP)


# --------------------------------------------------------------------------
# 2. the wall clock, and the fall-through to VizieR
# --------------------------------------------------------------------------

def test_a_hung_attempt_is_interrupted_rather_than_waited_out(fake_gaia_module):
    """o131's failure mode, in miniature: a launcher that never returns.  The
    bound has to INTERRUPT the blocked call -- a mechanism that only stops
    WAITING (a thread pool) leaves the hang to wedge interpreter exit."""
    fake_gaia_module(_FakeGaia(async_result=_gaia_table(10), hang_s=30.0))
    t0 = time.monotonic()
    with pytest.raises(B.GaiaTapTimeout):
        B._query_gaia_esa_tap(266.84, -28.33, 0.17, retries=1,
                              attempt_timeout_s=1.0, budget_s=4.0)
    assert time.monotonic() - t0 < 20.0, "the hang was waited out, not bounded"


def test_the_phase_budget_bounds_retries_and_backoff_together(fake_gaia_module):
    """A per-attempt bound alone is not enough.  At the old ``retries=6`` and 2
    launchers that is 12 attempts plus 2*(5+10+15+20+25+30) = 210 s of sleeps,
    so a 300 s per-attempt bound still allowed ~25 minutes -- o131's duration.
    The budget caps the phase however the attempts fail."""
    fake_gaia_module(_FakeGaia(async_result=_gaia_table(10), hang_s=30.0))
    t0 = time.monotonic()
    with pytest.raises(B.GaiaTapTimeout):
        B._query_gaia_esa_tap(266.84, -28.33, 0.17, retries=6,
                              attempt_timeout_s=1.0, budget_s=3.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 12.0, f"phase ran {elapsed:.1f} s against a 3 s budget"


def test_a_timed_out_esa_leg_falls_through_to_vizier(monkeypatch, fake_gaia_module):
    """The whole point of the order: a hung or capped ESA TAP must cost seconds
    and then hand off, not block the tile.  Here VizieR answers and the
    provenance recorded is VizieR's."""
    fake_gaia_module(_FakeGaia(async_result=_gaia_table(10), hang_s=30.0))
    monkeypatch.setattr(B, 'GAIA_BACKENDS', {
        'vizier': lambda ra, dec, radius: _gaia_table(9937),
        'esa-tap': B._query_gaia_esa_tap})
    # force ESA first so the fall-through is what is under test, not the default
    t, src = B.query_gaia(266.84, -28.33, 0.17, backends='esa-tap,vizier',
                          retries=1, attempt_timeout_s=1.0, budget_s=3.0)
    assert src == 'vizier'
    assert len(t) == 9937


def test_vizier_is_tried_first_by_default(monkeypatch):
    """VizieR is uncapped (``ROW_LIMIT=-1``) and bounded (``Vizier.TIMEOUT=60``);
    the ESA TAP is neither, and its two distinguishing features -- a server-side
    cone and ``ref_epoch`` -- are respectively matched by ``Vizier.query_region``
    and never read (``GAIA_EPOCH`` is hardcoded).  So VizieR leads and the ESA
    TAP is the fallback, which also means a healthy build never opens a TAP
    connection at all."""
    order = []

    def vz(ra, dec, radius):
        order.append('vizier')
        return _gaia_table(9937)

    def esa(ra, dec, radius, **kw):
        order.append('esa-tap')
        return _gaia_table(10)

    monkeypatch.setattr(B, 'GAIA_BACKENDS', {'vizier': vz, 'esa-tap': esa})
    assert B.resolve_gaia_backends() == ('vizier', 'esa-tap')
    t, src = B.query_gaia(266.84, -28.33, 0.17)
    assert src == 'vizier'
    assert order == ['vizier'], f"the ESA TAP was contacted anyway: {order}"


def test_the_wall_clock_bound_restores_the_previous_sigalrm_handler():
    """Clearing the timer is not enough -- the builder must not leave its own
    SIGALRM handler installed for whatever runs after it."""
    import signal
    sentinel = signal.getsignal(signal.SIGALRM)
    with B.wall_clock_bound(5.0, "probe"):
        assert signal.getsignal(signal.SIGALRM) is not sentinel
    assert signal.getsignal(signal.SIGALRM) is sentinel
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_the_wall_clock_bound_repeats_so_a_swallowed_alarm_still_lands():
    """astroquery wraps ``self.start()`` in ``except Exception:  # ignore``
    (``utils/tap/model/job.py:323-329``), so a ONE-SHOT alarm delivered inside
    that window is swallowed and never fires again -- the bound would silently
    evaporate.  A repeating interval fires again."""
    fired = []
    with B.wall_clock_bound(0.3, "probe", repeat=0.3):
        deadline = time.monotonic() + 3.0
        while True:
            try:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            except B.GaiaTapTimeout:
                fired.append(1)      # swallow it, as astroquery's job.start does
    assert len(fired) >= 2, (
        f"the alarm fired {len(fired)} time(s) in 3 s at a 0.3 s interval; a "
        f"one-shot bound evaporates the moment astroquery swallows the first")


def test_a_zero_bound_means_unbounded_and_installs_nothing():
    """``--gaia-tap-timeout 0`` is the deliberate opt-out, and must not leave a
    timer armed."""
    import signal
    with B.wall_clock_bound(0, "probe"):
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_the_bounds_are_env_overridable(fake_gaia_module, monkeypatch):
    """An operator has to be able to loosen or tighten the bound on a running
    sweep without editing code."""
    monkeypatch.setenv('GAIA_TAP_TIMEOUT_S', '1')
    monkeypatch.setenv('GAIA_TAP_BUDGET_S', '2')
    fake_gaia_module(_FakeGaia(async_result=_gaia_table(10), hang_s=30.0))
    t0 = time.monotonic()
    with pytest.raises(B.GaiaTapTimeout):
        B._query_gaia_esa_tap(266.84, -28.33, 0.17, retries=6)
    assert time.monotonic() - t0 < 12.0


# --------------------------------------------------------------------------
# 3. a total Gaia outage does not raise: the build goes VIRAC-only
# --------------------------------------------------------------------------

def test_every_gaia_path_failing_returns_none_instead_of_raising(monkeypatch):
    """The old code ended in ``return _query_gaia_vizier(...)``, so a VizieR
    outage propagated and the tile got no reference catalog at all.  For a
    program whose ``reference_catalog_required`` is True that blocks the
    reduction of the tile -- over a cross-check that cannot gate anything."""
    monkeypatch.setattr(B, 'GAIA_BACKENDS', {
        'vizier': lambda *a, **k: (_ for _ in ()).throw(RuntimeError("VizieR 500")),
        'esa-tap': lambda *a, **k: (_ for _ in ()).throw(B.GaiaTapTimeout("hung"))})
    t, src = B.query_gaia(266.84, -28.33, 0.17)
    assert t is None
    assert src == 'none'


def test_the_virac_only_outcome_is_announced_loudly(monkeypatch, capsys):
    """Silently dropping the Gaia leg is how sgrb2 carried a 22%-complete Gaia
    subset for three months unnoticed.  The outcome has to be in the log AND in
    the file."""
    monkeypatch.setattr(B, 'GAIA_BACKENDS', {
        'vizier': lambda *a, **k: (_ for _ in ()).throw(RuntimeError("VizieR 500")),
        'esa-tap': lambda *a, **k: (_ for _ in ()).throw(B.GaiaTapTimeout("hung"))})
    B.query_gaia(266.84, -28.33, 0.17)
    out = capsys.readouterr().out
    assert B.VIRAC_ONLY_BANNER in out
    assert 'VizieR 500' in out and 'hung' in out


def test_a_virac_only_table_builds_and_records_its_provenance():
    """The VIRAC-only refcat is a real, usable refcat: every VIRAC2 row kept
    (there is no Gaia to de-duplicate against, and ``match_to_catalog_sky``
    refuses a length-0 catalog outright), the coverage floor still applied, and
    NGAIA=0 / GAIASRC='none' written so the thinness is READ rather than
    inferred from a round row count."""
    v = _virac_table(20000)
    ref = B.build_refcat_table(None, 'none', v, 2026.6968, 0.02)
    assert len(ref) == 20000
    assert set(np.asarray(ref['source']).astype(str)) == {'VIRAC2'}
    assert ref.meta['NGAIA'] == 0
    assert ref.meta['NVIRAC'] == 20000
    assert ref.meta['GAIASRC'] == 'none'
    assert 'VIRAC2-ONLY' in ref.meta['NOTE']
    assert 'skycoord' in ref.colnames


def test_a_virac_only_table_is_what_the_downstream_consumers_expect():
    """Both Gaia-aware consumers degrade rather than fail, which is why no Gaia
    floor is kept here.  ``load_reference_catalog`` must still report the file
    as DENSE (the VIRAC2 component is what sets that) and hand back an EMPTY
    sparse set, which is the state in which ``measure_offset`` returns None ->
    ``sparse_untrustworthy`` -> ``cross_gross_ok=True``."""
    from jwst_gc_pipeline.photometry.visit_consensus import load_reference_catalog
    import tempfile
    v = _virac_table(20000)
    ref = B.build_refcat_table(None, 'none', v, 2026.6968, 0.02)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'gaia_virac2_refcat_epoch2026.70_o134.fits')
        ref.write(path)
        loaded = load_reference_catalog(path)
    assert loaded['dense'] is True, "a VIRAC-only refcat must still read as dense"
    assert len(loaded['sparse']) == 0
    assert len(loaded['all']) == 20000


def test_a_gaia_backed_table_still_fills_and_labels_as_before():
    """The VIRAC-only path is an addition, not a change: with Gaia present the
    output is the same two-component catalog, with the same 0.3" fill rule."""
    g, v = _gaia_table(3000), _virac_table(20000)
    ref = B.build_refcat_table(g, 'vizier', v, 2026.6968, 0.02)
    labels = np.asarray(ref['source']).astype(str)
    assert (labels == 'GaiaDR3').sum() == 3000
    assert (labels == 'VIRAC2').sum() == ref.meta['NVIRAC']
    assert ref.meta['NGAIA'] == 3000
    assert ref.meta['GAIASRC'] == 'vizier'
    assert 'VIRAC2-ONLY' not in ref.meta['NOTE']
    # the fill rule still drops the VIRAC2 rows with a Gaia match inside 0.3"
    assert ref.meta['NVIRAC'] <= 20000


def test_the_coverage_floor_still_refuses_a_broken_virac_query():
    """Gaia may not block; VIRAC2 still must.  A truncated VIRAC2 response is
    the failure the density floor exists for (issue #415 gap 4), and going
    VIRAC-only must not weaken it."""
    with pytest.raises(B.ThinReferenceCoverageError):
        B.build_refcat_table(None, 'none', _virac_table(300), 2026.6968, 0.1)


def test_backends_none_skips_gaia_entirely_without_a_query(monkeypatch):
    """An operator running a sweep during a Gaia outage should be able to say so
    up front rather than paying the timeout 139 times."""
    called = []
    monkeypatch.setattr(B, 'GAIA_BACKENDS', {
        'vizier': lambda *a, **k: called.append('vizier'),
        'esa-tap': lambda *a, **k: called.append('esa-tap')})
    t, src = B.query_gaia(266.84, -28.33, 0.17, backends='none')
    assert (t, src) == (None, 'none')
    assert called == []


def test_an_unknown_backend_name_is_an_error_not_a_silent_skip():
    """A typo in ``--gaia-backends`` must not read as 'no Gaia today'."""
    with pytest.raises(ValueError) as ex:
        B.resolve_gaia_backends('vizer')
    assert 'vizer' in str(ex.value)


def test_the_backend_order_is_env_overridable(monkeypatch):
    monkeypatch.setenv('GAIA_BACKENDS', 'esa-tap')
    assert B.resolve_gaia_backends() == ('esa-tap',)
    monkeypatch.setenv('GAIA_BACKENDS', 'none')
    assert B.resolve_gaia_backends() == ()
