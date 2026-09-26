#!/usr/bin/env python
"""Build a dense Gaia+VIRAC2 absolute astrometric reference catalog for ANY Galactic-Center field,
by querying Vizier (VIRAC2 II/387) and Gaia DR3 over the field footprint.

Standing policy (feedback_reference_frame_policy): GC fields -> VIRAC2 positions PM-propagated
per-star from the VIRAC2 reference epoch 2014.0 to the observation epoch; Gaia DR3 (PM-propagated
from 2016.0) provides the absolute frame where it is complete. The combined catalog = every Gaia DR3
source + every VIRAC2 source with no Gaia match within 0.3", all at the observation epoch.

VIRAC2 is the reference; the Gaia leg may not block a tile
----------------------------------------------------------
In the GC, Gaia DR3 defines the absolute FRAME but is far too sparse to be the
reference CATALOG (CLAUDE.md, memory ``gc-gaia-frame-not-catalog``): a 9' GC cone
holds 152k-185k VIRAC2 rows against 5.6k-10.5k Gaia.  Exactly two places read the
``source`` column and treat GaiaDR3 rows differently, and BOTH are non-gating
cross-checks:

* ``photometry.visit_consensus.load_reference_catalog`` returns the Gaia rows as
  ``sparse``, which feeds ``measure_reference_tie``'s ~100 mas GROSS backstop
  (``REFERENCE_CROSSCHECK_GROSS_MAS``).  With no Gaia rows at all,
  ``measure_offset`` returns ``None`` -> ``res_b is None`` ->
  ``sparse_untrustworthy`` -> ``cross_gross_ok=True``: the backstop ceases to
  exist, it does not block.  ``dense`` stays True because the VIRAC2 component
  is what sets it.
* ``scripts/release/check_interframe_overlap.py`` builds its Gaia leg as
  ``gaia = rc[gm] if gm.any() else None`` and ``continue``s on ``None``; the leg
  is labelled "[diagnostic, non-gating: Gaia too sparse]" at the use site.

Nothing else keys on the label -- no photometry, PSF, merging or release-staging
path -- and a repo-wide grep finds no ``n_gaia``/``NGAIA`` floor anywhere.  So a
Gaia outage must cost the build its cross-check and NOTHING else: this builder
writes a VIRAC-only catalog and says so loudly, rather than dying and leaving a
tile with no reference at all.  Program 10678 has 139 tiles to build and
``reference_catalog_required`` is True for it, so a Gaia hang that blocks a tile
blocks the reduction of that tile.

Why VizieR is tried BEFORE the ESA TAP
--------------------------------------
Measured on astroquery 0.4.11 in this environment:

* the ESA TAP has no timeout at ANY layer.  The string "timeout" does not occur
  in ``astroquery/utils/tap/`` or ``astroquery/gaia/`` outside their tests; the
  connections are built as bare ``HTTPSConnection(host, port)`` with no
  ``timeout=`` (``utils/tap/conn/tapconn.py:761``) over a process default of
  ``None``; and the async wait is ``while True: ... time.sleep(0.5)`` with no
  deadline (``utils/tap/model/job.py:327-338``).  Tile o131 hung on it for 25+
  minutes with nothing to stop it and had to be killed.
* the SYNC launcher silently truncates to 2000 rows: ``TapPlus.launch_job``
  injects ``TOP 2000`` (``utils/tap/core.py:294``), and ``Gaia.ROW_LIMIT = -1``
  is a no-op because neither ``launch_job`` nor ``launch_job_async`` accepts or
  consults ``maxrec``.  That cap reached disk twice on 2026-09-12 (o134: 2000
  Gaia instead of 8794; o135: 2000 instead of 9937) and once in production
  (sgrb2 2026-06-18: 2000 of a live 9488).  The truncation is AZIMUTHAL, not a
  random thinning -- the response is source_id/HEALPix ordered, so o135's capped
  Gaia read octant max/min 25.9 against the rebuild's 1.62 and left exactly ONE
  Gaia star inside the MIRI F770W footprint (225 in the rebuild).
* VizieR is bounded (``Vizier.TIMEOUT = 60`` per request), uncapped
  (``ROW_LIMIT = -1``), and already carries the far bigger half of this build:
  ``query_virac2`` pulls 152k-185k rows per cone from the same service with no
  fallback at all, so the build already dies if VizieR is down.  Positions agree
  with the ESA TAP to 0.0036 mas median (1999 of 2000 o134 rows matched within
  0.1 mas) and ``refmag`` is identical, so nothing is lost by preferring it.
* the ESA TAP's two distinguishing features are not used: the server-side cone
  is also what ``Vizier.query_region`` does, and ``ref_epoch`` is selected here
  and never read (``GAIA_EPOCH`` is hardcoded at 2016.0 and the PM propagation
  is done locally).

The ESA TAP is kept as the FALLBACK, under a wall-clock bound, so a VizieR
outage is still covered.

Usage:
    python build_gaia_virac2_refcat_byquery.py --base /orange/adamginsburg/jwst/sgrb2 \
        --epoch 2024.685 --ra 266.835 --dec -28.398 --radius 0.1 --out-epoch-tag 2024.68
"""
import argparse
import contextlib
import os
import signal
import threading
import time

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.astrometry_utils import farr, prop
from jwst_gc_pipeline.photometry.reference_uncertainty import (
    sigma_pred_mas, SIGMA_PRED_COLUMN, SIGMA_POS_RA_COLUMN, SIGMA_POS_DEC_COLUMN,
    SIGMA_PM_RA_COLUMN, SIGMA_PM_DEC_COLUMN)

GAIA_EPOCH = 2016.0    # Gaia DR3 reference epoch
VIRAC2_EPOCH = 2014.0  # VIRAC2 reference epoch (Smith+2025 II/387: fixed at 2014.0)

#: Rows ``astroquery``'s SYNC TAP launcher silently truncates a response to
#: (``taputils.set_top_in_query(query, 2000)`` at ``utils/tap/core.py:294``).
#: A GC cone returning EXACTLY this many Gaia rows is the TOP-N signature, never
#: the sky: the capped o134/o135/sgrb2 files read 2000 where the uncapped query
#: reads 8794/9937/9488.
GAIA_SYNC_ROW_CAP = 2000

#: An explicit ``TOP`` in the ADQL, because ``set_top_in_query`` only injects
#: when the query has none -- verified on astroquery 0.4.11: a query already
#: carrying ``SELECT TOP 3000000 ...`` comes back unchanged, so this PREVENTS
#: the sync cap rather than only detecting it.  Two orders above the densest GC
#: cone (10480 Gaia, gc-treasury o133).
GAIA_ADQL_TOP = 3_000_000

#: Wall clock ONE ESA TAP attempt may consume, seconds.  Env ``GAIA_TAP_TIMEOUT_S``.
GAIA_TAP_ATTEMPT_TIMEOUT_S = 120.0

#: Wall clock the WHOLE ESA TAP phase may consume, seconds.  Env ``GAIA_TAP_BUDGET_S``.
#: A per-attempt bound alone does not bound the phase: at 120 s x 2 launchers x
#: the old ``retries=6``, plus the 210 s of ``sleep(5 * (i + 1))`` those retries
#: burn, the worst case is still ~25 minutes before the fallback is reached --
#: which is exactly how long o131 held the queue.
GAIA_TAP_BUDGET_S = 300.0

#: ESA TAP attempts per launcher.  Was 6, which cost 2 * (5+10+15+20+25+30) =
#: 210 s of sleeps alone before the VizieR fallback line was ever reached.
GAIA_TAP_RETRIES = 2

#: Gaia backends, in the order they are tried.  Overridable with
#: ``--gaia-backends`` / env ``GAIA_BACKENDS``; ``none`` skips Gaia entirely.
DEFAULT_GAIA_BACKENDS = ('vizier', 'esa-tap')


class ThinReferenceCoverageError(RuntimeError):
    """The queried reference coverage is far below anything the sky can explain.

    This is a BROKEN-QUERY guard, not a thin-sky guard, and the distinction is
    the reason it is expressed as a DENSITY rather than a row count.  Measured
    2026-08-23 over all 139 program-10678 tile centres with a VizieR cone at
    0.02 deg against II/387 and I/355:

        thinnest tile   GC_130 (l=0.641, b=+0.006)   2094 VIRAC2 per cone
        median                                       4356
        densest         GC_49  (l=359.874, b=-0.030) 6184
        brick's refcat centre, same cone             2908  (its same-star tie reads ~0.6 mas)

    The whole survey spans a factor of 3.0 and the thinnest tile carries 0.72x
    the density of the field whose tie is known good, so no floor set for thin
    SKY would fire on this program -- VVV's bulge coverage is uniform over it,
    and the gradient is stellar density falling away from Sgr A*.  What a floor
    can still catch is a query that went wrong: a truncated VizieR response, a
    wrong --radius, a cone placed off the field.  Those return orders of
    magnitude less, so the floor sits at roughly HALF the thinnest real tile and
    two orders above a broken query, and as a density it applies unchanged at
    any radius (an absolute row count tuned for the CMZ would trip on legitimate
    non-GC fields -- ngc6334's refcats are ~1.4 MB against sgrb2's 8.4 MB for
    comparable radii).  Issue #415 gap 4.

    It cannot see a Gaia truncation, and is not asked to: VIRAC2 supplies ~96%
    of the rows, so capped o134 still recorded 2.41e6 deg^-2 against the 8e5
    floor (the rebuild 2.51e6 -- a 4% change).  The Gaia leg is guarded by
    ``GaiaRowCapError`` and recorded as ``ref.meta['NGAIA']`` instead, because a
    Gaia FLOOR would be a Gaia BLOCK, which the GC policy forbids.
    """


class GaiaRowCapError(RuntimeError):
    """An ESA TAP response came back at exactly ``GAIA_SYNC_ROW_CAP`` rows.

    Replaces the guard that never fired.  The old test was

        for launcher in (Gaia.launch_job_async, Gaia.launch_job):
            ...
            if launcher is Gaia.launch_job and len(res) == 2000:

    and ``astroquery.gaia.Gaia`` is an INSTANCE of ``GaiaClass``, so every
    ``Gaia.launch_job`` attribute access builds a fresh bound method: on
    astroquery 0.4.11 ``Gaia.launch_job is Gaia.launch_job`` -> ``False`` while
    ``==`` -> ``True``.  The first ``and`` term was therefore always False and
    ``len(res)`` was never consulted for EITHER launcher -- broader than issue
    #856's reading, which had it guarding the sync path only.  The raise also
    sat INSIDE the ``try`` whose ``except Exception`` was two lines below, so
    even a live guard would have been swallowed, printed as a failed attempt and
    retried.

    Now the count is tested on the RESULT, outside the try, after every
    launcher, and a capped response abandons the ESA leg rather than being
    retried (the same query returns the same cap) or written.  The test is NOT
    applied to the VizieR result: with ``ROW_LIMIT = -1`` an exactly-2000-row
    VizieR cone is legitimate.
    """


class GaiaTapTimeout(RuntimeError):
    """An ESA TAP attempt, or the ESA TAP phase as a whole, ran out of wall clock."""


# Usable reference sources per square degree, below which the query is treated
# as broken.  1000 per 0.02 deg cone = 1000 / (pi * 0.02**2) ~ 8e5 deg^-2.
MIN_REF_DENSITY_PER_SQDEG = 8.0e5


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw in (None, ''):
        return float(default)
    return float(raw)


@contextlib.contextmanager
def wall_clock_bound(seconds, what, repeat=5.0):
    """Interrupt the enclosed block with ``GaiaTapTimeout`` after ``seconds``.

    WHY A SIGNAL ALARM, and not any of the gentler options:

    * there is no timeout to pass.  ``GaiaClass.launch_job``/``launch_job_async``
      expose no such parameter, and the string "timeout" does not appear
      anywhere in astroquery 0.4.11's ``utils/tap/`` or ``gaia/`` outside their
      tests -- the TAP stack has no timeout at any layer to configure.
    * ``socket.setdefaulttimeout`` would bound the blocked read, but it is
      process-GLOBAL and this same process pulls 152k-185k VIRAC2 rows from
      VizieR in one request; a bound tight enough to catch a TAP hang would
      abort the legitimate VIRAC2 response, which is the leg that must not fail.
    * a ``ThreadPoolExecutor`` bounds only the WAIT, not the work.  Its worker
      threads are non-daemon and joined at interpreter exit, so a TAP read
      blocked in ``conn.getresponse()`` still wedges the process on the way out
      -- the hang is deferred to exit rather than removed.

    ``SIGALRM`` is delivered to the blocked read itself, which is the only
    mechanism here that both interrupts it and lets the process exit.

    Two details that matter and are easy to get wrong:

    * the interval REPEATS (``repeat``).  astroquery wraps ``self.start()`` in
      ``except Exception:  # ignore`` (``utils/tap/model/job.py:323-329``), so a
      one-shot alarm landing inside that window is swallowed and, being
      one-shot, never fires again -- the bound would silently evaporate.
    * the previous handler is RESTORED, not merely cleared, so the builder does
      not leave a SIGALRM handler installed for whatever runs after it.

    ``setitimer``/``signal`` are main-thread only.  Today's only caller is this
    module's ``main()`` in a serial subprocess (``build_treasury_refcats.py``
    runs one tile at a time via ``subprocess.run``), so that holds; off the main
    thread the block runs UNBOUNDED and says so, because Gaia is a non-gating
    cross-check and losing the bound is safer than losing the leg.  A future
    threaded caller needs a different mechanism, not a silent pass.
    """
    if not seconds or float(seconds) <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        print(f"  WARNING: {what} is running off the main thread, so its "
              f"{float(seconds):g} s wall-clock bound cannot be installed "
              f"(signal.setitimer is main-thread only); proceeding UNBOUNDED",
              flush=True)
        yield
        return

    seconds = float(seconds)

    def _fire(signum, frame):
        raise GaiaTapTimeout(
            f"{what} exceeded its {seconds:g} s wall-clock bound")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds, float(repeat))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


#: II/387 (VIRAC2, Smith+2025) columns carrying the per-star position and
#: proper-motion UNCERTAINTY (mas / mas yr^-1), queried alongside the position
#: and PM themselves so ``build_refcat_table`` can propagate a predicted
#: reference-position sigma to the observation epoch (issue #965 item 1,
#: following #957: the m2 same-star region map flags 45" cells at a tolerance
#: the reference's own per-star scatter can reach on its own, with no way to
#: tell a well-measured VIRAC2 star from a poorly-measured one).  Verified by
#: cone query against II/387/virac2 metadata (``Vizier(columns=['**'])``):
#: ``e_RAJ2000``/``e_DEJ2000`` are in mas, ``e_pmRA``/``e_pmDE`` in mas/yr.
VIRAC2_SIGMA_COLUMNS = ['e_RAJ2000', 'e_DEJ2000', 'e_pmRA', 'e_pmDE']


def query_virac2(ra, dec, radius):
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    Vizier.columns = (['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE', 'Jmag', 'Hmag', 'Ksmag']
                      + VIRAC2_SIGMA_COLUMNS)
    res = Vizier.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius * u.deg,
                              catalog='II/387/virac2')
    if not res:
        raise RuntimeError("VIRAC2 query returned nothing")
    return res[0]


def _query_gaia_vizier(ra, dec, radius):
    """Gaia DR3 from VizieR (I/355/gaiadr3) -- the DEFAULT Gaia backend.

    Uncapped (``ROW_LIMIT = -1``) and bounded (``Vizier.TIMEOUT = 60`` per
    request), which is the pair of properties the ESA TAP lacks.  Also the only
    path that works from compute nodes where the Gaia ESA TAP is firewalled.
    Returns a table with ESA-TAP-style column names.
    """
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    # e_RA_ICRS/e_DE_ICRS (mas) and e_pmRA/e_pmDE (mas/yr) verified present on
    # I/355/gaiadr3 by cone-query metadata (issue #965 item 1); carried through
    # under ESA-TAP-style names so build_refcat_table need not know which Gaia
    # backend answered.
    Vizier.columns = ['RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Gmag',
                      'e_RA_ICRS', 'e_DE_ICRS', 'e_pmRA', 'e_pmDE']
    res = Vizier.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius * u.deg,
                              catalog='I/355/gaiadr3')
    if not res:
        raise RuntimeError("VizieR Gaia DR3 query returned nothing")
    t = res[0]
    t.rename_column('RA_ICRS', 'ra'); t.rename_column('DE_ICRS', 'dec')
    t.rename_column('pmRA', 'pmra'); t.rename_column('pmDE', 'pmdec')
    t.rename_column('Gmag', 'phot_g_mean_mag')
    t.rename_column('e_RA_ICRS', 'ra_error'); t.rename_column('e_DE_ICRS', 'dec_error')
    t.rename_column('e_pmRA', 'pmra_error'); t.rename_column('e_pmDE', 'pmdec_error')
    return t


def gaia_adql(ra, dec, radius, top=GAIA_ADQL_TOP):
    """The cone query, carrying its own ``TOP`` so astroquery cannot inject 2000."""
    return (f"SELECT TOP {int(top)} ra,dec,pmra,pmdec,phot_g_mean_mag,ref_epoch,"
            "ra_error,dec_error,pmra_error,pmdec_error "
            "FROM gaiadr3.gaia_source "
            f"WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{radius}))=1")


def _query_gaia_esa_tap(ra, dec, radius, retries=None,
                        attempt_timeout_s=None, budget_s=None):
    """Gaia DR3 from the ESA TAP -- the FALLBACK backend, under a wall clock.

    Every attempt is bounded individually (``wall_clock_bound``) and the phase
    as a whole is bounded by ``budget_s``, so no combination of hangs, retries
    and back-off sleeps can hold a tile longer than the budget.  The row-cap
    test runs on the RESULT of BOTH launchers, outside the ``try``, so it cannot
    be swallowed by the retry handler.
    """
    from astroquery.gaia import Gaia
    # NOTE: no `Gaia.ROW_LIMIT = -1` here.  It read as if it handled the cap and
    # did not: neither launch_job nor launch_job_async accepts or consults
    # `maxrec`, so ROW_LIMIT only affects cone_search/query_object.  The explicit
    # ADQL `TOP` is what actually prevents the sync truncation.
    retries = GAIA_TAP_RETRIES if retries is None else int(retries)
    if attempt_timeout_s is None:
        attempt_timeout_s = _env_float('GAIA_TAP_TIMEOUT_S',
                                       GAIA_TAP_ATTEMPT_TIMEOUT_S)
    if budget_s is None:
        budget_s = _env_float('GAIA_TAP_BUDGET_S', GAIA_TAP_BUDGET_S)
    q = gaia_adql(ra, dec, radius)
    deadline = (time.monotonic() + float(budget_s)) if float(budget_s) > 0 else None
    last = None
    for i in range(max(1, int(retries))):
        # async first: it is the launcher with no server-side TOP injection.
        # Both are tested for the cap regardless -- see GaiaRowCapError.
        for name in ('launch_job_async', 'launch_job'):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise GaiaTapTimeout(
                    f"Gaia ESA TAP budget of {float(budget_s):g} s exhausted "
                    f"after {i} round(s); last error: {last}")
            bound = float(attempt_timeout_s)
            if remaining is not None:
                bound = min(bound, remaining)
            launcher = getattr(Gaia, name)
            job = None
            try:
                with wall_clock_bound(bound, f"Gaia ESA TAP {name}"):
                    # the Job is captured BEFORE get_results() so a timed-out
                    # attempt can at least name the server job it abandoned.
                    # SIGALRM interrupts OUR read; the job keeps running at ESA,
                    # so each timed-out retry leaves one behind.
                    job = launcher(q)
                    res = job.get_results()
            except GaiaTapTimeout as e:
                last = e
                jobid = getattr(job, 'jobid', None)
                orphan = (f" -- abandoned ESA job {jobid}, which keeps running "
                          f"server-side" if jobid else "")
                print(f"  Gaia ESA TAP attempt {i} ({name}) TIMED OUT: "
                      f"{e}{orphan}", flush=True)
                continue    # a hang is not made better by sleeping on it
            except Exception as e:
                # The TAP stack raises an open-ended set (requests, http.client,
                # ValueError out of the VOTable parse, astroquery's own), so this
                # is broad by necessity; the two conditions worth naming --
                # timeout and row cap -- are handled explicitly around it.
                last = e
                print(f"  Gaia ESA TAP attempt {i} ({name}) failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                nap = 5.0 * (i + 1)
                if deadline is not None:
                    nap = min(nap, max(0.0, deadline - time.monotonic()))
                if nap > 0:
                    time.sleep(nap)
                continue
            # OUTSIDE the try: a capped response must not be caught by the
            # handler above and retried as if it were a transient failure.
            if len(res) == GAIA_SYNC_ROW_CAP:
                raise GaiaRowCapError(
                    f"Gaia ESA TAP {name} returned EXACTLY {GAIA_SYNC_ROW_CAP} "
                    f"rows for the cone ({ra}, {dec}) r={radius} deg.  That is "
                    f"the TOP-{GAIA_SYNC_ROW_CAP} signature, not the sky: the "
                    f"capped o134/o135/sgrb2 catalogs read {GAIA_SYNC_ROW_CAP} "
                    f"where the uncapped query reads 8794/9937/9488, and the "
                    f"truncation is AZIMUTHAL (o135 Gaia octant max/min 25.9 vs "
                    f"1.62 rebuilt, ONE Gaia star left inside the MIRI F770W "
                    f"footprint against 225).  Abandoning the ESA leg rather "
                    f"than writing it.")
            return res
    raise GaiaTapTimeout(
        f"Gaia ESA TAP failed all {max(1, int(retries))} x 2 attempts; "
        f"last error: {last}")


#: name -> callable(ra, dec, radius) for the Gaia backends.
GAIA_BACKENDS = {
    'vizier': _query_gaia_vizier,
    'esa-tap': _query_gaia_esa_tap,
}


def resolve_gaia_backends(spec=None):
    """``'vizier,esa-tap'`` / ``'none'`` / ``None`` -> the tuple to try in order."""
    if spec is None:
        spec = os.environ.get('GAIA_BACKENDS') or ''
    if isinstance(spec, (list, tuple)):
        names = [str(s).strip().lower() for s in spec]
    else:
        names = [s.strip().lower() for s in str(spec).split(',')]
    names = [n for n in names if n]
    if not names:
        return tuple(DEFAULT_GAIA_BACKENDS)
    if names == ['none']:
        return ()
    unknown = [n for n in names if n not in GAIA_BACKENDS]
    if unknown:
        raise ValueError(f"unknown Gaia backend(s) {unknown}; known: "
                         f"{sorted(GAIA_BACKENDS)} or 'none'")
    return tuple(names)


#: The banner a VIRAC-only build prints.  Named so a test can pin that the
#: outcome is ANNOUNCED and not merely survived.
VIRAC_ONLY_BANNER = "NO GAIA BACKEND ANSWERED -- building this refcat VIRAC2-ONLY."


def query_gaia(ra, dec, radius, backends=None, retries=None,
               attempt_timeout_s=None, budget_s=None):
    """Gaia DR3 over the cone from the first backend that answers.

    Returns ``(table, provenance)``.  ``(None, 'none')`` means no Gaia path
    answered -- which is NOT an error here: the caller writes a VIRAC-only
    catalog, because the Gaia component feeds only non-gating cross-checks (see
    the module docstring) and a tile with no reference catalog at all cannot
    reduce.  Raising was the old behaviour, and it is what let a Gaia outage
    block a tile.

    VizieR is tried first; the module docstring carries the measurements behind
    that order.
    """
    order = resolve_gaia_backends(backends)
    if not order:
        print("  Gaia skipped (--gaia-backends none): VIRAC2-only refcat",
              flush=True)
        return None, 'none'
    failures = []
    for name in order:
        fn = GAIA_BACKENDS[name]
        try:
            if name == 'esa-tap':
                t = fn(ra, dec, radius, retries=retries,
                       attempt_timeout_s=attempt_timeout_s, budget_s=budget_s)
            else:
                t = fn(ra, dec, radius)
        except GaiaRowCapError as e:
            failures.append(f"{name}: CAPPED -- {e}")
            print(f"  Gaia backend {name} returned a CAPPED response: {e}",
                  flush=True)
            continue
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"  Gaia backend {name} failed ({type(e).__name__}: {e})",
                  flush=True)
            continue
        if t is None or len(t) == 0:
            failures.append(f"{name}: returned 0 rows")
            print(f"  Gaia backend {name} returned 0 rows", flush=True)
            continue
        print(f"  Gaia DR3 from {name}: {len(t)} rows", flush=True)
        return t, name
    print("\n" + "!" * 72, flush=True)
    print(f"!! {VIRAC_ONLY_BANNER}", flush=True)
    for f in failures:
        print(f"!!   {f}", flush=True)
    print("!! VIRAC2 is the reference catalog, so the TIE itself is unaffected.\n"
          "!! What is lost is the sparse-Gaia GROSS cross-check (the ~100 mas\n"
          "!! backstop in measure_reference_tie) and check_interframe_overlap's\n"
          "!! diagnostic Gaia leg.  Both are non-gating: an unmeasurable sparse\n"
          "!! leg sets cross_gross_ok=True, so the backstop ceases to exist\n"
          "!! rather than blocking.  ref.meta GAIASRC='none' and NGAIA=0 record\n"
          "!! it in the file.  Rebuild with --force once a Gaia service answers.",
          flush=True)
    print("!" * 72 + "\n", flush=True)
    return None, 'none'


def check_reference_coverage(n_usable, radius_deg, context="",
                             min_density=MIN_REF_DENSITY_PER_SQDEG):
    """Refuse to write a refcat whose usable source density is below the floor.

    ``n_usable`` counts the sources that survive the finite-position mask, i.e.
    the rows a tie can actually be measured against, not the rows the query
    returned.  Returns the measured density so a caller can record it.
    """
    area = np.pi * float(radius_deg) ** 2
    if area <= 0:
        raise ValueError(f"refcat coverage check: non-positive radius {radius_deg}")
    density = float(n_usable) / area
    if density < min_density:
        raise ThinReferenceCoverageError(
            f"reference coverage {density:.3g} usable sources/deg^2 "
            f"({n_usable} within {radius_deg} deg) is below the "
            f"{min_density:.3g} deg^-2 floor{' for ' + context if context else ''}.  "
            f"Every program-10678 tile measured 1.7e6-4.9e6 deg^-2 and the brick's "
            f"working refcat 2.3e6, so this is a BROKEN QUERY -- a truncated VizieR "
            f"response, the wrong --radius, or a cone off the field -- rather than "
            f"thin sky.  Check the query before building; --min-ref-density 0 "
            f"records a deliberate override.")
    return density


def refcat_filename(epoch_tag, obs_token=None):
    """``gaia_virac2_refcat_epoch<tag>[_o<obs>].fits``.

    The token is what ``astrometry_utils.pick_refcat`` matches on to give each
    observation its OWN catalog in a shared field directory; without one it
    falls back to the alphabetically last file for every observation alike.
    ``o`` is not doubled if the caller already passed ``o037``, and the number
    is zero-padded to three digits so ``37`` and ``037`` name one file.
    """
    if obs_token in (None, ''):
        return f'gaia_virac2_refcat_epoch{epoch_tag}.fits'
    token = str(obs_token).lstrip('oO')
    if token.isdigit():
        token = f'{int(token):03d}'
    return f'gaia_virac2_refcat_epoch{epoch_tag}_o{token}.fits'


def build_refcat_table(gaia_table, gaia_src, virac_table, epoch, radius,
                       context="", min_ref_density=MIN_REF_DENSITY_PER_SQDEG):
    """Assemble the refcat from the two query results.

    Split out of ``main`` so the VIRAC-only path and the Gaia provenance keys
    are testable without a query or a filesystem.  ``gaia_table`` may be
    ``None`` (no Gaia backend answered), in which case every VIRAC2 row is kept
    and ``NGAIA`` is 0.
    """
    dt_gaia = epoch - GAIA_EPOCH
    dt_virac = epoch - VIRAC2_EPOCH

    def _sigma_columns(tbl, ra_err_col, dec_err_col, pmra_err_col, pmdec_err_col,
                       dt_yr, label):
        """``(sigma_pos_ra, sigma_pos_dec, sigma_pm_ra, sigma_pm_dec,
        sigma_pred)`` mas / mas-yr^-1 arrays aligned to ``tbl``, or all-NaN with
        a printed note when the query result carries none of the four columns
        (e.g. an older cached query result, or a Gaia backend that changed its
        schema) -- a MISSING sigma must never be mistaken for a KNOWN-zero one,
        so it is NaN, not 0, and every consumer that gates or weights on it
        (``same_star_region_map``) already treats NaN as "unknown -- keep the
        pair, current behaviour" rather than as "perfectly known".
        """
        cols = (ra_err_col, dec_err_col, pmra_err_col, pmdec_err_col)
        if not all(c in tbl.colnames for c in cols):
            missing = [c for c in cols if c not in tbl.colnames]
            print(f"  NOTE: {label} query result is missing {missing} -- "
                  f"sigma_pred_mas will be NaN for these {len(tbl)} row(s) "
                  f"(pre-#965 refcat build, or a backend/schema change)")
            nan = np.full(len(tbl), np.nan)
            return nan, nan.copy(), nan.copy(), nan.copy(), nan.copy()
        s_pos_ra = farr(tbl[ra_err_col])
        s_pos_dec = farr(tbl[dec_err_col])
        s_pm_ra = farr(tbl[pmra_err_col])
        s_pm_dec = farr(tbl[pmdec_err_col])
        s_pred = sigma_pred_mas(s_pos_ra, s_pos_dec, s_pm_ra, s_pm_dec, dt_yr)
        return s_pos_ra, s_pos_dec, s_pm_ra, s_pm_dec, s_pred

    if gaia_table is None:
        gaia_sc = SkyCoord([], [], unit=u.deg, frame='icrs')
        gaia_mag = np.zeros(0, dtype=float)
        gaia_sigma = tuple(np.zeros(0, dtype=float) for _ in range(5))
    else:
        g = gaia_table
        gra, gdec = prop(farr(g['ra']), farr(g['dec']),
                         farr(g['pmra']), farr(g['pmdec']), dt_gaia)
        gfin = np.isfinite(gra) & np.isfinite(gdec)
        gaia_sc = SkyCoord(gra[gfin] * u.deg, gdec[gfin] * u.deg)
        gaia_mag = farr(g['phot_g_mean_mag'])[gfin]
        gaia_sigma_full = _sigma_columns(
            g, 'ra_error', 'dec_error', 'pmra_error', 'pmdec_error',
            dt_gaia, 'Gaia DR3')
        gaia_sigma = tuple(np.asarray(s)[gfin] for s in gaia_sigma_full)

    v = virac_table
    vra, vdec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']),
                     farr(v['pmRA']), farr(v['pmDE']), dt_virac)
    vfin = np.isfinite(vra) & np.isfinite(vdec)
    virac_sc = SkyCoord(vra[vfin] * u.deg, vdec[vfin] * u.deg)
    vJ = farr(v['Jmag'])[vfin]
    virac_sigma_full = _sigma_columns(
        v, 'e_RAJ2000', 'e_DEJ2000', 'e_pmRA', 'e_pmDE', dt_virac, 'VIRAC2')
    virac_sigma = tuple(np.asarray(s)[vfin] for s in virac_sigma_full)

    # Coverage floor BEFORE anything is written (issue #415 gap 4).  Counts the
    # sources that survive the finite-position masks -- the rows a tie can be
    # measured against -- not the rows the query returned.
    density = check_reference_coverage(len(gaia_sc) + int(vfin.sum()), radius,
                                      context=context,
                                      min_density=min_ref_density)
    print(f"reference coverage: {density:.3g} usable sources/deg^2 "
          f"(floor {min_ref_density:.3g})")

    if len(gaia_sc):
        idx, sep, _ = virac_sc.match_to_catalog_sky(gaia_sc)
        fill = sep > 0.3 * u.arcsec
    else:
        # match_to_catalog_sky refuses a length-0 catalog ("The catalog for
        # coordinate matching cannot be a scalar or length-0"), and with no Gaia
        # there is nothing to de-duplicate against: every VIRAC2 row is kept.
        fill = np.ones(len(virac_sc), dtype=bool)
    print(f"Gaia DR3: {len(gaia_sc)} sources; VIRAC2 fill (no Gaia <0.3\"): "
          f"{fill.sum()} of {vfin.sum()}")

    def _add_sigma_columns(tbl, sigma_tuple):
        (s_pos_ra, s_pos_dec, s_pm_ra, s_pm_dec, s_pred) = sigma_tuple
        tbl[SIGMA_POS_RA_COLUMN] = s_pos_ra
        tbl[SIGMA_POS_DEC_COLUMN] = s_pos_dec
        tbl[SIGMA_PM_RA_COLUMN] = s_pm_ra
        tbl[SIGMA_PM_DEC_COLUMN] = s_pm_dec
        tbl[SIGMA_PRED_COLUMN] = s_pred

    parts = []
    if len(gaia_sc):
        rows_gaia = Table()
        rows_gaia['RA'] = gaia_sc.ra.deg
        rows_gaia['DEC'] = gaia_sc.dec.deg
        rows_gaia['source'] = np.full(len(gaia_sc), 'GaiaDR3', dtype='U8')
        rows_gaia['refmag'] = gaia_mag
        _add_sigma_columns(rows_gaia, gaia_sigma)
        parts.append(rows_gaia)

    rows_v = Table()
    rows_v['RA'] = virac_sc.ra.deg[fill]
    rows_v['DEC'] = virac_sc.dec.deg[fill]
    rows_v['source'] = np.full(int(fill.sum()), 'VIRAC2', dtype='U8')
    rows_v['refmag'] = vJ[fill]
    _add_sigma_columns(rows_v, tuple(np.asarray(s)[fill] for s in virac_sigma))
    parts.append(rows_v)

    ref = vstack(parts) if len(parts) > 1 else parts[0]
    ref['skycoord'] = SkyCoord(ref['RA'] * u.deg, ref['DEC'] * u.deg)
    ref.meta['VERSION'] = 'gaia_dr3+virac2_fill+sigma'
    ref.meta['FRAME'] = 'Gaia DR3 (ICRS); VIRAC2 (II/387) tied to Gaia DR3 ~5 mas'
    ref.meta['EPOCH'] = epoch
    ref.meta['V2EPOCH'] = VIRAC2_EPOCH
    ref.meta['GAEPOCH'] = GAIA_EPOCH
    ref.meta['REFDENS'] = density
    ref.meta['SIGMACOL'] = (f'{SIGMA_PRED_COLUMN} = sqrt(sigma_pos^2 + '
                            f'(dt*sigma_pm)^2) per axis, RA/Dec combined in '
                            f'quadrature; dt = epoch - source epoch (VIRAC2 '
                            f'{VIRAC2_EPOCH}, Gaia {GAIA_EPOCH}); NaN where the '
                            f'query result carried no error columns')
    # Provenance for the Gaia leg.  Without these a capped file was
    # indistinguishable from a good one except by its round row count -- which is
    # why the o134/o135 cap had to be INFERRED rather than read (issue #856), and
    # why sgrb2's 2026-06-18 refcat carried it unnoticed for three months.
    ref.meta['GAIASRC'] = gaia_src
    ref.meta['NGAIA'] = int(len(gaia_sc))
    ref.meta['NVIRAC'] = int(fill.sum())
    ref.meta['NOTE'] = (f'GC reference-frame policy: Gaia DR3 abs frame + VIRAC2 NIR fill, per-star '
                        f'PM-propagated (VIRAC2 from {VIRAC2_EPOCH}, Gaia from {GAIA_EPOCH}) to {epoch}.')
    if not len(gaia_sc):
        ref.meta['NOTE'] += (' VIRAC2-ONLY: no Gaia backend answered, so the '
                             'sparse-Gaia gross cross-check is absent.  That '
                             'check is non-gating (an unmeasurable sparse leg '
                             'sets cross_gross_ok=True), so the tie stands; '
                             'rebuild with --force to restore it.')
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='target basepath (writes <base>/catalogs/...)')
    ap.add_argument('--epoch', type=float, required=True, help='observation epoch (jyear)')
    ap.add_argument('--ra', type=float, required=True)
    ap.add_argument('--dec', type=float, required=True)
    ap.add_argument('--radius', type=float, default=0.1, help='query radius (deg)')
    ap.add_argument('--out-epoch-tag', default=None, help='epoch tag in filename, e.g. 2024.68')
    ap.add_argument('--obs-token', default=None, metavar='NNN',
                    help='observation number to stamp into the filename '
                         '(gaia_virac2_refcat_epoch<tag>_o<NNN>.fits).  REQUIRED '
                         'for a field whose observations share one directory: '
                         'pick_refcat hands an untokened catalog to every '
                         'observation, which for tiles arcminutes apart is the '
                         'wrong sky (gc2211 o023 took a -9.28" correction that '
                         'way).')
    ap.add_argument('--gaia-backends', default=None, metavar='LIST',
                    help="Gaia backends in the order to try, e.g. "
                         "'vizier,esa-tap' (the default), 'esa-tap', or 'none' "
                         "to build VIRAC2-only.  VizieR is first because it is "
                         "uncapped and bounded while the ESA TAP is neither; "
                         "see the module docstring.  Env GAIA_BACKENDS.")
    ap.add_argument('--gaia-tap-timeout', type=float, default=None,
                    metavar='SECONDS',
                    help='wall clock for ONE ESA TAP attempt (default '
                         f'{GAIA_TAP_ATTEMPT_TIMEOUT_S:g}; env GAIA_TAP_TIMEOUT_S)')
    ap.add_argument('--gaia-tap-budget', type=float, default=None,
                    metavar='SECONDS',
                    help='wall clock for the WHOLE ESA TAP phase, retries and '
                         f'back-off included (default {GAIA_TAP_BUDGET_S:g}; '
                         'env GAIA_TAP_BUDGET_S)')
    ap.add_argument('--min-ref-density', type=float,
                    default=MIN_REF_DENSITY_PER_SQDEG, metavar='PER_SQDEG',
                    help='refuse to build below this usable-source density '
                         '(deg^-2).  A BROKEN-QUERY guard: every program-10678 '
                         'tile measures 1.7e6-4.9e6 and the brick refcat 2.3e6, '
                         'so the default sits ~2x below the thinnest real sky '
                         'and two orders above a truncated query.  0 disables '
                         'it, and setting it is the record of that decision.  '
                         'It is pooled Gaia+VIRAC2 and so cannot see a Gaia '
                         'truncation -- NGAIA in the output meta is what shows '
                         'that.')
    args = ap.parse_args()
    tag = args.out_epoch_tag or f'{args.epoch:.2f}'
    out = f'{args.base}/catalogs/{refcat_filename(tag, args.obs_token)}'

    # Gaia FIRST but never fatal: a failure here costs a non-gating cross-check,
    # while a failure of query_virac2 below costs the reference itself.
    g, gaia_src = query_gaia(args.ra, args.dec, args.radius,
                             backends=args.gaia_backends,
                             attempt_timeout_s=args.gaia_tap_timeout,
                             budget_s=args.gaia_tap_budget)
    v = query_virac2(args.ra, args.dec, args.radius)

    ref = build_refcat_table(
        g, gaia_src, v, args.epoch, args.radius,
        context=f"{args.base} epoch {args.epoch} at ({args.ra}, {args.dec})",
        min_ref_density=args.min_ref_density)
    ref.write(out, overwrite=True)
    ref.write(out.replace('.fits', '.ecsv'), overwrite=True)
    print(f"Wrote {out}: {len(ref)} sources ({ref.meta['NGAIA']} Gaia "
          f"[{gaia_src}] + {ref.meta['NVIRAC']} VIRAC2 fill)")


if __name__ == '__main__':
    main()
