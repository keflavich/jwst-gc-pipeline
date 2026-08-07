"""Sanity checks for the per-visit offsets table that ``fix_alignment`` consumes.

The operative ``Offsets_JWST_Brick<pid>_VIRAC2locked.csv`` is ad-hoc/hand-curated
(no clean single builder; the .bak trail crosstie/2stage/PREF410MSPLIT/pervisit/
v001fix). That curation once **collapsed brick-1182 visit-001 onto visit-002's value**
(both ~+1.9" for a visit truly ~20" off) -- every independent MEASUREMENT table had
v001 = -17.5" correct, but the collapsed locked table is the one that got applied, so
half the mosaic stayed ~20" off.

The tell of that failure: two DISTINCT visits of the same filter carrying (near-)
identical offsets. Independent per-visit pointing errors do not agree to a few mas by
chance -- especially not across every filter at once. This module flags that pattern
so a collapsed table cannot be silently applied again.

This is a cheap STATIC check (no re-measurement). The strong dynamic check -- remeasure
each row vs VIRAC2 with the window-swept helper -- lives in the standalone validator
(brick/analysis/verify_v001_fix_independent.py) and in
``photometry.astrometry_offsets.measure_offset``.
"""
import numpy as np


def flag_collapsed_visits(offsets_tbl, tol_arcsec=0.02):
    """Flag filters whose distinct visits carry (near-)identical offsets.

    Parameters
    ----------
    offsets_tbl : astropy Table
        Must have ``Visit``, ``Filter``, and ``dra (arcsec)`` / ``ddec (arcsec)``
        (or ``dra`` / ``ddec``) columns.
    tol_arcsec : float
        Two visits whose offsets agree within this are treated as suspiciously
        identical (default 0.02" = 20 mas; real per-visit pointing errors differ by
        much more).

    Returns
    -------
    list of dict
        One entry per suspicious (filter, visit-pair): ``dict(filter, visit_a,
        visit_b, sep_arcsec, dra, ddec)``.  Empty list = clean.
    """
    cols = offsets_tbl.colnames
    if 'Visit' not in cols or 'Filter' not in cols:
        return []
    dc = 'dra (arcsec)' if 'dra (arcsec)' in cols else ('dra' if 'dra' in cols else None)
    ec = 'ddec (arcsec)' if 'ddec (arcsec)' in cols else ('ddec' if 'ddec' in cols else None)
    if dc is None or ec is None:
        return []

    vis = np.asarray(offsets_tbl['Visit'])
    filt = np.asarray(offsets_tbl['Filter'])
    dra = np.asarray(offsets_tbl[dc], dtype=float)
    ddec = np.asarray(offsets_tbl[ec], dtype=float)

    issues = []
    for f in np.unique(filt):
        fm = filt == f
        visits = np.unique(vis[fm])
        # per-visit mean offset (tables may be per-exposure)
        vals = {}
        for v in visits:
            vm = fm & (vis == v)
            vals[v] = (float(np.nanmedian(dra[vm])), float(np.nanmedian(ddec[vm])))
        vlist = list(vals)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                a, b = vlist[i], vlist[j]
                sep = float(np.hypot(vals[a][0] - vals[b][0], vals[a][1] - vals[b][1]))
                if sep <= tol_arcsec:
                    issues.append(dict(filter=str(f), visit_a=str(a), visit_b=str(b),
                                       sep_arcsec=sep, dra=vals[a][0], ddec=vals[a][1]))
    return issues


class CollapsedOffsetsTableError(RuntimeError):
    """Raised when an offsets table has visits collapsed onto one value."""


class DivergedColumnPairError(ValueError):
    """The two (dra, ddec) copies disagree by something no writer accounts for."""


#: Float round-trip through CSV, and nothing physical.  Measured across all ten
#: live locked tables the reconstruction agrees to <0.1 mas.
PAIR_PROV_TOL_MAS = 0.5

#: The apply loop converts on-sky mas to a Delta-alpha offset by dividing by
#: cos(dec), and dec is not stored per row -- so the RA-axis gap is bounded
#: rather than exact.  Every field here is Galactic Centre or nearer the
#: equator, so |dec| < 30 deg.
_COS_DEC_MIN = 0.8660254037844387


def flag_diverged_column_pairs(offsets_tbl, tol_mas=PAIR_PROV_TOL_MAS):
    """Rows whose two column pairs differ by something ``prov_*`` cannot explain.

    A locked table carries ``dra``/``ddec`` AND ``dra (arcsec)``/``ddec
    (arcsec)``, and they are NOT two copies of one number -- that reading cost
    a re-reduction (#319).  The bare pair is the AS-BUILT value the offsets
    builder wrote; the ``(arcsec)`` pair is as-built PLUS every correction the
    m2 checkpoint has since accumulated, which is exactly what
    ``prov_dra_added_mas`` / ``prov_ddec_added_mas`` record.  So a gap is
    normal and its SIZE is not the signal:

        gc2211  o023 F277W exp1   ddec gap 14986.2 mas   prov_ddec 14986.2 mas
                                  dra  gap -7163.5 mas   prov_dra  -6269.7 mas
                                                         / cos(28.9 deg) = -7164.1

    Measured over all ten live locked tables (1164 rows): 678 diverge, **0** of
    them
    unexplained.  What this flags is the invariant BREAKING -- a writer that
    updates one pair and not the other, which is the mechanism the issue
    suspected and the only way the two can ever mean different things.

    A row is therefore legitimately in exactly one of TWO states, and both pass:

      * ``gap == 0`` -- the pairs are in sync.  A row starts here (the builder
        ends with ``t['dra (arcsec)'] = t['dra']``) and RETURNS here the moment
        ``update_offsets_table`` touches it, because that writer HEALS an
        explained gap into the plain pair before applying the new correction.
        ``prov_*`` keeps accumulating past the heal, so after a heal the gap is
        0 while ``prov_*`` is nonzero -- checking ``gap == prov`` alone would
        flag every corrected row on every table (it did: 8 tests, CI red).
      * ``gap == prov`` -- never healed, so the plain pair is still the as-built
        value and the whole correction history sits in the gap.

    There is no legitimate third state: an unhealed row that receives a
    correction is healed first, so it cannot accumulate provenance while
    keeping a stale gap.  A gap that is neither 0 nor the recorded provenance
    means one pair moved by an amount the other never got.

    Returns a list of offending rows; empty means the invariant holds.
    """
    import numpy as np
    cn = set(offsets_tbl.colnames)
    if not {"dra", "ddec", "dra (arcsec)", "ddec (arcsec)"} <= cn:
        return []                      # single-pair table: nothing to check
    d_gap = (np.asarray(offsets_tbl["dra (arcsec)"], dtype=float)
             - np.asarray(offsets_tbl["dra"], dtype=float)) * 1000.0
    c_gap = (np.asarray(offsets_tbl["ddec (arcsec)"], dtype=float)
             - np.asarray(offsets_tbl["ddec"], dtype=float)) * 1000.0
    prov_d = (np.nan_to_num(np.asarray(offsets_tbl["prov_dra_added_mas"],
                                       dtype=float))
              if "prov_dra_added_mas" in cn else np.zeros(len(offsets_tbl)))
    prov_c = (np.nan_to_num(np.asarray(offsets_tbl["prov_ddec_added_mas"],
                                       dtype=float))
              if "prov_ddec_added_mas" in cn else np.zeros(len(offsets_tbl)))
    # state 1: the pairs are in sync (freshly built, or healed by a correction)
    in_sync = (np.abs(d_gap) <= tol_mas) & (np.abs(c_gap) <= tol_mas)
    # state 2: never healed -- the gap IS the accumulated provenance
    dec_ok = np.abs(c_gap - prov_c) <= tol_mas
    lo = np.minimum(prov_d, prov_d / _COS_DEC_MIN) - tol_mas
    hi = np.maximum(prov_d, prov_d / _COS_DEC_MIN) + tol_mas
    ra_ok = (d_gap >= lo) & (d_gap <= hi)
    bad = ~(in_sync | (dec_ok & ra_ok))
    out = []
    for i in np.flatnonzero(bad):
        out.append(dict(
            row=int(i),
            visit=str(offsets_tbl["Visit"][i]) if "Visit" in cn else "?",
            filter=str(offsets_tbl["Filter"][i]) if "Filter" in cn else "?",
            exposure=(int(offsets_tbl["Exposure"][i])
                      if "Exposure" in cn else -1),
            dra_gap_mas=float(d_gap[i]), prov_dra_mas=float(prov_d[i]),
            ddec_gap_mas=float(c_gap[i]), prov_ddec_mas=float(prov_c[i])))
    return out


def assert_offsets_table_sane(offsets_tbl, tol_arcsec=0.02, context="",
                              raise_on_issue=False, raise_on_diverged=False):
    """Warn (or raise) if ``offsets_tbl`` shows the visit-collapse signature.

    Returns the issue list (empty = clean).  Set ``raise_on_issue`` (or env
    ``OFFSETS_TABLE_COLLAPSE_RAISE=1``) to raise instead of warn.

    The as-built/as-corrected divergence has its OWN switch
    (``raise_on_diverged`` / ``OFFSETS_TABLE_DIVERGENCE_RAISE=1``) and warns by
    default, because the two findings differ in severity: a collapse means the
    shift the reducer applies is wrong, while a divergence means only that the
    audit trail no longer says how the (still self-consistent) applied pair got
    there.  Sharing one switch made every existing ``raise_on_issue=True``
    caller stop on the weaker finding.
    """
    import os
    import warnings
    issues = flag_collapsed_visits(offsets_tbl, tol_arcsec=tol_arcsec)
    if issues:
        lines = [f"  {i['filter']}: visits {i['visit_a']} & {i['visit_b']} both "
                 f"({i['dra']:+.4f},{i['ddec']:+.4f})\" (agree to {i['sep_arcsec']*1000:.1f} mas)"
                 for i in issues]
        msg = (f"COLLAPSED OFFSETS TABLE{(' ' + context) if context else ''}: distinct "
               f"visits share (near-)identical offsets -- the brick-1182 v001 failure "
               f"signature (a visit's real offset was overwritten by another's). Do NOT "
               f"trust this table; re-measure the flagged visits with a window-swept "
               f"histogram (photometry.astrometry_offsets.measure_offset).\n" + "\n".join(lines))
        if raise_on_issue or os.environ.get('OFFSETS_TABLE_COLLAPSE_RAISE') == '1':
            raise CollapsedOffsetsTableError(msg)
        warnings.warn(msg)
    # The as-built / as-corrected invariant.  This is the SAME failure family as
    # the collapse above -- a table that reads plausibly while the pipeline
    # consumes something else -- and it is checked here because this is the one
    # function every writer and the release gate already call.
    diverged = flag_diverged_column_pairs(offsets_tbl)
    if diverged:
        lines = [f"  {d['filter']} {d['visit']} exp{d['exposure']}: "
                 f"dra gap {d['dra_gap_mas']:+.1f} mas vs prov "
                 f"{d['prov_dra_mas']:+.1f}; ddec gap {d['ddec_gap_mas']:+.1f} "
                 f"vs prov {d['prov_ddec_mas']:+.1f}" for d in diverged[:8]]
        dmsg = ("OFFSETS TABLE PROVENANCE BROKEN"
                f"{(' ' + context) if context else ''}: {len(diverged)} row(s) "
                "where `(arcsec) - plain` is NOT the accumulated `prov_*`.\n"
                + "\n".join(lines)
                + ("\n  ..." if len(diverged) > 8 else "")
                + "\n\nThe two pairs are NOT copies: `dra`/`ddec` is the "
                  "AS-BUILT value the offsets builder wrote, and `dra "
                  "(arcsec)`/`ddec (arcsec)` is as-built PLUS every correction "
                  "since -- which is what `prov_*` records (issue #319).  The "
                  "live tables bear that out: 678 of 1164 rows differ and every "
                  "one is "
                  "reconstructed to <0.1 mas.\n"
                  "A row is legitimately either in sync (gap 0 -- as built, or "
                  "healed by `update_offsets_table` before its next "
                  "correction) or never healed (gap == prov).  A gap that is "
                  "NEITHER means a writer updated one pair and not the other, "
                  "which costs the AUDIT TRAIL rather than the shift.  A "
                  "warning rather than a stop, because the reducer "
                  "reads the `(arcsec)` pair and that pair is still "
                  "self-consistent; what is lost is the ability to say how it "
                  "got there.  Re-derive with "
                  "scripts/reduction/reconcile_offsets_column_pairs.py.")
        if (raise_on_diverged
                or os.environ.get('OFFSETS_TABLE_DIVERGENCE_RAISE') == '1'):
            raise DivergedColumnPairError(dmsg)
        warnings.warn(dmsg)
        issues = list(issues) + [dict(kind="diverged_column_pair", **d)
                                 for d in diverged]
    return issues
