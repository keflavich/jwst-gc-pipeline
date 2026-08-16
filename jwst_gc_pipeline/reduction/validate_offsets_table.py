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


#: The provenance record of what a correction added is an ON-SKY separation.
#: Right ascension also has a COORDINATE offset, differing by cos(declination)
#: -- ~14% at Galactic Centre declinations -- and the table's own ``dra``
#: columns hold that one, so the name has to say which is which.  Tables
#: written before the convention was in the name use ``prov_*_added_mas`` and
#: are renamed on their next correction; both spellings are read here.
_PROV_ONSKY_RA = "prov_dra_onsky_mas"
_PROV_ONSKY_DEC = "prov_ddec_onsky_mas"
#: The declination each correction's cos(dec) conversion used.
_PROV_DEC_DEG = "prov_dec_deg"


def _prov_onsky_names(colnames):
    """``(ra, dec)`` provenance column names for whichever spelling is present."""
    ra = (_PROV_ONSKY_RA if _PROV_ONSKY_RA in colnames else "prov_dra_added_mas")
    dec = (_PROV_ONSKY_DEC if _PROV_ONSKY_DEC in colnames
           else "prov_ddec_added_mas")
    return ra, dec


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
#: cos(dec).  A row that records ``prov_dec_deg`` gives that factor back exactly;
#: this is the FALLBACK bound for any row whose cell is BLANK -- which includes
#: every row migration NaN-filled, not only rows predating the column.
#: Every field here is Galactic Centre or nearer the equator, so |dec| < 30 deg.
_COS_DEC_MIN = 0.8660254037844387


def flag_diverged_column_pairs(offsets_tbl, tol_mas=PAIR_PROV_TOL_MAS):
    """Rows whose two column pairs differ by something ``prov_*`` cannot explain.

    A locked table carries ``dra``/``ddec`` AND ``dra (arcsec)``/``ddec
    (arcsec)``, and they are NOT two copies of one number -- that reading cost
    a re-reduction (#319).  The bare pair is the AS-BUILT value the offsets
    builder wrote; the ``(arcsec)`` pair is as-built PLUS every correction the
    m2 checkpoint has since accumulated, which is exactly what
    ``prov_dra_onsky_mas`` / ``prov_ddec_onsky_mas`` record (called
    ``prov_*_added_mas`` before the convention was put in the name -- see
    ``astrometry_checkpoint.migrate_prov_column_names``).  So a gap is
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
    _pra, _pdec = _prov_onsky_names(cn)
    prov_d = (np.nan_to_num(np.asarray(offsets_tbl[_pra],
                                       dtype=float))
              if _pra in cn else np.zeros(len(offsets_tbl)))
    prov_c = (np.nan_to_num(np.asarray(offsets_tbl[_pdec],
                                       dtype=float))
              if _pdec in cn else np.zeros(len(offsets_tbl)))
    # state 1: the pairs are in sync (freshly built, or healed by a correction)
    in_sync = (np.abs(d_gap) <= tol_mas) & (np.abs(c_gap) <= tol_mas)
    # state 2: never healed -- the gap IS the accumulated provenance
    dec_ok = np.abs(c_gap - prov_c) <= tol_mas
    # With the declination the conversion used recorded on the row, the
    # coordinate offset a provenance entry implies is EXACT, and right
    # ascension is checked as strictly as declination.  Without it the
    # factor is only bounded to [_COS_DEC_MIN, 1] -- a ~14% window, wide
    # enough for a corruption of exactly that size to pass.  A masked or
    # empty cell reads as ABSENT, never as declination zero.
    _known = np.zeros(len(offsets_tbl), dtype=bool)
    _cosd = np.ones(len(offsets_tbl))
    if _PROV_DEC_DEG in cn:
        _dec = np.ma.filled(np.ma.asarray(offsets_tbl[_PROV_DEC_DEG],
                                          dtype=float), np.nan)
        _cosd = np.cos(np.radians(_dec))
        _known = np.isfinite(_dec) & (np.abs(_cosd) > 1e-6)
    # milliarcseconds throughout here: prov_d and d_gap are both mas in this
    # function (unlike _heal_column_pairs, which works in arcsec).
    _exact = prov_d / np.where(_known, _cosd, 1.0)
    lo = np.where(_known, _exact - tol_mas,
                  np.minimum(prov_d, prov_d / _COS_DEC_MIN) - tol_mas)
    hi = np.where(_known, _exact + tol_mas,
                  np.maximum(prov_d, prov_d / _COS_DEC_MIN) + tol_mas)
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


class BroadcastProvenanceError(RuntimeError):
    """One correction was written to every visit instead of the one it measured."""


#: Two visits' corrections agreeing this closely are the same number, not two
#: measurements.  Real per-visit m2 corrections differ by far more: across the
#: ten live tables the per-filter spread of the on-sky RA provenance between
#: distinct visits is 1.6-970 mas wherever it was measured per visit.
BROADCAST_PROV_TOL_MAS = 0.05

#: Below this a "correction" is a no-op and agreeing about it means nothing --
#: two visits that were both never corrected both read 0.
BROADCAST_PROV_MIN_MAS = 50.0


def flag_broadcast_provenance(offsets_tbl, tol_mas=BROADCAST_PROV_TOL_MAS,
                              min_mas=BROADCAST_PROV_MIN_MAS):
    """Filters whose distinct visits carry the IDENTICAL ``prov_*`` correction.

    A correction is measured for one visit of one filter.  Physically distinct
    pointings cannot need the same one to within a fraction of a mas, so when
    they all carry it, a per-filter value was broadcast across the visits
    instead of applied to the visit it belongs to.

    This is what happened to gc2211 (#284).  Its five observations are 0.3-17.6
    arcmin apart and measurably in five different states, and every one carried
    the same pair:

        F200W  o023 o028 o046 o049 o050   prov (-2470.1, +2825.4) mas
        F277W  o023 o028 o046 o049 o050   prov (-7031.7, +15009.7) mas

    The reducer reads ``dra (arcsec)`` = as-built + ``prov_*``, so a broadcast
    correction lands on all five products while the as-built pair -- which
    reproduced an independent swept-histogram measurement of each region to
    21-56 mas in RA and 1-26 mas in Dec -- stays correct and unread.  That is
    the same shape as the brick-1182 curation collapse and the sickle #270
    revert: the value the pixels were built from is the WRONG one, and the
    surviving good value is in the column nothing reads.

    Distinct from :func:`flag_collapsed_visits`, which flags identical
    *offsets*: a table can be built per-visit correctly and then have one
    correction smeared across it, which is exactly this.

    Returns a list of dicts, one per (filter, axis-pair) group.  Empty = clean.
    """
    import numpy as np
    cn = set(offsets_tbl.colnames)
    if not {"Visit", "Filter"} <= cn:
        return []
    _pra, _pdec = _prov_onsky_names(cn)
    if not {_pra, _pdec} <= cn:
        return []
    vis = np.asarray([str(v) for v in offsets_tbl["Visit"]])
    filt = np.asarray([str(f) for f in offsets_tbl["Filter"]])
    pd_ = np.nan_to_num(np.asarray(offsets_tbl[_pra], dtype=float))
    pc_ = np.nan_to_num(np.asarray(offsets_tbl[_pdec], dtype=float))
    out = []
    for f in np.unique(filt):
        fm = filt == f
        visits = np.unique(vis[fm])
        if len(visits) < 2:
            continue
        vals = {}
        for v in visits:
            m = fm & (vis == v)
            vals[v] = (float(np.nanmedian(pd_[m])), float(np.nanmedian(pc_[m])))
        mags = [np.hypot(*p) for p in vals.values()]
        if max(mags) < min_mas:
            continue                       # nothing was corrected; agreeing is not news
        spread = max(np.hypot(a[0] - b[0], a[1] - b[1])
                     for a in vals.values() for b in vals.values())
        if spread <= tol_mas:
            first = vals[visits[0]]
            out.append(dict(filter=str(f), n_visits=int(len(visits)),
                            visits=[str(v) for v in visits],
                            prov_dra_mas=first[0], prov_ddec_mas=first[1],
                            spread_mas=float(spread)))
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
    # One correction smeared across every visit of a filter.  Unlike the
    # divergence above this DOES change the shift the reducer applies, so it
    # rides the same switch as the collapse rather than the audit-trail one.
    broadcast = flag_broadcast_provenance(offsets_tbl)
    if broadcast:
        blines = [f"  {b['filter']}: {b['n_visits']} visits all carry prov "
                  f"({b['prov_dra_mas']:+.1f},{b['prov_ddec_mas']:+.1f}) mas "
                  f"(spread {b['spread_mas']:.3f} mas) -- {', '.join(b['visits'])}"
                  for b in broadcast]
        bmsg = ("BROADCAST OFFSETS PROVENANCE"
                f"{(' ' + context) if context else ''}: a per-visit correction "
                "is identical across DISTINCT visits, so it was applied to "
                "every visit rather than the one it was measured on.\n"
                + "\n".join(blines)
                + "\n\nThe reducer reads `dra (arcsec)` = as-built + `prov_*`, "
                  "so a broadcast correction lands on every product of that "
                  "filter while the as-built pair stays correct and unread "
                  "(gc2211 #284: five observations 0.3-17.6 arcmin apart, all "
                  "carrying one pair, mosaics 0.15-22 arcsec off VIRAC2 while "
                  "`dra`/`ddec` reproduced each region to <60 mas).  Do NOT "
                  "reconcile this table -- that would copy the bad pair over "
                  "the good one.  REVERT it: "
                  "scripts/reduction/revert_broadcast_provenance.py.")
        if raise_on_issue or os.environ.get('OFFSETS_TABLE_COLLAPSE_RAISE') == '1':
            raise BroadcastProvenanceError(bmsg)
        warnings.warn(bmsg)
        issues = list(issues) + [dict(kind="broadcast_provenance", **b)
                                 for b in broadcast]
    return issues
