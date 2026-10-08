"""Write m2's module-common per-exposure offsets into a module-locked offsets table.

The m2 checkpoint records every exposure x detector catalog's offset from its
(visit, filter) consensus, but it writes to the offsets table only the tail: the
mean of a row's MISALIGNED detectors (> 2 mas and significant), and only when
that mean reaches the field's ``m2_correction_floors`` floor (4 mas on
sgrb2/brick/sgra, 8 mas on sgrc/cloudc).  Every offset below those lines stays
in the frames, the mosaics and the catalogs.  ``m2_registration`` (opt-in)
removes it from the merged catalogs at m3+; this module puts the part a
module-locked table can carry into the table, so the next regeneration bakes it
into the frames and the mosaics as well.

What is written
---------------
One correction per offsets-table row: the MEAN over ALL of the row's detectors
of m2's recorded vs-consensus offset (an SW row is the four detectors of one
module; an LW row is one detector).  The members go through
``update_offsets_table(..., pool=True)``, so the pooling, the spread refusal and
the magnitude ceilings are the ones the m2 APPLY path uses.  The difference is
the membership: m2 pools only the misaligned detectors, which is the mean of a
selected subset.  On the live records the two differ by a median of 0.54 mas
(max 1.24) on sgrb2 F212N and 0.95 mas (max 1.13) on sgrc F212N, over the rows
holding at least one misaligned detector.

The per-detector remainder is not written: a module-family row cannot carry it
(#697).  Measured on the live m2 records (per-axis rms of the recorded offsets
-> of what remains once each row's mean is removed): sgrb2 F212N 1.51 -> 0.50
mas, sgrc F212N 1.11 -> 0.57, brick F212N 0.62 -> 0.41.  On LW rows the
remainder is zero by construction (sgrb2 F480M 0.89 -> 0).

Why one write converges
-----------------------
The row mean is a property of the frames' WCS, so it repeats and it follows the
table.  sgrb2 F212N: two m2 runs on frames whose baked offsets differ by a 47.8
mas global re-bake agree per exposure to 0.07 mas (median).  sgrc F480M: across
re-reductions whose rows changed by up to 11 mas, the recorded offsets moved by
the table change to within 0.05-0.24 mas (median).  After a regeneration the
written part reads ~0, the per-detector remainder is unchanged, and a second
pass writes nothing above ``min_mas``.

Guards (the row is skipped and the reason reported)
---------------------------------------------------
* **m2_actionable**: m2's own floor rule sends this row to the APPLY path
  (``m2_registration._m2_actionable_keys``).  Writing it here as well would add
  it twice if both ran on the same frames.
* **incomplete**: a detector the record holds for this (visit, module family,
  filter) is refused, ambiguous or absent in this exposure.  The mean of a
  subset carries the missing detector's per-detector term (up to ~0.9 mas).
* **stale**: the baked RAOFFSET/DEOFFSET m2 recorded for the row's detectors
  differ from each other, or from the row ``fix_alignment`` reads
  (``unified_alignment.locked_row_match``, ``dra (arcsec)``/``ddec (arcsec)``).
  The frames were built from another table, so adding the measured offset to
  the row would not put them where m2 measured they belong.  This is also why a
  second write-back from the same record is a no-op: after the first, the rows
  no longer match the record.
* **row_mismatch**: the reader (``locked_row_match``) and the writer
  (``_match_rows``) do not resolve the row's detectors to one and the same row.
* **below_min**: the row mean is under ``min_mas`` (default 0.5 mas).  After the
  per-detector term, what stays per exposure is ~0.3 mas per axis (sgrb2 / sgrc
  / brick F212N: 0.28 / 0.32 / 0.26), so the mean of four carries ~0.15 mas per
  axis; 0.5 mas is ~2.5 sigma on the magnitude.

  The threshold sits below m2's own floor (4 or 8 mas) because the two
  thresholds govern different quantities.  m2's floor is set for the
  per-detector distortion term, which a module-locked row cannot express:
  applying the misaligned detectors' mean of that term leaves the term in the
  frames, and the re-tie loop does not converge (ASTROMETRY_CHECKPOINTS.md,
  "The m2 correction floor").  The all-member row mean is the module-common
  part.  The row expresses it exactly and it repeats across runs (above), so
  its threshold follows its measurement noise.

Whole-record refusals: a stage other than m2, ``passed`` not True, or a record
correction whose visit token disagrees with the proposal/observation given.

Reaching the frames
-------------------
``fix_alignment`` treats a frame whose baked offset is within
``RAOFFSET_DISAGREE_TOL_ARCSEC`` (default 0.05" = 50 mas) of its row as current
(``unified_alignment.check_alignment_stale``), so mas-scale rows do not reach
existing frames through the realign-delta path.  They reach them through a
regeneration from ``_cal``, or a ``fix_alignment`` rerun with that tolerance set
below the smallest written change (``WritebackPlan.realign_tol_arcsec``).  The
im0 mosaics are not stale-tagged: nothing in them changed, and they stay as good
as the m2 verdict they passed.
"""
from dataclasses import dataclass, field

import numpy as np

#: ``prov_stage`` of the rows this writes, and the backup suffix.
WRITEBACK_STAGE = "m2-writeback"
#: Smallest row mean written, on-sky mas (see the module docstring).
DEFAULT_MIN_MAS = 0.5
#: The column pair ``fix_alignment`` reads from a module-locked table.
TABLE_RA_COL = "dra (arcsec)"
TABLE_DEC_COL = "ddec (arcsec)"

STATUSES = ("write", "below_min", "m2_actionable", "incomplete", "stale",
            "row_mismatch")


class M2WritebackError(RuntimeError):
    """The record cannot be written back at all."""


@dataclass
class WritebackRow:
    """One offsets-table row's verdict."""
    visit: str
    exposure: int
    family: str
    filtername: str
    vgroup: object
    detectors: tuple
    dra_mas: float = float("nan")
    ddec_mas: float = float("nan")
    status: str = ""
    reason: str = ""
    table_row: int = -1
    baked_ra: float = float("nan")
    baked_dec: float = float("nan")

    @property
    def mag_mas(self):
        return float(np.hypot(self.dra_mas, self.ddec_mas))


@dataclass
class WritebackPlan:
    """What ``apply_writeback`` would write, and why every other row is left."""
    record_path: str
    filtername: object
    min_mas: float
    dec_deg: float
    rows: list = field(default_factory=list)
    corrections: list = field(default_factory=list)

    def counts(self):
        return {s: sum(r.status == s for r in self.rows) for s in STATUSES}

    def written(self):
        return [r for r in self.rows if r.status == "write"]

    def realign_tol_arcsec(self):
        """A ``RAOFFSET_DISAGREE_TOL_ARCSEC`` under which every written row is
        seen as stale by ``fix_alignment``, or ``None`` when nothing is written.

        ``check_alignment_stale`` compares the bulk and the jitter component
        separately, with a strict ``>``.  A row's change splits between them,
        so the larger of the two is at least half of it; 0.4 x the smallest
        per-axis coordinate change keeps every written row above the line."""
        rows = self.written()
        if not rows:
            return None
        cosd = np.cos(np.radians(self.dec_deg))
        change = [max(abs(r.dra_mas / cosd), abs(r.ddec_mas)) / 1000.0
                  for r in rows]
        return 0.4 * min(change)


def full_visit_token(proposal_id, observation, visit):
    """``jw<proposal><obs><visit>`` for a bare record visit number."""
    from .astrometry_checkpoint import assert_visit_token
    try:
        tok = f"jw{int(proposal_id):05d}{int(observation):03d}{int(visit):03d}"
    except ValueError as ex:
        raise M2WritebackError(
            f"proposal {proposal_id!r} / observation {observation!r} / visit "
            f"{visit!r} do not form one JWST visit id; a joint "
            f"multi-observation record cannot be written back -- write each "
            f"observation's own record") from ex
    return assert_visit_token(tok, f"m2 write-back {proposal_id} obs {observation}")


def _row_of(key):
    from .astrometry_checkpoint import _module_family
    return (key[0], key[1], _module_family(key[2]), key[3],
            key[4] if len(key) > 4 else None)


def _reader_mask(tbl, row, detector):
    """The rows ``fix_alignment`` reads for one detector of ``row``."""
    from ..reduction.unified_alignment import locked_row_match
    return locked_row_match(tbl, visit=row.visit, exposure=row.exposure,
                            filtername=row.filtername, module=detector,
                            vgroup=row.vgroup)


def _check_record_visits(record, tokens, record_path):
    """A correction the record wrote with a full visit id must name a token
    built here, or the proposal/observation given is not the record's.  Bare
    visit numbers (untokened legacy records: cloudc ``'1'``/``'2'``) carry no
    observation and cannot contradict it."""
    from .astrometry_checkpoint import VISIT_TOKEN_RE
    built = set(tokens.values())
    for corr in record.get("corrections") or []:
        visit = str(corr.get("visit"))
        if VISIT_TOKEN_RE.match(visit) and visit not in built:
            raise M2WritebackError(
                f"{record_path or 'record'}: correction visit "
                f"{visit!r} is not among the visits built from the "
                f"proposal/observation given ({sorted(built)})")


def writeback_plan(record, tbl, proposal_id, observation, dec_deg,
                   min_mas=DEFAULT_MIN_MAS, record_path="", filtername=None):
    """Decide, per offsets-table row, what the m2 record lets this write.

    ``record`` is an m2 checkpoint record dict, ``tbl`` the module-locked
    offsets table, ``dec_deg`` the declination for the cos(dec) conversion.
    """
    from .astrometry_checkpoint import _match_rows
    from .m2_registration import (BAKED_OFFSET_MATCH_ARCSEC,
                                  registration_from_record)

    if record.get("passed") is not True:
        raise M2WritebackError(
            f"{record_path or 'record'}: passed={record.get('passed')!r}; only a "
            f"passed m2 record is written back")
    missing = [c for c in ("Visit", "Filter", TABLE_RA_COL, TABLE_DEC_COL)
               if c not in tbl.colnames]
    if missing:
        raise M2WritebackError(f"offsets table lacks {missing}; a module-locked "
                               f"table carries {TABLE_RA_COL!r}/{TABLE_DEC_COL!r}")
    if not np.isfinite(dec_deg):
        raise M2WritebackError(f"dec_deg={dec_deg!r} is not finite")
    # env={}: an operator's merge-time ceiling is not a write-back setting.
    reg = registration_from_record(record, record_path=record_path,
                                   filtername=filtername, env={})
    plan = WritebackPlan(record_path=record_path, filtername=filtername,
                         min_mas=float(min_mas), dec_deg=float(dec_deg))

    every_key = set(reg.entries) | set(reg.refused) | set(reg.ambiguous)
    tokens = {k[0]: full_visit_token(proposal_id, observation, k[0])
              for k in every_key}
    _check_record_visits(record, tokens, record_path)

    known = {}
    rows = {}
    for key in every_key:
        visit, _, fam, filt, _ = _row_of(key)
        known.setdefault((visit, fam, filt), set()).add(key[2])
        rows.setdefault(_row_of(key), [])
    for key, ent in reg.entries.items():
        rows[_row_of(key)].append((key, ent))

    # The record's date, not its filename: `_latest.json` is overwritten by the
    # next m2 run, so only the date still names this measurement later.
    source = f"{WRITEBACK_STAGE} module-common (m2 {reg.record_date or '?'})"
    for rkey in sorted(rows, key=lambda r: tuple(str(x) for x in r)):
        visit, exposure, fam, filt, vgroup = rkey
        members = sorted(rows[rkey], key=lambda m: m[0][2])
        dets = tuple(m[0][2] for m in members)
        row = WritebackRow(visit=tokens[visit], exposure=int(exposure),
                           family=fam, filtername=filt, vgroup=vgroup,
                           detectors=dets)
        if members:
            row.dra_mas = float(np.mean([m[1]["dra"] for m in members]))
            row.ddec_mas = float(np.mean([m[1]["ddec"] for m in members]))
        plan.rows.append(row)
        corrs = [dict(visit=row.visit, exposure=row.exposure, module=key[2],
                      filtername=filt, vgroup=vgroup,
                      dra_onsky_mas=ent["dra"], ddec_onsky_mas=ent["ddec"],
                      dec_deg=float(dec_deg), source=source)
                 for key, ent in members]

        if any(m[0] in reg.actionable for m in members):
            row.status = "m2_actionable"
            row.reason = ("m2's floor rule sends this row to the APPLY path "
                          f"(floor {reg.floor_mas:g} mas)")
            continue
        expected = known[(visit, fam, filt)]
        if set(dets) != expected:
            row.status = "incomplete"
            row.reason = (f"no certified offset for "
                          f"{sorted(expected - set(dets))} in this exposure")
            continue
        baked = np.array([(m[1]["raoffset_meta"], m[1]["deoffset_meta"])
                          for m in members])
        if np.ptp(baked, axis=0).max() > BAKED_OFFSET_MATCH_ARCSEC:
            row.status = "stale"
            row.reason = "the row's detectors carry different baked offsets"
            continue
        # Each detector must resolve to ONE row, the same row, for the reader
        # (what fix_alignment bakes) and for the writer (what the update edits).
        read, wrote = set(), set()
        for c in corrs:
            mask = _reader_mask(tbl, row, c["module"])
            read.add(tuple(int(i) for i in np.flatnonzero(mask)))
            wrote.add(tuple(sorted(int(i) for i in _match_rows(c, tbl))))
        if len(read) != 1 or read != wrote or len(next(iter(read))) != 1:
            row.status = "row_mismatch"
            row.reason = f"reader rows {sorted(read)}, writer rows {sorted(wrote)}"
            continue
        row.table_row = next(iter(read))[0]
        row.baked_ra, row.baked_dec = float(baked[0, 0]), float(baked[0, 1])
        t_ra = float(tbl[TABLE_RA_COL][row.table_row])
        t_dec = float(tbl[TABLE_DEC_COL][row.table_row])
        if (abs(t_ra - row.baked_ra) > BAKED_OFFSET_MATCH_ARCSEC
                or abs(t_dec - row.baked_dec) > BAKED_OFFSET_MATCH_ARCSEC):
            row.status = "stale"
            row.reason = (f"frames baked ({baked[0, 0]:+.6f},{baked[0, 1]:+.6f})\" "
                          f"but the table row reads ({t_ra:+.6f},{t_dec:+.6f})\"")
            continue
        if row.mag_mas < min_mas:
            row.status = "below_min"
            continue
        row.status = "write"
        plan.corrections.extend(corrs)
    return plan


def apply_writeback(offsets_path, plan):
    """Write ``plan``'s rows to ``offsets_path`` (the writer keeps a backup).

    Every written row is re-read first and must still hold the baked offset the
    plan verified; a table changed since the plan was made is refused rather
    than added to.  Returns the corrected table, or ``None`` when the plan
    writes nothing.
    """
    from astropy.table import Table
    from .astrometry_checkpoint import update_offsets_table
    from .m2_registration import BAKED_OFFSET_MATCH_ARCSEC

    if not plan.corrections:
        return None
    now = Table.read(offsets_path)
    moved = []
    for row in plan.written():
        idx = np.flatnonzero(_reader_mask(now, row, row.detectors[0]))
        if (len(idx) != 1
                or abs(float(now[TABLE_RA_COL][idx[0]]) - row.baked_ra)
                > BAKED_OFFSET_MATCH_ARCSEC
                or abs(float(now[TABLE_DEC_COL][idx[0]]) - row.baked_dec)
                > BAKED_OFFSET_MATCH_ARCSEC):
            moved.append(f"{row.visit} exp {row.exposure} {row.family} "
                         f"{row.filtername}")
    if moved:
        raise M2WritebackError(
            f"{offsets_path}: {len(moved)} row(s) changed since the plan was "
            f"made ({moved[:3]}{'...' if len(moved) > 3 else ''}); NOT writing. "
            f"Re-plan from the current table.")
    return update_offsets_table(offsets_path, plan.corrections, WRITEBACK_STAGE,
                                pool=True)
