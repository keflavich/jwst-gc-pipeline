"""Register each per-frame catalog onto its visit consensus at the per-filter merge.

The m2 checkpoint (``astrometry_checkpoint.run_visit_checkpoint``) measures every
exposure x detector catalog against its (visit, filter) consensus with the
offset-histogram estimator and records the result in
``checkpoint_m2_<FILTER><obs>_latest.json``.  It only ACTS on the tail: an
exposure becomes a correction when it is more than
``EXPOSURE_CONSENSUS_TOL_MAS`` (2 mas) off and significant, and that correction
is then applied only above the field's ``m2_correction_floors`` floor (4 mas on
sgrb2/brick/sgra, 8 mas on sgrc/cloudc).  Everything below those lines is
recorded and left in the catalogs.

That residual is not small next to the PSF-fit precision.  Measured on sgrb2
F212N (jw05365-o001, m2 record of 2026-08-29), the recorded per-exposure
vs-consensus offsets reach 3.8 mas with formal errors of 0.02 mas, repeat within
a dither block to 0.1-0.2 mas, and are mostly common to the four detectors of a
module.  They set the across-exposure position scatter of the merged catalog:
the per-star floor is 1.35 mas as merged and 0.62 mas after adding each
catalog's recorded offset, against 0.42 mas with a full per-catalog affine.
F480M goes 0.79 -> 0.66 mas.  The excess F212N scatter is why F212N reads WORSE
than F480M in the merged ``std_ra``/``std_dec`` although its PSF is half the
size.

The per-frame catalogs are in pixel space with a sky position computed from the
frame's GWCS, so a measured registration can be applied to them after the fact
(the same reasoning ``merge_catalogs.shift_individual_catalog`` documents for
the offsets table).  This module does that: it adds m2's own certified
per-exposure offset to each per-frame catalog's sky positions before
``combine_singleframe`` averages them.

What it does NOT change
-----------------------
* The frames, the im0/i2d mosaics and the offsets table.  The mosaics keep the
  sub-floor per-exposure offsets until the frames are regenerated with a table
  that carries them.
* The checkpoints.  Every m2..m6 checkpoint measures the per-frame catalogs on
  disk, which this never writes, so the m3..m6 frozen-stage movement checks see
  exactly what they saw before.
* The absolute frame.  ``build_visit_consensus`` median-re-centres the
  consensus, so the median per-exposure offset is ~0 and the registered catalog
  sits in the consensus frame -- the frame the consensus->reference tie measured.

Guards (each one leaves the catalog as it is and says why)
-----------------------------------------------------------
* **stale**: the catalog's baked ``RAOFFSET``/``DEOFFSET`` differs from the value
  m2 recorded for that exposure.  The frame was regenerated after m2 measured
  it, so the recorded offset describes a different WCS.
* **refused**: m2 did not stand behind the number
  (``astrometry_checkpoint._m2_exposure_untrustworthy``: ``ok`` False,
  ``unverified``, ``alias_suspect``, ``alias_rejected``, or past the
  per-exposure magnitude bound), or the exposure had no internal tie.
* **m2_actionable**: m2's own decision rule classifies the offset as one it
  corrects through the offsets table and stops the run for (it belongs to a
  regeneration).  The rule is the one ``cataloging.py`` applies: the exposure is
  ``misaligned``, and the mean of the misaligned exposures sharing its
  (visit, exposure, module family, filter, vgroup) row -- what
  ``pool_corrections_to_table_granularity`` writes on the module-locked channel
  -- reaches the record's ``correction_floor_mas`` (``>=``, as
  ``_floor_actionable_corrections``).  The per-DETECTOR value is not tested
  against the floor: pooling discards the per-detector spread before the floor
  is applied (#697), so a detector can sit above the floor in a row m2 passed.
  sgrb2 F212N exposures 18-20 nrcb3 are 4.08-4.13 mas in a row that pools to
  2.97 mas under a 4 mas floor.  On the ``consensus`` channel m2 does not pool,
  so this rule is looser there than m2's; the stale guard covers that case,
  because anything m2 acts on is regenerated with a new baked offset.
* **over_cap**: optional operator ceiling, ``ASTROM_M2_REGISTRATION_MAX_MAS``
  (on-sky mas, per exposure).  Unset by default.
* **absent**: no m2 entry for this exposure key.
* **ambiguous**: two record entries carry the same exposure key with different
  values.

Opt-in: ``ASTROM_MERGE_M2_REGISTRATION=1`` turns the step on; unset (or 0)
leaves every merge as it was.  It is off by default because a registered
catalog no longer matches its ``_i2d`` mosaic at the sub-floor level (sgrb2
F212N: median 1.8, max 4.1 mas) until the same offsets are written into the
offsets table and the frames regenerated -- the place this pipeline puts the
astrometric solution so that catalogs and images share it.
"""
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

#: Environment switch, OFF unless set to 1/true/yes/on (see module docstring).
M2_REGISTRATION_ENV = "ASTROM_MERGE_M2_REGISTRATION"
#: Optional operator ceiling on a registered offset, in on-sky mas.
M2_REGISTRATION_MAX_ENV = "ASTROM_M2_REGISTRATION_MAX_MAS"
#: A catalog's baked RAOFFSET/DEOFFSET (arcsec) must match the m2 record's to
#: this tolerance.  Both are copies of one FITS card, so a real match is exact;
#: 1e-6 arcsec (0.001 mas) absorbs float formatting only.
BAKED_OFFSET_MATCH_ARCSEC = 1e-6

STATUSES = ("applied", "stale", "refused", "m2_actionable", "over_cap",
            "absent", "ambiguous")


@dataclass
class M2Registration:
    """What the m2 record lets the merge apply, per exposure key.

    ``entries`` maps an ``exposure_key`` tuple to a dict with ``dra``/``ddec``
    (the on-sky correction to ADD, mas -- m2's ``vs_consensus`` convention,
    consensus minus exposure) and ``raoffset_meta``/``deoffset_meta`` (the baked
    offsets of the catalog m2 measured, arcsec).  ``refused`` maps the keys m2
    measured but did not certify to the reason.  ``actionable`` holds the keys
    m2's floor rule assigns to the offsets table (see the module docstring).
    """
    record_path: str
    record_date: str
    record_passed: object
    floor_mas: float
    ceiling_mas: object = None
    entries: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)
    ambiguous: set = field(default_factory=set)
    actionable: set = field(default_factory=set)


_ON = ("1", "true", "yes", "on")
_OFF = ("", "0", "false", "no", "off")


def registration_enabled(env=None):
    """True only when ``ASTROM_MERGE_M2_REGISTRATION`` is 1/true/yes/on.

    Any other non-empty value raises: a typo in an opt-in switch that silently
    read as OFF would leave a run unregistered while its operator believed
    otherwise.
    """
    env = os.environ if env is None else env
    raw = str(env.get(M2_REGISTRATION_ENV, "")).strip().lower()
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    raise ValueError(f"{M2_REGISTRATION_ENV}={raw!r}: use 1/true/yes/on to "
                     f"enable or leave unset/0 to disable")


def _ceiling_mas(env):
    raw = str(env.get(M2_REGISTRATION_MAX_ENV, "")).strip()
    return float(raw) if raw else None


def _floor_mas(record):
    """The floor m2 decided with.  A record without one (pre-floor runs) was
    decided at 0, where every misaligned exposure is actionable."""
    floor = (record.get("tolerances") or {}).get("correction_floor_mas")
    return float(floor) if floor is not None and np.isfinite(floor) else 0.0


def _misaligned(entry):
    flag = entry.get("misaligned")
    if flag is not None:
        return bool(flag)
    # Legacy record without the flag: the consensus tolerance alone.
    from .visit_consensus import EXPOSURE_CONSENSUS_TOL_MAS
    return bool(np.hypot(entry["dra"], entry["ddec"]) > EXPOSURE_CONSENSUS_TOL_MAS)


def _m2_actionable_keys(certified, floor_mas):
    """Keys m2's floor rule sends to the offsets table.

    ``certified`` maps key -> record entry.  Misaligned exposures are pooled by
    (visit, exposure, module family, filter, vgroup) with the mean, as
    ``pool_corrections_to_table_granularity`` does, and a row is actionable when
    the pooled on-sky magnitude reaches the floor.
    """
    from .astrometry_checkpoint import _module_family
    rows = {}
    for key, entry in certified.items():
        if not _misaligned(entry):
            continue
        row = (key[0], key[1], _module_family(key[2])) + tuple(key[3:])
        rows.setdefault(row, []).append((key, entry["dra"], entry["ddec"]))
    out = set()
    for members in rows.values():
        pooled = np.hypot(np.mean([m[1] for m in members]),
                          np.mean([m[2] for m in members]))
        if pooled >= floor_mas:
            out.update(m[0] for m in members)
    return out


def registration_from_record(record, record_path="", filtername=None, env=None):
    """Build an ``M2Registration`` from an m2 checkpoint record dict.

    ``filtername`` restricts the entries to one filter (needed for a
    mixed-filter ``_all`` record); ``None`` keeps every entry.
    """
    from .astrometry_checkpoint import (
        MAX_CORRECTION_ARCSEC, _m2_exposure_untrustworthy, _positive_env_float)

    env = os.environ if env is None else env
    stage = str(record.get("stage", ""))
    if stage != "m2":
        raise ValueError(f"{record_path or 'record'}: stage {stage!r} is not m2; "
                         f"only the m2 record is the frozen per-exposure solution")
    limit = _positive_env_float("ASTROM_MAX_CORRECTION_ARCSEC",
                                MAX_CORRECTION_ARCSEC)
    want = str(filtername).upper() if filtername else None
    reg = M2Registration(record_path=record_path,
                         record_date=str(record.get("date", "")),
                         record_passed=record.get("passed"),
                         floor_mas=_floor_mas(record),
                         ceiling_mas=_ceiling_mas(env))
    certified = {}
    for visit in record.get("visits", []) or []:
        for entry in visit.get("exposures", []) or []:
            key = tuple(entry.get("key", []) or [])
            if len(key) < 4:
                continue
            if want is not None and str(key[3]).upper() != want:
                continue
            dra, ddec = entry.get("dra"), entry.get("ddec")
            if dra is None or ddec is None or not (np.isfinite(dra)
                                                   and np.isfinite(ddec)):
                reg.refused[key] = "m2 recorded no finite offset for this exposure"
                continue
            refusal = _m2_exposure_untrustworthy(entry, limit=limit)
            if refusal is None and entry.get("internal_tie") is False:
                reg.refused[key] = ("m2 found no internal tie for this exposure "
                                    "(its only check is the reference tie)")
                continue
            if refusal is not None:
                reg.refused[key] = refusal.reason
                continue
            value = dict(dra=float(dra), ddec=float(ddec),
                         raoffset_meta=float(entry.get("raoffset_meta") or 0.0),
                         deoffset_meta=float(entry.get("deoffset_meta") or 0.0))
            # The same key twice with IDENTICAL values is one measurement
            # recorded twice and is kept; two different values cannot both be
            # this frame's offset, so neither is applied.
            if key in reg.entries and reg.entries[key] != value:
                reg.ambiguous.add(key)
            reg.entries[key] = value
            certified[key] = dict(dra=float(dra), ddec=float(ddec),
                                  misaligned=entry.get("misaligned"))
    # A key that is certified in one entry and refused in another is the same
    # duplicate-entry case.  An ambiguous key also leaves its row's pool: its
    # value is unknown, so it cannot decide whether its siblings' row is
    # actionable.
    reg.ambiguous |= set(reg.entries) & set(reg.refused)
    for key in reg.ambiguous:
        reg.entries.pop(key, None)
        reg.refused.pop(key, None)
        certified.pop(key, None)
    reg.actionable = _m2_actionable_keys(certified, reg.floor_mas)
    return reg


def load_m2_registration(record_dir, filtername, obs_token="", env=None):
    """Load the m2 registration for ``filtername`` from ``record_dir``.

    Returns ``None`` (and says so) when the step is disabled or no m2 record
    exists -- the merge then runs exactly as before.  The record is located by
    ``astrometry_checkpoint._m2_record_path``, the reader the frozen-stage
    checkpoints use, so the merge and the checkpoints read the same file.
    """
    env = os.environ if env is None else env
    if not registration_enabled(env):
        print(f"m2 registration: off (opt-in; set {M2_REGISTRATION_ENV}=1); "
              f"per-frame catalogs are merged as written", flush=True)
        return None
    from .astrometry_checkpoint import _m2_record_path
    path = _m2_record_path(record_dir, filtername, obs_token)
    if path is None:
        print(f"m2 registration: no m2 record for {filtername}{obs_token} in "
              f"{record_dir}; per-frame catalogs are merged as written",
              flush=True)
        return None
    with open(path) as fh:
        record = json.load(fh)
    reg = registration_from_record(record, record_path=path,
                                   filtername=filtername, env=env)
    print(f"m2 registration: {os.path.basename(path)} (date {reg.record_date}, "
          f"passed={reg.record_passed}): {len(reg.entries)} certified "
          f"exposure offset(s), {len(reg.refused)} refused, "
          f"{len(reg.ambiguous)} ambiguous, {len(reg.actionable)} m2-actionable "
          f"(floor {reg.floor_mas:g} mas)"
          + (f"; ceiling {reg.ceiling_mas:g} mas" if reg.ceiling_mas is not None
             else ""), flush=True)
    return reg


#: Sky-position columns a per-frame catalog can carry.  ``combine_singleframe``
#: reads ``skycoord_centroid`` for DAO tables and ``skycoord`` for crowdsource
#: ones, and ``shift_individual_catalog`` prefers ``skycoord`` -- so every one
#: present is shifted, and the merge reads a registered position whichever it
#: picks.
SKYCOORD_COLUMNS = ("skycoord", "skycoord_centroid")


def registration_for_merge(merge_label, record_dir, filtername, proposal_id,
                           field, cutout=False, env=None):
    """The registration a per-filter merge at ``merge_label`` should apply.

    ``None`` at the correction stages (``CORRECTION_STAGES``: m1/m2/m12 are the
    measurement, and the solution is frozen from m3 on) and for cutout runs,
    which have no checkpoint record.  Otherwise ``load_m2_registration`` with
    the checkpoint's own observation token (``consensus_obs_token``), computed
    only past that gate so a correction-stage merge never builds it.  This is
    the whole decision ``cataloging.run_manual_pipeline`` makes, kept here so
    it is testable without driving that function.
    """
    from .astrometry_checkpoint import CORRECTION_STAGES
    if merge_label in CORRECTION_STAGES or cutout:
        return None
    from .consensus_catalog import consensus_obs_token
    return load_m2_registration(record_dir, filtername,
                                consensus_obs_token(proposal_id, field), env=env)


def _skycoord_colnames(tbl):
    names = [name for name in SKYCOORD_COLUMNS if name in tbl.colnames]
    if not names:
        raise KeyError(f"per-frame catalog has no skycoord/skycoord_centroid "
                       f"column ({tbl.colnames[:8]}...)")
    return names


def _stamp(tbl, status, dra=0.0, ddec=0.0, reg=None):
    tbl.meta["M2REGST"] = status
    tbl.meta["M2REGRA"] = float(dra)     # on-sky mas ADDED (0 unless applied)
    tbl.meta["M2REGDE"] = float(ddec)
    if reg is not None:
        tbl.meta["M2REGREC"] = os.path.basename(reg.record_path)


def apply_m2_registration(tables, registration, context=""):
    """Shift each per-frame catalog by its m2 vs-consensus offset, in place.

    Returns a summary dict: counts per status, the applied offsets
    (``applied_mas``: list of (key, dra, ddec)), and a ``lines`` list of the
    per-exposure reasons for everything not applied.  Every table gets
    ``M2REGST``/``M2REGRA``/``M2REGDE``/``M2REGREC`` in its meta.
    """
    from .visit_consensus import _meta_lookup, exposure_key

    counts = Counter({s: 0 for s in STATUSES})
    applied, lines = [], []
    for tbl in tables:
        key = exposure_key(tbl)
        if key in registration.ambiguous:
            counts["ambiguous"] += 1
            lines.append(f"{key}: two different m2 entries carry this key")
            _stamp(tbl, "ambiguous", reg=registration)
            continue
        if key in registration.refused:
            counts["refused"] += 1
            lines.append(f"{key}: {registration.refused[key]}")
            _stamp(tbl, "refused", reg=registration)
            continue
        ent = registration.entries.get(key)
        if ent is None:
            counts["absent"] += 1
            lines.append(f"{key}: no m2 entry")
            _stamp(tbl, "absent", reg=registration)
            continue
        ra0 = float(_meta_lookup(tbl, "RAOFFSET", default=0.0))
        de0 = float(_meta_lookup(tbl, "DEOFFSET", default=0.0))
        if (abs(ra0 - ent["raoffset_meta"]) > BAKED_OFFSET_MATCH_ARCSEC
                or abs(de0 - ent["deoffset_meta"]) > BAKED_OFFSET_MATCH_ARCSEC):
            counts["stale"] += 1
            lines.append(
                f"{key}: baked offset ({ra0:+.6f},{de0:+.6f})\" differs from the "
                f"({ent['raoffset_meta']:+.6f},{ent['deoffset_meta']:+.6f})\" m2 "
                f"measured -- the frame was regenerated after m2; re-run m2")
            _stamp(tbl, "stale", reg=registration)
            continue
        dra, ddec = ent["dra"], ent["ddec"]
        if key in registration.actionable:
            counts["m2_actionable"] += 1
            lines.append(f"{key}: ({dra:+.2f},{ddec:+.2f}) mas pools to a row at "
                         f"or over the {registration.floor_mas:g} mas m2 floor -- "
                         f"an offsets-table correction (regenerate the frames)")
            _stamp(tbl, "m2_actionable", reg=registration)
            continue
        if (registration.ceiling_mas is not None
                and np.hypot(dra, ddec) > registration.ceiling_mas):
            counts["over_cap"] += 1
            lines.append(f"{key}: ({dra:+.2f},{ddec:+.2f}) mas is over the "
                         f"{M2_REGISTRATION_MAX_ENV}={registration.ceiling_mas:g} "
                         f"mas ceiling")
            _stamp(tbl, "over_cap", reg=registration)
            continue
        for col in _skycoord_colnames(tbl):
            sc = tbl[col]
            dec = sc.dec.to(u.deg)
            cosd = np.cos(dec.to(u.rad).value)
            new_ra = sc.ra.to(u.deg) + (dra / 3.6e6 / cosd) * u.deg
            new_dec = dec + (ddec / 3.6e6) * u.deg
            tbl[col] = SkyCoord(ra=new_ra, dec=new_dec, frame=sc.frame)
        counts["applied"] += 1
        applied.append((key, dra, ddec))
        _stamp(tbl, "applied", dra, ddec, reg=registration)

    tag = f" [{context}]" if context else ""
    if applied:
        mags = np.hypot([a[1] for a in applied], [a[2] for a in applied])
        size = (f"; |offset| median {np.median(mags):.2f}, max {mags.max():.2f} "
                f"mas")
    else:
        size = ""
    print(f"m2 registration{tag}: "
          + ", ".join(f"{s} {counts[s]}" for s in STATUSES if counts[s])
          + f" of {len(tables)} per-frame catalog(s){size}", flush=True)
    for line in lines[:20]:
        print(f"  m2 registration{tag}: not applied -- {line}", flush=True)
    if len(lines) > 20:
        print(f"  m2 registration{tag}: ... and {len(lines) - 20} more not "
              f"applied", flush=True)
    return dict(counts=dict(counts), applied_mas=applied, lines=lines,
                record=registration.record_path)
