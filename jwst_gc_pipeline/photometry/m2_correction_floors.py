"""Per-field m2 correction floor.

The m2 visit-consensus checkpoint corrects and STOPS when an exposure sits more
than ``EXPOSURE_CONSENSUS_TOL_MAS`` (2 mas) from its visit consensus.  Some
fields have a per-exposure scatter of that same size, from SIAF/DVA-class
systematics the module-locked offsets tables cannot express.  For those, the
2 mas tolerance is below the noise: the tail of the field's own scatter reads as
misalignment, the run corrects the detector MEANS -- which is what the previous
cycle already did -- and stops.  The next cycle measures the same scatter and
stops again.  It does not converge.

``ASTROM_M2_CORRECTION_FLOOR_MAS`` exists for exactly that, and its default of 0
is what keeps biting: the floor lives only in an operator's memory, so any run
submitted without it fails.  Measured cost on brick 1182:

    job 37614271   2026-08-xx   F115W, 35 corrections, floor unset
    job 39884095   2026-08-22   F115W, floor unset -- consensus scatter
                                2.274 mas, all 96 exposure offsets in
                                0.19-2.65 mas with no outlier, 21 flagged
                                "misaligned" for exceeding 2 mas

Both times the applied correction was ~1 mas and the whole chain died at m12,
taking m3-m7 with it as DependencyNeverSatisfied.

So the floor becomes a per-FIELD property, derived from that field's own
measured scatter and recorded here with the issue that set it.  The environment
variable still wins when set, because an operator overriding deliberately is a
different act from a default nobody remembered.

This does NOT weaken the checkpoint.  Corrections are always measured, always
recorded, and always written to the record; the floor governs only the
stop/apply decision, and only for residuals in the class the table cannot
express.  The consensus->reference tie is exempt from the floor entirely
(``REFERENCE_TIE_SOURCE_SUFFIX`` / ``_is_whole_consensus_shift``) -- a rigid
whole-visit shift IS expressible, so it is always actionable no matter how
small.  A field absent from the table keeps the strict 2 mas behaviour.
"""
import os
import warnings

#: env var an operator sets to override the per-field default
FLOOR_ENV = 'ASTROM_M2_CORRECTION_FLOOR_MAS'


class UnregisteredM2FloorWarning(UserWarning):
    """A field is running on an env-var floor with no entry of its own.

    That is the state every retroactively-added entry was in first, and each
    one cost a stopped chain to discover: sgra (#494, m12 stopped twice),
    w51 (#508), arches (#512), gc2211_o028 (#533).  In each case the field had
    been run with ``ASTROM_M2_CORRECTION_FLOOR_MAS`` set by hand for weeks, the
    records said so in ``correction_floor_source``, and nothing surfaced it
    until a run was submitted without the variable and died at m12.

    So the warning fires at the moment the override is USED, which is the
    moment the evidence for an entry exists and the operator is present to act
    on it -- rather than at the next run, when neither is true.
    """


def _warn_unregistered(target, floor):
    warnings.warn(
        f"m2 correction floor for {target!r} is {floor} mas from "
        f"${FLOOR_ENV}, and {target!r} has NO entry in PER_FIELD_FLOOR_MAS. "
        f"The next run submitted without that variable gets the strict 0.0 "
        f"default and stops at m12 on this field's own per-exposure scatter "
        f"(sgra, w51, arches and gc2211_o028 each paid a stopped chain to "
        f"learn this). Register the field in "
        f"jwst_gc_pipeline/photometry/m2_correction_floors.py with the "
        f"measured distribution beside it, or record there why it is "
        f"deliberately absent.",
        UnregisteredM2FloorWarning, stacklevel=3)

#: Per-field floor in mas, keyed by target.  Each entry is a measurement of THAT
#: field's per-exposure scatter, not a preference -- raise one only with the
#: scatter that justifies it.
#:
#: 8.0 fields do not reach a fixed point at 4.0: cloudc emitted 40 corrections
#: while measuring 0.00 mas pass-to-pass (see `cloudc-retie-fixed-point-floor8`).
PER_FIELD_FLOOR_MAS = {
    'brick': 4.0,               # F115W scatter 2.27 mas; jobs 37614271, 39884095
    'sgrb2': 4.0,
    'sickle': 4.0,
    'cloudef': 4.0,
    'cloudef_controlfield': 4.0,   # same program + optics as cloudef (2092)
    'cloudc': 8.0,              # never converges at 4.0
    'sgrc': 8.0,                # issue #261
    # sgra F115W: consensus scatter 1.84 mas over 96 measurements, and its m2
    # emitted a single 2.08 mas correction -- credible (contrast 976, rank
    # 77/96) but the same size as the field's own scatter, so it is not a
    # displacement the module-locked table should express.  It stopped the m12
    # finalize twice (jobs 39933168, 39972201) after a full regeneration had
    # already been run for it.  Same shape as brick, whose scatter is 2.27.
    'sgra': 4.0,
    # w51: set from ALL of its filters, not the first one that tripped.
    #
    # 4.0 was originally chosen from F140M alone (max correction 3.64 mas).  The
    # very next filter, F162M, exceeded it at 5.42 -- because the m2 checkpoint
    # stops at the FIRST filter with an actionable correction, so only one
    # filter had ever been measured.  A WARN_ONLY measurement pass (job
    # 40016510) recorded all of them:
    #
    #     filter  scatter   n   p90    max   ncorr
    #     F182M    1.00    64   1.58   1.98    0
    #     F187N    1.05    64   1.07   1.94    0
    #     F210M    0.95    64   1.34   1.98    0
    #     F335M    1.07    16   0.60   1.22    0
    #     F360M    1.04    16   0.10   1.14    0
    #     F405N    1.30    16   1.73   1.74    0
    #     F410M    1.04    16   0.56   1.33    0
    #     F480M    1.29    16   0.57   1.64    0
    #     F140M    1.40    64   1.56   3.64    5
    #     F162M    1.58    64   3.49   5.42   11
    #     F444W    1.29    16   8.74   9.58   15
    #
    # EIGHT of eleven filters are clean -- no corrections at all, residuals under
    # 2 mas.  The per-detector systematic class tops out at 5.42 (F162M nrcb1
    # exposures 1-4; F140M nrcb2 exposures 1-4).  6.0 covers that class and
    # nothing else.
    #
    # It deliberately does NOT cover F444W, whose 15-of-16 displaced exposures at
    # p90 8.74 are a different animal: too large for the SIAF/DVA class, present
    # on nearly every exposure rather than a few detectors, and NOT a module
    # split (`module_antisymmetry` detected=False, A-B = -1.6 mas on the current
    # record).  F444W should keep stopping the run until it is understood; a
    # floor of 10 would have hidden it, which is the argument against choosing
    # the floor to make a field green.
    'w51': 6.0,
    # arches: every m2 record it has ever written carries
    # `correction_floor_mas: 4.0` from the env var, so the field has only ever
    # run with an operator remembering to set it.  Measured from the records
    # themselves (2026-08-23, `astrometry_checkpoints/*_latest.json`), each
    # correction as hypot(dra_onsky_mas, ddec_onsky_mas):
    #
    #     record         filter  scatter   n   min    med    max
    #     2026-08-22     F212N    1.23    38   2.00   2.94   3.93
    #     2026-08-22     F323N    1.25    13   2.20   3.44   3.78
    #     2026-08-01     F212N    1.18    43   2.00   2.99   3.91
    #     2026-08-01     F323N    1.53    13   2.52   3.34   4.22
    #
    # 107 corrections, all but one under 4 mas, against a consensus scatter of
    # 1.2-1.5 -- the brick shape (scatter 2.27, corrections to 3.39), one band
    # tighter.  The two 2026-08-22 records read `passed: true` with
    # `correction_floor_source: env`; the same measurements at the 0.0 default
    # are 51 actionable corrections and an m12 stop.
    #
    # 4.0 deliberately does NOT cover the single 4.22 mas F323N correction of
    # 2026-08-01.  Choosing 5.0 to swallow it would be picking a constant to
    # keep a field green, which is the habit the w51 entry above argues against;
    # if it recurs it should stop the run and be looked at.
    'arches': 4.0,
    # gc2211 o046 and o050: the two pointings whose corrections are entirely the
    # per-exposure scatter class.  Same records, same measurement:
    #
    #     field        filter  scatter    n   min    med    max
    #     gc2211_o046  F200W    0.70     31   2.01   2.53   3.13
    #     gc2211_o046  F277W    6.91     14   2.17   2.56   3.52
    #     gc2211_o050  F200W    0.68     13   2.00   2.31   2.84
    #     gc2211_o050  F277W    0.76      0     --     --     --
    #
    # 58 corrections, every one under 4 mas.  Both were run with the env var set;
    # at the default they stop at m12 the way brick did twice.
    #
    # The other three 2211 pointings were NOT registered, because their measured
    # corrections were not this class and a floor would have been a guess about
    # them rather than a measurement of them: o023 median 118 mas (its four
    # exposures are trailed -- excluded outright in #493), o028 median 28 mas
    # with a coherent ~200 mas exposure-2 displacement on seven of eight
    # detectors, and o049 only three corrections, 5.4-22.4 mas, from the
    # joint-lowest-contrast cells in its whole record.  See #484.
    'gc2211_o046': 4.0,
    'gc2211_o050': 4.0,
    # o028 NOW QUALIFIES, and the paragraph above is exactly why it did not
    # before.  That "~200 mas exposure-2 displacement" was never on the sky: m2
    # wrote it into the offsets table on 2026-07-23 and the frames were
    # regenerated carrying it.  The tell was F277W -- LW is simultaneous with SW,
    # its rows were never corrected, and it put exposure 2 within 2.9 mas of
    # exposure 3, which no real pointing error can do.  The 36 affected rows were
    # reverted to their pre-m2 values (identical across all eight 2026-07-22/23
    # backups) and F150W was regenerated from _cal.
    #
    # Re-measured afterwards with APPLY=0, both filters:
    #
    #     field        filter  scatter    n   min    med    max
    #     gc2211_o028  F150W    0.86      0     --     --     --
    #     gc2211_o028  F277W    2.60     20   2.19   2.70   4.33
    #
    # F150W emits NOTHING where it previously emitted 192 corrections at a median
    # of 28.47 mas: those were the pipeline chasing its own earlier write.  What
    # is left is F277W's ordinary per-exposure scatter, the same class as the two
    # entries above.
    #
    # 6.0 rather than 5.0 -- the measured maximum is 4.33, and 5.0 clears it by
    # only 0.67 mas, inside the run-to-run variation these distributions show.
    # It also matches the w51 entry, set at 6.0 for a worst filter of 5.42.  The
    # siblings stay at 4.0 because their maxima are 3.1-3.5; o028's scatter is
    # genuinely a little wider, so it takes its own entry rather than theirs.
    'gc2211_o028': 6.0,
    # quintuplet is deliberately absent as well, and the reason is narrower than
    # "no measured distribution" -- that wording sent the next reader looking
    # for records that are there.
    #
    # Its FOUR `*_latest.json` records (2026-08-01 and 2026-08-15, F212N and
    # F323N) hold ZERO corrections at a consensus scatter of 1.16-1.32 mas.  But
    # it has 37 records in total, and the other 33 are not empty: 88 corrections,
    # 26 of them (30%) in the 2-4 mas band -- the band that is actionable at the
    # strict default and suppressed by a 4.0 floor -- and they are the
    # nrcb1-nrcb4 per-detector class against a 1.21-1.26 mas scatter.  That is
    # the arches shape, on the same instrument and the same two filters.
    #
    # What is true is that quintuplet is absent-and-FINE rather than
    # absent-and-broken.  Its 2026-07-30/31 records hold real 120 mas and 232 mas
    # displacements; those were applied, and the field has read zero corrections
    # for 21 consecutive records over three weeks since.  So submitting it today
    # without the env var does NOT stop the chain, which is not the case for
    # arches (51 corrections behind a `correction_floor_source: env` pass).
    #
    # It takes an entry when it next measures the 2-4 mas class -- which a re-tie
    # or a regeneration from _cal would put back, since that is where those
    # earlier records came from.
    #
    # o049 now has the measurement the paragraph above said it lacked.  The claim
    # there -- "only three corrections, 5.4-22.4 mas", not the scatter class --
    # was true of every record it had, and every one of those was taken with
    # `correction_floor_source: env` at 4.0.  A 4 mas floor cannot report the
    # sub-4 mas population, so the absence of that class was a property of the
    # measurement, not of the field.
    #
    # 2026-08-27 it ran at the default 0.0 (unintentionally -- there was no entry
    # here, and the relaunch did not set the env var), which is the first look at
    # o049 below 4 mas:
    #
    #     field        filter  n   min    med    max
    #     gc2211_o049  F200W   8   2.03   2.26   23.21
    #
    # Split by class, the three known corrections reproduce almost exactly --
    # 23.21 / 13.19 / 5.46 against 22.40 / 13.42 / 5.45 measured on 2026-08-22 --
    # and the five new ones are 2.03, 2.08, 2.22, 2.24, 2.29.  Four of those five
    # are nrca3 exposure 2 in four different vgroups reading +1.74..+1.93 dRA and
    # +1.08..+1.41 dDec, i.e. one systematic per-detector term, not four
    # independent pointing errors.  That is the o046/o050 shape (2.01-3.13 and
    # 2.00-2.84) on the same instrument and the same filter.
    #
    # 4.0, matching the siblings rather than taking its own entry: the measured
    # maximum of the scatter class is 2.29 and the smallest real correction is
    # 5.46, so the two populations are separated by more than 3 mas and the
    # constant is not being chosen to keep anything green.  The three real
    # corrections stay actionable.
    'gc2211_o049': 4.0,
}


def m2_correction_floor(target, env=None):
    """Effective floor in mas for ``target``, and where it came from.

    Returns ``(floor_mas, source)`` where source is ``'env'``, ``'per-field'``
    or ``'default'``.  The source is returned rather than inferred by the
    caller so the checkpoint record can say which it was: a pass that passed
    because the floor was RAISED BY HAND has to be distinguishable from one
    that passed at the field's standing floor.

    Using the env var on a field with NO entry emits
    ``UnregisteredM2FloorWarning``: that combination is precisely the state
    every retroactively-added entry was in, and the warning is the only moment
    at which both the evidence and the operator are present.  It does not
    change the returned floor -- the override still wins.
    """
    env = os.environ if env is None else env
    raw = env.get(FLOOR_ENV)
    if raw not in (None, ''):
        floor = float(raw)
        if target not in PER_FIELD_FLOOR_MAS and floor > 0:
            _warn_unregistered(target, floor)
        return floor, 'env'
    if target in PER_FIELD_FLOOR_MAS:
        return float(PER_FIELD_FLOOR_MAS[target]), 'per-field'
    return 0.0, 'default'
