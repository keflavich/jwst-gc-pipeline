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

#: env var an operator sets to override the per-field default
FLOOR_ENV = 'ASTROM_M2_CORRECTION_FLOOR_MAS'

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
}


def m2_correction_floor(target, env=None):
    """Effective floor in mas for ``target``, and where it came from.

    Returns ``(floor_mas, source)`` where source is ``'env'``, ``'per-field'``
    or ``'default'``.  The source is returned rather than inferred by the
    caller so the checkpoint record can say which it was: a pass that passed
    because the floor was RAISED BY HAND has to be distinguishable from one
    that passed at the field's standing floor.
    """
    env = os.environ if env is None else env
    raw = env.get(FLOOR_ENV)
    if raw not in (None, ''):
        return float(raw), 'env'
    if target in PER_FIELD_FLOOR_MAS:
        return float(PER_FIELD_FLOOR_MAS[target]), 'per-field'
    return 0.0, 'default'
