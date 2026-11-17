"""Guard on the rows that only the propagated S/N floor admits (issue #1025).

``manual_ext_snr_floor_propagated`` (#1016) applies the vetting S/N floor to
flux / flux_err_prop instead of flux / flux_err.  A row the per-frame floor
would have refused, and the propagated floor admits, is a *propagated-only*
row.  The rules are motivated by three subsets measured in the Brick F182M,
Sgr B2 F187N and W51 F187N replays of #1025 (docs/evidence/
faint_snr_floor_propagated), which are not the sets the rules remove.  They
are subsets of the rows the propagated floor adds over the #1018 replay, not
only of the rows the per-frame floor refuses.  (a) A union of three subsets
(``flags == 1`` with qfit > 0.2, regardless of the other routes; prominence
< 4, an unmeasured NaN prominence counted as low; qfit 0.2-0.6) has realness
0.02 / 0.31 / 0.37 against 1.30 / 1.10 / 0.76 for the other additions; the
largest part of that union (qfit 0.2-0.6, Brick 304 of 329) has no rule here.
(b) ``qfit * S/N < 3`` is a separate set (realness 0.45 / 0.38 / 0.66), on the
per-frame S/N where #1025 writes S/N_prop.  Rule 2 drops only the rows whose
sole star-like route is ``flags``; rule 3 keeps a NaN prominence.  Whether the
rules, as implemented, remove the low-realness rows is what the replay is for.

The guard acts on propagated-only rows alone.  With all three rules at very
large values and no later tier that re-admits a dropped row (multi-frame,
sky-clean, hysteresis, the satstar force-keep), the per-frame-floor catalogue
is reproduced row for row.  Each sub-rule has its own switch; with every
switch off the guard returns an all-False mask.  The guard does not run on the
MIRI path, which vets on prominence alone.
"""
import numpy as np


def prop_floor_guard_drop(prop_only, *, flags_only, prominence, qf, snr_per_frame,
                          prom_min=0.0, drop_flags=False, qfit_snr_min=0.0):
    """Mask of propagated-only rows to drop, and the per-rule counts.

    ``prop_only``: rows admitted by the propagated floor and refused by the
    per-frame floor (``keep & ~keep_per_frame``).  Rows kept by ``qfit <=
    qfit_max`` are never in it, because both floors keep them.

    ``flags_only``: rows whose only star-like route is the ``flags`` keep.

    Rules (each off at 0 / False):

    * ``drop_flags``: drop ``prop_only & flags_only``.
    * ``prom_min``: drop ``prop_only`` rows with a finite prominence below it.
      A NaN prominence (no data i2d, or within 10 px of its edge) is kept.
    * ``qfit_snr_min``: drop ``prop_only`` rows with ``qfit * S/N`` below it,
      S/N on flux / flux_err.  A non-finite product is kept.

    Returns ``(drop, counts)``.  ``counts`` holds ``prop_only`` and, per rule,
    the rows it selects (``flags``, ``prominence``, ``qfit_snr``) and
    ``dropped``, the union.  The per-rule counts overlap, so they do not add up
    to ``dropped``; ``prominence_only`` is the prominence rows the flags rule
    does not already select and ``qfit_snr_only`` the qfit * S/N rows neither
    of the other two selects, so a rule that adds nothing shows up in the log.
    ``dropped`` is the count before the later tiers, which can re-admit a row.
    """
    prop_only = np.asarray(prop_only, dtype=bool)
    n = prop_only.size
    zero = np.zeros(n, dtype=bool)
    r_flags = zero
    r_prom = zero
    r_qs = zero
    if drop_flags:
        r_flags = prop_only & np.asarray(flags_only, dtype=bool)
    if prom_min > 0:
        prominence = np.asarray(prominence, dtype=float)
        with np.errstate(invalid='ignore'):
            r_prom = (prop_only & np.isfinite(prominence)
                      & (prominence < float(prom_min)))
    if qfit_snr_min > 0:
        with np.errstate(invalid='ignore'):
            _ks = (np.asarray(qf, dtype=float)
                   * np.asarray(snr_per_frame, dtype=float))
            r_qs = prop_only & (_ks < float(qfit_snr_min))
    drop = r_flags | r_prom | r_qs
    counts = {
        'prop_only': int(prop_only.sum()),
        'flags': int(r_flags.sum()),
        'prominence': int(r_prom.sum()),
        'prominence_only': int((r_prom & ~r_flags).sum()),
        'qfit_snr': int(r_qs.sum()),
        'qfit_snr_only': int((r_qs & ~r_flags & ~r_prom).sum()),
        'dropped': int(drop.sum()),
    }
    return drop, counts
