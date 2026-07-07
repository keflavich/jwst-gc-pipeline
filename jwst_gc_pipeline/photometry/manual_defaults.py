"""Single source of truth for manual-pipeline tunable defaults.

Historically each knob's default lived in TWO places: the optparse
``default=`` in ``crowdsource_catalogs_long.make_parser`` and a literal
``getattr(options, name, <default>)`` fallback in ``cataloging.py`` (the
fallback fires when ``run_manual_pipeline`` is driven by a hand-built
options object, e.g. tests or ``opts_phase`` clones).  Editing one copy
silently left the other stale.  Both sides now read this dict.

This module must stay import-light (no package imports): it is imported by
both ``crowdsource_catalogs_long`` and ``cataloging``, which import each
other's helpers.
"""

MANUAL_DEFAULTS = {
    # -- fit QC (model/data-peak overshoot; NOTES_star_vs_extended_emission.md)
    'manual_overshoot_ratio': 1.2,
    'manual_overshoot_action': 'refit',
    # -- detection S/N thresholds
    'local_snr_threshold': 5.0,        # iter1 (m1) per-source local S/N
    'manual_iter2_local_snr': 3.0,     # residual / bg-subtracted passes
    # -- extended-emission vetting (_filter_extended_emission)
    'manual_ext_qfit_max': 0.2,
    'manual_ext_prom_min': -1.0,       # -1 = AUTO (3.0 on ext-emission NIRCam)
    'manual_ext_peak_over_bkg': 20.0,
    'manual_ext_local_snr_min': 5.0,
    'manual_ext_snr_high_keep': 20.0,
    'manual_ext_qfit_high_keep_max': 0.4,
    'manual_ext_qfit_recover_max': 0.2,  # == qfit_max -> recover tier NO-OP
    'manual_ext_recover_satstar_guard_arcsec': 2.0,
    'manual_ext_recover_prom_gate': True,
    'manual_ext_recover_prom_log_intercept': -0.77,
    'manual_ext_recover_prom_log_slope': 5.6,
    'manual_ext_nmatch_confirm': 0,      # OFF; star-field-only tool
    'manual_ext_nmatch_confirm_qfit_max': 0.6,
    'manual_ext_nmatch_confirm_maxpos_mas': 0.0,
    'manual_ext_nmatch_confirm_strong': 0,   # OFF; nmatch>=N keeps any qfit (+low_fit_quality flag)
    'manual_ext_low_fit_quality_qfit': 0.0,  # 0 = use nmatch qfit ceiling
    # -- per-frame residual-pass (m2..m6) daofind shape + detection-floor scale
    'manual_detect_threshold_scale': 1.0,    # scale on min-noise floor (1.0 = no-op)
    'manual_resid_roundlo': -1.0,            # loosened from -0.3 (blended companions)
    'manual_resid_roundhi': 1.0,
    'manual_resid_sharplo': 0.50,
    'manual_resid_sharphi': 1.00,
    # -- sky-clean keep tier (per-source deep-where-no-emission vetting)
    'manual_sky_clean_keep': True,
    'manual_sky_clean_max_sky_snr': 2.0,
    'manual_sky_clean_prom_min': 5.0,
    'manual_sky_clean_snr_min': 3.0,
    # -- per-frame residual-pass (m2..m6) daofind shape cuts + threshold scale
    'manual_detect_threshold_scale': 1.0,   # <1 = detect fainter (no headroom found)
    'manual_resid_roundlo': -1.0,      # loosened from -0.3 2026-07-06 (emission-safe)
    'manual_resid_roundhi': 1.0,
    'manual_resid_sharplo': 0.50,
    'manual_resid_sharphi': 1.00,
    # -- i2d residual-seed DAO shape cuts
    'manual_seed_round_max': 0.5,      # star fields: loosen to ~1.0
    'manual_seed_sharp_lo': 0.4,
    'manual_seed_sharp_hi': 1.2,
    # -- per-pass prominence reject (ext-emission NIRCam; 0 = off)
    'nircam_prom_m1': 0.0,
    'nircam_prom_m2': 0.0,
    'nircam_prom_m3plus': 0.0,
    # -- MIRI per-pass prominence reject (no CLI flag; miri_tuning schedules it)
    'miri_prominence_snr': 5.0,
    # -- emission-noise-floor detection (0 = off)
    'detect_noise_floor_box': 0,
    'detect_noise_floor_k': 5.0,
    'detect_noise_floor_i2dseed': 0,
    # -- coarse-background detection (0 = off; MIRI raw rounds use 51)
    'coarse_bg_box': 0,
    # -- grouping
    'manual_group_min_sep_fwhm': 2.0,
    # -- cross-band seed (m7, _build_crossband_seed)
    'manual_crossband_seed_min_filters': 2,
    'manual_crossband_seed_snr_min': 5.0,
    'manual_crossband_seed_qfit_max': 0.2,
    'manual_crossband_seed_max_sep_mas': 30.0,
}


def mopt(options, name):
    """``getattr(options, name)`` with the canonical MANUAL_DEFAULTS fallback.

    KeyError on an unknown ``name`` is deliberate: an option without an entry
    here must not silently invent a default.
    """
    return getattr(options, name, MANUAL_DEFAULTS[name])
