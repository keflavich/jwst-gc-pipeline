"""Which fields destreak, and therefore what the reduced frames are called.

Stage 1 writes ``*_destreak_o<obs>_crf.fits`` where it destreaks and
``*_align_o<obs>_crf.fits`` where it does not, and stage 2's ``--each-suffix``
has to name the right one. Both stages read the policy from here, so they
cannot disagree about it.
"""

#: Extended emission dominates these fields and no background map exists for
#: them yet, so destreaking is off: it opens outlier_detection coverage holes.
EXTENDED_EMISSION_FIELDS = ('w51', 'sickle', 'wd2', 'ngc6334')

#: Sickle overrides that per filter. Its short-wavelength filters destreak
#: ("the streaks are worse than the destreak artifacts"); its long-wavelength
#: filters stay on the plain aligned copy.
SICKLE_SHORTWAVE_FILTERS = (
    'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
    'F182M', 'F187N', 'F200W', 'F210M', 'F212N')


def destreaks(target, filtername, requested=True):
    """Whether stage 1 destreaks this (field, filter).

    ``requested`` is the run's own ``--no_destreak`` choice; the policy can
    only turn destreaking off.
    """
    if not requested:
        return False
    if str(target) == 'sickle':
        return str(filtername).upper() in SICKLE_SHORTWAVE_FILTERS
    return str(target) not in EXTENDED_EMISSION_FIELDS


def crf_suffix(target, filtername, obsid, requested=True):
    """The reduced-frame suffix stage 2 should photometer.

    ``destreak_o001_crf`` or ``align_o001_crf``.
    """
    token = 'destreak' if destreaks(target, filtername, requested) else 'align'
    return f'{token}_o{obsid}_crf'


def suffixes_by_filter(target, filternames, obsid, requested=True):
    """``{FILTER: suffix}`` for a whole filter list.

    Sickle needs this: its short and long filters take different suffixes, so
    no single ``--each-suffix`` is right for the observation.
    """
    return {f.upper(): crf_suffix(target, f, obsid, requested)
            for f in filternames}
