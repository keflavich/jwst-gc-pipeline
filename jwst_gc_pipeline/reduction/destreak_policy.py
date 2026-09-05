"""Which fields destreak, and therefore what the reduced frames are called.

NIRCam's stage 1 writes ``*_destreak_o<obs>_crf.fits`` where it destreaks and
``*_align_o<obs>_crf.fits`` where it does not, and stage 2's ``--each-suffix``
has to name the right one. Both stages read the policy from here, so they
cannot disagree about it.

The destreak/align lineage token is a NIRCam thing. ``PipelineMIRI`` and
``PipelineRerunNIRISS`` have no destreak step, and both name the per-exposure
frame straight off the ``_cal`` stem -- ``*_o<obs>_crf.fits``, with no lineage
token at all (``PipelineMIRI.py`` line 663, ``PipelineRerunNIRISS.py`` line
525). So the instrument, not just the field, decides the suffix.
"""
from jwst_gc_pipeline.photometry.naming import _instrument_from_filter

#: Extended emission dominates these fields and no background map exists for
#: them yet, so destreaking is off: it opens outlier_detection coverage holes.
EXTENDED_EMISSION_FIELDS = ('w51', 'sickle', 'wd2', 'ngc6334')

#: Sickle overrides that per filter. Its short-wavelength filters destreak
#: ("the streaks are worse than the destreak artifacts"); its long-wavelength
#: filters stay on the plain aligned copy.
SICKLE_SHORTWAVE_FILTERS = (
    'F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
    'F182M', 'F187N', 'F200W', 'F210M', 'F212N')


class UnknownInstrumentError(ValueError):
    """The instrument a lineage token was asked for did not resolve.

    Raised rather than answered, because both answers are wrong: guessing
    NIRCam asks a MIRI observation for a token no MIRI frame carries (#647),
    and guessing not-NIRCam drops the token from a NIRCam run and globs both
    of its co-resident lineages at once.
    """


#: The three instruments this project reduces.  ``naming._instrument_from_filter``
#: returns its ``instrument`` argument unchanged when it does not recognise it
#: (``''``, ``'nrcalong'``, a detector name), so the answer has to be checked
#: against this before it decides anything.
KNOWN_INSTRUMENTS = ('NIRCam', 'MIRI', 'NIRISS')


def _is_nircam(filtername, instrument=None):
    """Whether this (filter, instrument) is reduced by the NIRCam driver.

    Resolved through ``naming._instrument_from_filter``, which is the one place
    the project decides this: an explicit ``instrument`` wins, then the
    ``GC_INSTRUMENT_OVERRIDE`` a non-NIRCam job already exports, then the
    filter-name heuristic.  NIRISS shares its filter names with NIRCam, so a
    NIRISS run has to say so; MIRI's filter names are its own and resolve
    without being told.

    Fail-closed polarity: the lineage token is dropped only for a RESOLVED MIRI
    or NIRISS.  An instrument argument that resolves to none of the three --
    ``''``, ``'nrcalong'``, ``'NIRCAM '``, a typo -- raises
    :class:`UnknownInstrumentError` instead of falling through to "not NIRCam",
    which is what ``== 'NIRCam'`` alone did: it made the whole thing a
    whitelist whose default was to drop the token, so a caller that could not
    name its instrument silently got the untokened suffix and globbed brick's
    ``o001_crf`` and ``destreak_o001_crf`` together.
    """
    resolved = _instrument_from_filter(filtername, instrument=instrument)
    if resolved not in KNOWN_INSTRUMENTS:
        raise UnknownInstrumentError(
            f'instrument={instrument!r} (filter {filtername!r}) resolved to '
            f'{resolved!r}, which is not one of {KNOWN_INSTRUMENTS}. The '
            f'destreak/align lineage token is NIRCam-only, so this decides '
            f'which frames the run reads; pass instrument as one of '
            f"'nircam', 'miri', 'niriss'.")
    return resolved == 'NIRCam'


def destreaks(target, filtername, requested=True, instrument=None):
    """Whether stage 1 destreaks this (field, filter).

    ``requested`` is the run's own ``--no_destreak`` choice; the policy can
    only turn destreaking off.  Destreaking is a NIRCam stage-1 step, so it is
    off for every other instrument whatever the field asks for.
    """
    if not requested:
        return False
    if not _is_nircam(filtername, instrument):
        return False
    if str(target) == 'sickle':
        return str(filtername).upper() in SICKLE_SHORTWAVE_FILTERS
    return str(target) not in EXTENDED_EMISSION_FIELDS


def crf_suffix(target, filtername, obsid, requested=True, instrument=None):
    """The reduced-frame suffix stage 2 should photometer.

    ``destreak_o001_crf`` or ``align_o001_crf`` for NIRCam; a bare
    ``o001_crf`` for MIRI and NIRISS, whose drivers write no lineage token.
    Asking a MIRI observation for a ``destreak_``/``align_`` suffix globs a
    name no MIRI frame on disk carries, and cataloging then finds no frames.
    """
    if not _is_nircam(filtername, instrument):
        return f'o{obsid}_crf'
    token = 'destreak' if destreaks(target, filtername, requested,
                                    instrument) else 'align'
    return f'{token}_o{obsid}_crf'


def suffixes_by_filter(target, filternames, obsid, requested=True,
                       instrument=None):
    """``{FILTER: suffix}`` for a whole filter list.

    Sickle needs this: its short and long filters take different suffixes, so
    no single ``--each-suffix`` is right for the observation.
    """
    return {f.upper(): crf_suffix(target, f, obsid, requested, instrument)
            for f in filternames}
