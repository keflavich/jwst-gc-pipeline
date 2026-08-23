"""Canonical MAST product-name construction: the JWST filename prefix.

Named ``mast_names`` rather than ``naming`` because
``jwst_gc_pipeline/photometry/naming.py`` already exists with a neighbouring
purpose (per-frame filename tokens); two modules called ``naming`` one package
level apart read as the same module at an import site.

MAST zero-pads the proposal number to FIVE digits in every product name:
proposal 2221 ships as ``jw02221...`` and proposal 10678 as ``jw10678...``.
Every proposal whose data this pipeline has REDUCED has four digits, so ~96
call sites spelled the prefix as the literal ``f'jw0{proposal_id}'`` --
byte-identical to the padded form for a 4-digit proposal and wrong for a
5-digit one (``jw010678`` where MAST writes ``jw10678``): the MAST URI filter
selects zero uncals, every product glob matches nothing, and the m2 visit
token fails its own ``^jw\\d{11}$`` validator (issue #414).  Build the prefix
here instead; a grep-guard test (``test_no_jw0_prefix_literals.py``)
refuses any new ``jw0{`` literal.

Two 5-digit proposals are in scope, and the helper changes the prefix for
both: the GC Treasury program 10678 (first delivery 2026-08-16) and omegacen's
12587 (GO-8322/12587, still EXCLUSIVE_ACCESS, so no products on disk).  For
every proposal with data on disk today -- all 4-digit -- the prefix stays
byte-identical to the old spelling.

The same 4-digit assumption also lived in the reduction path as a HEADER
slice, ``header['PROGRAM'][1:5]``, which drops the fifth digit of a 5-digit
proposal; ``proposal_id_from_program`` below replaces it.

Vocabulary used above.  A JWST *visit token* is ``jw`` + proposal(5) +
observation(3) + visit(3), the 11 digits ``^jw\\d{11}$`` counts
(``jw02221001001``).  An *uncal* is the raw ramp product MAST serves, the
reduce's stage-1 input.  The *MAST URI filter* is the substring test the
reduce applies to each ``dataURI`` a MAST query returns, to keep only this
proposal-and-observation's uncals.  *m2* is the second merge stage of
cataloging, where per-exposure astrometry is re-verified.

The module body is stdlib-only, though importing it executes
``jwst_gc_pipeline/__init__.py``, which pulls numpy and astropy for the
provenance hook (the JWST stack is not among them).  A script that must run
with no package on the path spells the five-digit pad itself rather than
importing this: ``scripts/reduction/preflight_reduce_inputs.py`` does, and its
test asserts the two agree.
"""
import os
import re

__all__ = ["jw_prefix", "proposal_id_from_filename", "proposal_id_from_program",
           "proposal_id_from_datamodel"]

#: The proposal number is the first five digits after ``jw`` in every product
#: basename -- 4-digit proposals are stored zero-padded on disk, so one parse
#: covers both generations.
_JW_BASENAME_RE = re.compile(r"^jw(\d{5})")

#: Accepted proposal spellings: one to five decimal digits, leading zeros
#: allowed.  Everything else -- a sign, a thousands separator, an underscore
#: digit group, surrounding whitespace, a non-ASCII digit -- is refused before
#: ``int()`` gets to be lenient about it.
_PROPOSAL_RE = re.compile(r"\A[0-9]{1,5}\Z")


def jw_prefix(proposal_id):
    """``'jw02221'`` / ``'jw10678'`` -- the MAST filename prefix of a proposal.

    Zero-pads to five digits exactly as MAST does.  For a 4-digit proposal --
    every proposal with products on disk today -- this reproduces the old
    ``f'jw0{proposal_id}'`` spelling byte for byte (``jw_prefix(2221) ==
    'jw02221'``).  It diverges from that spelling in the two places the
    spelling was wrong: a 5-digit proposal (``jw10678``, not ``jw010678``) and
    a proposal below 1000 (``jw00618``, not ``jw0618``).

    Accepts an int or a string of one to five decimal digits, leading zeros
    allowed.  Raises ``ValueError`` on anything else, including the shapes
    ``int()`` alone would have accepted and silently normalised: a sign
    (``'+2221'``), an underscore digit group (``'2_221'``), surrounding
    whitespace, a non-ASCII digit, zero, a negative number, or a number wider
    than the five digits every JWST filename and visit token assumes.
    """
    text = proposal_id if isinstance(proposal_id, str) else str(proposal_id)
    if _PROPOSAL_RE.match(text) is None:
        raise ValueError(
            f"proposal_id {proposal_id!r} is not a JWST proposal number: "
            f"expected one to five decimal digits")
    pid = int(text)
    if pid == 0:
        raise ValueError(f"proposal_id {proposal_id!r} is not a real proposal")
    return f'jw{pid:05d}'


def proposal_id_from_program(program):
    """Proposal number read from a JWST ``PROGRAM`` header keyword.

    ``PROGRAM`` carries the same five-character zero-padded form MAST puts in
    the filename -- ``'02221'`` for proposal 2221, ``'10678'`` for the GC
    Treasury -- so the pipeline's unpadded key comes from stripping that pad.
    Three reduction sites took the fixed slice ``header['PROGRAM'][1:5]``
    instead, which drops the fifth digit and reads ``'0678'`` off a 10678
    frame, silently: ``destreak.add_background_map`` warns that the filter has
    no background map and returns the frame unchanged, and
    ``saturated_star_finding.get_psf`` looks for a merged PSF grid nobody
    wrote and falls back to a detector-specific one.

    Raises ``ValueError`` for a value that is not a proposal number.
    """
    # jw_prefix does the validating; strip its 'jw' back off to get the
    # unpadded key the pipeline passes around.
    return str(int(jw_prefix(str(program).strip())[2:]))


#: Filter/pupil wheel entries that hold no bandpass.  NIRCam spells the empty
#: slot ``CLEAR``, NIRISS spells its empty pupil ``CLEARP``.
_EMPTY_WHEEL = ('CLEAR', 'CLEARP')


def filtername_from_header(header):
    """The science bandpass a JWST ``FILTER``/``PUPIL`` header pair names.

    JWST spreads one bandpass over two wheels, and which wheel holds it is not
    fixed.  NIRCam parks its narrow and medium pupil-wheel bands behind a wide
    filter-wheel band -- ``FILTER='F444W', PUPIL='F405N'`` is an F405N
    exposure, and ``FILTER='F150W2', PUPIL='F162M'`` is an F162M one -- while a
    filter-wheel band flies with an empty pupil, ``FILTER='F212N',
    PUPIL='CLEAR'``.  NIRISS inverts the convention: its pupil-wheel bands read
    ``FILTER=<band>, PUPIL='CLEARP'``.  So neither keyword alone is the answer,
    and reading ``PUPIL`` first is wrong exactly as often as reading ``FILTER``
    first.

    Rule: whichever wheel is not empty holds the band; when both are real, the
    pupil wheel wins, because that is where NIRCam's narrow/medium bands live
    and the filter wheel is then only the blocking element.

    This is the pipeline's single spelling of that rule.
    ``reduction.filtering.get_filtername`` delegates here.  The reason it lives
    in ``mast_names`` rather than in ``filtering`` is import weight:
    ``filtering`` pulls in photutils, astroquery and pylab, which the reduction
    hot path (``reduction.destreak``) must not drag in to answer a question
    about two header keywords.

    ``reduction.destreak.add_background_map`` carries a second copy, which this
    does not yet replace -- that is PR #465, which is stacked on this one::

        filtername = hdu[0].header['PUPIL']
        if filtername in ('CLEAR', 'F444W') and hdu[0].header['FILTER'] in (
                'F405N', 'F466N', 'F410M', 'F212N', 'F187N', 'F182M'):
            filtername = hdu[0].header['FILTER']

    -- correct only for the six narrow/medium bands hardcoded in that tuple,
    which are brick's and Cloud C's.  Every CLEAR-pupil WIDE band falls through
    it and resolves to the literal ``'CLEAR'``: F115W, F200W, F356W and F444W
    all look up the background-map key ``'clear'``.

    That is one of TWO independent reasons the ``f200w``/``f356w``/``f444w``
    entries of ``destreak.background_mapping`` are unreachable, and for the
    frames actually on disk it is not the operative one.  Those three entries
    sit under the proposal-``'2221'`` key while naming proposal-1182 maps, and
    every wide-band brick frame on disk is ``jw01182`` -- a proposal that has
    no key in that mapping at all -- so the proposal lookup misses first and
    the filter key is never consulted.  (The files they name were also renamed
    ``.fits_stale`` in 2023.)  The filter rule is the lock that would still be
    shut after someone fixed the proposal key; it is also the lock that would
    silently swallow any NEW wide-band or Cloud E/F map (``f210m``, ``f360m``,
    ``f480m``), which is the case this fix is really for.

    Returns the band name stripped and upper-cased.  Real FITS headers are
    already in that form, so this is a no-op on every frame on disk and on
    both existing callers, but a hand-built ``{'FILTER': 'f212n'}`` now comes
    back ``'F212N'`` rather than verbatim.

    Raises ``ValueError`` when both wheels are empty or the keywords are
    missing, which no real science exposure has.
    """
    filtername = str(header.get('FILTER', '') or '').strip().upper()
    pupil = str(header.get('PUPIL', '') or '').strip().upper()

    if pupil and pupil not in _EMPTY_WHEEL:
        # A real pupil-wheel band. NIRCam narrow/medium behind a wide blocker,
        # or a filter-wheel band whose FILTER slot reads CLEAR.
        filtername = pupil
    if not filtername or filtername in _EMPTY_WHEEL:
        raise ValueError(
            f"header names no science bandpass: FILTER={header.get('FILTER')!r}, "
            f"PUPIL={header.get('PUPIL')!r}")
    return filtername


def proposal_id_from_filename(filename):
    """Proposal number parsed from a JWST product filename or path.

    ``jw02221001001_..._cal.fits`` -> ``'2221'`` and
    ``jw10678001001_..._cal.fits`` -> ``'10678'``: the first five digits after
    ``jw`` are the proposal on both the zero-padded 4-digit products and the
    5-digit ones.  The slice this replaces, ``basename(fn)[3:7]``, assumed the
    pad and read ``'0678'`` off a 10678 product.  Returns the unpadded string
    form the pipeline passes around, matching the old slice exactly on 4-digit
    products.  Raises ``ValueError`` for a basename with no ``jw<5-digit>``
    prefix, where the slice silently returned garbage.
    """
    base = os.path.basename(filename)
    match = _JW_BASENAME_RE.match(base)
    if match is None:
        raise ValueError(
            f"{base!r} does not start with a 'jw<5-digit proposal>' prefix; "
            f"cannot infer its proposal id")
    return str(int(match.group(1)))


def proposal_id_from_datamodel(model, filename=None):
    """Proposal number of an open JWST datamodel, read from its own header.

    ``model.meta.observation.program_number`` is the ``PROGRAM`` keyword, and
    it travels INSIDE the file: a frame that was renamed, copied into another
    field's tree, or written under a hand-built name still reports the
    proposal it was observed under, where ``proposal_id_from_filename``
    reports whatever the new name says.  That is the whole reason to prefer
    this over the filename parse (issue #440).

    ``filename`` is the fallback source, used only when the header carries no
    ``PROGRAM`` -- a hand-built model, or a product assembled outside the
    JWST calibration pipeline.  Passing ``None`` makes a header-less model an
    error instead.

    Returns the unpadded string form the pipeline passes around, so
    ``PROGRAM='02221'`` -> ``'2221'`` and ``PROGRAM='10678'`` -> ``'10678'``;
    a caller comparing against the literal ``'2221'`` keeps working.  Raises
    ``ValueError`` when neither source names a proposal.
    """
    meta = getattr(model, 'meta', None)
    observation = getattr(meta, 'observation', None)
    program = getattr(observation, 'program_number', None)
    if program is not None and str(program).strip():
        return proposal_id_from_program(program)
    if filename is None:
        raise ValueError(
            "datamodel carries no observation.program_number (PROGRAM) and no "
            "filename was given to fall back on; cannot infer its proposal id")
    return proposal_id_from_filename(filename)
