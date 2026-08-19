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

__all__ = ["jw_prefix", "proposal_id_from_filename", "proposal_id_from_program"]

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
