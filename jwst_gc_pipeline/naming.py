"""Canonical JWST filename-prefix construction.

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

The module itself is stdlib-only, so a script can defer the import to the
function that needs it and keep its own module-level imports light.  (The
import still executes ``jwst_gc_pipeline/__init__.py``, which pulls numpy and
astropy for the provenance hook; the JWST stack is not among them.)
"""
import os
import re

__all__ = ["jw_prefix", "proposal_id_from_filename"]

#: The proposal number is the first five digits after ``jw`` in every product
#: basename -- 4-digit proposals are stored zero-padded on disk, so one parse
#: covers both generations.
_JW_BASENAME_RE = re.compile(r"^jw(\d{5})")


def jw_prefix(proposal_id):
    """``'jw02221'`` / ``'jw10678'`` -- the MAST filename prefix of a proposal.

    Zero-pads to five digits exactly as MAST does.  For a 4-digit proposal --
    every proposal with products on disk today -- this reproduces the old
    ``f'jw0{proposal_id}'`` spelling byte for byte (``jw_prefix(2221) ==
    'jw02221'``).  It diverges from that spelling in the two places the
    spelling was wrong: a 5-digit proposal (``jw10678``, not ``jw010678``) and
    a proposal below 1000 (``jw00618``, not ``jw0618``).

    Accepts an int or a numeric string (with or without leading zeros).
    Raises ``ValueError`` on anything else: a non-numeric value, a negative
    number, or a number wider than the five digits every JWST filename and
    visit token assumes.
    """
    try:
        pid = int(str(proposal_id).strip())
    except ValueError as err:
        raise ValueError(
            f"proposal_id {proposal_id!r} is not a JWST proposal number"
        ) from err
    if pid < 0:
        raise ValueError(f"proposal_id {proposal_id!r} is negative")
    if pid > 99999:
        raise ValueError(
            f"proposal_id {proposal_id!r} has more than five digits; JWST "
            f"filenames and visit tokens (^jw\\d{{11}}$) assume at most five")
    return f'jw{pid:05d}'


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
