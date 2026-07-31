"""FROZEN copy of the pre-unification ``fix_alignment`` shift dispatch.

This exists for ONE purpose: to prove that
:func:`jwst_gc_pipeline.reduction.unified_alignment.resolve_shift` returns the
same total shift the old ``if/elif`` chain did, for every configured field.  It
is exercised only by ``tests/test_unified_alignment.py``.

DO NOT EXTEND THIS FILE.  New fields go in
:mod:`jwst_gc_pipeline.reduction.alignment_config`.  When a deliberate change to
an applied shift is made, the equivalence test for that field is updated (or
dropped) in the same commit, with the reason recorded there -- that way an
intended change is visible in review and an unintended one is not.

Faithfully reproduced: the branch conditions, the table filenames, the row
narrowing, the on-sky -> coordinate RA conversions, and the constants.

Deliberately omitted (side effects that do not affect the RETURNED value, and
which the unified path keeps): the DVA pre-correction, the generation
(``genlock``) checks, the offsets-table collapse validation, and the progress
prints.
"""

import os

import numpy as np
from astropy.table import Table

__all__ = ['legacy_shift', 'LEGACY_SUPPORTED']

#: (proposal, field) combinations the frozen dispatch handles with a real tie.
#: Everything else fell through to ``else`` and got (0, 0).
LEGACY_SUPPORTED = (
    ('1182', '004'), ('2221', '001'), ('2221', '002'),
    ('2092', '002'), ('3958', '007'),
    ('1979', None), ('1334', None), ('4147', None), ('6151', None),
)


def legacy_shift(fn, proposal_id, field, filtername, basepath,
                 refname=None, use_average=True):
    """Return ``(dra_arcsec, ddec_arcsec)`` exactly as the old dispatch did."""
    proposal_id = str(proposal_id)

    if ((field == '004' and proposal_id == '1182')
            or (field in ('001', '002') and proposal_id == '2221')):
        exposure = int(fn.split("_")[-3])
        thismodule = fn.split("_")[-2]
        visit = fn.split("_")[0]
        locked_tbl = (f'{basepath}/offsets/'
                      f'Offsets_JWST_Brick{proposal_id}_VIRAC2locked.csv')
        if os.path.exists(locked_tbl):
            offsets_tbl = Table.read(locked_tbl)
            match = ((offsets_tbl['Visit'] == visit)
                     & (offsets_tbl['Filter'] == filtername))
            if match.sum() > 1 and 'Exposure' in offsets_tbl.colnames:
                match = match & (offsets_tbl['Exposure'] == exposure)
            if match.sum() > 1 and 'Module' in offsets_tbl.colnames:
                match = match & ((offsets_tbl['Module'] == thismodule)
                                 | (offsets_tbl['Module'] == thismodule.strip('1234')))
            if match.sum() != 1:
                raise ValueError(
                    f"module-locked offset match={match.sum()} for {fn} "
                    f"(visit={visit}, exposure={exposure}, filter={filtername}); "
                    f"expected exactly 1 row in {locked_tbl}")
            row = offsets_tbl[match]
        elif use_average:
            if refname is None or 'bug' in refname.lower():
                raise ValueError("This is a disallowed reference file")
            tblfn = (f'{basepath}/offsets/'
                     f'Offsets_JWST_Brick{proposal_id}_{refname}_average.csv')
            offsets_tbl = Table.read(tblfn)
            match = (((offsets_tbl['Module'] == thismodule)
                      | (offsets_tbl['Module'] == thismodule.strip('1234')))
                     & (offsets_tbl['Filter'] == filtername))
            if 'Visit' in offsets_tbl.colnames:
                match &= (offsets_tbl['Visit'] == visit)
            row = offsets_tbl[match]
            if match.sum() != 1:
                raise ValueError(f"too many or too few matches for {fn} "
                                 f"(match.sum() = {match.sum()}).")
        else:
            if refname is None or 'bug' in refname.lower():
                raise ValueError("This is a disallowed reference file")
            tblfn = (f'{basepath}/offsets/'
                     f'Offsets_JWST_Brick{proposal_id}_{refname}.csv')
            offsets_tbl = Table.read(tblfn)
            match = ((offsets_tbl['Visit'] == visit)
                     & (offsets_tbl['Exposure'] == exposure)
                     & ((offsets_tbl['Module'] == thismodule)
                        | (offsets_tbl['Module'] == thismodule.strip('1234')))
                     & (offsets_tbl['Filter'] == filtername))
            row = offsets_tbl[match]
            if match.sum() != 1:
                raise ValueError(f"too many or too few matches for {fn} "
                                 f"(match.sum() = {match.sum()}).")
        return float(row['dra (arcsec)'][0]), float(row['ddec (arcsec)'][0])

    # NOTE: the original chain had an `elif (field == '002' and proposal_id ==
    # '2221')` branch here carrying hardcoded per-visit shifts (visit 001:
    # dDec +7.95", dRA +0.6"; visit 002: +3.85", +1.57") plus a short-wavelength
    # nrca term (+0.1", -0.23") for F212N/F187N/F182M.  It was UNREACHABLE -- the
    # first branch above already matches (2221, '002') -- so it never ran and is
    # not reproduced.  test_dead_2221_002_branch_is_unreachable pins that.

    if field == '002' and proposal_id == '2092':
        visit = fn.split('_')[0][-3:]
        if visit == '002':
            return 0.098, -0.171
        return 0.0, 0.0

    if field == '007' and proposal_id == '3958':
        gns = {'F187N': (-89.7, -34.2), 'F210M': (-88.5, -34.5),
               'F335M': (-89.5, -33.2), 'F470N': (-91.4, -33.9),
               'F480M': (-90.6, -33.1)}
        cdra, cddec = gns.get(filtername.upper(), (-90.0, -34.0))
        cosd = np.cos(np.radians(-28.805))
        return cdra / 1000.0 / cosd, cddec / 1000.0

    if proposal_id in ('1979', '1334'):
        gaia_tie = {
            ('jw01979002001', 'F150W2'): (104.7, -180.3),
            ('jw01979002001', 'F322W2'): (-442.9, -87.9),
            ('jw01979003001', 'F150W2'): (-2189.0, 370.7),
            ('jw01979003001', 'F322W2'): (-1914.7, 546.9),
            ('jw01334001001', 'F090W'): (-1832.1, -708.2),
            ('jw01334001001', 'F150W'): (-1853.5, -710.6),
            ('jw01334001001', 'F277W'): (-1852.1, -711.7),
            ('jw01334001001', 'F444W'): (-1852.7, -710.7),
        }
        key = (fn.split('_')[0], filtername.upper())
        cdra, cddec = gaia_tie.get(key, (0.0, 0.0))
        cosd = np.cos(np.radians(-26.427 if proposal_id == '1979' else 43.139))
        return cdra / 1000.0 / cosd, cddec / 1000.0

    if proposal_id in ('4147', '6151'):
        return _legacy_consensus(fn, basepath, proposal_id, filtername)

    return 0.0, 0.0


def _legacy_consensus(fn, basepath, proposal_id, filtername):
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        lookup_consensus_offset,
    )
    tblfn = (f'{basepath}/offsets/'
             f'Offsets_JWST_Brick{proposal_id}_consensus.csv')
    if not os.path.exists(tblfn):
        return 0.0, 0.0
    tbl = Table.read(tblfn)
    visit = fn.split('_')[0]
    exposure = int(fn.split('_')[-3])
    thismodule = fn.split('_')[-2]
    return lookup_consensus_offset(tbl, visit, exposure, thismodule, filtername)
