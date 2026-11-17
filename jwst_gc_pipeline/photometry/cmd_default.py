"""Write the default-CMD-selection flag onto merged catalogs (#1283).

The m7 cross-band merge writes ``{table}_qualcuts_oksep<token>.fits``, a row
subset.  m8 (forced fill) and m8_dedup never pass through that merge, so no
oksep subset exists for them, and a joint catalogue built from the m7 subsets
drops every F212N-only source whose F480M value comes from the forced fit.

Instead of a second row subset, the oksep test and the independent-detection
requirement are recorded as two boolean columns on the full table:

* ``oksep_pass``: more than one non-wide band has ``sep_<band>`` below 0.1".
* ``cmd_default``: ``oksep_pass`` AND ``n_filt_independent >= 2``.

The ``CMDDEF`` header card names the test that was applied.  The tests live in
``merge_catalogs.compute_cmd_default``; this module is the writer.  The columns
are recomputed on every call, so a rewrite of the table (a dedup rerun, a
changed ``n_filt_independent``) is followed by one more call and never needs
the old values.
"""
import os
import uuid

from astropy.io import fits
from astropy.table import Table

OKSEP_COL = 'oksep_pass'
CMD_DEFAULT_COL = 'cmd_default'
CMDDEF_CARD = 'CMDDEF'


_STRUCTURAL_KEYS = ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND')


def _write_keeping_headers(table, path, primary_header, extra_hdus):
    """Replace ``path`` with ``table`` plus the original PRIMARY header and extra HDUs.

    ``Table.write`` builds a fresh primary header, which would drop cards such
    as ``GCTAG`` that live there beside the table HDU's ``ROLLC*`` cards.  It
    is still the writer, because it serialises mixin columns (``skycoord_ref``)
    that ``fits.table_to_hdu`` refuses.  The temporary file has a unique name
    and keeps the ``.fits`` extension; the ``os.replace`` makes the swap
    atomic, as in ``atomic_io.write_table_atomic``.
    """
    directory = os.path.dirname(path) or '.'
    root, ext = os.path.splitext(os.path.basename(path))
    tmp = os.path.join(directory,
                       f'.{root}.tmp{os.getpid()}{uuid.uuid4().hex[:8]}{ext}')
    published = False
    try:
        table.write(tmp, overwrite=True)
        with fits.open(tmp, mode='update') as hdul:
            header = hdul[0].header
            for card in primary_header.cards:
                if card.keyword in _STRUCTURAL_KEYS:
                    continue
                if card.keyword in ('COMMENT', 'HISTORY', ''):
                    header.append(card, end=True)
                else:
                    header[card.keyword] = (card.value, card.comment)
            for hdu in extra_hdus:
                hdul.append(hdu)
        os.replace(tmp, path)
        published = True
    finally:
        if not published and os.path.exists(tmp):
            os.remove(tmp)


def annotate_cmd_default(path, max_sep_arcsec=None):
    """Add or refresh ``oksep_pass`` / ``cmd_default`` and ``CMDDEF`` in a FITS table.

    The rewrite is atomic and idempotent; rows, every other column and the
    primary and table header cards (``GCTAG``, ``ROLLC*``) are carried through.
    Returns ``(n_rows, n_oksep, n_cmd_default)``.
    """
    from jwst_gc_pipeline.photometry import merge_catalogs as _mc
    if max_sep_arcsec is None:
        max_sep_arcsec = _mc.OKSEP_MAX_SEP_ARCSEC
    # memmap=False: the table outlives the open file, and the file is replaced below.
    with fits.open(path, memmap=False) as hdul:
        primary_header = hdul[0].header.copy()
        extra_hdus = [h.copy() for h in hdul[2:]]
        table = Table.read(hdul[1])
    oksep, cmd_default = _mc.compute_cmd_default(table, max_sep_arcsec)
    table[OKSEP_COL] = oksep
    table[CMD_DEFAULT_COL] = cmd_default
    table.meta[CMDDEF_CARD] = _mc.cmd_default_rule(table)
    _write_keeping_headers(table, path, primary_header, extra_hdus)
    n_oksep, n_cmd = int(oksep.sum()), int(cmd_default.sum())
    print(f"cmd_default: {os.path.basename(path)} N={len(table)} "
          f"{CMDDEF_CARD}={table.meta[CMDDEF_CARD]} oksep={n_oksep} "
          f"cmd_default={n_cmd}", flush=True)
    return len(table), n_oksep, n_cmd


def annotate_cmd_default_with_subsets(merged_path):
    """``annotate_cmd_default`` on a cross-band table and its ``_qualcuts*`` subsets.

    The subsets are written by the merge before ``n_filt_independent`` exists;
    ``propagate_independent_detection`` copies it onto them, and this runs
    after that, so every file gets the same rule.  Returns
    ``{path: (n_rows, n_oksep, n_cmd_default)}``.
    """
    import glob
    stem = merged_path[:-len('.fits')] if merged_path.endswith('.fits') else merged_path
    paths = [merged_path] if os.path.exists(merged_path) else []
    paths += sorted(glob.glob(f'{stem}_qualcuts*.fits'))
    return {p: annotate_cmd_default(p) for p in paths}
