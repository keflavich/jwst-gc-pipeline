"""The ``_qualcuts*`` subsets must carry the independent-detection flags.

``merge_daophot`` writes the cross-band merged catalog and its quality-cut row
subsets in one call; ``annotate_independent_detection`` then rewrites only the
full table.  Every ``_qualcuts_oksep`` table on disk -- and so the treasury
joint catalog built from them -- lacked ``independently_detected_<filt>`` and
``n_filt_independent`` (10678 o105: m7 112 columns, qualcuts 109).
"""
import inspect

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.merge_catalogs import _qualcuts_oksep_suffix

OKSEP = _qualcuts_oksep_suffix('gc-treasury')


def _write_family(tmp_path, n=50, seed=1):
    rng = np.random.default_rng(seed)
    ra = 266.5 + rng.uniform(0, 0.01, n)
    dec = -28.9 + rng.uniform(0, 0.01, n)
    parent = Table()
    parent['skycoord_ref'] = SkyCoord(ra * u.deg, dec * u.deg)
    parent['flux_f212n'] = rng.uniform(1, 10, n)
    stem = str(tmp_path / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_hybpsf_m7_o105')
    keep = np.sort(rng.choice(n, n // 3, replace=False))
    # the subsets are written BEFORE annotate, i.e. without the flags
    parent[keep].write(f'{stem}{OKSEP}.fits')
    parent[keep[::-1]].write(f'{stem}_qualcuts.fits')   # order must not matter
    parent['independently_detected_f212n'] = rng.uniform(size=n) > 0.5
    parent['independently_detected_f480m'] = rng.uniform(size=n) > 0.5
    parent['n_filt_independent'] = (parent['independently_detected_f212n'].astype('i4')
                                    + parent['independently_detected_f480m'].astype('i4'))
    parent.write(f'{stem}.fits')
    return stem, parent, keep


def test_flags_copied_to_every_subset_row_for_row(tmp_path):
    stem, parent, keep = _write_family(tmp_path)
    done = cataloging.propagate_independent_detection(f'{stem}.fits')
    assert done == {f'{stem}_qualcuts.fits': len(keep),
                    f'{stem}{OKSEP}.fits': len(keep)}
    for path, order in ((f'{stem}{OKSEP}.fits', keep),
                        (f'{stem}_qualcuts.fits', keep[::-1])):
        sub = Table.read(path)
        for c in ('independently_detected_f212n', 'independently_detected_f480m',
                  'n_filt_independent'):
            np.testing.assert_array_equal(sub[c], parent[c][order])
        np.testing.assert_array_equal(sub['flux_f212n'], parent['flux_f212n'][order])


def test_foreign_subset_left_untouched(tmp_path):
    stem, parent, keep = _write_family(tmp_path)
    # a stale sibling from another run: one row is not in this merged table
    other = Table.read(f'{stem}{OKSEP}.fits')
    other['skycoord_ref'][0] = SkyCoord(1 * u.deg, 1 * u.deg)
    other.write(f'{stem}{OKSEP}.fits', overwrite=True)
    done = cataloging.propagate_independent_detection(f'{stem}.fits')
    assert done[f'{stem}{OKSEP}.fits'] is None
    assert 'n_filt_independent' not in Table.read(f'{stem}{OKSEP}.fits').colnames
    assert done[f'{stem}_qualcuts.fits'] == len(keep)


def test_unannotated_parent_is_a_no_op(tmp_path):
    stem, parent, keep = _write_family(tmp_path)
    parent.remove_columns(['independently_detected_f212n', 'independently_detected_f480m',
                           'n_filt_independent'])
    parent.write(f'{stem}.fits', overwrite=True)
    assert cataloging.propagate_independent_detection(f'{stem}.fits') == {}


def test_m7_finalize_propagates_after_annotating():
    src = inspect.getsource(cataloging.run_manual_pipeline)
    a = src.index('annotate_independent_detection(_xb,')
    p = src.index('propagate_independent_detection(_xb)')
    assert a < p < src.index("_stamp_catalog_provenance(\n                _xb, 'm7'")
