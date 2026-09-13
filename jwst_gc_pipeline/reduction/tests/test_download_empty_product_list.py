"""An observation with rows but no staged products must not abort its band.

MAST publishes an observation's rows before its products are stageable.  GC
Treasury tiles o129 and o130 were both in that state on 2026-09-12, minutes after
appearing: the obs table held 6 F480M rows and `get_product_list` returned an
EMPTY table with a NoResultsWarning.

`download_mast_observation` then did

    uncal &= fits['productType'] == 'SCIENCE'

where `uncal` came from a zero-length list comprehension -- so numpy made it
float64, and the `&=` raised

    TypeError: ufunc 'bitwise_and' not supported for the input types, and the
    inputs could not be safely coerced to any supported types according to the
    casting rule ''safe''

which aborted the whole band and skipped the tile.  Nothing is wrong in that
state and the next pass picks the products up, so it has to report and continue.

These tests exercise the mask arithmetic directly rather than the CLI, because
the failure is purely in the dtype of an empty mask -- no network is involved and
none is used here.
"""
import numpy as np
import pytest
from astropy.table import Table


def _uncal_mask(uris):
    """The mask as the script builds it, including the dtype that matters."""
    return np.array([u.endswith('_uncal.fits') and '_nrc' in u for u in uris],
                    dtype=bool)


def test_an_empty_uri_list_yields_a_BOOLEAN_mask():
    """Without dtype=bool numpy returns float64 for a zero-length list, and the
    `&=` one line later is the TypeError."""
    mask = _uncal_mask([])
    assert mask.dtype == np.bool_, (
        f'empty mask came back as {mask.dtype}; `&=` against a boolean column '
        f'raises TypeError')
    assert len(mask) == 0


def test_the_original_construction_is_the_one_that_breaks():
    """Pins the mechanism, so nobody reintroduces it by dropping dtype=bool."""
    bad = np.array([], dtype=float)
    with pytest.raises(TypeError, match='bitwise_and'):
        bad &= np.array([], dtype=bool)


def test_anding_an_empty_boolean_mask_against_an_empty_column_is_fine():
    products = Table({'dataURI': np.array([], dtype=str),
                      'productType': np.array([], dtype=str)})
    uncal = _uncal_mask(products['dataURI'])
    uncal &= np.asarray(products['productType'] == 'SCIENCE', dtype=bool)
    assert uncal.sum() == 0


def test_a_populated_table_still_selects_the_science_uncal():
    products = Table({
        'dataURI': np.array([
            'mast:JWST/product/jw10678129001_02101_00001_nrca1_uncal.fits',
            'mast:JWST/product/jw10678129001_02101_00001_nrca1_rate.fits',
            'mast:JWST/product/jw10678129001_02101_00001_nrca1_uncal.fits',
        ]),
        'productType': np.array(['SCIENCE', 'SCIENCE', 'INFO']),
    })
    uncal = _uncal_mask(products['dataURI'])
    uncal &= np.asarray(products['productType'] == 'SCIENCE', dtype=bool)
    assert list(uncal) == [True, False, False]
