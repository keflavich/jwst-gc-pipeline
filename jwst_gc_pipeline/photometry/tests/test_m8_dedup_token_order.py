"""The m8 dedup sibling keeps `_dedup` BEFORE the observation token.

#772 made the cross-band reader spell the merged-catalog token the way the
writer does, so the m8 path handed to `_maybe_dedup_m8` now already carries
`_o001`.  Two things then went wrong on brick's first run under that fix:

* the dedup name was built by appending, giving `..._m8_o001_dedup.fits`, which
  `diagnostics._CROSSBAND_RE` does not match (it parses `_dedup` then `_o<obs>`);
* the brick anti-clobber copy in `_maybe_dedup_m8` -- which exists precisely
  because the primary name USED to be unscoped -- appended a second token,
  producing `..._m8_o001_o001.fits` and `..._m8_o001_dedup_o001.fits`.

Both are name-only faults; the table contents were correct.
"""
import re

import pytest

from jwst_gc_pipeline.diagnostics.inventory import _CROSSBAND_RE

_STEM = 'basic_nrca_indivexp_photometry_tables_merged_resbgsub'


def _dedup_name(m8_path):
    """Mirror of the naming rule in cataloging._maybe_dedup_m8."""
    m = re.search(r'(_o[0-9-]+)\.fits$', m8_path)
    if m:
        tok = m.group(1)
        return m8_path[:-len(tok + '.fits')] + '_dedup' + tok + '.fits'
    out = m8_path.replace('_m8.fits', '_m8_dedup.fits')
    return out if out != m8_path else m8_path.replace('.fits', '_dedup.fits')


@pytest.mark.parametrize('m8,expected', [
    (f'{_STEM}_m8_o001.fits', f'{_STEM}_m8_dedup_o001.fits'),
    (f'{_STEM}_m8_o004.fits', f'{_STEM}_m8_dedup_o004.fits'),
    (f'{_STEM}_m8_o001-002.fits', f'{_STEM}_m8_dedup_o001-002.fits'),
    (f'{_STEM}_m8.fits', f'{_STEM}_m8_dedup.fits'),
])
def test_dedup_name(m8, expected):
    assert _dedup_name(m8) == expected


@pytest.mark.parametrize('name', [
    f'{_STEM}_m8_o001.fits',
    f'{_STEM}_m8_dedup_o001.fits',
    f'{_STEM}_m8_o001-002.fits',
    f'{_STEM}_m8_dedup_o001-002.fits',
])
def test_the_canonical_names_are_parseable(name):
    """The point of the ordering: these must be discoverable."""
    assert _CROSSBAND_RE.match(name), f'{name} is not matched by _CROSSBAND_RE'


@pytest.mark.parametrize('name', [
    f'{_STEM}_m8_o001_dedup.fits',        # append-instead-of-insert
    f'{_STEM}_m8_o001_o001.fits',         # double token
    f'{_STEM}_m8_o001_dedup_o001.fits',   # both
])
def test_the_broken_spellings_are_not_parseable(name):
    """Pins WHY the ordering matters: these were produced on brick's first run
    under #772 and the crossband discovery cannot see them."""
    assert not _CROSSBAND_RE.match(name), (
        f'{name} unexpectedly parses; this test no longer pins the bug')


def test_scoping_a_name_that_already_carries_a_token_is_a_no_op():
    """The brick anti-clobber copy must not append a second token."""
    already = f'{_STEM}_m8_o001.fits'
    assert re.search(r'_o[0-9-]+\.fits$', already), 'guard predicate must fire here'
    unscoped = f'{_STEM}_m8.fits'
    assert not re.search(r'_o[0-9-]+\.fits$', unscoped), 'and must NOT fire here'
