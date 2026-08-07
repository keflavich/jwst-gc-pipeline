"""Open a CRDS reference from the local cache instead of over the network.

The three reduction drivers each fetched the tweakreg pars reference as

    asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{name}')

which makes every reduce depend on STScI being reachable *at that moment*, for a
file that is already sitting in the CRDS cache.  The fetch goes through
fsspec/aiohttp, which has no retry, so a single 504 kills the job outright:

    aiohttp.client_exceptions.ClientResponseError: 504, message='Gateway Time-out',
      url='.../jwst_nircam_pars-tweakregstep_0048.asdf'
    FileNotFoundError: .../jwst_nircam_pars-tweakregstep_0048.asdf

That took out 4 of 8 sgrc filters on 38870453 and all 4 again on the re-run
38871288 (2026-08-07, issue #327), plus 3 of 11 sgrb2 tasks the day before --
with both references present and checksum-correct in
``/orange/adamginsburg/jwst/crds`` the entire time.  ``CRDS_CLIENT_RETRY_COUNT``
and ``CRDS_CLIENT_RETRY_DELAY_SECONDS`` (#315) do not help here: they govern
``crds.client``, and this path never goes through it.
"""
import os

import asdf

CRDS_REFERENCE_URL = 'https://jwst-crds.stsci.edu/unchecked_get/references/jwst'


def cached_reference_path(crds_dir, instrument, filename):
    """Where CRDS would keep ``filename`` for ``instrument`` under ``crds_dir``."""
    return os.path.join(crds_dir, 'references', 'jwst', instrument, filename)


def open_crds_reference(crds_dir, instrument, filename):
    """Open a CRDS reference, preferring the local cache.

    Falls back to the CRDS server only when the file genuinely is not cached,
    and prints which source was used so a stale or missing cache is not silent.

    The fallback is on ABSENCE, not on failure: a cached file that exists but is
    truncated or corrupt raises out of ``asdf.open`` rather than being quietly
    re-fetched.  That is deliberate -- a corrupt cache is a real problem, and
    papering over it with a silent network read would hide it for as long as
    STScI happens to be reachable, which is exactly the coupling this module
    exists to remove.  Repair the cache instead (``crds sync``, or copy from
    another field's cache and check the md5 against it).
    """
    cached = cached_reference_path(crds_dir, instrument, filename)
    if os.path.exists(cached):
        print(f"CRDS reference {filename}: using cached {cached}", flush=True)
        return asdf.open(cached)
    url = f'{CRDS_REFERENCE_URL}/{filename}'
    print(f"CRDS reference {filename}: NOT in {crds_dir}, fetching {url}",
          flush=True)
    return asdf.open(url)
