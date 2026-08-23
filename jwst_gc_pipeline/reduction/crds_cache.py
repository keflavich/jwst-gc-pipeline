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

**The network fallback is live, not hypothetical.**  Preferring the cache fixes
every reference the cache holds; it does nothing for one it does not.  Measured
2026-08-23: ``jwst_miri_pars-tweakregstep_0003.rmap`` -- the rmap
``PipelineMIRI`` loads by name -- resolves MIRI F770W to
``jwst_miri_pars-tweakregstep_0020.asdf``, and **no CRDS tree any field's
``CRDS_PATH`` resolves to holds that file**.  The 21 ``<field>/crds`` paths are
17 symlinks onto four real trees
(``/blue/adamginsburg/adamginsburg/jwst/brick/crds``,
``/orange/adamginsburg/jwst/crds``, ``.../w51/crds``, ``.../cloudc/crds``), and
none of the four has it; the archive's one copy sits in a 2024 personal tree
(``/orange/adamginsburg/jwst/sgrb2/NB/crds``) that no field points at.  So every
MIRI F770W reduce takes this fallback, and a 504 at that moment still kills the
task.  Hence the bounded retry below: the fetch that remains is the one that has
to survive a transient server error.

Retry policy, and why it is spelled this way:

* The 504 arrives as ``FileNotFoundError`` -- fsspec's ``_info`` converts every
  non-200 into one -- so a transient outage and a genuinely absent reference are
  INDISTINGUISHABLE at this layer.  Retrying therefore costs a real 404 the full
  ladder before it reports; ``CRDS_FETCH_RETRIES`` bounds that, and the default
  ladder (4 attempts, 5s growing linearly = 30s of sleeping) keeps a true miss
  fast relative to a reduce, whose tasks die at ~2.5 minutes on this fetch.
* The names the code reads are ``CRDS_FETCH_RETRIES`` and
  ``CRDS_FETCH_DELAY_SECONDS``, distinct from the ``CRDS_CLIENT_RETRY_*`` pair
  (#315) precisely because those govern a different downloader and setting them
  did not help here.  Two knobs with one name would have hidden that.
"""
import os
import time

import asdf

CRDS_REFERENCE_URL = 'https://jwst-crds.stsci.edu/unchecked_get/references/jwst'

#: Environment variable naming how many ATTEMPTS the network fallback makes.
CRDS_FETCH_RETRIES_ENV = 'CRDS_FETCH_RETRIES'
#: Environment variable naming the base delay between those attempts (seconds).
CRDS_FETCH_DELAY_ENV = 'CRDS_FETCH_DELAY_SECONDS'

DEFAULT_FETCH_RETRIES = 4
DEFAULT_FETCH_DELAY_SECONDS = 5.0


def _retryable_exceptions():
    """What a transient CRDS outage looks like from here.

    ``FileNotFoundError`` is an ``OSError``, and so are the connection and
    timeout errors urllib raises, so one entry covers the observed failure and
    its neighbours.  ``aiohttp.ClientError`` is NOT an ``OSError``; it is added
    when aiohttp is importable, for the case fsspec lets one through
    unconverted.
    """
    exceptions = [OSError]
    try:
        import aiohttp
    except ImportError:
        pass
    else:
        exceptions.append(aiohttp.ClientError)
    return tuple(exceptions)


def _env_number(name, default, cast):
    """Read ``name`` from the environment, falling back to ``default``.

    A malformed value is a configuration error worth seeing, so it is reported
    rather than silently replaced by the default -- the whole point of these
    knobs is that someone can tell whether the one they set is in effect.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} is not a number; unset it or give it one") from exc


def cached_reference_path(crds_dir, instrument, filename):
    """Where CRDS would keep ``filename`` for ``instrument`` under ``crds_dir``."""
    return os.path.join(crds_dir, 'references', 'jwst', instrument, filename)


def fetch_crds_reference(url, retries=None, delay_seconds=None, sleep=None):
    """``asdf.open(url)`` with a bounded retry, for the CRDS network fallback.

    Attempts ``retries`` times with a linearly growing delay, re-raising the
    LAST failure when they are exhausted so the traceback still names the real
    error.  Every attempt after the first is printed, so a log shows whether a
    reduce sailed through or clawed its way past an outage.
    """
    if sleep is None:
        # Resolved here rather than as a default argument so a test (or a
        # caller) that patches ``time.sleep`` is actually obeyed.
        sleep = time.sleep
    if retries is None:
        retries = _env_number(CRDS_FETCH_RETRIES_ENV, DEFAULT_FETCH_RETRIES, int)
    if delay_seconds is None:
        delay_seconds = _env_number(CRDS_FETCH_DELAY_ENV,
                                    DEFAULT_FETCH_DELAY_SECONDS, float)
    retries = max(1, int(retries))
    retryable = _retryable_exceptions()

    for attempt in range(1, retries + 1):
        try:
            return asdf.open(url)
        except retryable as exc:
            if attempt >= retries:
                raise
            wait = delay_seconds * attempt
            print(f"CRDS fetch {url}: attempt {attempt}/{retries} failed "
                  f"({type(exc).__name__}: {exc}); retrying in {wait:g}s",
                  flush=True)
            sleep(wait)


def open_crds_reference(crds_dir, instrument, filename):
    """Open a CRDS reference, preferring the local cache.

    Falls back to the CRDS server only when the file genuinely is not cached,
    and prints which source was used so a stale or missing cache is not silent.
    The fallback retries a transient failure rather than killing the reduce.
    """
    cached = cached_reference_path(crds_dir, instrument, filename)
    if os.path.exists(cached):
        print(f"CRDS reference {filename}: using cached {cached}", flush=True)
        return asdf.open(cached)
    url = f'{CRDS_REFERENCE_URL}/{filename}'
    print(f"CRDS reference {filename}: NOT in {crds_dir}, fetching {url}",
          flush=True)
    return fetch_crds_reference(url)
