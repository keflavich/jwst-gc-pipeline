"""SVO Filter Profile Service lookups that survive an SVO outage.

The cross-band merge (m3+ finalize), the reference-table builder, aperture
photometry and the plotting helpers all read Vega zero points and effective
wavelengths from ``astroquery.svo_fps.SvoFps.get_filter_list``.  That is a
network call to svo2.cab.inta-csic.es.  astroquery caches the response, but
only for a week; once the cache expires, every call goes back to the network.
When SVO cannot be reached, ``requests`` raises ``ConnectTimeout`` and the
finalize dies after its photometry is done (the 10678 o067 latest-versions
run, 2026-10-24: the m4 finalize lost its merge to a 60 s connect timeout).

``SvoFps`` here is a drop-in for astroquery's object.  ``get_filter_list``
asks SVO first, exactly as before.  If the request fails for a JWST query, it
returns the same table parsed from a copy of SVO's own VOTable response that
ships with the package (``data/svo_jwst_filter_list.xml``, the ``Facility=JWST``
list).  The copy goes through the same ``parse_single_table(...).to_table()``
call astroquery uses, so the fallback table has the same columns, dtypes,
units and values as the live one did on the day it was saved.

Behaviour:

* A live answer is used whenever SVO responds, so a zero point SVO revises is
  picked up as before.
* The fallback prints a ``WARNING`` naming the error and the copy's fetch date.
* After the first connection failure or timeout in a process the network is
  not tried again, so a merge that reads the list for every filter waits for
  one timeout, not one per call.  A response that arrives but is not a table
  (an HTML maintenance page, an empty answer) falls back for that call only.
* A ``ValueError`` counts as an SVO failure only when the VOTable parser raised
  it.  Any other ``ValueError`` (a bad ``timeout`` value, say) is a caller bug
  and raises; so does astroquery's ``InvalidQueryError`` for bad parameters.
* ``JWST_GC_SVO_OFFLINE=1`` skips the network entirely.
* A non-JWST query has no copy to fall back on and raises as before.
* Every other attribute (``get_transmission_data``, ``get_filter_index``, ...)
  is astroquery's, unchanged.

Refresh the copy while SVO is reachable::

    python -m jwst_gc_pipeline.svo_filters --write-vendored
"""
import datetime
import io
import os
import re
import sys
from importlib import resources

#: The ``Facility=JWST`` VOTable SVO returned, saved byte for byte.
VENDORED_RESOURCE = 'svo_jwst_filter_list.xml'

#: Date the vendored response was fetched from SVO.
VENDORED_FETCHED = '2026-10-17'

#: Environment switch: any value other than 0/false/no/off skips the network.
OFFLINE_ENV = 'JWST_GC_SVO_OFFLINE'

#: Seconds to wait for SVO before falling back.  astroquery's default is 60.
DEFAULT_TIMEOUT_S = 30


def _svo_unreachable_errors():
    """Exceptions that mean "SVO could not be reached".

    ``requests`` exceptions (connect/read timeouts, connection errors and the
    HTTPError from ``raise_for_status``) subclass ``OSError``.  astroquery
    raises its own ``TimeoutError``, a plain ``Exception``, from some paths.
    """
    from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError
    return (OSError, AstroqueryTimeoutError)


def _is_response_parse_error(err):
    """True for an error raised while parsing SVO's response.

    astroquery raises ``IndexError`` when the VOTable holds no table.  The
    VOTable parser raises ``ValueError`` (an ``E19`` for an HTML page, a plain
    ``ValueError`` for an empty or non-XML body) from inside
    ``astropy.io.votable``; a ``ValueError`` raised anywhere else is a caller
    bug and is not one of these.
    """
    if isinstance(err, IndexError):
        return True
    if not isinstance(err, ValueError):
        return False
    from astropy.io.votable.exceptions import VOWarning
    if isinstance(err, VOWarning):
        return True
    marker = os.path.join('astropy', 'io', 'votable')
    tb = err.__traceback__
    while tb is not None:
        if marker in tb.tb_frame.f_code.co_filename:
            return True
        tb = tb.tb_next
    return False


def _offline_requested(env=None):
    env = os.environ if env is None else env
    return str(env.get(OFFLINE_ENV, '0')).strip().lower() not in (
        '0', '', 'false', 'no', 'off')


def vendored_filter_list(instrument=None):
    """The saved ``Facility=JWST`` filter list, optionally cut to one instrument.

    Instrument names compare case-insensitively: SVO spells NIRSpec
    ``NIRSPEC`` in the table, and callers pass header spellings.
    """
    from astropy.io.votable import parse_single_table
    raw = resources.files('jwst_gc_pipeline.data').joinpath(
        VENDORED_RESOURCE).read_bytes()
    table = parse_single_table(io.BytesIO(raw)).to_table()
    table.meta['svo_source'] = f'vendored {VENDORED_RESOURCE} ({VENDORED_FETCHED})'
    if instrument is not None:
        keep = [str(v).upper() == str(instrument).upper()
                for v in table['Instrument']]
        table = table[keep]
        if len(table) == 0:
            raise IndexError(f'no JWST filters for instrument {instrument!r} '
                             f'in the vendored SVO list')
    return table


class _SvoFpsWithFallback:
    """astroquery's ``SvoFps`` with a vendored fallback for JWST filter lists."""

    TIMEOUT = DEFAULT_TIMEOUT_S

    def __init__(self):
        self._network_failed = None

    def __getattr__(self, name):
        from astroquery.svo_fps import SvoFps as _astroquery_svofps
        return getattr(_astroquery_svofps, name)

    def get_filter_list(self, facility, *, instrument=None, **kwargs):
        if str(facility).upper() != 'JWST':
            from astroquery.svo_fps import SvoFps as _astroquery_svofps
            return _astroquery_svofps.get_filter_list(
                facility, instrument=instrument, **kwargs)
        if _offline_requested():
            return vendored_filter_list(instrument)
        if self._network_failed is None:
            from astroquery.svo_fps import SvoFps as _astroquery_svofps
            kwargs.setdefault('timeout', self.TIMEOUT)
            try:
                return _astroquery_svofps.get_filter_list(
                    facility, instrument=instrument, **kwargs)
            except _svo_unreachable_errors() as err:
                self._network_failed = f'{type(err).__name__}: {err}'
                reason = (f'the SVO request failed with {self._network_failed}; '
                          f'not retrying in this process')
            except (IndexError, ValueError) as err:
                if not _is_response_parse_error(err):
                    raise
                reason = (f'SVO answered with something that is not a filter '
                          f'table: {type(err).__name__}: {err}')
        else:
            reason = (f'an earlier SVO request in this process failed with '
                      f'{self._network_failed}')
        print(f"WARNING: SVO filter list for {facility}"
              f"{'/' + instrument if instrument else ''} unavailable "
              f"({reason}); using the copy vendored in "
              f"jwst_gc_pipeline/data/{VENDORED_RESOURCE}, fetched "
              f"{VENDORED_FETCHED}", flush=True)
        return vendored_filter_list(instrument)


#: Drop-in replacement for ``astroquery.svo_fps.SvoFps``.
SvoFps = _SvoFpsWithFallback()


def write_vendored(path=None, timeout=DEFAULT_TIMEOUT_S):
    """Fetch the ``Facility=JWST`` list from SVO and save the raw VOTable.

    Writing the packaged copy (``path=None``) also sets ``VENDORED_FETCHED``
    in this module's source to today's date.
    """
    from astroquery.svo_fps import SvoFps as _astroquery_svofps
    from astropy.io.votable import parse_single_table
    response = _astroquery_svofps._request(
        'GET', _astroquery_svofps.SVO_MAIN_URL,
        params={'Facility': 'JWST', 'Instrument': None},
        timeout=timeout, cache=False)
    response.raise_for_status()
    parse_single_table(io.BytesIO(response.content)).to_table()  # must parse
    packaged = path is None
    if packaged:
        path = resources.files('jwst_gc_pipeline.data').joinpath(VENDORED_RESOURCE)
    with open(path, 'wb') as fh:
        fh.write(response.content)
    print(f'wrote {len(response.content)} bytes to {path}')
    if packaged:
        _stamp_fetched_date(datetime.date.today().isoformat())
    return path


def _stamp_fetched_date(date, source=__file__):
    """Rewrite the ``VENDORED_FETCHED = '...'`` line of ``source``."""
    with open(source, encoding='utf-8') as fh:
        text = fh.read()
    new, n = re.subn(r"^VENDORED_FETCHED = '[^']*'$",
                     f"VENDORED_FETCHED = '{date}'", text, flags=re.M)
    if n != 1:
        raise RuntimeError(f'expected one VENDORED_FETCHED line in {source}, '
                           f'found {n}')
    with open(source, 'w', encoding='utf-8') as fh:
        fh.write(new)
    print(f'set VENDORED_FETCHED = {date!r} in {source}')


if __name__ == '__main__':
    if '--write-vendored' in sys.argv[1:]:
        write_vendored()
    else:
        print(__doc__)
