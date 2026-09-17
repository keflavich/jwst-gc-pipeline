#!/usr/bin/env python
"""The guide stars the Treasury observations were actually taken on.

Every JWST exposure records the star FGS was locked on in its primary header
(`GDSTARID`, `GS_RA`, `GS_DEC`, `GS_MAG`, `GS_ORDER`, `GS_V3_PA`).  That is a
different thing from the guide stars APT PLANNED: the onboard ID sequence walks
down the candidate list when a star fails to acquire, so `GS_ORDER > 1` means
the first choice was not used.  Reading it from the delivered frames says which
star each visit was really on.

Two products, one file:

* ``sources`` -- one entry per distinct guide star, for the viewer's catalogue
  overlay.  A star that guided several visits appears ONCE, with all of them
  listed, because it is one star on the sky.
* ``by_obs`` -- the inverse lookup, so the focal-plane panel can name the guide
  star for the observation the user selected.

Positions are the catalogue positions as flown, in the frame the header
records; they are not re-measured here.

THE UNCERTAINTY COLUMNS ARE PUBLISHED IN THE HEADER'S OWN UNITS, UNCONVERTED
----------------------------------------------------------------------------
`GS_URA`/`GS_UDEC` are emitted under their keyword names with no unit in the
label, because the unit is not settled:

* The JWST keyword dictionary says **arcsec**, sourced from ``GSC:raErr``
  (`archive.stsci.edu/jwst/keyword/latest/meta_guidestar_gs_ura.html`).  The
  FITS card comment carries no unit, and `stdatamodels`' ``core.schema.yaml``
  declares none.
* The 34 delivered values fall in two groups 50x apart with nothing between
  them: 19 at 0.017 to 1.295, and 15 at 65.0 to 118.1.  Read as arcsec, the
  second group claims a 65 to 118 ARCSECOND catalogue position, which no guide
  star acquisition would survive.  Read as mas, both groups are ordinary.
* The split is not arbitrary.  Every star in the small group has a Gaia DR3
  source within 0.07" (median 0.032"); not one star in the large group has a
  Gaia source within 3".  So the two groups are Gaia-based and plate-based
  catalogue entries, which is exactly the population split that would carry
  two different precisions -- and, plausibly, two different unit conventions.

Converting on an assumption would publish a number that is wrong by 1000x for
one of the two groups.  The values go out as the header wrote them, named for
the keyword, with the viewer's note saying so.
"""
import argparse
import collections
import json
import os

DEFAULT_EXPOSURES = ('/orange/adamginsburg/jwst/releases/v1.8-2026.09/'
                     'gc-treasury/exposures')
DEFAULT_OUT = ('/orange/adamginsburg/jwst/htdocs_staging/'
               'jwst_gc_cat_guidestars.json')
#: Where the monitor records each observation's STScI visit status. A skipped
#: observation is the case this file cannot speak to on its own: it has no
#: frames, so the scan below never sees it.
DEFAULT_FOOTPRINTS = '/orange/adamginsburg/jwst/monitor/footprints.json'

#: Header keys read from extension 0. Missing ones are reported, never guessed.
GS_KEYS = ('GDSTARID', 'GS_RA', 'GS_DEC', 'GS_MAG', 'GS_ORDER', 'GS_V3_PA',
           'GS_URA', 'GS_UDEC', 'GSC_VER')


#: Position tolerance, in degrees, before two frames are said to disagree
#: about where a guide star is. 1e-7 deg is 0.36 mas -- below any real
#: difference and above float round-tripping through a FITS card.
_SAME_POSITION_DEG = 1e-7


def _keep(star, key, value, path, disagreements):
    """Keep the first value seen for a per-star field; report a later mismatch.

    These are properties of the STAR, so every frame that names it should
    agree. Where they do not, the scan says which frame disagreed rather than
    silently publishing whichever one `os.walk` reached last.
    """
    if key not in star:
        star[key] = value
        return
    old = star[key]
    if isinstance(old, float) and isinstance(value, float):
        same = abs(old - value) <= (_SAME_POSITION_DEG if key in ('ra', 'dec')
                                    else abs(old) * 1e-6 + 1e-12)
    else:
        same = old == value
    if not same:
        disagreements.append(f'{path}: {key} {value!r} != {old!r} kept')


def scan(exposures_dir):
    """``(per_star, per_visit, problems)`` from every frame under a tree.

    ``problems`` holds both frames with no guide star recorded and frames whose
    per-star values contradict an earlier frame's; both are reported rather
    than dropped.

    Reads every frame rather than one per visit: a visit can re-acquire on a
    different star mid-visit, and sampling one exposure would report whichever
    frame happened to sort first as if it covered the whole visit.
    """
    from astropy.io import fits

    per_star = collections.defaultdict(lambda: {'visits': set(), 'obs': set(),
                                                'frames': 0, 'orders': set(),
                                                'instruments': set()})
    per_visit = {}
    no_guidestar = []
    disagreements = []
    for root, _dirs, names in os.walk(exposures_dir):
        for name in sorted(names):
            if not name.endswith('.fits'):
                continue
            path = os.path.join(root, name)
            try:
                hdr = fits.getheader(path, 0)
            except (OSError, IndexError, ValueError) as err:
                no_guidestar.append(f'{path}: {type(err).__name__}: {err}')
                continue
            gsid = hdr.get('GDSTARID')
            ra, dec = hdr.get('GS_RA'), hdr.get('GS_DEC')
            if not gsid or ra is None or dec is None:
                no_guidestar.append(f'{path}: no GDSTARID/GS_RA/GS_DEC')
                continue
            visit = str(hdr.get('VISIT_ID') or '')
            obs = f"o{int(hdr.get('OBSERVTN', 0)):03d}" if hdr.get('OBSERVTN') \
                else ''
            star = per_star[str(gsid)]
            star['frames'] += 1
            # FIRST frame wins, and a later frame that disagrees is reported.
            # Last-wins over `os.walk` order made these six fields depend on
            # directory traversal order, in a scan whose stated reason for
            # reading every frame is that per-frame values can differ.
            _keep(star, 'ra', float(ra), path, disagreements)
            _keep(star, 'dec', float(dec), path, disagreements)
            for key in ('GS_MAG', 'GS_URA', 'GS_UDEC', 'GSC_VER'):
                if hdr.get(key) is not None:
                    _keep(star, key.lower(), hdr.get(key), path, disagreements)
            if visit:
                star['visits'].add(visit)
            if obs:
                star['obs'].add(obs)
            if hdr.get('GS_ORDER') is not None:
                star['orders'].add(int(hdr['GS_ORDER']))
            if hdr.get('INSTRUME'):
                star['instruments'].add(str(hdr['INSTRUME']))
            if visit:
                entry = per_visit.setdefault(visit, {
                    'visit': visit, 'observation': obs, 'stars': {}})
                entry['stars'].setdefault(str(gsid), 0)
                entry['stars'][str(gsid)] += 1
                if hdr.get('GS_V3_PA') is not None:
                    entry['pa_v3_guidestar'] = float(hdr['GS_V3_PA'])
    return per_star, per_visit, no_guidestar + disagreements


def _sig(value, digits=3):
    """Round to significant digits, or ``None``.

    The values span 0.017 to 118, so a fixed number of decimals either loses
    the small ones entirely or prints six meaningless digits on the large
    ones.
    """
    if value is None:
        return None
    value = float(value)
    if value == 0.0:
        return 0.0
    from math import floor, log10
    return round(value, -int(floor(log10(abs(value)))) + (digits - 1))


def unflown_observations(footprints_path):
    """Observations the plan holds and the telescope never took.

    These are the ones whose guide star cannot be reported at all.  A guide
    star is assigned by PPS at scheduling and appears only in delivered data:
    for 10678's seven Skipped observations MAST holds either nothing (six of
    them) or planned rows at ``calib_level = -1`` (o099), the APT export
    carries no guide-star element, and the visit status table gives status and
    a WOPR number and no star.  So they are listed as pointings with no guide
    star rather than left out -- a viewer that simply omits them says the
    survey has 34 guide stars and nothing failed.
    """
    if not os.path.isfile(footprints_path):
        return []
    with open(footprints_path) as fh:
        data = json.load(fh)
    out = []
    for entry in data.get('planned', []):
        status = str(entry.get('status') or '')
        if status.lower() not in ('skipped', 'failed', 'withdrawn'):
            continue
        number = str(entry.get('number') or '').strip()
        if not number.isdigit():
            continue
        out.append({'observation': f'o{int(number):03d}',
                    'status': status,
                    'target': entry.get('target') or '',
                    'ra': entry.get('ra'), 'dec': entry.get('dec')})
    return sorted(out, key=lambda e: e['observation'])


def to_document(per_star, per_visit, programme='10678', unflown=()):
    """The viewer's JSON: a catalogue plus the two inverse lookups."""
    sources = []
    for gsid in sorted(per_star):
        star = per_star[gsid]
        sources.append({
            'ra': star['ra'], 'dec': star['dec'],
            'guide star': gsid,
            'FGS mag': round(float(star['gs_mag']), 2)
            if star.get('gs_mag') is not None else None,
            'catalogue': star.get('gsc_ver', ''),
            # Named for the keyword, in the header's own units -- see the
            # module docstring. `round(..., 1)` published 0.0 for 8 of the 34
            # stars, which reads as a position known exactly on the column
            # added to say it is not.
            'GS_URA': _sig(star.get('gs_ura')),
            'GS_UDEC': _sig(star.get('gs_udec')),
            # `2` here means FGS fell through to the second candidate.
            'ID order': ','.join(str(o) for o in sorted(star['orders'])),
            # A star acquired at order > 1 is one the observatory fell through
            # to: at least one earlier candidate failed to acquire. That is the
            # only failure the delivered data records, and it is drawn in its
            # own colour.
            'fallback': max(star['orders']) > 1 if star['orders'] else False,
            'observations': ' '.join(sorted(star['obs'])),
            # The label the viewer prints beside the marker: the observation
            # numbers this star guided, without the `o` that repeats 34 times.
            'label': ' '.join(sorted(o.lstrip('o').lstrip('0') or '0'
                                     for o in star['obs'])),
            'visits': len(star['visits']),
            'frames': star['frames'],
        })

    by_obs = collections.defaultdict(list)
    for visit in sorted(per_visit):
        entry = per_visit[visit]
        for gsid, frames in sorted(entry['stars'].items()):
            by_obs[entry['observation']].append(
                {'visit': visit, 'guide_star': gsid, 'frames': frames,
                 'ra': per_star[gsid]['ra'], 'dec': per_star[gsid]['dec'],
                 'mag': per_star[gsid].get('gs_mag'),
                 # The V3 position angle OF THE GUIDE STAR for the science
                 # exposure: the attitude the visit was actually held at, as
                 # opposed to the planned PA the focal-plane overlay draws.
                 'pa_v3': entry.get('pa_v3_guidestar')})
    return {
        'name': f'JWST {programme} guide stars',
        'programme': programme,
        'n': len(sources),
        'n_fallback': sum(1 for s in sources if s['fallback']),
        'sources': sources,
        'by_obs': dict(by_obs),
        'unflown': list(unflown),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exposures', default=DEFAULT_EXPOSURES)
    ap.add_argument('--footprints', default=DEFAULT_FOOTPRINTS,
                    help='where the observation statuses come from')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--programme', default='10678')
    args = ap.parse_args(argv)

    if not os.path.isdir(args.exposures):
        raise SystemExit(f'no exposures tree at {args.exposures}')
    per_star, per_visit, missing = scan(args.exposures)
    if not per_star:
        raise SystemExit(f'no frame under {args.exposures} records a guide star')
    unflown = unflown_observations(args.footprints)
    doc = to_document(per_star, per_visit, args.programme, unflown)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(doc, fh, separators=(',', ':'))
    reused = sum(1 for s in doc['sources'] if s['visits'] > 1)
    print(f"{doc['n']} guide stars over {len(per_visit)} visits "
          f"-> {args.out}")
    print(f"  {reused} guided more than one visit; "
          f"{doc['n_fallback']} were not the first candidate")
    if unflown:
        names = ', '.join(e['observation'] for e in unflown)
        print(f'  {len(unflown)} observation(s) never flown, so no guide star '
              f'was ever recorded for them: {names}')
    if missing:
        print(f'  {len(missing)} frame(s) record no guide star:')
        for line in missing[:5]:
            print(f'    {line}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
