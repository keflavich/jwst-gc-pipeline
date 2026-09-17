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
* ``by_visit`` / ``by_obs`` -- the inverse lookup, so the focal-plane panel can
  name the guide star for the observation the user selected.

Positions are the catalogue positions from GSC 2.3 as flown, in the frame the
header records; they are not re-measured here.  The uncertainties in
``GS_URA``/``GS_UDEC`` are in mas and are carried through, since they are
routinely tens to hundreds of mas and a reader comparing a marker against a
JWST source needs to know that.
"""
import argparse
import collections
import json
import os

DEFAULT_EXPOSURES = ('/orange/adamginsburg/jwst/releases/v1.8-2026.09/'
                     'gc-treasury/exposures')
DEFAULT_OUT = ('/orange/adamginsburg/jwst/htdocs_staging/'
               'jwst_gc_cat_guidestars.json')

#: Header keys read from extension 0. Missing ones are reported, never guessed.
GS_KEYS = ('GDSTARID', 'GS_RA', 'GS_DEC', 'GS_MAG', 'GS_ORDER', 'GS_V3_PA',
           'GS_URA', 'GS_UDEC', 'GSC_VER')


def scan(exposures_dir):
    """``(per_star, per_visit, no_guidestar)`` from every frame under a tree.

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
            star['ra'], star['dec'] = float(ra), float(dec)
            for key in ('GS_MAG', 'GS_URA', 'GS_UDEC', 'GSC_VER'):
                if hdr.get(key) is not None:
                    star[key.lower()] = hdr.get(key)
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
    return per_star, per_visit, no_guidestar


def to_document(per_star, per_visit, programme='10678'):
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
            # mas, as the header writes them
            'sigma RA (mas)': round(float(star['gs_ura']), 1)
            if star.get('gs_ura') is not None else None,
            'sigma Dec (mas)': round(float(star['gs_udec']), 1)
            if star.get('gs_udec') is not None else None,
            # `2` here means FGS fell through to the second candidate.
            'ID order': ','.join(str(o) for o in sorted(star['orders'])),
            'observations': ' '.join(sorted(star['obs'])),
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
                 'mag': per_star[gsid].get('gs_mag')})
    return {
        'name': f'JWST {programme} guide stars',
        'programme': programme,
        'n': len(sources),
        'sources': sources,
        'by_obs': dict(by_obs),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exposures', default=DEFAULT_EXPOSURES)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--programme', default='10678')
    args = ap.parse_args(argv)

    if not os.path.isdir(args.exposures):
        raise SystemExit(f'no exposures tree at {args.exposures}')
    per_star, per_visit, missing = scan(args.exposures)
    if not per_star:
        raise SystemExit(f'no frame under {args.exposures} records a guide star')
    doc = to_document(per_star, per_visit, args.programme)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(doc, fh, separators=(',', ':'))
    reused = sum(1 for s in doc['sources'] if s['visits'] > 1)
    fallback = sum(1 for s in doc['sources'] if s['ID order'] != '1')
    print(f"{doc['n']} guide stars over {len(per_visit)} visits "
          f"-> {args.out}")
    print(f'  {reused} guided more than one visit; '
          f'{fallback} were not the first candidate')
    if missing:
        print(f'  {len(missing)} frame(s) record no guide star:')
        for line in missing[:5]:
            print(f'    {line}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
