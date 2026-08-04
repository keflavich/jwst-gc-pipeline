#!/usr/bin/env python
"""Turn a JWST APT file into sky polygons for the monitor's footprint view.

Written for **program 10678** — "The JWST/NIRCam Legacy Survey of the Galactic
Center" (Schoedel, Cycle 5, NIRCam + coordinated MIRI parallels, Flight Ready).
139 pointings, none observed yet.

The geometry follows the same recipe as the Roman GBTDS footprint page: take a
pointing (RA, Dec, PA_V3), build the observatory attitude with ``pysiaf``, and
project each aperture's corners through it.

Two things are easy to get wrong here and are done deliberately:

* **The attitude is anchored on the aperture APT points at, not on V1.**  APT's
  target coordinate is where the *selected aperture's* reference point lands.
  For this program that is the NIRCam full-array aperture, so the attitude is
  built from ``NRCALL_FULL``'s reference point — and the MIRI parallel is then
  projected through that SAME attitude, which is what makes it land where the
  parallel actually observes rather than on top of NIRCam.

* **PA_V3 is a RANGE, not a value.**  The program requests
  ``OrientRange V3PA 79–95°``; the scheduled angle is not known until the visit
  is planned.  The midpoint is used and the range is recorded in the output, so
  the page can say the footprints rotate within it rather than implying a
  precision the plan does not have.

Observed footprints come from the APT ``VisitStatus`` entries -- which live at
``JwstProposal/VisitStatuses``, a ROOT child, not inside ``Observation``; reading
them from the obvious place silently pins the observed count at zero forever.
They carry only a ``VisitId``, so they are matched to observations by the
observation number embedded in it.  Today every visit reads ``IMPLEMENTATION``,
so the layer is empty -- and it renders empty rather than being omitted.

    python scripts/monitoring/build_footprints.py 10678 --out footprints.json
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

APT_URL = 'https://www.stsci.edu/jwst-program-info/download/jwst/apt/{program}'

#: Apertures drawn per instrument.  The per-detector NIRCam apertures rather
#: than the single NRCALL outline: the module gap is real sky the survey does
#: not cover, and an outline hides it.
NIRCAM_APERTURES = ('NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL', 'NRCA4_FULL',
                    'NRCB1_FULL', 'NRCB2_FULL', 'NRCB3_FULL', 'NRCB4_FULL')
#: The aperture APT's target coordinate refers to; the attitude is anchored here.
NIRCAM_ANCHOR = 'NRCALL_FULL'
#: The ILLUMINATED imaging field, not MIRIM_FULL.  MIRIM_FULL is the whole
#: 1032x1024 detector (123" x 121") and includes the coronagraph/Lyot region,
#: which a FULL-subarray MiriImaging exposure does not image.  MIRIM_ILLUM is
#: 83" x 118" -- drawing MIRIM_FULL overstates the parallel's sky coverage by
#: ~30% in one axis.
MIRI_APERTURES = ('MIRIM_ILLUM',)


def fetch_apt(program, dest, force=False):
    """Download the APT file, unless a USABLE copy is already there.

    "Usable" means it opens as a zip: a truncated download is still >10 kB and
    the size check alone made every later run fail on the same corrupt file
    until a human deleted it.  ``force`` re-downloads regardless, which is what
    picking up newly-executed visits needs -- otherwise the cache silently
    pins the answer to whenever the file was first fetched.
    """
    import urllib.request
    if not force and os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        if zipfile.is_zipfile(dest):
            return dest
        print(f'{dest} is not a readable zip; re-downloading', file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or '.', exist_ok=True)
    with urllib.request.urlopen(APT_URL.format(program=program), timeout=120) as fh:
        data = fh.read()
    with open(dest, 'wb') as out:
        out.write(data)
    return dest


#: Statuses that mean a visit actually ran.  A Flight Ready program reads
#: IMPLEMENTATION for every visit, which is "planned", not "observed".
EXECUTED_STATUSES = ('executed', 'archived', 'completed', 'flight', 'succeeded')


def _tag(elem):
    return elem.tag.split('}')[-1]


_SEXA = re.compile(r'^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+'
                   r'([\d.]+)\s+([\d.]+)\s*$')


def parse_sexagesimal(value):
    """``'17 44 38.8689 -29 28 41.23'`` -> ``(266.161954, -29.478119)`` degrees."""
    m = _SEXA.match(value)
    if not m:
        raise ValueError(f'unparseable coordinate {value!r}')
    rh, rm, rs, dd, dm, ds = (float(g) for g in m.groups())
    ra = 15.0 * (rh + rm / 60.0 + rs / 3600.0)
    sign = -1.0 if value.strip().split()[3].startswith('-') else 1.0
    dec = sign * (abs(dd) + dm / 60.0 + ds / 3600.0)
    return ra, dec


def parse_apt(xml_path):
    """``{'targets': {...}, 'observations': [...], 'observed': [...]}``."""
    root = ET.parse(xml_path).getroot()

    targets = {}
    for elem in root.iter():
        if _tag(elem) != 'Target':
            continue
        kids = {_tag(c): c for c in elem}
        name = (kids['TargetName'].text or '').strip()
        coord = kids['EquatorialCoordinates'].attrib.get('Value', '')
        if name and coord:
            targets[name] = parse_sexagesimal(coord)

    # VisitStatus elements live at JwstProposal/VisitStatuses/VisitStatus -- a
    # SIBLING of DataRequests, not a descendant of Observation.  Scanning inside
    # each Observation (the obvious reading) can never reach one, so the observed
    # count was structurally pinned at 0 and would have stayed there forever
    # however far the program progressed.  VisitId is 11 digits:
    # <5 program><3 observation><3 visit>, so the observation number is digits
    # 5..8 -- the status carries no target name, so it must be mapped by number.
    executed_obs = set()
    for elem in root.iter():
        if _tag(elem) != 'VisitStatus':
            continue
        status = (elem.attrib.get('Status') or '').strip().lower()
        visit_id = (elem.attrib.get('VisitId') or '').strip()
        if status in EXECUTED_STATUSES and len(visit_id) >= 8:
            executed_obs.add(visit_id[5:8].lstrip('0') or '0')
    # VisitExecutions is the element that records actual execution; empty in a
    # Flight Ready program, but read so this does not need revisiting later.
    for elem in root.iter():
        if _tag(elem) != 'VisitExecution':
            continue
        visit_id = (elem.attrib.get('VisitId') or '').strip()
        if len(visit_id) >= 8:
            executed_obs.add(visit_id[5:8].lstrip('0') or '0')

    observations, observed = [], []
    for elem in root.iter():
        if _tag(elem) != 'Observation':
            continue
        kids = {_tag(c): c for c in elem}
        target_id = (kids.get('TargetID').text or '').strip() \
            if kids.get('TargetID') is not None else ''
        # 'TargetID' reads '1 GC_1'; the name is the remainder after the number
        name = target_id.split(None, 1)[1] if ' ' in target_id else target_id
        instrument = ((kids.get('Instrument').text or '').strip()
                      if kids.get('Instrument') is not None else '')
        orient = None
        for sub in elem.iter():
            if _tag(sub) == 'OrientRange':
                lo = sub.attrib.get('OrientMin', '')
                hi = sub.attrib.get('OrientMax', '')
                orient = (_deg(lo), _deg(hi))
        filters = [(_tag(c), (c.text or '').strip()) for c in elem.iter()
                   if _tag(c) in ('ShortFilter', 'LongFilter')]
        # Every observation dithers.  Only the NOMINAL position is drawn; the
        # real covered area is slightly larger, and FULLBOX exists precisely to
        # fill the module gap the drawn outline shows as a hole.  Recorded so
        # the page can say so instead of implying the outline is the coverage.
        dither = {}
        for c in elem.iter():
            if _tag(c) in ('PrimaryDitherType', 'SubpixelDitherType',
                           'PrimaryDithers', 'DitherSize'):
                dither[_tag(c)] = (c.text or '').strip()
        number = (kids.get('Number').text or '').strip() \
            if kids.get('Number') is not None else ''
        parallel = ((kids.get('CoordinatedParallelSet').text or '').strip()
                    if kids.get('CoordinatedParallelSet') is not None else '')
        observations.append({
            'number': number, 'target': name, 'instrument': instrument,
            'orient': orient, 'filters': dict(filters), 'parallel': parallel,
            'dither': dither})

        if (number.lstrip('0') or '0') in executed_obs:
            observed.append({'number': number, 'target': name,
                             'status': 'executed'})
    return {'targets': targets, 'observations': observations,
            'observed': observed, 'n_visit_status': len(executed_obs)}


def _deg(text):
    """``'79 Degrees'`` -> ``79.0``."""
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return None


def aperture_polygons(ra, dec, pa_v3, apertures, anchor, siaf_cache={}):
    """Sky corners of ``apertures`` for a pointing, via the observatory attitude.

    ``anchor`` is the aperture whose reference point sits at ``(ra, dec)`` --
    APT's target coordinate.  Every aperture, including the parallel
    instrument's, is projected through the attitude that anchor defines, which
    is what places the parallel correctly instead of on top of the prime.
    """
    import pysiaf
    from pysiaf.utils import rotations

    out = {}
    anchor_inst = 'NIRCam' if anchor.startswith('NRC') else 'MIRI'
    if anchor_inst not in siaf_cache:
        siaf_cache[anchor_inst] = pysiaf.Siaf(anchor_inst)
    anchor_ap = siaf_cache[anchor_inst][anchor]
    attitude = rotations.attitude(anchor_ap.V2Ref, anchor_ap.V3Ref,
                                  ra, dec, pa_v3)

    for name in apertures:
        inst = 'NIRCam' if name.startswith('NRC') else 'MIRI'
        if inst not in siaf_cache:
            siaf_cache[inst] = pysiaf.Siaf(inst)
        ap = siaf_cache[inst][name]
        ap.set_attitude_matrix(attitude)
        v2, v3 = ap.corners('tel', rederive=False)
        sky_ra, sky_dec = rotations.pointing(attitude, v2, v3)
        out[name] = [[round(float(a), 6), round(float(d), 6)]
                     for a, d in zip(sky_ra, sky_dec)]
    return out


def build(program, apt_path, pa_v3=None):
    parsed = parse_apt(apt_path)
    targets = parsed['targets']
    observed_targets = {o['target'] for o in parsed['observed']}

    orients = [o['orient'] for o in parsed['observations'] if o['orient']]
    lo = min(o[0] for o in orients) if orients else None
    hi = max(o[1] for o in orients) if orients else None
    if pa_v3 is None:
        pa_v3 = (lo + hi) / 2.0 if (lo is not None and hi is not None) else 0.0

    planned, observed = [], []
    for obs in parsed['observations']:
        if obs['instrument'] != 'NIRCAM':
            # The other 139 Observation elements are text-only entries under
            # LinkingRequirements/SameOrientLink, not a MIRI half -- this program
            # has no separate MIRI Observation; MIRI rides as the coordinated
            # parallel of each NIRCam observation.
            continue
        coord = targets.get(obs['target'])
        if coord is None:
            continue
        ra, dec = coord
        nircam = aperture_polygons(ra, dec, pa_v3, NIRCAM_APERTURES,
                                   NIRCAM_ANCHOR)
        miri = aperture_polygons(ra, dec, pa_v3, MIRI_APERTURES, NIRCAM_ANCHOR)
        rec = {'target': obs['target'], 'number': obs['number'],
               'ra': round(ra, 6), 'dec': round(dec, 6),
               'filters': obs['filters'],
               'nircam': list(nircam.values()),
               'miri': list(miri.values())}
        (observed if obs['target'] in observed_targets else planned).append(rec)

    dithers = {}
    for o in parsed['observations']:
        for k, v in (o.get('dither') or {}).items():
            dithers.setdefault(k, set()).add(v)

    return {'program': str(program),
            'dither': {k: sorted(v) for k, v in dithers.items()},
            'title': 'JWST/NIRCam Legacy Survey of the Galactic Center',
            'pa_v3': pa_v3, 'pa_v3_range': [lo, hi],
            'n_planned': len(planned), 'n_observed': len(observed),
            'nircam_apertures': list(NIRCAM_APERTURES),
            'miri_apertures': list(MIRI_APERTURES),
            'planned': planned, 'observed': observed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('program', nargs='?', default='10678')
    ap.add_argument('--apt', default=None, help='existing .aptx (else download)')
    ap.add_argument('--out', default='footprints.json')
    ap.add_argument('--force', action='store_true',
                    help='re-download the APT file and re-extract, rather than '
                         'reusing a cached copy (needed to pick up visits that '
                         'have executed since)')
    ap.add_argument('--pa-v3', type=float, default=None,
                    help='override the PA_V3 used (default: midpoint of the '
                         'program OrientRange)')
    args = ap.parse_args(argv)

    workdir = os.path.dirname(os.path.abspath(args.out)) or '.'
    aptx = args.apt or os.path.join(workdir, f'{args.program}.aptx')
    fetch_apt(args.program, aptx, force=args.force)

    xml_name = f'{args.program}.xml'
    xml_path = os.path.join(workdir, xml_name)
    if args.force or not os.path.exists(xml_path):
        with zipfile.ZipFile(aptx) as zf:
            zf.extract(xml_name, workdir)

    data = build(args.program, xml_path, pa_v3=args.pa_v3)
    with open(args.out, 'w') as fh:
        json.dump(data, fh)
    print(f"{args.out}: {data['n_planned']} planned, {data['n_observed']} observed "
          f"(PA_V3 {data['pa_v3']:.1f}°, range {data['pa_v3_range']})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
