"""Which uncovered SATSTAR_NEAR leftover stars get a far-companion exemption row, per S/N floor.

usage: python cover_targets.py BUILD_OUTDIR LEFTOVERS_045.json LEFTOVERS_105.json OUT.json

BUILD_OUTDIR is the output directory of build_exempt_seed.py run with the S/N floors 50 30 40 60 80 (the rows of the first floor are the ones it writes
as exempt_<module>.fits); the band tables band_<module>_snr<s>.fits of its o<T> subdirectories hold the seed of every floor.  The targets are the 22 stars of the control m7
residual of gc-treasury 10678 F212N (o045 9, o105 13) that the leftover classification calls SATSTAR_NEAR with no m7 catalog row (NOROW) or a row that
does not subtract the star (ROW); (x, y) are the mosaic pixels of the leftover list.  Prints, for each target, the angular distance (arcsec) to the
nearest row an exemption adds at each floor; a row within 0.05" (about one pixel at the 0.031" mosaic scale) counts as covering the star.
"""
import json
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

TGT = {'045': [(7311, 2154), (6355, 3168), (1253, 131), (6277, 2642), (4979, 3391), (6092, 3774), (6471, 2248), (2970, 3509), (5070, 4424)],
       '105': [(434, 3684), (7030, 3275), (3245, 3878), (6137, 3868), (7485, 2672), (1435, 2408), (4952, 3836), (6760, 471), (9794, 833),
               (11328, 2324), (5188, 2780), (7097, 961), (6588, 1271)]}
FLOORS = (30, 40, 50, 60, 80)
BUILD, LEFT, OUT = sys.argv[1], {'045': sys.argv[2], '105': sys.argv[3]}, sys.argv[4]
res = {}
for tile in ('045', '105'):
    by = {(r['x'], r['y']): r for r in json.load(open(LEFT[tile]))}
    adds = {f: [] for f in FLOORS}
    for mod in ('nrca', 'nrcb'):
        off = Table.read(f'{BUILD}/o{tile}/band_{mod}_snr0.fits')
        nx = int(np.sum(np.asarray(off['seed_origin']).astype(str) == 'crossband'))
        ofs = SkyCoord(off['skycoord'])[nx:]
        for f in FLOORS:
            on = Table.read(f'{BUILD}/o{tile}/band_{mod}_snr{f:g}.fits')
            onsc = SkyCoord(on['skycoord'])[nx:]
            _, sep, _ = onsc.match_to_catalog_sky(ofs)
            adds[f].append(onsc[sep.to_value(u.mas) > 1e-3])
    allsc = {f: SkyCoord([a for a in adds[f] if len(a)]) for f in FLOORS}
    print(f'== o{tile}: rows added by floor ' + ', '.join(f'S/N>={f}: {len(allsc[f])}' for f in FLOORS))
    for k in TGT[tile]:
        r = by[k]
        c = SkyCoord(r['ra'] * u.deg, r['dec'] * u.deg)
        got = []
        for f in FLOORS:
            s = c.separation(allsc[f]).to_value(u.arcsec)
            got.append(f'{s.min():.3f}' if len(s) else '-')
        print(f'  {k} resid {r["resid_peak"]:7.1f} {r["cls"]:18s} nearest exemption row (arcsec) at floors {FLOORS}: {got}')
        res[f'{tile}|{k[0]},{k[1]}'] = got
json.dump(res, open(OUT, 'w'))
