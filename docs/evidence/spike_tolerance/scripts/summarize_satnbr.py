"""Tables of the SATURATED-neighbour measurement from satnbr_<tile>_<band>.json (written by scan_satnbr.py).

usage: python -I summarize_satnbr.py OUTDIR > summary.txt"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

OUT = sys.argv[1]
TILES = ['o045', 'o105', 'o118', 'o130', 'o138']
BANDS = ['F212N', 'F480M']
TAGS = ['prod5', 'optnm', 'optfin']
THR = [50, 100, 200, 500]
SINGLE, CONFIRMED = 0.15, 0.5
RADIUS_ARCSEC = None


def conf(b):
    if b['n_other_frames'] == 0:
        return 'none'
    f = b['other_frac_max']
    return 'single' if f < SINGLE else ('confirmed' if f >= CONFIRMED else 'between')


def unit(ra, dec):
    ra, dec = np.radians(np.atleast_1d(ra)), np.radians(np.atleast_1d(dec))
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T


def chord_of(arcsec):
    return 2.0 * np.sin(np.radians(arcsec / 3600.0) / 2.0)


data = {}
for t in TILES:
    for b in BANDS:
        with open(f'{OUT}/satnbr_{t}_{b}.json') as fh:
            data[(t, b)] = json.load(fh)
RADIUS_ARCSEC = next(iter(data.values()))['match_arcsec']
FLOOR = next(iter(data.values()))['floor']
SHIP = next(iter(data.values()))['ship_min']
print(f'finder tolerance max_unusable = {next(iter(data.values()))["max_unusable"]}; blocked spikes kept from excess {FLOOR:g} MJy/sr; '
      f'recipe spike floor {SHIP:g} MJy/sr; match radius {RADIUS_ARCSEC} arcsec; confirmation: single < {SINGLE}, confirmed >= {CONFIRMED} '
      f'(3x3 maximum at the sky position in the other frames, fraction of the excess)\n')

print('## Table 1: blocked spikes per tile-band (a pixel the finder with SATURATED cleared returns and the finder as shipped does not)')
print('| tile | band | frames | blocked | >= 300 | >= 200 | single | between | confirmed | no other frame | isolated flagged px (blob < 5 px) | in a big blob (>= 25 px) | JUMP_DET in >= 6 neighbours |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
tot = dict(n=0, n300=0, n200=0, single=0, between=0, confirmed=0, none=0, iso=0, bigb=0, jump=0)
for (t, b), d in data.items():
    bl = d['blocked']
    c = [conf(x) for x in bl]
    n300 = sum(x['excess'] >= 300 for x in bl)
    n200 = sum(x['excess'] >= 200 for x in bl)
    iso = sum(x['max_comp_area'] < 5 for x in bl)
    bigb = sum(x['max_comp_area'] >= d['big_blob_px'] for x in bl)
    jump = sum(x['n_jump_nb'] >= 6 for x in bl)
    print(f'| {t} | {b} | {d["n_frames"]} | {len(bl)} | {n300} | {n200} | {c.count("single")} | {c.count("between")} | {c.count("confirmed")} | {c.count("none")} | {iso} | {bigb} | {jump} |')
    for k, v in (('n', len(bl)), ('n300', n300), ('n200', n200), ('single', c.count('single')), ('between', c.count('between')),
                 ('confirmed', c.count('confirmed')), ('none', c.count('none')), ('iso', iso), ('bigb', bigb), ('jump', jump)):
        tot[k] += v
print(f'| all | all | | {tot["n"]} | {tot["n300"]} | {tot["n200"]} | {tot["single"]} | {tot["between"]} | {tot["confirmed"]} | {tot["none"]} | {tot["iso"]} | {tot["bigb"]} | {tot["jump"]} |\n')

print('## Table 2: leftovers of the residual mosaics with a blocked spike within the match radius (all eight tile-bands, the excess of the spike is any >= floor)')
print('| render | leftovers >= | n | with blocked spike | with blocked >= 300 | of those: single | with shipped spike (>= 300) | with blocked and no shipped | with neither |')
print('|---|---|---|---|---|---|---|---|---|')
for tag in TAGS:
    for thr in THR:
        n = nb = nb300 = nsingle = nship = nbonly = nneither = 0
        for (t, b), d in data.items():
            bl = d['blocked']
            for r in d['leftovers'][tag]:
                if r['resid_peak'] < thr:
                    continue
                n += 1
                has_b = bool(r['blocked'])
                has_b300 = any(bl[j]['excess'] >= 300 for j in r['blocked'])
                has_s = bool(r['shipped'])
                nb += has_b
                nb300 += has_b300
                nsingle += any(conf(bl[j]) == 'single' for j in r['blocked'])
                nship += has_s
                nbonly += has_b and not has_s
                nneither += (not has_b) and (not has_s)
        print(f'| {tag} | {thr} | {n} | {nb} | {nb300} | {nsingle} | {nship} | {nbonly} | {nneither} |')
print()

print('## Table 2b: the same for the renders, per tile-band at >= 100 MJy/sr (n, with blocked, with blocked >= 300, with shipped)')
print('| tile | band | ' + ' | '.join(f'{tag}' for tag in TAGS) + ' |')
print('|---|---|' + '---|' * len(TAGS))
for (t, b), d in data.items():
    cells = []
    for tag in TAGS:
        rows = [r for r in d['leftovers'][tag] if r['resid_peak'] >= 100]
        bl = d['blocked']
        cells.append(f'{len(rows)}, {sum(bool(r["blocked"]) for r in rows)}, '
                     f'{sum(any(bl[j]["excess"] >= 300 for j in r["blocked"]) for r in rows)}, {sum(bool(r["shipped"]) for r in rows)}')
    print(f'| {t} | {b} | ' + ' | '.join(cells) + ' |')
print()

print('## Table 3: blocked spikes with excess >= 300 and a leftover of the render within the match radius (converse direction)')
print('| render | leftover >= | blocked >= 300 | with a leftover | single-frame ones with a leftover |')
print('|---|---|---|---|---|')
for tag in TAGS:
    for thr in (50, 100, 200):
        nbl = nwith = nsw = 0
        for (t, b), d in data.items():
            bl = d['blocked']
            hit = set()
            for r in d['leftovers'][tag]:
                if r['resid_peak'] >= thr:
                    hit.update(r['blocked'])
            for j, x in enumerate(bl):
                if x['excess'] < 300:
                    continue
                nbl += 1
                if j in hit:
                    nwith += 1
                    nsw += conf(x) == 'single'
        print(f'| {tag} | {thr} | {nbl} | {nwith} | {nsw} |')
print()

print('## Chance matches: the blocked spikes shifted by 1.0 to 3.0 arcsec in eight directions, leftovers >= 100 of optfin and optnm with a blocked spike (mean over the 8 shifts, all tile-bands)')
rad = chord_of(RADIUS_ARCSEC)
for tag in ('optnm', 'optfin'):
    counts = []
    real = 0
    for shift in (1.0, 1.5, 2.0, 2.5, 3.0):
        for ang in np.arange(8) * np.pi / 4:
            n = 0
            for (t, b), d in data.items():
                if not d['blocked']:
                    continue
                dec0 = np.array([x['dec'] for x in d['blocked']])
                ra0 = np.array([x['ra'] for x in d['blocked']])
                dec1 = dec0 + shift / 3600.0 * np.sin(ang)
                ra1 = ra0 + shift / 3600.0 * np.cos(ang) / np.cos(np.radians(dec0))
                tree = cKDTree(unit(ra1, dec1))
                for r in d['leftovers'][tag]:
                    if r['resid_peak'] >= 100 and tree.query_ball_point(unit(r['ra'], r['dec'])[0], rad):
                        n += 1
            counts.append(n)
    for (t, b), d in data.items():
        real += sum(1 for r in d['leftovers'][tag] if r['resid_peak'] >= 100 and r['blocked'])
    print(f'{tag}: real matches {real}; shifted matches mean {np.mean(counts):.2f}, max {max(counts)} over {len(counts)} shifts')
print()

print('## Table 4: leftovers >= 100 of the masked render (optfin) with a blocked spike, largest first')
print('| tile | band | leftover (x, y) | residual peak | class | spike frame | spike (x, y) | spike excess | sigma | SATURATED neighbours | their flux / excess | blob area | JUMP_DET neighbours | other frames: n, max fraction | spike class |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
rows = []
for (t, b), d in data.items():
    bl = d['blocked']
    for r in d['leftovers']['optfin']:
        if r['resid_peak'] >= 100 and r['blocked']:
            x = bl[r['blocked'][0]]
            rows.append((r['resid_peak'], t, b, r, x))
for pk, t, b, r, x in sorted(rows, key=lambda v: -v[0]):
    fr = x['frame'].replace('jw10678', '').replace('_destreak_' + t + '_crf.fits', '')
    sf = ', '.join(f'{v:.2f}' for v in x['sat_nb_frac'])
    print(f'| {t} | {b} | ({r["x"]}, {r["y"]}) | {pk:.0f} | {r["cls"]} | {fr} | ({x["x"]}, {x["y"]}) | {x["excess"]:.0f} | {x["nsigma"]:.0f} | {x["n_sat_nb"]} | {sf} | '
          f'{x["max_comp_area"]} | {x["n_jump_nb"]} | {x["n_other_frames"]}, {x["other_frac_max"]:.2f} | {conf(x)} |')
print()

print('## Table 5: blocked spikes of excess >= 300 that another frame confirms (fraction >= 0.5): the cores at risk if a SATURATED neighbour were tolerated')
print('| tile | band | frame | (x, y) | excess | sigma | SATURATED neighbours | their flux / excess | blob area | other frames: n, max fraction | distance to vetted row (arcsec) | nmatch | distance to satstar row (arcsec) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for (t, b), d in data.items():
    for x in sorted(d['blocked'], key=lambda v: -v['excess']):
        if x['excess'] >= 300 and conf(x) == 'confirmed':
            fr = x['frame'].replace('jw10678', '').replace('_destreak_' + t + '_crf.fits', '')
            sf = ', '.join(f'{v:.2f}' for v in x['sat_nb_frac'])
            print(f'| {t} | {b} | {fr} | ({x["x"]}, {x["y"]}) | {x["excess"]:.0f} | {x["nsigma"]:.0f} | {x["n_sat_nb"]} | {sf} | {x["max_comp_area"]} | '
                  f'{x["n_other_frames"]}, {x["other_frac_max"]:.2f} | {x["d_vetted_arcsec"]:.3f} | {x["nmatch_nearest"]:.0f} | {x["d_satstar_arcsec"]:.3f} |')
