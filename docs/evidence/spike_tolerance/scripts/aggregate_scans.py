"""Tables of the spike-finder tolerance scans: counts, the added spikes, the star cores at risk and the limit they set.
usage: python -I aggregate_scans.py SCAN_DIR [--out FILE] [--confirm 0.5] [--kmax 4]
SCAN_DIR holds spikes_<tile>_<band>.json (written by ``scan_spikes.py`` of this directory).  READ ONLY."""
import argparse
import glob
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument('scan_dir')
ap.add_argument('--out', default=None)
ap.add_argument('--confirm', type=float, default=0.5,
                help='a core is a source when another frame holds at least this fraction of its excess at the same sky position')
ap.add_argument('--kmax', type=int, default=4, help='largest tolerance the star-core list carries (the scan script KDETAIL)')
args = ap.parse_args()

runs = []
for path in sorted(glob.glob(os.path.join(args.scan_dir, 'spikes_*.json'))):
    with open(path) as fh:
        runs.append(json.load(fh))
runs.sort(key=lambda r: (r['band'], r['tile']))
lines = []


def out(text=''):
    lines.append(text)
    print(text)


def fnum(v, fmt):
    return 'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)


out('Table 1: spikes found per tile and band, as shipped and with the tolerance k (finder run on every frame; MIN_EXCESS 300 MJy/sr)')
out('tile band   frames  shipped | k=1  k=2  k=3  k=4  k=5  k=6   (totals; the added count is the difference to shipped)')
for r in runs:
    t = r['totals']['tolerant_by_k']
    out(f"{r['tile']} {r['band']}  {r['n_frames']:>5}  {r['totals']['shipped']:>7} | " + '  '.join(f"{t[str(k)]:>3}" for k in range(1, 7)))
out()

out(f'Table 2: star cores at risk.  A core is a local maximum of a complete 3x3 box with excess >= {runs[0]["min_excess"]:g} MJy/sr and >= 5 sigma.')
out('Dropping its k brightest neighbours it passes the isolation test (brightest remaining neighbour <= 0.15 of the excess) = "passes at k".')
out('"star-like" = brightest neighbour >= 0.3 of the excess.  "source" = another frame holds >= %.2f of the excess at the same sky position.' % args.confirm)
out('tile band   cores   star-like | passes at k=0..7 (all)                    | star-like passes at k=0..7')
for r in runs:
    c = r['starcore']
    out(f"{r['tile']} {r['band']} {c['n_candidates']:>7} {c['n_starlike']:>9} | " + ' '.join(f'{v:>5}' for v in c['pass'])
        + ' | ' + ' '.join(f'{v:>5}' for v in c['pass_starlike']))
out()

out('Table 3: the cores that fail the isolation test as shipped and pass with their k brightest neighbours missing (k = 1 .. %d), by the other frames' % args.kmax)
out('tile band   k | cores | source (other frame >= %.2f) | untestable (no other frame) | single-frame' % args.confirm)
limit_ok = {k: True for k in range(1, args.kmax + 1)}
by_k_rows = []
for r in runs:
    for k in range(1, args.kmax + 1):
        cores = [v for v in r['starcore']['vulnerable'] if v['kmin'] <= k]
        src = [v for v in cores if v['n_other_frames'] and v['other_frac_max'] >= args.confirm]
        untest = [v for v in cores if not v['n_other_frames']]
        single = len(cores) - len(src) - len(untest)
        out(f"{r['tile']} {r['band']}  {k} | {len(cores):>5} | {len(src):>27} | {len(untest):>27} | {single:>12}")
        if src or untest:
            limit_ok[k] = False
        by_k_rows.append((r['tile'], r['band'], k, len(cores), len(src), len(untest)))
out()
safe = [k for k in range(1, args.kmax + 1) if all(limit_ok[j] for j in range(1, k + 1))]
out('Limit rule: the largest k for which no tile and band has a source or an untestable core among its cores that pass at k or less.')
out('  ok by k: ' + ', '.join(f'k={k}: {"yes" if limit_ok[k] else "NO"}' for k in limit_ok))
out(f'  largest safe k: {max(safe) if safe else 0}')
out()
out('Sources and untestable cores at k = 1 .. %d:' % args.kmax)
for r in runs:
    for v in sorted(r['starcore']['vulnerable'], key=lambda v: (v['kmin'], v['frame'], v['y'], v['x'])):
        if (v['n_other_frames'] and v['other_frac_max'] >= args.confirm) or not v['n_other_frames']:
            if v['kmin'] <= 2:
                out(f"  {r['tile']} {r['band']} kmin {v['kmin']} {v['frame']} ({v['x']}, {v['y']}) excess {v['excess']:.0f} brightest neighbours {v['r'][0]:.3f} {v['r'][1]:.3f} "
                    f"{v['r'][2]:.3f}, other frames {v['n_other_frames']} max fraction {fnum(v['other_frac_max'], '.2f')}, nearest multi-frame row {v['d_multiframe_fwhm']:.2f} FWHM")
out()

out('Table 4: the added spikes (found with the tolerance and not shipped), every tile and band, tolerance up to 6')
out('Columns: frame, pixel (x, y), excess MJy/sr, sigma, brightest usable neighbour / excess, missing neighbours (DQ values), distance to the nearest SATURATED pixel (px),')
out('distance to the nearest DO_NOT_USE pixel outside the 3x3 box (px), nearest vetted row (FWHM, nmatch), nearest vetted row with nmatch >= 2 (FWHM), nearest satstar row (FWHM),')
out('other frames covering the position and the largest fraction of the excess they hold there.')
n_added = {}
for r in runs:
    for a in sorted(r['added'], key=lambda a: (a['unusable'], a['frame'], a['y'], a['x'])):
        n_added.setdefault((r['tile'], r['band'], a['unusable']), []).append(a)
        out(f"{r['tile']} {r['band']} k={a['unusable']} {a['frame']} ({a['x']}, {a['y']}) exc {a['excess']:.0f} {a['nsigma']:.0f} sigma nb {a['nbfrac']:.4f} "
            f"missing {[m['dq'] for m in a['missing']]} dsat {fnum(a['d_sat_px'], '.0f')} ddnu {fnum(a['d_dnu_outside_px'], '.0f')} "
            f"vetted {a['d_vetted_fwhm']:.2f} (nmatch {a['nmatch_nearest']:.0f}) multi {a['d_multiframe_fwhm']:.2f} satstar {a['d_satstar_fwhm']:.1f} "
            f"others {a['n_other_frames']} max {fnum(a['other_frac_max'], '.2f')}")
out()

out('Table 5: added spikes within 1 FWHM of a catalog row, by tolerance')
out('tolerance | added | within 1 FWHM of a vetted row | of a vetted row with nmatch >= 2 | of a satstar row | other frame >= %.2f of the excess' % args.confirm)
for k in range(1, 7):
    sel = [a for r in runs for a in r['added'] if a['unusable'] == k]
    cum = [a for r in runs for a in r['added'] if a['unusable'] <= k]
    for label, s in ((f'k={k} only', sel), (f'k<={k}', cum)):
        out(f"{label:>9} | {len(s):>5} | {sum(a['d_vetted_fwhm'] < 1 for a in s):>29} | {sum(a['d_multiframe_fwhm'] < 1 for a in s):>33} | "
            f"{sum(a['d_satstar_fwhm'] < 1 for a in s):>16} | {sum(bool(a['n_other_frames']) and a['other_frac_max'] >= args.confirm for a in s):>10}")
out()
out('Table 6: the other-frame fraction of the cores that fail the isolation test as shipped and pass with their k brightest neighbours missing,')
out('by the largest fraction of the excess that another frame holds at the same sky position (cores with no other frame are in the last column)')
edges = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
labels = ['<0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5', '0.5-0.8', '>=0.8']
out('  k | ' + ' | '.join(f'{lab:>7}' for lab in labels) + ' | no other frame | smallest fraction at or above 0.3 | largest fraction below 0.3')
for k in range(1, args.kmax + 1):
    fr = [v['other_frac_max'] for r in runs for v in r['starcore']['vulnerable'] if v['kmin'] <= k and v['n_other_frames']]
    none = sum(1 for r in runs for v in r['starcore']['vulnerable'] if v['kmin'] <= k and not v['n_other_frames'])
    counts = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else float('inf')
        counts.append(sum(1 for f in fr if lo <= f < hi))
    above = [f for f in fr if f >= 0.3]
    below = [f for f in fr if f < 0.3]
    out(f'  {k} | ' + ' | '.join(f'{c:>7}' for c in counts) + f" | {none:>15} | {fnum(min(above) if above else None, '.3f'):>33} | {fnum(max(below) if below else None, '.3f'):>26}")
out()
out('Table 7: the spikes added at k = 1 (the shipped limit), summed over the tile-bands')
added1 = [a for r in runs for a in r['added'] if a['unusable'] == 1]
out(f"  count {len(added1)} (F212N {sum(1 for r in runs if r['band'] == 'F212N' for a in r['added'] if a['unusable'] == 1)}, "
    f"F480M {sum(1 for r in runs if r['band'] == 'F480M' for a in r['added'] if a['unusable'] == 1)})")
out(f"  excess {min(a['excess'] for a in added1):.0f} to {max(a['excess'] for a in added1):.0f} MJy/sr, "
    f"sigma {min(a['nsigma'] for a in added1):.0f} to {max(a['nsigma'] for a in added1):.0f}")
out(f"  brightest usable neighbour / excess: largest {max(a['nbfrac'] for a in added1):.4f}, median {sorted(a['nbfrac'] for a in added1)[len(added1) // 2]:.4f} (limit 0.15)")
dqs = {}
for a in added1:
    for m in a['missing']:
        key = (m['dq'], 'finite' if m['finite'] else 'NaN')
        dqs[key] = dqs.get(key, 0) + 1
out('  DQ of the missing neighbour (value, finite or NaN): ' + ', '.join(f'{k[0]} {k[1]} x{v}' for k, v in sorted(dqs.items())))
fr = [a['other_frac_max'] for a in added1 if a['n_other_frames']]
out(f"  frames that cover the position: {sum(1 for a in added1 if a['n_other_frames'])} of {len(added1)} have another frame, the largest fraction of the excess in another frame is {max(fr):.3f}")
out(f"  within 1 FWHM of a vetted row: {sum(a['d_vetted_fwhm'] < 1 for a in added1)}, of a vetted row with nmatch >= 2: {sum(a['d_multiframe_fwhm'] < 1 for a in added1)}, "
    f"of a satstar row: {sum(a['d_satstar_fwhm'] < 1 for a in added1)}")
out(f"  nearest SATURATED pixel: closest {min(a['d_sat_px'] for a in added1):.0f} px")
if args.out:
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
