"""Sum the hysteresis keeps that the cataloging logs report, per field, band and phase.

Each phase that applies the keep prints
``[<phase>:<band>] previous-phase keep: +N vetted in the previous phase ...``
(``_filter_extended_emission``), and with the satstar guard
``...; M refused within R" of a saturated star``.  N and M count sources
over the whole cutout (not only the evaluated inner box).  The table gives
``N`` per phase, or ``N / M`` when the logs carry the guard count.

usage: python keep_counts.py <log_dir> <variant>
       reads <log_dir>/ref_<field>_<variant>_s*.log
"""
import glob
import os
import re
import sys
from collections import defaultdict

PAT = re.compile(r'\[(\w+):(\w+)\] previous-phase keep: \+(\d+)(?:.*?; (\d+) refused)?')


def main(logdir, variant):
    sums = defaultdict(int)
    refused = defaultdict(int)
    guard = False
    nlogs = defaultdict(int)
    for path in sorted(glob.glob(os.path.join(logdir, f'ref_*_{variant}_s*.log'))):
        field = re.match(rf'ref_(\w+?)_{variant}_s\d+', os.path.basename(path)).group(1)
        nlogs[field] += 1
        with open(path) as fh:
            for line in fh:
                m = PAT.search(line)
                if m:
                    sums[(field, m.group(2), m.group(1))] += int(m.group(3))
                    if m.group(4) is not None:
                        guard = True
                        refused[(field, m.group(2), m.group(1))] += int(m.group(4))
    phases = sorted({k[2] for k in sums}, key=lambda p: int(p[1:]))
    print(f'hysteresis keeps{" / satstar-guard refusals" if guard else ""}, '
          f'variant {variant}, summed over runs (whole cutout)')
    print('| field / band | runs | ' + ' | '.join(phases) + ' | total |')
    print('|---' * (len(phases) + 3) + '|')
    for field, band in sorted({k[:2] for k in sums}):
        row = [sums.get((field, band, p), 0) for p in phases]
        cells = [str(v) for v in row] + [str(sum(row))]
        if guard:
            ref = [refused.get((field, band, p), 0) for p in phases]
            cells = [f'{c} / {r}' for c, r in zip(cells, ref + [sum(ref)])]
        print(f'| {field} {band} | {nlogs[field]} | ' + ' | '.join(cells) + ' |')
    for field in sorted(nlogs):
        tot = sum(v for k, v in sums.items() if k[0] == field)
        print(f'| {field} (all bands) | {nlogs[field]} | ' + ' | '.join('' for _ in phases) + f' | {tot} |')


if __name__ == '__main__':
    main(*sys.argv[1:])
