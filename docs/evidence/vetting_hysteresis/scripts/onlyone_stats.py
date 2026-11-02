"""Data / residual S/N at the sources one variant's final catalog has and
the other's lacks (the ab_gallery_m7.py candidates), pooled over seeds.

usage: python onlyone_stats.py <varA> <varB> <field:band> [<field:band> ...]
Per field/band: n, how many have data S/N >= 5 / < 3 (PSF-matched, data
mosaic), A-residual S/N >= 5 / 2-5 / <= -3, B-residual S/N >= 5 / <= -3,
and distance to the nearest variant-B satstar.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_gallery_m7 as G  # noqa: E402


def main(va, vb, *specs):
    _, fields = G.RF.load_config()
    print(f'sources in {vb} m7 vetted with no {va} m7 vetted source within 1 px (all seeds)')
    print('| field / band | n | data S/N >= 5 | data S/N < 3 | A resid >= 5 | A resid 2-5 | A resid <= -3 '
          '| B resid >= 5 | B resid <= -3 | satstar <= 0.2" | satstar 0.2-0.5" |')
    print('|---' * 11 + '|')
    for s in specs:
        name, band = s.split(':')
        spec = fields[name]
        r = []
        for seed in [0] + list(spec['seeds']):
            c = G.candidates(spec, name, band.upper(), seed, va, vb)
            r += list(zip(c['snr_data'], c['snr_a'], c['snr_b'], c['sat_dist_as']))
        r = np.array(r, float).reshape(-1, 4)
        d, a, b, sd = r.T
        print(f'| {name} {band.upper()} | {len(r)} | {np.sum(d >= 5)} | {np.sum(d < 3)} | {np.sum(a >= 5)} | '
              f'{np.sum((a >= 2) & (a < 5))} | {np.sum(a <= -3)} | {np.sum(b >= 5)} | {np.sum(b <= -3)} | '
              f'{np.sum(sd <= 0.2)} | {np.sum((sd > 0.2) & (sd <= 0.5))} |')


if __name__ == '__main__':
    main(*sys.argv[1:])
