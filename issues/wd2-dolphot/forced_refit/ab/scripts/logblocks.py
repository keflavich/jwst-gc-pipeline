"""Pair log blocks (flag/refit/phantom-drop/nonpositive-drop counts) with per-frame
catalogs by final length L = M - phantom - nonpos; sum per band and arm."""
import re, glob, os, collections
import numpy as np
from astropy.io import fits
H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/forced_refit/ab'
blocks = {'fr0': [], 'fr1': []}
for f in glob.glob(f'{H}/logs/wd2-fr_*.out'):
    arm = None; cur = None
    for l in open(f, errors='replace'):
        if l.startswith('ARM fr'):
            arm = l.split()[1]
        m = re.match(r'\[m7\] model/data-peak overshoot \(>1\.2x\): (\d+)/(\d+) fits', l)
        if m and arm:
            cur = dict(flag=int(m.group(1)), M=int(m.group(2)), refit=0, ph=0, neg=0); blocks[arm].append(cur); continue
        if cur is None: continue
        m = re.match(r'\[m7\] refit (\d+) overshooting', l)
        if m: cur['refit'] = int(m.group(1))
        m = re.match(r'\[m7\] dropped (\d+) phantom', l)
        if m: cur['ph'] = int(m.group(1))
        m = re.match(r'\[m7\] dropped (\d+) non-positive', l)
        if m: cur['neg'] = int(m.group(1))
print({a: len(v) for a, v in blocks.items()})
for arm in blocks:
    byL = collections.defaultdict(list)
    for bl in blocks[arm]:
        byL[bl['M'] - bl['ph'] - bl['neg']].append(bl)
    S = collections.defaultdict(lambda: collections.Counter())
    unm = 0
    for b in ['F150W', 'F187N', 'F200W', 'F277W']:
        for g in glob.glob(f'{H}/tree_{arm}/{b}/*_m7_daophot_basic.fits'):
            n = fits.getheader(g, 1)['NAXIS2']
            c = byL.get(n, [])
            if len(c) != 1:
                unm += 1; continue
            for k, v in c[0].items(): S[b][k] += v
            S[b]['nframes'] += 1
    print(arm, 'unmatched/ambiguous frames', unm)
    for b in S: print(' ', b, dict(S[b]))
