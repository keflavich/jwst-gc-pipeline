"""Compare selections: per band/detector Gaia stars/pairs, rms, m (16-84) and beta; reads summary_<TAG>.pkl."""
import pickle, sys
import numpy as np
tags = sys.argv[2:]
field = sys.argv[1]
DETS = [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]
for t in tags:
    P = pickle.load(open(f'summary_{t}.pkl', 'rb'))
    S, B = P['S'], P['BETA']
    print(f'\n##### {t}')
    bands = sorted({k[0] for k in S})
    print('band  ' + ' '.join(f'{d[3:]:>20s}' for d in DETS))
    for b in bands:
        row = []
        for d in DETS:
            s = S.get((b, d))
            row.append('-'.rjust(20) if s is None else f'{s["n"]:2d}/{s["npair"]:3d} {s["s"]:4.1f} {s["m"]:4.0f}({np.percentile(s["mb"],16):.0f}-{np.percentile(s["mb"],84):.0f})'.rjust(20))
        print(f'{b:5s} ' + ' '.join(row))
    print('(entries: Ngaia/Npairs rms_mas m(16-84 arcsec))')
    for (b, g), v in sorted(B.items()):
        if g in ('A2+A3', 'A1-A4'):
            print(f'  beta {b:6s} {g:6s} {v[0]:+.2f} +- {v[1]:.2f}')
