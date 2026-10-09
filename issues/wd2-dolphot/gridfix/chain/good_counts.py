"""Per band: dolphot stars with a good value (|dm| < 0.3) in arm A and arm B, gained/lost, and the dolphot-mag
distribution of the stars that change status.  usage: python analyze_g.py gridfix/chain/good_counts.py main2 main2gt"""
import sys
import numpy as np
import analyze as an
a, b = sys.argv[1:3]
an.ZPWIN.update(an.zp_windows())
A, B = an.Arm(a), an.Arm(b)
print(f'| band | good {a} | good {b} | net | gained / lost | median dolphot mag gained / lost | lost with \\|dm_{a}\\| in 0.2-0.3 | lost with no {b} value |')
print('|---|---|---|---|---|---|---|---|')
tot = np.zeros(2, int)
for band in an.BANDS:
    dA = np.where(A.matched, A.dm(band), np.nan); dB = np.where(B.matched, B.dm(band), np.nan)
    gA = np.abs(dA) < 0.3; gB = np.abs(dB) < 0.3
    gain = gB & ~gA; lost = gA & ~gB
    ref = A.ref[band]
    tot += [gA.sum(), gB.sum()]
    mg = np.nanmedian(ref[gain]) if gain.any() else np.nan
    ml = np.nanmedian(ref[lost]) if lost.any() else np.nan
    print(f'| F{band} | {gA.sum()} | {gB.sum()} | {gB.sum()-gA.sum():+d} | {gain.sum()} / {lost.sum()} | {mg:.2f} / {ml:.2f} | '
          f'{(lost & (np.abs(dA) > 0.2)).sum()} | {(lost & ~np.isfinite(dB)).sum()} |')
print(f'| total | {tot[0]} | {tot[1]} | {tot[1]-tot[0]:+d} | | | | |')
