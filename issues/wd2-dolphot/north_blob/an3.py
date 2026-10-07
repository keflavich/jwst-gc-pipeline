import numpy as np
from an import ca, fl
z = np.load('offs.npz')
mag = fl(ca['mag_vega_f150w'])
sat_all = np.ma.filled(ca['replaced_saturated_f150w'], 0).astype(bool)
for b, c in [('150W', (0.0, 0.68)), ('162M', (0.0, 0.70)), ('200W', (0.08, 0.75))]:
    si, x, y, A, B = [z[f'{b}_{k}'] for k in ['si', 'x', 'y', 'A', 'B']]
    smag = mag[si]
    print('band', b)
    for lo, hi in [(-99, 12), (12, 14), (14, 16), (16, 17), (17, 18), (18, 19), (19, 99)]:
        m = (smag >= lo) & (smag < hi)
        sm = sat_all & (mag >= lo) & (mag < hi)
        R = 0.12
        inb = m & (np.hypot(x - c[0], y - c[1]) < R)
        mir = m & (np.hypot(x + c[0], y + c[1]) < R)
        print(f'  mag {lo}-{hi}: nsat={sm.sum()} blob {inb.sum()} mirror {mir.sum()} excess/sat {(inb.sum()-mir.sum())/max(sm.sum(),1):+.3f}; neither {(inb&~A&~B).sum()} vs {(mir&~A&~B).sum()}')
