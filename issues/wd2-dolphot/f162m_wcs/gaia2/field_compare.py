"""Compare the module-B / module-A (tr-, curl+) vectors between wd2 and wd1 after rotating the wd2 vector by +-dPA
(J_sky = R(phi) J_det acts on the output (sky) side only, so (tr-, curl+) rotates by phi)."""
import pickle
import numpy as np
K = 62.0; RAD = 206264.806
dPA = 284.867 - 140.820   # PA_V3 wd1 - wd2 (deg)
a = pickle.load(open('summary_wd2_q03s10.pkl', 'rb'))['S']
b = pickle.load(open('summary_wd1_q03s10.pkl', 'rb'))['S']
bands = ['f212n', 'f187n', 'f164n']
def rot(v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
m = lambda v: np.hypot(*v) / K * RAD
print(f'dPA(wd1-wd2) = {dPA:.1f} deg; vectors in mas/pix; mean over {bands}')
print('det    wd2 v (tr-,curl+) m   | wd1 v m  | wd2 rotated by +dPA: m, resid-to-wd1 m | rotated by -dPA: resid m')
for d in [f'nrc{x}{i}' for x in 'ab' for i in range(1, 5)]:
    va = np.mean([a[(bb, d)]['v'] for bb in bands if (bb, d) in a], axis=0)
    vb = np.mean([b[(bb, d)]['v'] for bb in bands if (bb, d) in b], axis=0)
    rp, rm = rot(va, dPA), rot(va, -dPA)
    print(f'{d} ({va[0]:+.4f},{va[1]:+.4f}) {m(va):5.1f} | ({vb[0]:+.4f},{vb[1]:+.4f}) {m(vb):5.1f} | rot+: ({rp[0]:+.4f},{rp[1]:+.4f}) resid {m(rp-vb):5.1f} | rot-: resid {m(rm-vb):5.1f} | unrotated resid {m(va-vb):5.1f}')
