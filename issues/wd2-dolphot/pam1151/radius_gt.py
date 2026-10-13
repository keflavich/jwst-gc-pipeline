"""Radius dependence of the (e) psf - ap_area two-regressor fit, both arms; pred spread per band. usage: python radius_gt.py"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, f'{Q}/apclosure')
sys.argv = sys.argv[:1]
import importlib.util
spec = importlib.util.spec_from_file_location('ap', f'{Q}/apclosure/analyze_ap.py')
src = open(f'{Q}/apclosure/analyze_ap.py').read().replace('\nmain()\n', '\n')
ap = importlib.util.module_from_spec(spec)
exec(compile(src, spec.origin, 'exec'), ap.__dict__)
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {'main2 (pre-#1154 loader)': f'{Q}/apclosure/frames_{{b}}.ecsv',
        'main2gt (#1154 loader)': f'{HERE}/frames_main2gt_{{b}}.ecsv'}
class G:
    Q, HERE, ARMS = Q, HERE, ARMS


for band in ('200W', '212N', '150W', '250M', '300M', '323N', '410M', '405N'):
    for arm in G.ARMS:
        ap.HERE = f'{G.Q}/apclosure' if 'pre' in arm else f'{G.HERE}/gt'
        rec = ap.stars(band)
        out = []
        for r in ap.RADII:
            D = ap.diffs(band, rec, r)
            c, lo, hi, n, rho = ap.fit2(D['e'], D['pred'], D['predT'])
            out.append(f'r{r}: A {c[0]:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}] B {c[1]:+.2f}')
        print(f'F{band} {arm[:7]:7s} predσ {ap.rs(D["pred"]):.4f} corr {rho:+.2f} | ' + ' | '.join(out), flush=True)
