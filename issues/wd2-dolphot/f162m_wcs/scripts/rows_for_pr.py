"""ECSV rows (mean of two fields, sem = |row_1 - row_2| / (2 sqrt 2), the
per-component standard error of a two-point mean) for F162M and F164N."""
import importlib.util
import re
import sys

import numpy as np

WT = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-ffsign'
sys.path.insert(0, WT)
spec = importlib.util.spec_from_file_location('solver', f'{WT}/scripts/analysis/solve_filter_frame_offsets.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
DETS = ['NRCA1', 'NRCA2', 'NRCA3', 'NRCA4', 'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4']
SRC = {'F162M': [('datascale_f162m_run1.txt', 141.01), ('datascale_sgrc_f162m_e2.txt', 91.50)],
       'F164N': [('datascale_other.txt', 141.01), ('datascale_wd1_f164n.txt', 284.71)]}


def c0(path, band):
    txt = open(path).read()
    m = re.search(rf'### {band.lower()} - f212n.*?\n.*?\n(.*?)(?:\n\s*\n|\n###|\Z)', txt, re.S)
    return {p[0].upper(): (float(p[2]), float(p[3]))
            for p in (ln.split() for ln in m.group(1).splitlines()) if p and re.fullmatch(r'nrc[ab][1-4]', p[0])}


for band, srcs in SRC.items():
    rows = [solver.sky_residual_to_instrument_correction(c0(p, band), roll) for p, roll in srcs]
    for d in DETS:
        a, b = np.array(rows[0][d]), np.array(rows[1][d])
        mu = 0.5 * (a + b)
        sem = np.linalg.norm(a - b) / (2 * np.sqrt(2))
        print(f'{d} {band} F212N instrument {mu[0]:.3f} {mu[1]:.3f} 2 {sem:.3f}')
