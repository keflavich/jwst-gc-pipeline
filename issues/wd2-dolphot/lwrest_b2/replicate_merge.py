"""Replicate combine_singleframe (read-only import of the main2kfpk repo) for one band and
map each pre-replacement merged row to its final row in the per-band merged catalog.
usage: python replicate_merge.py 277W  -> pre_BAND.ecsv-like pickle-free outputs: pre_map_BAND.ecsv (small)"""
import sys, glob, re, warnings, io, contextlib
import numpy as np
from astropy.table import Table
import astropy.units as u
warnings.filterwarnings('ignore')
REPO = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2kfpk'
sys.path.insert(0, REPO)
from jwst_gc_pipeline.photometry import merge_catalogs as MC
assert MC.__file__.startswith(REPO)
b = sys.argv[1].upper().lstrip('F')
T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2kfpk'
fns = sorted(glob.glob(f'{T}/F{b}/f{b.lower()}_*_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits'))
print(len(fns), flush=True)
tabs = []
for fn in fns:
    t = Table.read(fn)
    t.meta['exposure'] = fn.split('_exp')[-1][:5]
    tabs.append(t)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pre = MC.combine_singleframe(tabs, filtername=f'f{b.lower()}', nanaverage=MC.nanaverage_numpy)
print(len(pre), pre.colnames[:12])
pre.write(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits', overwrite=True)
