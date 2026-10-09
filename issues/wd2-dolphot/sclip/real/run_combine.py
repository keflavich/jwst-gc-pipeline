"""Run the pipeline's combine_singleframe on the main2kfpk per-frame m7 daophot tables of one band,
with whatever jwst_gc_pipeline is first on PYTHONPATH (main 3393a5f5 via wt-rshare, or the
f67ed28a fix via wt-sclip).  Read-only on the tree; writes one npz to OUT.
usage: PYTHONPATH=<worktree> python run_combine.py <band e.g. 277W> <tag>"""
import sys, glob, os, warnings
import numpy as np
from astropy.table import Table
warnings.filterwarnings('ignore')
from jwst_gc_pipeline.photometry import merge_catalogs as mc
print('merge_catalogs from', mc.__file__, flush=True)
T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2kfpk'
OUT = '/blue/adamginsburg/adamginsburg/tmp/claude-3663/sclip_real'
os.makedirs(OUT, exist_ok=True)
b, tag = sys.argv[1], sys.argv[2]
fns = sorted(glob.glob(f'{T}/F{b}/f{b.lower()}_*_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits'))
tabs = []
for fn in fns:
    t = Table.read(fn)
    if 'exposure' not in t.meta:
        t.meta['exposure'] = fn.split('_exp')[-1][:5]
    tabs.append(t)
print(len(tabs), 'frames', [len(t) for t in tabs], flush=True)
out = mc.combine_singleframe(tabs, offsets_table=None, filtername=f'F{b}')
print('output rows', len(out), flush=True)
c = out['skycoord_avg']
np.savez(f'{OUT}/comb_{tag}_{b}.npz', ra=np.asarray(c.ra.deg), dec=np.asarray(c.dec.deg),
         flux=np.asarray(out['flux_fit_avg'], float), nmatch=np.asarray(out['nmatch']),
         nmatch_good=np.asarray(out['nmatch_good']))
