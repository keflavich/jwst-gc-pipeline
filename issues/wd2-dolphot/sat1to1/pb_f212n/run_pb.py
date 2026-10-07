"""F212N per-band merge for the #1122 A/B.  The code under test comes from
PYTHONPATH (wd2main2 = main 5434f5e7, sat1to1 = PR #1122).

Guard: the (mtime, size) of every file in tree_mainfcbg/F212N and
tree_mainfcbg/catalogs is recorded before the merge and compared after it.
"""
import os
import sys

ARM = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
T = f'{HERE}/tree_{ARM}'
SRC = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_mainfcbg'
assert os.environ.get('GC_BASEPATH_OVERRIDE') == T, os.environ.get('GC_BASEPATH_OVERRIDE')


def snapshot():
    out = {}
    for d in (f'{SRC}/F212N', f'{SRC}/F212N/pipeline', f'{SRC}/catalogs'):
        for name in os.listdir(d):
            p = os.path.realpath(f'{d}/{name}')
            if os.path.isfile(p):
                st = os.stat(p)
                out[p] = (st.st_mtime_ns, st.st_size)
    return out


before = snapshot()
from jwst_gc_pipeline.photometry import merge_catalogs as mc  # noqa: E402
print('merge_catalogs from', mc.__file__, flush=True)
print('has _one_to_one_satstar_pairs:', hasattr(mc, '_one_to_one_satstar_pairs'), flush=True)
from jwst_gc_pipeline.photometry.observation_merge import merge_frames_for_observation  # noqa: E402

tbl = merge_frames_for_observation(
    '3523', '005', module='merged', filtername='f212n', method='dao',
    suffix='_basic', target='wd2', basepath=T, iteration_label='m7',
    bgsub=False, desat=False, epsf=False, blur=False, resbgsub=True,
    group=False, fwhm_basepath=T, n_spatial_chunks=1,
    merge_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
print('rows', len(tbl), flush=True)

after = snapshot()
changed = [p for p in before if after.get(p) != before[p]]
added = [p for p in after if p not in before]
print(f'GUARD: {len(changed)} tree_mainfcbg file(s) changed, {len(added)} added', flush=True)
for p in changed + added:
    print('  GUARD:', p, flush=True)
sys.exit(1 if changed else 0)
