"""Run one m7 fan-out shard for the staged bands while building the m7 cross-band seed
from all 16 wd2 bands' m6 catalogs, as the full 16-band run does.
usage: python run_m7.py <crowdsource_catalogs_long args...>
env: PIPE_ROOT (repo), GC_BASEPATH_OVERRIDE (tree)."""
import os
import runpy
import sys

REPO = os.environ['PIPE_ROOT']
TREE = os.environ['GC_BASEPATH_OVERRIDE']
H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/dvafix/'
assert TREE.startswith(H) and os.path.realpath(TREE).startswith(H), TREE
sys.path.insert(0, REPO)
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402
assert C.__file__.startswith(REPO), C.__file__

ALL16 = ['F115W', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F212N',
         'F250M', 'F277W', 'F300M', 'F323N', 'F335M', 'F405N', 'F410M', 'F466N']
_orig = C._build_crossband_seed


def _seed_all16(cut_bp, modules, filternames, options, **kw):
    print(f'[run_m7] cross-band seed from all 16 bands (run filters: {filternames})', flush=True)
    return _orig(cut_bp, modules, ALL16, options, **kw)


C._build_crossband_seed = _seed_all16
print('CODE', C.__file__, 'TREE', TREE, 'HANDOFF', os.environ.get('DAOPHOT_HANDOFF_UNACCEPTED_SAT'),
      'KEEP_FINITE', os.environ.get('SATSTAR_ZF_KEEP_FINITE'), flush=True)
sys.argv = ['crowdsource_catalogs_long'] + sys.argv[1:]
runpy.run_module('jwst_gc_pipeline.photometry.crowdsource_catalogs_long', run_name='__main__')
