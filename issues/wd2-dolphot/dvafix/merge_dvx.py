"""Per-band m7 merge for the dvafix arms (tree_ARM), production arguments.  Guard: every
production file the tree symlinks (stage manifest) keeps its size and mtime.
usage: python merge_dvx.py ARM BAND"""
import glob
import json
import os
import sys

ARM, BAND = sys.argv[1:3]
H = os.path.dirname(os.path.abspath(__file__))
T = f'{H}/tree_{ARM}'
assert os.environ.get('GC_BASEPATH_OVERRIDE') == T, os.environ.get('GC_BASEPATH_OVERRIDE')
man = {}
for _f in glob.glob(f'{H}/manifest_*.json'):
    man.update(json.load(open(_f)))


def guard(tag):
    bad = []
    for p, (size, mt) in man.items():
        st = os.stat(p)
        if st.st_size != size or st.st_mtime_ns != mt:
            bad.append(p)
    print(f'GUARD {tag}: {len(bad)} of {len(man)} production files changed', flush=True)
    for p in bad[:20]:
        print('  GUARD:', p, flush=True)
    return bad


b0 = guard('before merge')
from jwst_gc_pipeline.photometry.observation_merge import merge_frames_for_observation  # noqa: E402
import jwst_gc_pipeline  # noqa: E402
print('code', jwst_gc_pipeline.__file__, flush=True)
tbl = merge_frames_for_observation(
    '3523', '005', module='merged', filtername=BAND.lower(), method='dao',
    suffix='_basic', target='wd2', basepath=T, iteration_label='m7',
    bgsub=False, desat=False, epsf=False, blur=False, resbgsub=True,
    group=False, fwhm_basepath=T, n_spatial_chunks=1,
    merge_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
print('rows', len(tbl), flush=True)
b1 = guard('after merge')
sys.exit(1 if (b0 or b1) else 0)
