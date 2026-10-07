"""Run only the m7 cross-band merge (merge_daophot, vetted) on a scratch tree.
usage: python run_merge.py ARM REPO"""
import sys
arm, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from jwst_gc_pipeline.photometry import merge_catalogs as M
print('merge_catalogs from', M.__file__, flush=True)
assert M.__file__.startswith(repo)
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
BANDS = ['f115w', 'f150w', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f212n', 'f250m', 'f277w', 'f300m',
         'f323n', 'f335m', 'f405n', 'f410m', 'f466n']
M.merge_daophot(module='merged', daophot_type='basic', indivexp=True, desat=False, bgsub=False, blur=False,
                resbgsub=True, iteration_label='m7', target='wd2', basepath=f'{Q}/tree_{arm}',
                ref_filter='f410m', filternames_override=BANDS, field='005', progid='3523', vetted=True)
