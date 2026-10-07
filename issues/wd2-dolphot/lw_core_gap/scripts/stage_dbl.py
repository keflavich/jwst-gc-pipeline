"""Stage F277W-only trees for the deblend A/B (symlinks to production inputs,
never anything m7/m8 writes).  Same rules as Q_integ/stage_trees.py, then the
F277W m6 smoothed-bg symlink is replaced by a copy from tree_integbg (as in
stage_main2.sh, for --residual-bg-median-size=0).
usage: python stage_dbl.py ARM [ARM ...]"""
import json
import os
import re
import shutil
import sys

R = '/orange/adamginsburg/jwst/wd2'
Q = f'{R}/dolphot_benchmark/Q_integ'
H = f'{Q}/f277w_gap/dbl'
ARMS = sys.argv[1:]
BANDS = os.environ.get('BANDS', 'F277W').split()
DENY = re.compile(r'(_m7(?![0-9])|m7_|_m8|m8_|wingcal|consolidated_satstar|crossband_seed|_perframe_markers)')
COPY = re.compile(r'(_mergedcat_grid_.*\.asdf$|^gaia.*refcat.*\.fits$)')
SKIP_SUFFIX = ('_asn.json',)
manifest = {}


def place(src, dstdir):
    name = os.path.basename(src)
    dst = os.path.join(dstdir, name)
    if os.path.lexists(dst):
        os.remove(dst)
    if COPY.search(name):
        shutil.copy2(src, dst)
        return
    os.symlink(src, dst)
    real = os.path.realpath(src)
    st = os.stat(real)
    manifest[real] = [st.st_size, st.st_mtime_ns]


def stage_dir(srcdir, dstdir):
    os.makedirs(dstdir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(srcdir)):
        src = os.path.join(srcdir, name)
        if not os.path.isfile(src) or DENY.search(name) or name.endswith(SKIP_SUFFIX):
            continue
        place(src, dstdir)
        n += 1
    return n


for arm in ARMS:
    T = f'{H}/tree_{arm}'
    assert T.startswith(H)
    for b in BANDS:
        print(arm, b, stage_dir(f'{R}/{b}/pipeline', f'{T}/{b}/pipeline'), flush=True)
    if os.environ.get('STAGE_COMMON', '1') == '1':
        for d in ('catalogs', 'psfs', 'regions_'):
            print(arm, d, stage_dir(f'{R}/{d}', f'{T}/{d}'), flush=True)
        os.makedirs(f'{T}/offsets', exist_ok=True)
        for f in os.listdir(f'{R}/offsets'):
            shutil.copy2(f'{R}/offsets/{f}', f'{T}/offsets/{f}')
    for b in BANDS:
        name = (f'jw03523-o005_t001_nircam_clear-{b.lower()}-merged_resbgsub_m6_daophot_basic_'
                f'mergedcat_residual_smoothed_bg_i2d.fits')
        src = f'{Q}/tree_mainfcbg/{b}/pipeline/{name}'
        assert os.path.isfile(src) and not os.path.islink(src), src
        dst = f'{T}/{b}/pipeline/{name}'
        if os.path.islink(dst):
            os.remove(dst)
        assert not os.path.exists(dst), dst
        shutil.copy2(src, dst)
        print(arm, 'copied m6 smoothed bg', name, flush=True)
json.dump(manifest, open(f'{H}/manifest_{"_".join(ARMS)}_{"_".join(BANDS)}.json', 'w'))
print(len(manifest), 'symlink targets in manifest')
