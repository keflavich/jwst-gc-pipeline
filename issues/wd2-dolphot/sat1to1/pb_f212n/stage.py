"""Stage tree_pb0 / tree_pb1 for the F212N per-band merge A/B (#1122 review).

Each tree gets real directories holding per-file symlinks into tree_mainfcbg,
so every file the merge writes lands in the new tree.  catalogs/ starts empty.
"""
import os
import sys

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
SRC = f'{Q}/tree_mainfcbg'
HERE = os.path.dirname(os.path.abspath(__file__))

for arm in sys.argv[1:] or ['pb0', 'pb1']:
    T = f'{HERE}/tree_{arm}'
    os.makedirs(f'{T}/F212N/pipeline', exist_ok=True)
    os.makedirs(f'{T}/catalogs', exist_ok=True)
    n = 0
    for name in sorted(os.listdir(f'{SRC}/F212N')):
        if name == 'pipeline' or not name.endswith('_m7_daophot_basic.fits'):
            continue
        dst = f'{T}/F212N/{name}'
        if not os.path.lexists(dst):
            os.symlink(f'{SRC}/F212N/{name}', dst)
            n += 1
    m = 0
    for name in sorted(os.listdir(f'{SRC}/F212N/pipeline')):
        src = f'{SRC}/F212N/pipeline/{name}'
        if os.path.isdir(src):
            continue
        dst = f'{T}/F212N/pipeline/{name}'
        if not os.path.lexists(dst):
            os.symlink(src, dst)
            m += 1
    for sub in ('offsets',):
        os.makedirs(f'{T}/{sub}', exist_ok=True)
        for name in sorted(os.listdir(f'{SRC}/{sub}')):
            dst = f'{T}/{sub}/{name}'
            if not os.path.lexists(dst):
                os.symlink(f'{SRC}/{sub}/{name}', dst)
    print(f'{arm}: {n} per-frame catalogs, {m} pipeline files linked')
