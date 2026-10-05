"""S_seedtreas: one gc-treasury frame through remove_saturated_stars with the production
satstar environment for extended-emission NIRCam (as cataloging sets it), for the
SATSTAR_ZF_RCURVE_SATCHECK on/off check (x SATSTAR_ZF_KEEP_FINITE on/off).
usage: run_frame_rc.py <tree> <band> <crf basename>; env REPO_PATH = repo.
Writes only inside <tree>."""
import os, sys
REPO = os.environ['REPO_PATH']
sys.path.insert(0, REPO)
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/repos/webbpsf/data/')
W = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/S_seedtreas/'
tree, band, base = sys.argv[1:4]
assert os.path.realpath(tree).startswith(W) and tree.startswith(W), tree
fn = f'{tree}/{band}/pipeline/{base}'
assert os.path.islink(fn), fn
assert os.path.realpath(fn).startswith(f'/orange/adamginsburg/jwst/gc-treasury/{band}/pipeline/')
from jwst_gc_pipeline.photometry import cataloging  # noqa
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf  # noqa
assert ssf.__file__.startswith(REPO), ssf.__file__
# the extended-emission NIRCam defaults cataloging applies before the satstar stage
for k, v in (('NIRCAM_SATSTAR_TIGHT_BOUND', '1'), ('SATSTAR_COMPONENT_OVERLAP_FRAC', '0.5'),
             ('NIRCAM_SATSTAR_RECOVERED_CAP', '1'), ('SATSTAR_ERR_BKG_SCATTER', '1')):
    os.environ.setdefault(k, v)
cataloging._default_nircam_satstar_lock_env(os.environ, True)
sw = ssf.satstar_fit_switches()
print('CODE', ssf.__file__, 'SWITCHES', sw, 'SIG', ssf.satstar_fit_switch_signature(fn), flush=True)
print('ENV', {k: v for k, v in os.environ.items() if k.startswith(('SATSTAR', 'NIRCAM_SATSTAR'))}, flush=True)
ssf.remove_saturated_stars(fn, overwrite=True, path_prefix=f'{tree}/psfs', file_suffix='_rctest',
                           deblend_with_zeroframe=False)
