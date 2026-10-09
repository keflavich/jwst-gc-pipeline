"""Round-7f satstar refit (rstar/run_frames7f.py, unchanged) with the cached STPSF grid read by the #1154 loader
(psf_grid_io.load_stpsf_grid, each plane at its computed position) instead of the pre-#1154 transposed loader.
With the fixed grid, satstar dm against dolphot should carry pred only (the dolphot pixel-area double count, #1153),
so the #1150 test is whether the Hf slope against 2.5 log10(flat), after subtracting pred, is near 0 on all four
F150W/F200W x nrcb1/nrcb3 subsets.  usage: python run_frames7g.py BAND DET EXP  -> flatarea/out7g/"""
import sys
import importlib.util
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, f'{Q}/satrefit/rstar')
sys.path.insert(0, f'{Q}/satrefit')
import satrefit_core as C  # noqa: E402

PGI_FILE = ('/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-zfflatm/'
            'jwst_gc_pipeline/photometry/psf_grid_io.py')
_spec = importlib.util.spec_from_file_location('psf_grid_io_main', PGI_FILE)
PGI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PGI)


def load_grid_fixed(psfdir, header, lw):
    det = header['DETECTOR'].lower()
    det = {'nrcalong': 'nrca5', 'nrcblong': 'nrcb5'}.get(det, det)
    filt = C.S.get_filtername(header).lower()
    fov = 1024 if lw else 512
    fn = f'{psfdir}/nircam_{det}_{filt}_fovp{fov}_samp2_npsf16.fits'
    g = PGI.load_stpsf_grid(fn)
    C.log('fixed-loader grid', fn, 'xypos[:3]', g.grid_xypos[:3])
    return g, fn


C.load_grid = load_grid_fixed
import run_frames7f as R7  # noqa: E402
R7.OUT = f'{Q}/flatarea/out7g'

if __name__ == '__main__':
    R7.run_refit(sys.argv[1], sys.argv[2], int(sys.argv[3]))
