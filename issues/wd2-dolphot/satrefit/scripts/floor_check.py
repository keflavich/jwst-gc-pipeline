"""Check what the production wing self-calibration does on wd2 with the stock severity floors (F200W 4000; F150W/F250M/F300M none)."""
import sys, numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
import satrefit_core as C
S = C.S
for band, vg, det, lw in [('200W', '12101', 'nrcb1', False), ('150W', '10101', 'nrcb1', False)]:
    fn = f'{C.Q}/tree_main2/F{band}/pipeline/jw03523005001_{vg}_00001_{det}_align_o005_crf.fits'
    P = C.prep_frame(fn)
    hdr = P['header']
    grid, _ = C.load_grid(C.Q + '/tree_main2/psfs', C.fits.getheader(fn), lw)
    mim = np.nan_to_num(C.fits.getdata(fn.replace('.fits', '') + '_resbgsub_m7_satstar_model.fits').astype(float))
    fl = S._resolve_satstar_severity_floor(hdr['FILTER'])
    print('CHECK', band, 'severity floor', fl, flush=True)
    r = S._wing_selfcal(P['data'] - mim, P['err'], P['saturated'], grid, [1, 2, 3, 4, 5, 6, 8], fwhm_pix=1.6, severity_floor=fl)
    print('CHECK', band, 'result', r, flush=True)
