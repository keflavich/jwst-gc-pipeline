"""#1148 review (rimcount layout, all frames): rim value under the measured curve (SATSTAR_ZF_R_HEADER=0) and the header rate (=1),
against the pedestal-aware expectation R_header g0 + B, on the frames that move most.

B is the crf level not tracked by group 0: B_dn = median(crf / R_header - g0) over far-field pixels
(edt >= 25, not SATURATED or DO_NOT_USE) with 500 <= g0 < 2000, in group-0 DN.  The rim expectation
is R_header (g0 + B_dn).  PR branch code (jwst-gc-pipeline-rcexcl), cap on, read-only.
"""
import os
import sys
import numpy as np
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
os.environ['NIRCAM_SATSTAR_RECOVERED_CAP'] = '1'
os.environ.pop('SATSTAR_ZF_KEEP_FINITE', None)
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-rcexcl')
from astropy.io import fits  # noqa: E402
from scipy import ndimage  # noqa: E402
from jwst_gc_pipeline.reduction import saturated_star_finding as S  # noqa: E402
from stdatamodels.jwst.datamodels import dqflags  # noqa: E402

JW = '/orange/adamginsburg/jwst'
WD2 = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2'
FR = [('brick', 'F356W', 'nrcalong', f'{JW}/brick/F356W/pipeline/jw01182004001_02101_00001_nrcalong_destreak_o004_crf.fits'),
      ('brick', 'F356W', 'nrcblong', f'{JW}/brick/F356W/pipeline/jw01182004001_02101_00001_nrcblong_destreak_o004_crf.fits'),
      ('brick', 'F444W', 'nrcalong', f'{JW}/brick/F444W/pipeline/jw01182004001_04101_00001_nrcalong_destreak_o004_crf.fits'),
      ('brick', 'F444W', 'nrcblong', f'{JW}/brick/F444W/pipeline/jw01182004001_04101_00001_nrcblong_destreak_o004_crf.fits')]


def add_survey(field, band, det):
    import json
    for r in json.load(open('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/survey_fields/survey_fields.json')):
        if (r['field'], r['band'], r['det']) == (field, band, det):
            FR.append((field, band, det, r['file']))
            return


for f, b, d in (('sgrb2', 'F210M', 'nrca2'), ('sgrb2', 'F210M', 'nrcb1'), ('sgrb2', 'F150W', 'nrca2'),
                ('sgrb2', 'F480M', 'nrcblong'), ('sgrc', 'F360M', 'nrcblong')):
    add_survey(f, b, d)
FR += [('wd2', 'F150W', 'nrcb1', f'{WD2}/F150W/pipeline/jw03523005001_10101_00001_nrcb1_align_o005_crf.fits'),
       ('wd2', 'F150W', 'nrcb3', f'{WD2}/F150W/pipeline/jw03523005001_10101_00001_nrcb3_align_o005_crf.fits'),
       ('wd2', 'F250M', 'nrcblong', f'{WD2}/F250M/pipeline/jw03523005001_04101_00001_nrcblong_align_o005_crf.fits'),
       ('wd2', 'F300M', 'nrcblong', f'{WD2}/F300M/pipeline/jw03523005001_12101_00001_nrcblong_align_o005_crf.fits')]


def run(fn, flag):
    os.environ['SATSTAR_ZF_R_HEADER'] = flag
    fh = fits.open(fn, memmap=False)
    header = fh[0].header
    fh['DQ'].data = S.correct_dq_first_group_saturation(fh['DQ'].data, fn, header.get('INSTRUME', ''))
    sw = S.satstar_fit_switches()
    zf = S._find_zeroframe_for(fn) if S._zeroframe_fit_enabled() else None
    ff = S._find_first_frame_for(fn) if (zf is not None and sw['first_frame']) else None
    g0s = None
    if zf is not None and (sw['g0_groupdq'] or ff is not None):
        g0s = S._find_group0_saturation_for(fn, do_not_use=(ff is not None))
    data = np.array(fh['SCI'].data, float)
    data[np.isnan(fh['VAR_POISSON'].data)] = 0
    photmjsr = fh['SCI'].header.get('PHOTMJSR', header.get('PHOTMJSR'))
    Rh = S.zeroframe_header_R(header, photmjsr)
    out, deep, rim, _ = S.zeroframe_fit_anchor(data.copy(), fh['DQ'].data, zf, group0_saturated=g0s,
                                              first_frame=ff, R_header=Rh)
    return dict(data=data, out=out, rim=rim, dq=np.array(fh['DQ'].data), zf=zf, Rh=Rh)



GB = [(0, 2000), (2000, 5000), (5000, 15000), (15000, 70000)]
print('| field | band | det | B_dn 500-2000 | SAT rewritten off / on | buffer off / on | buffer only-off / only-on '
      '| buffer only-off: data / R_hdr (g0 + B) | SAT on/off by g0 <2k / 2-5k / 5-15k / >15k (n) |')
print('|---|---|---|---|---|---|---|---|---|')
for field, band, det, fn in FR:
    if not os.path.exists(fn):
        print('SKIP', fn, flush=True)
        continue
    a = run(fn, '0')
    h = run(fn, '1')
    zf, Rh, crf, dq = a['zf'], a['Rh'], a['data'], a['dq']
    sat = (dq & dqflags.pixel['SATURATED']) != 0
    dnu = (dq & dqflags.pixel['DO_NOT_USE']) != 0
    edt = ndimage.distance_transform_edt(~sat)
    far = np.isfinite(crf) & np.isfinite(zf) & ~sat & ~dnu & (edt >= 25) & (crf != 0)
    s = far & (zf >= 500) & (zf < 2000)
    B = float(np.median(crf[s] / Rh - zf[s])) if s.sum() >= 50 else float('nan')
    s0, s1 = a['rim'] & sat, h['rim'] & sat
    b0, b1 = a['rim'] & ~sat, h['rim'] & ~sat
    oo = b0 & ~b1
    infl = np.median(crf[oo] / (Rh * (zf[oo] + B))) if oo.any() else np.nan
    both = s0 & s1 & (a['out'] > 0)
    cells = []
    for lo, hi in GB:
        m = both & (zf >= lo) & (zf < hi)
        cells.append(f"{np.median(h['out'][m] / a['out'][m]):.3f} ({int(m.sum())})" if m.any() else '- (0)')
    print(f"| {field} | {band} | {det} | {B:+.0f} | {int(s0.sum())} / {int(s1.sum())} | {int(b0.sum())} / {int(b1.sum())} "
          f"| {int(oo.sum())} / {int((b1 & ~b0).sum())} | {infl:.3f} | {' / '.join(cells)} |", flush=True)
