"""#1148 review note A: rim (SATURATED) vs dilation-buffer pixels rewritten under the measured
curve (SATSTAR_ZF_R_HEADER=0) and the header rate (=1), wd2 exposure-1 frames, PR branch code."""
import os
import sys
import numpy as np
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
os.environ['NIRCAM_SATSTAR_RECOVERED_CAP'] = '1'
os.environ.pop('SATSTAR_ZF_KEEP_FINITE', None)
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-rcexcl')
from astropy.io import fits  # noqa: E402
from jwst_gc_pipeline.reduction import saturated_star_finding as S  # noqa: E402
from stdatamodels.jwst.datamodels import dqflags  # noqa: E402

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
TREE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2'
FR = [('150W', 'nrcb1'), ('150W', 'nrcb3'), ('200W', 'nrcb1'), ('200W', 'nrcb3'),
      ('250M', 'nrcblong'), ('300M', 'nrcblong')]


def frame(band, det):
    import glob
    c = sorted(glob.glob(f'{TREE}/F{band}/pipeline/jw*_00001_{det}_*crf.fits'))
    c = [f for f in c if 'resbgsub' not in f and 'm7' not in f]
    return c[0]


def run(fn, flag):
    os.environ['SATSTAR_ZF_R_HEADER'] = flag
    fh = fits.open(fn, memmap=False)
    header = fh[0].header
    fh['DQ'].data = S.correct_dq_first_group_saturation(fh['DQ'].data, fn, header.get('INSTRUME', ''))
    sw = S.satstar_fit_switches()
    zf = S._find_zeroframe_for(fn) if S._zeroframe_fit_enabled() else None
    ff = S._find_first_frame_for(fn) if (zf is not None and sw['first_frame']) else None
    g0 = None
    if zf is not None and (sw['g0_groupdq'] or ff is not None):
        g0 = S._find_group0_saturation_for(fn, do_not_use=(ff is not None))
    data = np.array(fh['SCI'].data, float)
    data[np.isnan(fh['VAR_POISSON'].data)] = 0
    photmjsr = fh['SCI'].header.get('PHOTMJSR', header.get('PHOTMJSR'))
    out, deep, rim, _ = S.zeroframe_fit_anchor(data.copy(), fh['DQ'].data, zf, group0_saturated=g0,
                                              first_frame=ff, R_header=S.zeroframe_header_R(header, photmjsr))
    sat = (fh['DQ'].data & dqflags.pixel['SATURATED']) != 0
    return data, out, rim, sat


print('| frame | rim SAT off / on | buffer off / on | buffer only-on / only-off | SAT rim value on/off (median) | buffer-only-on: data / R_hdr g0 (median) |')
print('|---|---|---|---|---|---|')
for band, det in FR:
    fn = frame(band, det)
    d, o0, r0, sat = run(fn, '0')
    _, o1, r1, _ = run(fn, '1')
    b0, b1 = r0 & ~sat, r1 & ~sat
    s0, s1 = r0 & sat, r1 & sat
    both = s0 & s1 & (o0 > 0)
    ratio = np.median(o1[both] / o0[both]) if both.any() else np.nan
    new = b1 & ~b0
    infl = np.median(d[new] / o1[new]) if new.any() else np.nan
    print(f'| F{band} {det} | {int(s0.sum())} / {int(s1.sum())} | {int(b0.sum())} / {int(b1.sum())} | '
          f'{int(new.sum())} / {int((b0 & ~b1).sum())} | {ratio:.4f} | {infl:.3f} |', flush=True)
