"""Which vetting test drops a star that the previous phase kept (vetting flicker).

For each reference run (field:variant:seed:band) take the phase-loss rows of
category ``vetted_out`` (phase K vetted the star, phase K+1 fitted it and its
vetting dropped it).  Replay ``_filter_extended_emission`` on the phase K and
the phase K+1 merged catalogs with the pipeline options of the run, capture
the function's intermediate masks (sys.setprofile on its return), and record,
for the merged row matched to the star in each phase, every input and every
keep branch.  The replayed kept count is checked against the vetted file on
disk.

usage: WT=<worktree> LOST_DIR=<dir> python vet_gates.py <out.fits> <field:variant:seed:band> ...

WT is the checkout whose cataloging code is replayed; LOST_DIR holds the
``<field>_<variant>_s<seed>_<band>_lost.fits`` tables that
docs/evidence/m7_companion_ratio/scripts/phase_loss.py writes.
"""
import glob
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'm7_companion_ratio', 'scripts'))
sys.path.insert(0, os.environ['WT'])
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402
from jwst_gc_pipeline.photometry import reference_fields as RF  # noqa: E402
from jwst_gc_pipeline.photometry.manual_defaults import mopt  # noqa: E402
from phase_loss import phase_files  # noqa: E402

assert os.path.realpath(C.__file__).startswith(os.path.realpath(os.environ['WT'])), C.__file__
LOST = os.environ['LOST_DIR']


def _capture(func, *args, **kw):
    """Call ``func`` and return (result, its locals at return)."""
    box = {}
    code = func.__code__

    def prof(frame, event, arg):
        if event == 'return' and frame.f_code is code:
            box.update(frame.f_locals)
    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        out = func(*args, **kw)
    finally:
        sys.setprofile(old)
    return out, box


def vet_kwargs(opts):
    """The NIRCam keyword set of the vetting call in run_manual_pipeline."""
    ext = float(mopt(opts, 'manual_ext_prom_min'))
    ext = (3.0 if C._is_extended_emission(opts) else 0.0) if ext < 0 else ext
    g = lambda k: mopt(opts, k)  # noqa: E731
    return dict(
        qfit_max=float(g('manual_ext_qfit_max')), peak_over_bkg=float(g('manual_ext_peak_over_bkg')),
        star_prom_min=float(g('manual_ext_star_prom_min')), qfit_snr_k=float(g('manual_ext_qfit_snr_k')),
        star_prom_peak_min=float(g('manual_ext_star_prom_peak_min')),
        star_prom_robust_min=C._auto_star_prom_robust_min(g('manual_ext_star_prom_robust_min'), opts),
        star_prom_robust_conc=float(g('manual_ext_star_prom_robust_conc')),
        min_prominence=0.0, local_snr_min=float(g('manual_ext_local_snr_min')),
        snr_floor_propagated=bool(g('manual_ext_snr_floor_propagated')),
        snr_high_keep=float(g('manual_ext_snr_high_keep')),
        qfit_high_keep_max=float(g('manual_ext_qfit_high_keep_max')),
        qfit_recover_max=float(g('manual_ext_qfit_recover_max')),
        recover_satstar_guard_arcsec=float(g('manual_ext_recover_satstar_guard_arcsec')),
        recover_prom_gate=bool(g('manual_ext_recover_prom_gate')),
        recover_prom_log_intercept=float(g('manual_ext_recover_prom_log_intercept')),
        recover_prom_log_slope=float(g('manual_ext_recover_prom_log_slope')),
        nmatch_confirm=int(g('manual_ext_nmatch_confirm')),
        nmatch_confirm_qfit_max=float(g('manual_ext_nmatch_confirm_qfit_max')),
        nmatch_confirm_maxpos_mas=float(g('manual_ext_nmatch_confirm_maxpos_mas')),
        nmatch_confirm_strong=int(g('manual_ext_nmatch_confirm_strong')),
        low_fit_quality_qfit=float(g('manual_ext_low_fit_quality_qfit')),
        ext_prom_min=ext,
        ext_prom_exempt_qfit=float(g('manual_ext_prom_exempt_qfit')),
        ext_prom_exempt_snr=float(g('manual_ext_prom_exempt_snr')),
        ext_prom_exempt_prom_min=float(g('manual_ext_prom_exempt_prom_min')),
        sky_clean_keep=bool(g('manual_sky_clean_keep')),
        sky_clean_max_sky_snr=float(g('manual_sky_clean_max_sky_snr')),
        sky_clean_prom_min=float(g('manual_sky_clean_prom_min')),
        sky_clean_snr_min=float(g('manual_sky_clean_snr_min')),
        sky_clean_local_arcsec=float(g('manual_sky_clean_local_arcsec')),
        sky_clean_local_max_err=float(g('manual_sky_clean_local_max_err')),
        struct_x=0.0, struct_y=0.0)


def replay(basic_path, vetted_path, data_path, kw, label):
    merged = Table.read(basic_path)
    with fits.open(data_path) as dh:
        hdu = dh['SCI'] if 'SCI' in dh else dh[0]
        d, w = hdu.data.astype(float), WCS(hdu.header)
        e = dh['ERR'].data.astype(float) if 'ERR' in dh else None
    out, loc = _capture(C._filter_extended_emission, merged, data_i2d_image=d, ww_i2d=w,
                        err_i2d_image=e, label=label, **kw)
    n_disk = len(Table.read(vetted_path))
    n = len(merged)
    z = np.zeros(n, bool)
    qf, snr, snr_floor = loc['qf'], loc['snr'], loc['snr_floor']
    cols = dict(
        qfit=qf, snr=snr, snr_floor=snr_floor, flags=loc['flg'], group_size=loc['gsz'],
        prominence=loc['prominence'], prominence_robust=loc['prominence_robust'],
        peak_sb=loc['peaksb'], local_bkg=loc['lbk'],
        qfit_ok=qf <= kw['qfit_max'],
        peak_or_prom=np.asarray(loc['_peak_branch'], bool),
        bright_isolated=np.asarray(loc['bright_isolated'], bool),
        star_like=np.asarray(loc['star_like'], bool),
        snr_ok=(np.isfinite(snr_floor) & (snr_floor >= kw['local_snr_min'])) | ~np.isfinite(snr_floor),
        q_refused=(np.asarray(loc['_q_refused'], bool) if loc.get('_q_refused') is not None else z),
        sky_clean=np.asarray(merged['sky_clean'], bool) if 'sky_clean' in merged.colnames else z,
        sc_keep=np.asarray(loc['_sc_keep'], bool) if '_sc_keep' in loc else z,
        overshoot=(np.asarray(merged['model_overshoot'], bool)
                   if 'model_overshoot' in merged.colnames else z),
        keep=np.asarray(loc['keep'], bool))
    # the prominence keep on its own (before the qfit noise bound and the peak test)
    with np.errstate(invalid='ignore'):
        cols['prom_ge_min'] = np.isfinite(loc['prominence']) & (loc['prominence'] >= kw['star_prom_min'])
        cols['qbound'] = np.where(np.isfinite(snr) & (snr > 0),
                                  np.hypot(kw['qfit_max'], kw['qfit_snr_k'] / snr), kw['qfit_max'])
        lb = loc['lbk']
        cols['peaksb_ok'] = np.isfinite(loc['peaksb']) & (lb > 0) & (loc['peaksb'] > kw['peak_over_bkg'] * lb)
    return SkyCoord(merged['skycoord']), cols, int(cols['keep'].sum()), n_disk


def reason(c, i, kw):
    """The first test that fails for row i, in the order the keep is built."""
    if c['keep'][i]:
        return 'kept'
    if c['overshoot'][i] and c['star_like'][i]:
        return 'overshoot'
    if c['star_like'][i] and not c['snr_ok'][i]:
        return 'snr_floor'
    if not c['star_like'][i]:
        if c['prom_ge_min'][i] and c['q_refused'][i]:
            return 'prom_keep:qfit_bound'
        if c['peaksb_ok'][i] and not c['peak_or_prom'][i]:
            return 'peak:prom<4'
        if c['snr'][i] >= kw['snr_high_keep'] and c['qfit'][i] < kw['qfit_high_keep_max']:
            return 'not_star_like:group>1'
        return 'not_star_like'
    return 'other'


def kept_by(c, i):
    """The keep branches that pass for row i ('+'-joined)."""
    names = [('qfit', c['qfit_ok'][i]), ('peak|prom', c['peak_or_prom'][i]),
             ('bright_iso', c['bright_isolated'][i]), ('flags1', c['flags'][i] == 1),
             ('sky_clean', c['sc_keep'][i])]
    return '+'.join(n for n, v in names if v) or '-'


def main(out, *specs):
    _, fields = RF.load_config()
    rows = []
    for s in specs:
        name, variant, seed, filt = s.split(':')
        seed, FILT = int(seed), filt.upper()
        spec = fields[name]
        d = RF.run_dir(spec, variant, seed)
        f = FILT.lower()
        hit = glob.glob(f'{d}/catalogs/{f}_merged*_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')[0]
        obstok = re.match(rf'{f}_merged(.*)_indivexp', os.path.basename(hit)).group(1)
        files = phase_files(f'{d}/catalogs', f'{d}/{FILT}/pipeline', FILT, 'merged', obstok)
        lost = Table.read(f'{LOST}/{name}_{variant}_s{seed}_{f}_lost.fits')
        lost = lost[np.char.startswith(np.asarray(lost['category']).astype(str), 'vetted_out')]
        if not len(lost):
            continue
        opts = SimpleNamespace(target=spec['target'])
        kw = vet_kwargs(opts)
        cache = {}
        lsc = SkyCoord(np.asarray(lost['ra']) * u.deg, np.asarray(lost['dec']) * u.deg)
        rec = {k: [] for k in ('prev', 'next')}
        for which, phcol in (('prev', 'last_phase'), ('next', 'next_phase')):
            for ph in np.unique(np.asarray(lost[phcol]).astype(str)):
                if ph not in cache:
                    sc, cols, n_keep, n_disk = replay(files[ph]['basic'], files[ph]['vetted'],
                                                      files['data'], kw, f'{name}:{ph}:{FILT}')
                    print(f'{name} {variant} s{seed} {FILT} {ph}: replay kept {n_keep}, '
                          f'on disk {n_disk}', flush=True)
                    cache[ph] = (sc, cols)
        for i in range(len(lost)):
            r = dict(field=name, variant=variant, seed=seed, filt=FILT,
                     last_phase=str(lost['last_phase'][i]), next_phase=str(lost['next_phase'][i]),
                     category=str(lost['category'][i]), starlike_left=bool(lost['starlike_left'][i]),
                     ra=float(lost['ra'][i]), dec=float(lost['dec'][i]))
            for which, ph in (('prev', r['last_phase']), ('next', r['next_phase'])):
                sc, cols = cache[ph]
                jj, sep, _ = lsc[i:i + 1].match_to_catalog_sky(sc)
                j = int(jj[0])
                r[f'{which}_sep_mas'] = float(sep.to_value(u.mas)[0])
                for k, v in cols.items():
                    r[f'{which}_{k}'] = v[j]
                r[f'{which}_reason'] = reason(cols, j, kw)
                r[f'{which}_kept_by'] = kept_by(cols, j)
            rows.append(r)
    t = Table(rows=rows)
    t.write(out, overwrite=True)
    for k in ('prev_reason', 'next_reason'):
        vals, cnt = np.unique(np.asarray(t[k]).astype(str), return_counts=True)
        print(k, dict(zip(vals, cnt.tolist())))


if __name__ == '__main__':
    main(*sys.argv[1:])
