"""Full-field replay of the m6 vetting for #1016 (Brick 2221/o001 F182M).

Runs _filter_extended_emission on the production m6 merged (unvetted)
catalog with the pipeline-default options, the S/N floors per-frame (main)
or on flux_err_prop (#1016), with and without the sky-clean tier, and
splits the newly passing sources by the keep path that admitted them.

usage: python replay_vetting.py <pipeline_worktree> <outdir> [filter]
"""
import json
import os
import sys
import time

WT, OUT = sys.argv[1], sys.argv[2]
FILT = sys.argv[3] if len(sys.argv) > 3 else 'F182M'
sys.path.insert(0, WT)
import numpy as np                                       # noqa: E402
from astropy.io import fits                              # noqa: E402
from astropy.table import Table                          # noqa: E402
from astropy import wcs                                  # noqa: E402
from jwst_gc_pipeline.photometry import cataloging as C  # noqa: E402
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS as MD  # noqa: E402

assert os.path.realpath(C.__file__).startswith(os.path.realpath(WT)), C.__file__
PROD = '/orange/adamginsburg/jwst/brick'
fl = FILT.lower()
merged = Table.read(f'{PROD}/catalogs/{fl}_merged_o001_indivexp_merged_resbgsub_m6_dao_basic.fits')
merged['rowid'] = np.arange(len(merged))
dpath = f'{PROD}/{FILT}/pipeline/jw02221-o001_t001_nircam_clear-{fl}-merged_data_i2d.fits'
with fits.open(dpath) as dh:
    d_i2d = dh['SCI'].data.astype(float)
    ww = wcs.WCS(dh['SCI'].header)

KW = dict(qfit_max=MD['manual_ext_qfit_max'], peak_over_bkg=MD['manual_ext_peak_over_bkg'],
          min_prominence=0.0, local_snr_min=MD['manual_ext_local_snr_min'],
          snr_high_keep=MD['manual_ext_snr_high_keep'],
          qfit_high_keep_max=MD['manual_ext_qfit_high_keep_max'],
          qfit_recover_max=MD['manual_ext_qfit_recover_max'],
          recover_satstar_guard_arcsec=MD['manual_ext_recover_satstar_guard_arcsec'],
          recover_prom_gate=MD['manual_ext_recover_prom_gate'],
          recover_prom_log_intercept=MD['manual_ext_recover_prom_log_intercept'],
          recover_prom_log_slope=MD['manual_ext_recover_prom_log_slope'],
          nmatch_confirm=MD['manual_ext_nmatch_confirm'],
          nmatch_confirm_qfit_max=MD['manual_ext_nmatch_confirm_qfit_max'],
          nmatch_confirm_maxpos_mas=MD['manual_ext_nmatch_confirm_maxpos_mas'],
          nmatch_confirm_strong=MD['manual_ext_nmatch_confirm_strong'],
          low_fit_quality_qfit=MD['manual_ext_low_fit_quality_qfit'],
          ext_prom_min=0.0,
          sky_clean_max_sky_snr=MD['manual_sky_clean_max_sky_snr'],
          sky_clean_prom_min=MD['manual_sky_clean_prom_min'],
          sky_clean_snr_min=MD['manual_sky_clean_snr_min'],
          struct_x=0.0, struct_y=0.0)
assert MD['manual_sky_clean_keep'] is True


def run(prop, sky):
    t0 = time.time()
    v = C._filter_extended_emission(merged.copy(), data_i2d_image=d_i2d, ww_i2d=ww,
                                    snr_floor_propagated=prop, sky_clean_keep=sky,
                                    label=f'replay prop={prop} sky={sky}', **KW)
    print(f'prop={prop} sky={sky}: {len(v)} kept ({time.time() - t0:.0f}s)', flush=True)
    return v


runs = {(p, s): run(p, s) for p in (False, True) for s in (False, True)}
ids = {k: set(np.asarray(v['rowid']).tolist()) for k, v in runs.items()}
full_off, full_on = ids[(False, True)], ids[(True, True)]
main_off, main_on = ids[(False, False)], ids[(True, False)]
new = full_on - full_off
via_main = new & main_on
via_sky = new - main_on
# sky-clean floor kept per-frame, local floor propagated
full_mixed = main_on | full_off
lost = full_off - full_on

# columns (prominence, peak_sb, sky_clean ...) for every row that any run kept
cols = Table(merged, copy=True)
for c in ('prominence', 'peak_sb', 'sky_clean', 'local_emission_snr'):
    cols[c] = np.full(len(cols), np.nan)
for v in runs.values():
    r = np.asarray(v['rowid'])
    for c in ('prominence', 'peak_sb', 'sky_clean', 'local_emission_snr'):
        if c in v.colnames:
            cols[c][r] = np.asarray(v[c], dtype=float)
qf = np.asarray(cols['qfit'], float)
flg = np.asarray(cols['flags'], float)
pk = np.asarray(cols['peak_sb'], float)
lb = np.asarray(cols['local_bkg'], float)
reason = np.full(len(cols), '', dtype='U16')
reason[(qf <= KW['qfit_max'])] = 'qfit<=0.2'
r2 = (reason == '') & np.isin(flg, [1.0])
reason[r2] = 'flags==1'
r3 = (reason == '') & np.isfinite(pk) & (lb > 0) & (pk > KW['peak_over_bkg'] * lb)
reason[r3] = 'peakSB'
reason[reason == ''] = 'other'

group = np.full(len(cols), '', dtype='U16')
group[list(full_off)] = 'vetted_main'
group[list(via_main)] = 'new_local_floor'
group[list(via_sky)] = 'new_sky_clean'
group[list(lost)] = 'lost'
cols['group'] = group
cols['star_like_reason'] = reason
fe, fep = np.asarray(cols['flux_err'], float), np.asarray(cols['flux_err_prop'], float)
fx = np.asarray(cols['flux'], float)
cols['snr_frame'] = fx / fe
cols['snr_prop'] = fx / fep
keep_rows = group != ''
cols[keep_rows].write(f'{OUT}/replay_{fl}.fits', overwrite=True)

summary = dict(n_merged=len(merged), n_vetted_main=len(full_off), n_vetted_pr=len(full_on),
               n_vetted_skyclean_per_frame=len(full_mixed), n_lost=len(lost),
               n_new=len(new), n_new_via_local_floor=len(via_main), n_new_via_sky_clean=len(via_sky),
               main_only_no_skyclean=dict(off=len(main_off), on=len(main_on)))
for g in ('new_local_floor', 'new_sky_clean'):
    k = group == g
    summary[g] = dict(
        reason={r: int((reason[k] == r).sum()) for r in ('qfit<=0.2', 'flags==1', 'peakSB', 'other')},
        qfit_gt_0p2=int((qf[k] > 0.2).sum()),
        snr_frame_median=float(np.nanmedian(cols['snr_frame'][k])),
        snr_prop_median=float(np.nanmedian(cols['snr_prop'][k])),
        prominence_median=float(np.nanmedian(np.asarray(cols['prominence'], float)[k])),
        nmatch_median=float(np.nanmedian(np.asarray(cols['nmatch'], float)[k])))
print(json.dumps(summary, indent=1), flush=True)
with open(f'{OUT}/replay_{fl}.json', 'w') as fh:
    json.dump(summary, fh, indent=1)
