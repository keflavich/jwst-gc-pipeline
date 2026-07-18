"""M92 sparse-field deep-mask wing-calibration campaign — STAGE 2: C(r) curves.

From the stage-1 stacks (m92_deep_stacks.py: sub-pixel-registered empirical
E and STPSF-model M 2D PSF stacks, 151x151, center (75,75)), measure the
masked-core flux bias

    C(r_mask) = amp(M -> E, circle r < r_mask masked) / amp(M -> E, unmasked)

with mask radii extended to 26 px (the production regime where crowded-field
self-cal has NO calibrators and clamps at ~10 px).  Two fit-region flavors:

  * annulus : uniform-weight LSQ on r_mask <= r <= 50 px (the geometry of the
              existing calcurves/ product, calibrate_masked_core_bias.py)
  * prodwin : uniform-weight LSQ on the production fit window — square box of
              half-width min(2*r_mask+10, 50) px minus the masked circle,
              normalized by an unmasked 11x11-core fit (the _wing_selfcal
              truth fit geometry in reduction/saturated_star_finding.py)

Each flavor is computed with and without the phantom "segment blob" region
(OPD R2022062004 mirror-tilt artifact in 12/20 M92 STPSF grids, ~2.2% of
model flux at (+25,-25) det px, r~14 px — see
docs/reports/SATSTAR_WING_CALIBRATION_REPORT.md).  The blob lands at
r~35 px, INSIDE the deep-mask fit annulus, so the noblob variant is the
transferable calibration; a per-config blob metric (model-minus-azimuthal
median flux in the blob aperture) says how much each grid is affected.

Errors: the same curve on each per-frame median stack -> madstd across
frames (also the frame-to-frame variation production cares about).

Outputs (small, committed): calcurves_deep_m92/calcurve_deep_<filt>_<det>.ecsv,
calcurves_deep_m92_all.ecsv, summary_deep_m92.json, and a printed comparison
of C(r=10..25) against the production clamp-at-C(10) extrapolation
(delta_mag = -2.5 log10(C(r)/C(10))).

Usage: python m92_deep_calcurve.py [stacksdir] [outdir]
"""
import glob
import json
import os
import re
import sys
import numpy as np
from astropy.table import Table, vstack

STACKS = (sys.argv[1] if len(sys.argv) > 1
          else '/orange/adamginsburg/jwst/m92/wingcal_deep_stacks')
OUT = (sys.argv[2] if len(sys.argv) > 2
       else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'calcurves_deep_m92'))

CEN = 75                       # stacks are 151x151
FIT_R = 50.0                   # production-like fit region radius [px]
CORE_HALF = 5                  # 11x11 truth-fit box (prodwin normalization)
BLOB_DX, BLOB_DY, BLOB_R = 25.0, -25.0, 14.0
R_MASK = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18,
                   20, 22, 24, 25, 26], dtype=float)
R_REF = 10.0                   # production clamp radius in crowded fields

YY, XX = np.mgrid[0:2 * CEN + 1, 0:2 * CEN + 1]
RR = np.hypot(XX - CEN, YY - CEN)
BLOB = np.hypot(XX - (CEN + BLOB_DX), YY - (CEN + BLOB_DY)) < BLOB_R
BOXR = np.maximum(np.abs(XX - CEN), np.abs(YY - CEN))


def lsq_amp(data, model, sel):
    d, m = data[sel], model[sel]
    if sel.sum() < 10 or not np.isfinite(d).all():
        ok = np.isfinite(d) & np.isfinite(m)
        d, m = d[ok], m[ok]
        if len(d) < 10:
            return np.nan
    s = np.sum(m * m)
    return float(np.sum(d * m) / s) if s > 0 else np.nan


def curves(E, M, noblob):
    """Return dict of flavor -> C(r) arrays for one stack pair."""
    base = (RR <= FIT_R) & np.isfinite(E) & np.isfinite(M)
    if noblob:
        base &= ~BLOB
    core = base & (BOXR <= CORE_HALF)
    amp0_ann = lsq_amp(E, M, base)
    amp0_core = lsq_amp(E, M, core)
    ann, prod = [], []
    for r in R_MASK:
        sel_ann = base & (RR >= r)
        ann.append(lsq_amp(E, M, sel_ann) / amp0_ann)
        half = min(2 * r + 10, FIT_R)
        sel_prod = base & (RR >= r) & (BOXR <= half)
        prod.append(lsq_amp(E, M, sel_prod) / amp0_core)
    return {'ann': np.array(ann), 'prod': np.array(prod),
            'amp0_ann': amp0_ann, 'amp0_core': amp0_core}


def blob_metric(S, base):
    """Excess flux in the blob aperture over the azimuthal median profile,
    as a fraction of total stack flux inside FIT_R."""
    prof = np.full(S.shape, np.nan)
    for r0 in np.arange(0, FIT_R + 1):
        sel = base & (RR >= r0) & (RR < r0 + 1) & ~BLOB
        if sel.sum() > 5:
            prof[base & (RR >= r0) & (RR < r0 + 1)] = np.median(S[sel])
    bsel = base & BLOB & np.isfinite(prof)
    tot = np.nansum(np.where(base, S, np.nan))
    if bsel.sum() == 0 or not np.isfinite(tot) or tot <= 0:
        return np.nan
    return float(np.nansum(S[bsel] - prof[bsel]) / tot)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f'{STACKS}/wingratio_m92_*.npz'))
    if not files:
        print(f'no stacks found in {STACKS}')
        return
    tables, summary = [], {}
    for fn in files:
        m = re.search(r'wingratio_m92_(nrc[ab]\d)_(f\d+w)\.npz',
                      os.path.basename(fn))
        if not m:
            continue
        det, filt = m.groups()
        z = np.load(fn, allow_pickle=True)
        E = np.asarray(z['med_dstack'], float)
        M = np.asarray(z['med_mstack'], float)
        nstars = int(np.sum(z['nstars_frames']))
        base = (RR <= FIT_R) & np.isfinite(E) & np.isfinite(M)

        cb = curves(E, M, noblob=False)
        cn = curves(E, M, noblob=True)

        # per-frame scatter (annulus, noblob flavor)
        per_frame = []
        for Ef, Mf in zip(z['dstack_frames'], z['mstack_frames']):
            Ef, Mf = np.asarray(Ef, float), np.asarray(Mf, float)
            if np.isfinite(Ef).sum() < 1000:
                continue
            per_frame.append(curves(Ef, Mf, noblob=True)['ann'])
        per_frame = np.array(per_frame)
        if len(per_frame) >= 2:
            fscat = 1.4826 * np.nanmedian(
                np.abs(per_frame - np.nanmedian(per_frame, axis=0)), axis=0)
            ferr = fscat / np.sqrt(len(per_frame))
        else:
            fscat = np.full(len(R_MASK), np.nan)
            ferr = fscat

        bm_model = blob_metric(M, base)
        bm_data = blob_metric(E, base)

        t = Table({
            'r_mask': R_MASK,
            'bias_ann': cb['ann'], 'bias_ann_noblob': cn['ann'],
            'bias_prodwin': cb['prod'], 'bias_prodwin_noblob': cn['prod'],
            'frame_scatter': fscat, 'err': ferr,
        })
        t['filter'] = filt.upper()
        t['detector'] = det
        t.meta = {
            'filter': filt.upper(), 'detector': det, 'field': 'M92 GO-1334',
            'n_stars': nstars, 'n_frames': int(len(z['frame_names'])),
            'source_stack': fn, 'fit_region_radius_px': FIT_R,
            'amp_unmasked_ann': cb['amp0_ann'],
            'amp_unmasked_core11': cb['amp0_core'],
            'blob_excess_frac_model': bm_model,
            'blob_excess_frac_data': bm_data,
            'bias_def': ('C(r_mask) = uniform-weight LSQ amplitude of the '
                         'STPSF-model stack fit to the empirical stack with '
                         'the circle r < r_mask masked, divided by the '
                         'unmasked amplitude.  >1 = masked-core (satstar) '
                         'flux overestimated.'),
            'flavors': ('ann: fit region r_mask<=r<=50 px; prodwin: '
                        'production _wing_selfcal window (square half-width '
                        'min(2r+10,50)) normalized by unmasked 11x11 core '
                        'fit; *_noblob: OPD phantom-blob circle (+25,-25) '
                        'r<14 px excluded from the fit'),
            'errors': ('frame_scatter = madstd of bias_ann_noblob across '
                       'per-frame stacks; err = frame_scatter/sqrt(n_frames)'),
        }
        t.write(f'{OUT}/calcurve_deep_{filt}_{det}.ecsv', overwrite=True,
                format='ascii.ecsv')
        tables.append(t)

        def at(r, arr):
            return float(arr[np.argmin(np.abs(R_MASK - r))])

        iref = np.argmin(np.abs(R_MASK - R_REF))
        cref = cn['ann'][iref]
        summary[f'{filt}_{det}'] = {
            'n_stars': nstars,
            'blob_excess_frac_model': bm_model,
            'blob_excess_frac_data': bm_data,
            'C_ann_noblob': {int(r): at(r, cn['ann'])
                             for r in (5, 8, 10, 12, 15, 16, 20, 25)},
            'C_prodwin_noblob': {int(r): at(r, cn['prod'])
                                 for r in (5, 8, 10, 12, 15, 16, 20, 25)},
            'C_ann_withblob': {int(r): at(r, cb['ann'])
                               for r in (10, 15, 20, 25)},
            'err_at': {int(r): at(r, ferr) for r in (10, 15, 20, 25)},
            'clamp_residual_mag': {
                int(r): float(-2.5 * np.log10(at(r, cn['ann']) / cref))
                for r in (12, 15, 20, 25)},
        }
        print(f'\n{filt.upper()} {det}  ({nstars} stars, '
              f'blob model excess {bm_model * 100:.2f}%'
              f' / data {bm_data * 100:.2f}%)')
        print('  r_mask  C_ann(noblob)  C_prodwin(noblob)  C_ann(blob)  '
              '+-frame   clamp-resid [mag]')
        for r in (5, 8, 10, 12, 15, 20, 25):
            i = np.argmin(np.abs(R_MASK - r))
            resid = -2.5 * np.log10(cn['ann'][i] / cref) if r > R_REF else 0.0
            print(f'  {r:5.0f}   {cn["ann"][i]:10.3f}   {cn["prod"][i]:14.3f}'
                  f'   {cb["ann"][i]:10.3f}   {fscat[i]:7.3f}   {resid:8.3f}',
                  flush=True)

    comb = vstack(tables, metadata_conflicts='silent')
    comb.meta = {'note': 'see per-config calcurve_deep_<filt>_<det>.ecsv',
                 'bias_def': tables[0].meta['bias_def'],
                 'flavors': tables[0].meta['flavors']}
    comb.write(f'{OUT}/calcurves_deep_m92_all.ecsv', overwrite=True,
               format='ascii.ecsv')

    # per-filter medians across detectors.  The deep-radius bias correlates
    # strongly with stellar density (detectors near the cluster core show a
    # steep spurious rise from crowding contamination of the stacked wings,
    # with tiny frame-to-frame scatter — a field property, not noise), so
    # ALSO aggregate over the 4 SPARSEST detectors per filter: that subset is
    # the campaign's headline deep C(r).
    for filt in sorted({k.split('_')[0] for k in summary}):
        ks = [k for k in summary if k.startswith(f'{filt}_nrc')]
        sparse = sorted(ks, key=lambda k: summary[k]['n_stars'])[:4]
        for label, sel in (('median', ks), ('sparse_median', sparse)):
            med = {r: float(np.median([summary[k]['C_ann_noblob'][r]
                                       for k in sel]))
                   for r in (10, 12, 15, 20, 25)}
            resid = {r: float(np.median([summary[k]['clamp_residual_mag'][r]
                                         for k in sel]))
                     for r in (12, 15, 20, 25)}
            summary[f'{filt}_{label}'] = {
                'C_ann_noblob': med, 'clamp_residual_mag': resid,
                'detectors': [k.split('_')[1] for k in sel]}
            print(f'\n{filt.upper()} {label} over {len(sel)} detectors '
                  f'({", ".join(k.split("_")[1] for k in sel)}): '
                  + '  '.join(f'C({r})={med[r]:.3f}'
                              for r in (10, 15, 20, 25))
                  + '  |  clamp residual '
                  + '  '.join(f'{r}px:{resid[r]:+.3f}mag'
                              for r in (15, 20, 25)))

    with open(f'{OUT}/summary_deep_m92.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'\nwrote {OUT}/calcurves_deep_m92_all.ecsv (+ per-config) and '
          f'summary_deep_m92.json')


if __name__ == '__main__':
    main()
