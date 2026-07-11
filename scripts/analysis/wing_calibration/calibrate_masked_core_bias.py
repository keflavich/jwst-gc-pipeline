"""CALIBRATION PRODUCT: flux bias of fitting a PSF model to the stacked
empirical PSF with the inner X brightest pixels masked (saturated-core proxy).

Per filter:
  E(x,y) = empirical stacked PSF (median stack of bright unsaturated isolated
           stars, LocalBackground-subtracted, core-flux-normalized, sub-pixel
           registered) -- 101x101.
  M(x,y) = STPSF GriddedPSFModel evaluated at the same detector positions and
           sub-pixel phases, pushed through IDENTICAL stacking machinery.
  For each mask level X (inner X brightest pixels of E masked):
     bias flavors (uniform-weight LSQ amplitude on unmasked px, r <= 50 px,
     mimicking the production satstar fit region):
       (a) amp(M -> E)        : what production satstar fitting effectively does
       (b) amp(E -> E) == 1   : control, validates machinery
       (c) amp(M -> M) == 1   : pure-model self-consistency
       (d) amp(M -> null)     : null stack = model*A + real ERR noise pushed
                                through the full stacking machinery (brick
                                filters only) -- isolates machinery bias
  bias_model_to_data = amp(M->E, mask X) / amp(M->E, no mask), so 1.0 means
  "masking the core does not change the fitted flux".  amp0 itself is ~1 by
  construction (E is normalized by the per-star core LSQ flux of M).

Outputs (in this directory): calcurve_<filt>.ecsv per filter,
calcurves_all.ecsv combined, calcurves_bias.png figure.
"""
import json
import numpy as np
from astropy.table import Table, vstack

SCR = ('/blue/adamginsburg/adamginsburg/tmp/claude-3663/'
       '-blue-adamginsburg-adamginsburg-repos-jwst-gc-pipeline/'
       'd046d21c-902e-4ed6-b230-4b20ac93f54c/scratchpad')
OUT = f'{SCR}/agent_calcurves'

FIT_R = 50.0          # production-like fit region radius [px]
HALF = 50             # stacks are 101x101, center (50, 50)

# mask levels: log-spaced 5..2000 px, plus exact circular-equivalents of the
# masked-core star-fit experiment radii r = 3, 4, 6, 8, 12 px (N = pi r^2)
N_LOG = np.unique(np.round(np.logspace(np.log10(5), np.log10(2000), 24))
                  ).astype(int)
N_EXP = np.array([int(round(np.pi * r * r)) for r in (3, 4, 6, 8, 12)])
N_LEVELS = np.unique(np.concatenate([N_LOG, N_EXP]))

SOURCES = {
    # filt: (label, path, kind)  kind: 'stack2d' (101x101 + null) | 'gc' (151)
    'f090w': ('M92 nrca1',   f'{SCR}/agent_globular/wingratio_m92_nrca1_f090w.npz', 'gc'),
    'f150w': ('M92 nrca1',   f'{SCR}/agent_globular/wingratio_m92_nrca1_f150w.npz', 'gc'),
    'f182m': ('brick nrca1', f'{SCR}/agent_stack2d/stack2d_f182m_nrca1.npz', 'stack2d'),
    'f187n': ('brick nrca1', f'{SCR}/agent_stack2d/stack2d_f187n_nrca1.npz', 'stack2d'),
    'f212n': ('brick nrca1', f'{SCR}/agent_stack2d/stack2d_f212n_nrca1.npz', 'stack2d'),
    'f405n': ('brick nrcalong', f'{SCR}/agent_stack2d/stack2d_f405n_nrcalong.npz', 'stack2d'),
    'f410m': ('brick nrcalong', f'{OUT}/stack2d_f410m_nrcalong.npz', 'stack2d'),
    'f466n': ('brick nrcalong', f'{OUT}/stack2d_f466n_nrcalong.npz', 'stack2d'),
}


def load_stacks(path, kind):
    z = np.load(path, allow_pickle=True)
    if kind == 'stack2d':
        E = np.asarray(z['data_stack'], float)
        M = np.asarray(z['model_stack'], float)
        NUL = np.asarray(z['null_stack'], float)
        n = int(z['n_stars'])
    else:  # gc: 151x151, center (75,75) -> crop to 101x101
        s = np.s_[75 - HALF:75 + HALF + 1, 75 - HALF:75 + HALF + 1]
        E = np.asarray(z['med_dstack'], float)[s]
        M = np.asarray(z['med_mstack'], float)[s]
        NUL = None
        n = len(z['meta'])
    return E, M, NUL, n


def lsq_amp(data, model, sel):
    """Uniform-weight LSQ amplitude of model fit to data on selected px."""
    d, m = data[sel], model[sel]
    return float(np.sum(d * m) / np.sum(m * m))


def curves_for(E, M, NUL):
    yy, xx = np.mgrid[0:101, 0:101]
    rr = np.hypot(xx - HALF, yy - HALF)
    base = (rr <= FIT_R) & np.isfinite(E) & np.isfinite(M)
    if NUL is not None:
        base &= np.isfinite(NUL)

    # rank pixels of E (brightest first) inside the fit region
    idx = np.flatnonzero(base.ravel())
    order = idx[np.argsort(E.ravel()[idx])[::-1]]

    amp0 = lsq_amp(E, M, base)
    nul0 = lsq_amp(NUL, M, base) if NUL is not None else np.nan

    rows = []
    for N in np.concatenate([[0], N_LEVELS]):
        sel = base.copy()
        if N > 0:
            masked = order[:N]
            sel.ravel()[masked] = False
            r_eq = float(np.sqrt(N / np.pi))
            r_max = float(rr.ravel()[masked].max())
        else:
            r_eq, r_max = 0.0, 0.0
        amp = lsq_amp(E, M, sel)
        ctrlE = lsq_amp(E, E, sel)
        ctrlM = lsq_amp(M, M, sel)
        nul = lsq_amp(NUL, M, sel) / nul0 if NUL is not None else np.nan
        # circular-mask flavor (star-fit-experiment-like geometry): mask
        # r < r_eq instead of the N brightest px (same masked area)
        selc = base & (rr >= r_eq) if N > 0 else base
        ampc = lsq_amp(E, M, selc)
        rows.append((N, r_eq, r_max, amp / amp0, ampc / amp0, amp,
                     ctrlE, ctrlM, nul))
    return rows, amp0


def main():
    tables = []
    summary = {}
    for filt, (label, path, kind) in SOURCES.items():
        E, M, NUL, nstars = load_stacks(path, kind)
        rows, amp0 = curves_for(E, M, NUL)
        t = Table(rows=rows, names=[
            'N_masked_px', 'r_mask_equiv', 'r_mask_max',
            'bias_model_to_data', 'bias_circmask', 'amp_raw',
            'control_self_data', 'control_self_model', 'null_machinery'])
        t['filter'] = filt.upper()
        t.meta = {
            'filter': filt.upper(), 'field_det': label, 'n_stars': nstars,
            'source_stack': path, 'fit_region_radius_px': FIT_R,
            'amp_unmasked': amp0,
            'bias_def': ('LSQ amplitude of STPSF-model stack fit to empirical '
                         'stack with inner N brightest px masked, divided by '
                         'the unmasked amplitude. >1 = flux overestimated.'),
            'r_mask_equiv_def': 'sqrt(N_masked/pi) [px]',
            'r_mask_max_def': 'largest radius among masked px [px]',
            'bias_circmask_def': ('same fit but masking the circle '
                                  'r < r_mask_equiv instead of the N '
                                  'brightest px (star-fit-experiment '
                                  'geometry)'),
            'controls': ('control_self_* = same-mask self-fit (==1 validates '
                         'machinery); null_machinery = model+ERR-noise stack '
                         'pushed through identical stacking, fit vs model, '
                         'normalized to its unmasked amp (brick only)'),
        }
        t.write(f'{OUT}/calcurve_{filt}.ecsv', overwrite=True, format='ascii.ecsv')
        tables.append(t)

        ctrl_dev = max(abs(t['control_self_data'] - 1).max(),
                       abs(t['control_self_model'] - 1).max())
        summary[filt] = {
            'n_stars': nstars, 'amp_unmasked': amp0,
            'ctrl_maxdev': float(ctrl_dev),
            'bias_at': {int(N): float(t['bias_model_to_data'][t['N_masked_px'] == N][0])
                        for N in N_EXP},
            'bias_circ_at': {int(N): float(t['bias_circmask'][t['N_masked_px'] == N][0])
                             for N in N_EXP},
            'null_at': {int(N): (float(t['null_machinery'][t['N_masked_px'] == N][0])
                                 if NUL is not None else None)
                        for N in N_EXP},
        }
        print(f'{filt.upper():6s} ({label}, {nstars} stars) amp0={amp0:.4f} '
              f'ctrl_maxdev={ctrl_dev:.2e}')
        for r, N in zip((3, 4, 6, 8, 12), N_EXP):
            b = summary[filt]['bias_at'][int(N)]
            bc = summary[filt]['bias_circ_at'][int(N)]
            nl = summary[filt]['null_at'][int(N)]
            nls = f' null={nl:.3f}' if nl is not None else ''
            print(f'    r_mask={r:2d}px (N={N:4d}): brightN={b:.3f} '
                  f'circ={bc:.3f}{nls}')

    comb = vstack(tables, metadata_conflicts='silent')
    comb.meta = {'note': 'see per-filter calcurve_<filt>.ecsv for provenance',
                 'bias_def': tables[0].meta['bias_def']}
    comb.write(f'{OUT}/calcurves_all.ecsv', overwrite=True, format='ascii.ecsv')
    with open(f'{OUT}/summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'\nwrote {OUT}/calcurves_all.ecsv (+ per-filter) and summary.json')


if __name__ == '__main__':
    main()
