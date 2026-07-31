"""Build the fit-footprint flux-correction calibration table from the sweep array.

For every frame in cal_manifest.tsv and every swept size, match the satstars to
the size-81 reference (truth) by sky position and record the correction factor
that recovers the truth flux:

    R(size, r_core, filter) = flux_81 / flux_size

r_core = sqrt(sat_area / pi). The output table (correction_data.ecsv) + diagnostic
figure (R vs size, colored by r_core, one panel per filter) are the input to the
correction-model fit. Read-only wrt /orange.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = os.path.dirname(__file__)
MANIFEST = os.path.join(HERE, 'cal_manifest.tsv')
OUT = os.environ.get('CORR_OUT', '/blue/adamginsburg/adamginsburg/tmp/fitsweep_cal')
REF_SIZE = 81
MATCH_ARCSEC = 0.3
MIN_FLUX = 1e3   # ignore junk-faint rows


def frame_table(label, outdir):
    cats = {}
    for f in sorted(glob.glob(f'{outdir}/satstar_size*.fits')):
        size = int(os.path.basename(f).replace('satstar_size', '').replace('.fits', ''))
        cats[size] = Table.read(f)
    if REF_SIZE not in cats:
        print(f'  [skip] {label}: no size-{REF_SIZE} reference ({sorted(cats)})')
        return None
    ref = cats[REF_SIZE]
    filt = label.split('_')[1]
    if 'sat_area' not in ref.colnames:
        print(f'  [warn] {label}: no sat_area column; skipping')
        return None
    infov = ~np.asarray(ref['outside_fov_seed'], bool) if 'outside_fov_seed' in ref.colnames \
        else np.ones(len(ref), bool)
    good = infov & (np.asarray(ref['flux_fit'], float) > MIN_FLUX)
    ref = ref[good]
    sc_ref = SkyCoord(ref['skycoord_fit'])
    rows = []
    for size, cat in cats.items():
        if size == REF_SIZE or not len(cat):
            continue
        sc = SkyCoord(cat['skycoord_fit'])
        idx, sep, _ = sc_ref.match_to_catalog_sky(sc)
        for k in range(len(ref)):
            if sep[k].arcsec > MATCH_ARCSEC:
                continue
            fref = float(ref['flux_fit'][k])
            fsz = float(cat['flux_fit'][idx[k]])
            if not (np.isfinite(fref) and np.isfinite(fsz) and fsz > 0 and fref > 0):
                continue
            rcore = np.sqrt(max(float(ref['sat_area'][k]), 1) / np.pi)
            rows.append((filt, label, size, rcore, fref, fsz, fref / fsz,
                         float(ref['flux_fit'][k]),
                         sc_ref[k].separation(sc[idx[k]]).to(u.mas).value))
    if not rows:
        return None
    t = Table(rows=rows, names=('filter', 'frame', 'size', 'r_core', 'flux_ref',
                                'flux_size', 'R', 'flux', 'dpos_mas'))
    print(f'  {label}: {len(t)} matched (size,star) rows, r_core '
          f'{t["r_core"].min():.1f}..{t["r_core"].max():.1f}')
    return t


def main():
    labels = []
    for line in open(MANIFEST):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4:
            labels.append((p[0], p[3]))
    tabs = []
    for label, outdir in labels:
        t = frame_table(label, outdir)
        if t is not None:
            tabs.append(t)
    if not tabs:
        raise SystemExit('[corr] no calibration rows found (array not done?)')
    T = vstack(tabs)
    os.makedirs(OUT, exist_ok=True)
    T.write(f'{OUT}/correction_data.ecsv', overwrite=True)
    print(f'\n[corr] wrote {OUT}/correction_data.ecsv  ({len(T)} rows)')

    filters = sorted(set(T['filter']))
    sizes = sorted(set(int(s) for s in T['size']))
    print('\n=== median R = flux_81/flux_size  by (filter, size) ===')
    print('  filter   ' + ' '.join(f'{s:>7d}' for s in sizes))
    for filt in filters:
        row = []
        for s in sizes:
            m = (T['filter'] == filt) & (T['size'] == s)
            row.append(f'{np.median(T["R"][m]):7.3f}' if m.sum() else f'{"--":>7}')
        print(f'  {filt:8s} ' + ' '.join(row))

    fig, axes = plt.subplots(1, len(filters), figsize=(5 * len(filters), 5), squeeze=False)
    for j, filt in enumerate(filters):
        a = axes[0, j]
        m = T['filter'] == filt
        sc = a.scatter(T['size'][m], T['R'][m], c=T['r_core'][m], s=10, alpha=0.5,
                       cmap='viridis', vmin=0, vmax=np.percentile(T['r_core'], 98))
        for s in sizes:
            ms = m & (T['size'] == s)
            if ms.sum():
                a.plot(s, np.median(T['R'][ms]), 'r_', ms=18, mew=2)
        a.axhline(1.0, color='0.5', ls='--')
        a.set_xlabel('fit size (px)'); a.set_ylabel('R = flux_81 / flux_size')
        a.set_title(f'{filt}  (n={int(m.sum())})'); a.set_ylim(0.8, 3)
        plt.colorbar(sc, ax=a, label='r_core (px)')
    fig.suptitle('Fit-footprint flux-correction calibration (R>1 = small box underestimates)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f'{OUT}/correction_data.png', dpi=120)
    print(f'[corr] wrote {OUT}/correction_data.png')


if __name__ == '__main__':
    main()
