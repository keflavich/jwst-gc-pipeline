"""Close pairs in the m7 band seed, split by where each member came from.

The m7 seed of each band (``_build_m7_band_seed``) is the cross-band seed plus
that band's own m6 vetted sources.  An own-band source joins when it lies at
least ``manual_crossband_seed_max_sep_mas`` (30 mas) from every cross-band
position; the companion cut then drops it only when a much brighter pool source
(> 1/COMPRAT times its flux) lies within COMPFWHM.  Nothing deduplicates the
union at a fraction of the FWHM, so two seeds 30-60 mas apart can describe one
star.  This script counts nearest-neighbour pairs closer than 0.5 FWHM in the
written seed (``crossband_seed_manual_<module>_<filt>_vetted.fits``) by origin.

usage: python seed_pairs.py <catalog-dir> <out-prefix> FILT:fwhm_mas [FILT:fwhm_mas ...]
"""
import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

MAX_SEP_MAS = 30.0


def pairs(path, fwhm_mas):
    t = Table.read(path)
    hdr = fits.getheader(path, 1)
    sc = SkyCoord(t['skycoord'])
    # local tangent-plane offsets in mas around the field centre
    c = SkyCoord(np.median(sc.ra), np.median(sc.dec))
    off = sc.transform_to(c.skyoffset_frame())
    xy = np.c_[off.lon.to_value('mas'), off.lat.to_value('mas')]
    d, j = cKDTree(xy).query(xy, k=2)
    sep, nbr = d[:, 1], j[:, 1]
    origin = np.asarray(t['seed_origin']).astype(str)
    close = np.flatnonzero((sep < 0.5 * fwhm_mas) & (np.arange(len(t)) < nbr))  # each pair once
    combo = ['+'.join(sorted((origin[a], origin[b]))) for a, b in zip(close, nbr[close])]
    u, n = np.unique(combo, return_counts=True)
    fr = np.asarray(t['flux'], float)
    ratio = fr[nbr[close]] / fr[close]
    ratio = np.where(ratio > 1, 1 / ratio, ratio)
    return dict(path=os.path.basename(path), n=len(t), fwhm_mas=fwhm_mas,
                header={k: hdr[k] for k in ('COMPFWHM', 'COMPRAT', 'NCOMPAN', 'DEDUPMAS') if k in hdr},
                origins=dict(zip(*np.unique(origin, return_counts=True))),
                frac_nn_lt_0p5fwhm=float(np.mean(sep < 0.5 * fwhm_mas)),
                n_pairs_lt_0p5fwhm=int(len(close)),
                pairs_by_origin=dict(zip(u.tolist(), n.tolist())),
                pair_sep_mas_p10_50_90=np.percentile(sep[close], [10, 50, 90]).tolist() if len(close) else None,
                pair_faint_over_bright_p16_50_84=np.percentile(ratio, [16, 50, 84]).tolist() if len(close) else None), \
        sep, origin, nbr


def main(catdir, outp, specs):
    res = {}
    fig, axs = plt.subplots(1, len(specs), figsize=(5.5 * len(specs), 4))
    axs = np.atleast_1d(axs)
    for ax, spec in zip(axs, specs):
        filt, fwhm = spec.split(':')
        fwhm = float(fwhm)
        path = os.path.join(catdir, f'crossband_seed_manual_merged_{filt.lower()}_vetted.fits')
        summ, sep, origin, nbr = pairs(path, fwhm)
        summ = {k: ({kk: int(vv) for kk, vv in v.items()} if k == 'origins' else v) for k, v in summ.items()}
        res[filt] = summ
        bins = np.arange(0, 1.5 * fwhm, 2.0)
        own = np.char.startswith(origin, 'own') | (origin == 'i2d')
        for lab, m in (('both cross-band', ~own & ~own[nbr]), ('cross-band + own-band', own ^ own[nbr]),
                       ('both own-band', own & own[nbr])):
            ax.hist(sep[m], bins, histtype='step', lw=1.5, label=f'{lab} ({m.sum()})')
        ax.axvline(MAX_SEP_MAS, color='k', ls=':', lw=1)
        ax.text(MAX_SEP_MAS, 0.96, ' max_sep_mas 30', fontsize=7, transform=ax.get_xaxis_transform())
        ax.axvline(0.5 * fwhm, color='r', ls='--', lw=1)
        ax.text(0.5 * fwhm, 0.88, ' 0.5 FWHM', fontsize=7, color='r', transform=ax.get_xaxis_transform())
        ax.set_yscale('log')
        ax.set_xlabel('separation to nearest other seed (mas)')
        ax.set_ylabel('N seeds')
        ax.set_title(f'brick {filt} m7 band seed (n={summ["n"]}), FWHM {fwhm:.0f} mas', fontsize=9)
        ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(outp + '.png', dpi=100)
    json.dump(res, open(outp + '.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
