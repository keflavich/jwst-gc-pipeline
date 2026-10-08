"""dm vs dolphot SNR and vs our SNR, to separate selection (Eddington) bias from a bkg offset."""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

tag = sys.argv[1] if len(sys.argv) > 1 else 'mainfcbg'
t = Table.read(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_{tag}.fits')
ref = Table.read('/orange/adamginsburg/jwst/wd2/catalogs/wd2_nircam_wf_mf_nf.ecsv')
c1 = SkyCoord(t['RA'], t['DEC'], unit='deg')
c2 = SkyCoord(ref['RA'], ref['DEC'], unit='deg')
idx, sep, _ = c1.match_to_catalog_sky(c2)
sepm = sep.to(u.mas).value
print('sep>1mas:', (sepm > 1).sum(), 'max', sepm.max())
rr = np.asarray(ref['MAG200W'], float)[idx]
tt = np.asarray(t['ref_200W'], float)
both = np.isfinite(rr) & np.isfinite(tt)
print('ref_200W mismatch rows:', (np.abs(rr - tt)[both] > 1e-6).sum(), 'of', both.sum())
bands = ['200W', '212N', '277W', '300M', '335M', '410M', '405N', '466N']
snr_edges = np.array([3, 5, 7, 10, 15, 20, 30, 50, 100, 300, 1e5])
for which in ('dolphot', 'ours'):
    print(f'== dm median/MAD vs {which} SNR (tag={tag})')
    print('band  ' + ' '.join(f'{lo:>11.0f}' for lo in snr_edges[:-1]))
    for b in bands:
        r = np.asarray(t[f'ref_{b}'], float)
        o = np.asarray(t[f'our_{b}'], float)
        sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
        if which == 'dolphot':
            e = np.asarray(ref[f'ERRMAG{b}'], float)[idx]
            snr = 1.0857 / e
        else:
            snr = np.asarray(t[f'our_flux_{b}'], float) / np.asarray(t[f'our_flux_err_{b}'], float)
        ok = np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90) & np.isfinite(snr)
        dm = o - r
        cells = []
        for lo, hi in zip(snr_edges[:-1], snr_edges[1:]):
            s = ok & (snr >= lo) & (snr < hi)
            if s.sum() < 20:
                cells.append(f'{"":>11}')
                continue
            d = dm[s]
            med = np.median(d)
            cells.append(f'{med:+.3f}/{1.4826*np.median(np.abs(d-med)):.3f}')
        print(f'{b:5s} ' + ' '.join(cells))
