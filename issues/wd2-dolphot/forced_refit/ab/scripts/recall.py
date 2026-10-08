"""Fraction of dolphot stars with a pipeline row within 0.08" (not replaced_saturated) per arm, by dolphot mag."""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from classify import load, dsk, A
for b in ['150W', '187N', '200W', '277W']:
    ref = A.ref[b]; have = np.where(np.isfinite(ref))[0]
    out = {}
    for arm in ('fr0', 'fr1'):
        t, s, f = load(arm, b)
        j, d, _ = dsk[have].match_to_catalog_sky(s)
        out[arm] = d.arcsec < 0.08
    print(f'F{b}: ' + '; '.join(
        f'{lo}-{hi}: N={int(((ref[have]>=lo)&(ref[have]<hi)).sum())} fr0 {out["fr0"][(ref[have]>=lo)&(ref[have]<hi)].mean():.4f} fr1 {out["fr1"][(ref[have]>=lo)&(ref[have]<hi)].mean():.4f}'
        for lo, hi in ((10, 18.6), (18.6, 21), (21, 23), (23, 30))))
