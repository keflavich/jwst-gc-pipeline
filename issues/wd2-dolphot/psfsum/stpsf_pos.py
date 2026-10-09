"""Direct stpsf check: compute a distorted PSF at an asymmetric detector
position and compare its sum with the SIAF pixel area at (x, y) and (y, x)."""
import numpy as np
import stpsf
from siafarea import siaf_area

nrc = stpsf.NIRCam()
nrc.filter = 'F200W'
nrc.detector = 'NRCA1'
for pos in [(100, 1900), (1900, 100)]:
    nrc.detector_position = pos
    psf = nrc.calc_psf(fov_pixels=101, oversample=2, monochromatic=2.0e-6)
    names = [h.name for h in psf]
    out = []
    for h in psf:
        hd = h.header
        out.append(f"{h.name}: DET_X={hd.get('DET_X')} DET_Y={hd.get('DET_Y')} "
                   f"OVERSAMP={hd.get('OVERSAMP')} sum/os2={h.data.sum() / hd.get('OVERSAMP', 1)**2:.5f}")
    a_xy = siaf_area('nrca1', pos[0], pos[1])
    a_yx = siaf_area('nrca1', pos[1], pos[0])
    print(f"detector_position={pos}: siaf area at (x,y)={a_xy:.6g} at (y,x)={a_yx:.6g} ratio xy/yx={a_xy / a_yx:.5f}")
    for o in out:
        print('   ', o)
    s_ov = psf['OVERSAMP'].data.sum() / 4
    s_od = psf['OVERDIST'].data.sum() / 4
    print(f"    OVERDIST/OVERSAMP sum ratio = {s_od / s_ov:.5f}")
