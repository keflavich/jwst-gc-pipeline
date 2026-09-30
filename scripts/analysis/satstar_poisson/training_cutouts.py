"""Cutouts of the brightness-sequence stars (README section 5) with cutout.prep.

    python training_cutouts.py [<datadir>, default ../data]

Needs jw10678<obs>001_02101_0000<d>_nrcblong_cal.fits for d = 1..6 of each visit below
(e.g. `python s3data.py --outdir ../data 10678 116 nrcblong cal`).  Writes
trn_<obs>_e<d>.npz, one per dither on which the star falls; a dither whose GWCS inverse
of the star is not finite (star off the detector) is skipped with a message, so the
dither number d is kept in the file name (loo_crop.py --prefix handles the gaps).

RA/Dec are the GWCS world coordinates (stdatamodels meta.wcs, never the SIP header) of
the star's dither-1 detector position given in the comment; they round-trip through the
GWCS inverse to < 0.001 px.  Rounded to 7 decimals they reproduce the cutouts used for
README section 5 with identical pixel data and the star position within 0.0015 px
(checked on obs 042).  Dithers that contain the star: 042 has 1,2,3,6; 070 and 046
have 1,2 only (too few for the LOO, see README section 5).
"""
import os
import sys

import numpy as np
import stdatamodels.jwst.datamodels as dm

from cutout import prep

HALFWIDTH = 512
#        obs:  (RA [deg],     Dec [deg])      dither-1 NRCBLONG (x, y)
STARS = {'116': (266.7874176, -28.4989889),  # (1140, 1619)
         '069': (266.5180850, -28.8803394),  # (1187, 1391)
         '126': (266.9368967, -28.4434759),  # (196, 1814)
         '078': (266.4671392, -28.7884783),  # (1165, 469)
         '063': (266.4482998, -28.9244746),  # (705, 1565)
         '042': (266.3223522, -29.0735319),  # (1536, 975)
         '070': (266.4843813, -28.8525569),  # (1705, 1463)
         '046': (266.4615126, -29.0724150),  # (1707, 735)
         }


def main(datadir='../data'):
    for obs, (ra, dec) in STARS.items():
        for d in range(1, 7):
            fn = os.path.join(datadir, f'jw10678{obs}001_02101_{d:05d}_nrcblong_cal.fits')
            with dm.open(fn) as m:
                xt, yt = m.meta.wcs.invert(ra, dec)
            if not (np.isfinite(xt) and np.isfinite(yt)):
                print(f'skip obs {obs} dither {d}: star off the detector', flush=True)
                continue
            prep(fn, ra, dec, HALFWIDTH, f'trn_{obs}_e{d}.npz')


if __name__ == '__main__':
    main(*sys.argv[1:2])
