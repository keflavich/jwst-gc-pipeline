"""Realness proxy: recovery in an INDEPENDENT observation of the same field.

Brick 2221/o001 (F182M ...) and 1182/o004 (F200W, F444W) are separate visits
with separate detectors and pointings.  A source that is real in F182M is
detected in F200W at a rate that depends on its brightness; a spurious one
(noise peak, PSF artifact, emission knot misfit) is matched only by chance.
For each group the match fraction within ``r_mas`` is compared, at matched
F182M flux, with the already-vetted catalog and with the chance rate from
the same positions shifted by a few arcsec.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

SHIFTS_AS = [(2.1, 0.0), (0.0, 2.3), (-1.9, 1.1), (1.3, -2.2)]


def _shift(sc, dx_as, dy_as):
    return SkyCoord(sc.ra + dx_as * u.arcsec / np.cos(sc.dec), sc.dec + dy_as * u.arcsec)


def in_footprint(sc, ref, r_as=1.0):
    """Inside the reference catalog's footprint: a reference source within r_as."""
    _, sep, _ = sc.match_to_catalog_sky(ref)
    return sep.arcsec < r_as


def match_fraction(sc, ref, r_mas=60.0):
    """(fraction matched, chance fraction) of positions ``sc`` against ``ref``."""
    if len(sc) == 0:
        return np.nan, np.nan
    _, sep, _ = sc.match_to_catalog_sky(ref)
    real = float(np.mean(sep.to_value(u.mas) < r_mas))
    ch = []
    for dx, dy in SHIFTS_AS:
        _, s2, _ = _shift(sc, dx, dy).match_to_catalog_sky(ref)
        ch.append(np.mean(s2.to_value(u.mas) < r_mas))
    return real, float(np.mean(ch))


def binned(groups, ref, edges, r_mas=60.0):
    """groups: {name: (SkyCoord, flux)} -> {name: [(n, frac, chance, (frac-chance)/(1-chance))...]}"""
    out = {}
    for name, (sc, flux) in groups.items():
        rows = []
        lf = np.log10(np.where(flux > 0, flux, np.nan))
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = (lf >= lo) & (lf < hi)
            f, c = match_fraction(sc[k], ref, r_mas) if k.sum() else (np.nan, np.nan)
            purity = (f - c) / (1 - c) if np.isfinite(f) and c < 1 else np.nan
            rows.append((int(k.sum()), f, c, purity))
        out[name] = rows
    return out
