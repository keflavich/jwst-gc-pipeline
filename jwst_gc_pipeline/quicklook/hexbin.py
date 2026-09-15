"""Hexagonal density binning, computed once here and drawn in the browser.

Why bin server-side at all: the whole-sample colour-magnitude diagram is
~10^6-10^7 stars across every pointing, and the point of the page is that
hovering a footprint swaps the overlay instantly.  Shipping the stars
themselves would be tens of megabytes and would re-bin on every hover; shipping
the *bins* is a few hundred kilobytes and the hover is a redraw.

The grid is defined in NORMALISED coordinates -- ``u = (x - xmin)/(xmax-xmin)``,
``v = (y - ymin)/(ymax-ymin)`` -- rather than in magnitudes.  A colour axis
spans ~5 mag and a magnitude axis ~15, so a hexagon that is regular in data
space is a sliver on screen; normalising means the hexagons are regular in the
drawn box, which is what a reader is actually looking at.  The browser
un-normalises with the same ``extent``, so the two sides cannot drift.

Pointy-top axial hexagons, the standard cube-rounding assignment.  Every field
is binned on the SAME grid as the whole sample, so a per-field overlay lands
exactly on the cells of the background it is drawn over -- if each field chose
its own extent, the overlay would be a differently-scaled picture sitting on
top of an unrelated one.
"""
import math

import numpy as np

#: Cells across the normalised x range.  60 puts ~2000 hexagons in the box:
#: fine enough to show the giant branch and the reddening vector as structure,
#: coarse enough that a single pointing (10^4-10^5 stars) fills cells rather
#: than speckling them, and small enough to ship (~30 kB per field as JSON).
DEFAULT_NX = 60


def hex_radius(nx=DEFAULT_NX):
    """Circumradius, in normalised units, of a grid ``nx`` hexagons wide.

    Horizontal centre-to-centre spacing of pointy-top hexagons is
    ``sqrt(3) * R``, so ``nx`` of them span ``nx * sqrt(3) * R = 1``.
    """
    return 1.0 / (nx * math.sqrt(3.0))


def assign(u, v, radius):
    """Axial ``(q, r)`` hexagon index for each normalised point.

    Standard pixel-to-hex for a pointy-top layout followed by cube rounding.
    Returns two int arrays.  Points outside [0, 1] are still assigned -- the
    caller clips, because whether an out-of-range star is dropped or piled onto
    the edge bin is a decision about the plot, not about the geometry.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    qf = (math.sqrt(3.0) / 3.0 * u - v / 3.0) / radius
    rf = (2.0 / 3.0 * v) / radius
    # cube rounding: round all three axes, then fix up whichever moved most
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = np.rint(xf), np.rint(yf), np.rint(zf)
    dx, dy, dz = np.abs(rx - xf), np.abs(ry - yf), np.abs(rz - zf)
    # Three branches, of which only two touch the axes we return: correcting
    # the y component leaves q and r alone, so it is absent rather than a no-op.
    fix_x = (dx > dy) & (dx > dz)
    fix_z = ~fix_x & ~(dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(fix_z, -rx - ry, rz)
    return rx.astype(np.int64), rz.astype(np.int64)


def centre(q, r, radius):
    """Normalised ``(u, v)`` centre of axial cell ``(q, r)``."""
    return (radius * math.sqrt(3.0) * (q + r / 2.0), radius * 1.5 * r)


class Grid:
    """A fixed binning: an extent in data units plus a hexagon size.

    Built once from the whole sample and then reused for every field, so all
    the maps on the page share one geometry.
    """

    def __init__(self, xmin, xmax, ymin, ymax, nx=DEFAULT_NX):
        if not (xmax > xmin and ymax > ymin):
            raise ValueError(f"degenerate extent: x {xmin}..{xmax}, y {ymin}..{ymax}")
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.nx = int(nx)
        self.radius = hex_radius(self.nx)

    @property
    def extent(self):
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def normalise(self, x, y):
        u = (np.asarray(x, dtype=float) - self.xmin) / (self.xmax - self.xmin)
        v = (np.asarray(y, dtype=float) - self.ymin) / (self.ymax - self.ymin)
        return u, v

    def bin(self, x, y):
        """Counts per cell as ``[[q, r, count], ...]``, plus the peak count.

        Points outside the extent are dropped rather than clamped: they would
        otherwise pile into a bright edge cell that reads as a real feature.
        """
        u, v = self.normalise(x, y)
        keep = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        if not keep.any():
            return [], 0, 0
        q, r = assign(u[keep], v[keep], self.radius)
        key = np.stack([q, r], axis=1)
        uniq, counts = np.unique(key, axis=0, return_counts=True)
        cells = [[int(a), int(b), int(c)] for (a, b), c in zip(uniq, counts)]
        cells.sort(key=lambda c: (c[1], c[0]))
        return cells, int(counts.max()), int(keep.sum())

    def to_json(self):
        return {"extent": self.extent, "nx": self.nx, "radius": self.radius}


def percentile_extent(x, y, lo=0.2, hi=99.8, pad=0.04):
    """Plot limits from percentiles, so a handful of outliers do not set them.

    A crowded-field catalog has a tail of unphysical colours (a marginal
    detection paired with a non-detection).  Taking min/max would put the whole
    giant branch in three pixels; percentiles keep the structure and let the
    tail fall off the edge, which is where it belongs on a quicklook.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        raise ValueError("need at least two finite points to set an extent")
    xlo, xhi = np.percentile(x[good], [lo, hi])
    ylo, yhi = np.percentile(y[good], [lo, hi])
    xpad, ypad = (xhi - xlo) * pad, (yhi - ylo) * pad
    return (float(xlo - xpad), float(xhi + xpad),
            float(ylo - ypad), float(yhi + ypad))
