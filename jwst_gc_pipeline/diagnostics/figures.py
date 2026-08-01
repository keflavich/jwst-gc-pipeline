"""The record a figure builder hands back to the write-up generator."""

import os
from dataclasses import dataclass, field as _dcfield


@dataclass
class FigureResult:
    """One finished figure, plus everything the LaTeX needs to discuss it.

    *measurements* is the bridge between the plotting code and the prose: the
    write-up quotes numbers out of it rather than re-deriving them, so the
    text and the figure can never disagree.
    """

    key: str
    path: str
    caption: str
    section: str
    measurements: dict = _dcfield(default_factory=dict)
    notes: list = _dcfield(default_factory=list)

    @property
    def relpath(self):
        return os.path.join('figures', os.path.basename(self.path))

    @property
    def label(self):
        return f'fig:{self.key}'


def save(fig, outdir, key, fmt='pdf'):
    """Write *fig* into ``outdir/figures`` and return the path."""
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, f'{key}.{fmt}')
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
