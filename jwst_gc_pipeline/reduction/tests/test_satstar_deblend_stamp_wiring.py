"""``get_saturated_stars`` stamps the deblend counts onto the satstar catalogue
meta, only for a frame the deblend ran on (#1196).

``build_deblended_source_records`` is replaced by a stub that fills ``stats``
(its own counts are tested in ``test_satstar_deblend_psf_veto.py``), so the
tests pin the wiring: the stub's stats reach ``table.meta`` as plain integers,
and a frame fitted without the deblend carries no ``DBL*`` key.  The fit itself
is the real one on a synthetic two-star frame with a Gaussian PSF.
"""
import numpy as np
from photutils.psf import CircularGaussianPRF

import jwst_gc_pipeline.reduction.satstar_deblend as SD
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF

from .test_satstar_recovered_cap import _two_star_frame

class _Prf(CircularGaussianPRF):
    """The pipeline calls ``evaluate`` without the ``fwhm`` argument."""

    def evaluate(self, x, y, flux, x_0, y_0, fwhm=None):
        fwhm = self.fwhm.value if fwhm is None else fwhm
        return super().evaluate(x, y, flux, x_0, y_0, fwhm)


_KEYS = ('DBLNCMP', 'DBLNSPL', 'DBLFAIL', 'DBLNVET')


def _run(monkeypatch, *, zeroframe, deblend):
    fh, sat = _two_star_frame()
    n = sat.shape[0]
    calls = []

    def _records(saturated, sources, coms, sizes, zf, data, fwhm, stats=None,
                 **kw):
        calls.append(kw)
        stats.update(n_components=1, n_split=1, n_failed=0, n_vetoed=3)
        return [{'com': c, 'label': 1, 'forced': False,
                 'sat_area': int(sat.sum())}
                for c in [(100.0, 93.0), (100.0, 107.0)]]

    monkeypatch.setattr(SD, 'build_deblended_source_records', _records)
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor',
                        lambda data, dq, zf, **kw: (data, sat.copy(),
                                                    np.zeros_like(sat), None))
    monkeypatch.setattr(SSF, 'get_psf',
                        lambda header, **kw: _Prf(fwhm=1.6))
    # the stpsf generator is built up front and reads data files absent on CI
    monkeypatch.setattr(SSF.stpsf, 'NIRCam', lambda *a, **kw: object())
    zf = np.ones((n, n)) if zeroframe else None
    tab = SSF.get_saturated_stars(fh, zeroframe=zf, zeroframe_deblend=deblend,
                                  plot=False)
    return tab, calls


def test_the_counts_reach_the_catalogue_meta_as_integers(monkeypatch):
    tab, calls = _run(monkeypatch, zeroframe=True, deblend=True)
    assert len(calls) == 1
    assert [tab.meta[k] for k in _KEYS] == [1, 1, 0, 3]


def test_a_frame_fitted_without_the_deblend_carries_no_counts(monkeypatch):
    for zeroframe, deblend in ((True, False), (False, True)):
        tab, calls = _run(monkeypatch, zeroframe=zeroframe, deblend=deblend)
        assert calls == []
        assert not any(k in tab.meta for k in _KEYS)
