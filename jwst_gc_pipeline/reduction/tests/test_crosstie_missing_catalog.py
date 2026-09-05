"""``crosstie_offset`` refuses to report a zero shift it never measured.

The JWST<->JWST cross-tie is a hand-maintained constant: ``--remeasure-crosstie``
prints a paste-ready ``'f115w': (+0.01868, -0.00080),`` block and the operator
copies it into ``CROSSTIE``.  When the catalog glob resolved to nothing the printer
emitted ``(+0.00000, +0.00000)`` in that same form, so a lookup miss reached the
table as a measured zero -- replacing brick/1182 constants of 18-21 mas, the size
the cross-tie exists to remove.

A measurement that RAN and declined (too few pairs, too few vetted core matches)
still returns (0,0): that is a separate severity and this module does not touch it.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.reduction import build_virac2_offsets as bvo


def _write_cat(path, n=400, ra0=266.5, dec0=-28.7, dra_arcsec=0.0, ddec_arcsec=0.0,
               seed=0):
    """A vetted-catalog-shaped file: skycoord + flux, optionally rigidly shifted."""
    rng = np.random.default_rng(seed)
    ra = ra0 + rng.uniform(-0.01, 0.01, n) + dra_arcsec / 3600.0
    dec = dec0 + rng.uniform(-0.01, 0.01, n) + ddec_arcsec / 3600.0
    t = Table({'skycoord': SkyCoord(ra=ra, dec=dec, unit='deg'),
               'flux': np.full(n, 1000.0)})
    path.parent.mkdir(parents=True, exist_ok=True)
    t.write(path, overwrite=True)
    return path


@pytest.fixture
def keyed_region(tmp_path, monkeypatch):
    """A region keyed in CROSSTIE, with its basepath under tmp_path."""
    rc = dict(proposal='1182', field='004', basepath=str(tmp_path / 'field'),
              filts={'f115w': ('F115W', 2022.0, 'm2')})
    cfg = dict(master_cat=str(tmp_path / 'master' / 'f212n_master.fits'),
               master_name='2221 F212N',
               shifts={'f115w': (+0.01868, -0.00080)})
    monkeypatch.setitem(bvo.REGION, '_test1182', rc)
    monkeypatch.setitem(bvo.CROSSTIE, '_test1182', cfg)
    return rc, cfg


def test_missing_src_catalog_raises(keyed_region, tmp_path):
    """Master on disk, src glob matches nothing -> raise, naming both patterns."""
    rc, cfg = keyed_region
    _write_cat(tmp_path / 'master' / 'f212n_master.fits')
    with pytest.raises(bvo.CrosstieCatalogMissingError) as exc:
        bvo.crosstie_offset('f115w', rc)
    msg = str(exc.value)
    assert 'f212n_master.fits' in msg          # the master pattern that DID resolve
    assert 'f115w_merged_indivexp_merged' in msg   # the src pattern that did not
    assert 'no file matches' in msg


def test_missing_master_catalog_raises(keyed_region, tmp_path):
    """Src on disk, master glob matches nothing -> raise."""
    rc, cfg = keyed_region
    _write_cat(tmp_path / 'field' / 'catalogs' /
               'f115w_merged_indivexp_merged_x_m2_dao_basic_vetted.fits')
    with pytest.raises(bvo.CrosstieCatalogMissingError):
        bvo.crosstie_offset('f115w', rc)


def test_allow_missing_restores_zero(keyed_region, capsys):
    """The explicit escape hatch still returns (0,0) with the WARN line."""
    rc, cfg = keyed_region
    assert bvo.crosstie_offset('f115w', rc, allow_missing=True) == (0.0, 0.0)
    assert 'APPLYING 0 (WARN)' in capsys.readouterr().out


def test_unkeyed_region_still_returns_zero(tmp_path):
    """A region with no CROSSTIE entry has nothing to tie: (0,0), no raise."""
    rc = dict(proposal='9999', field='001', basepath=str(tmp_path), filts={})
    assert bvo.crosstie_offset('f200w', rc) == (0.0, 0.0)


def test_a_real_offset_still_measures(keyed_region, tmp_path):
    """The refusal did not swallow the measurement: a planted 20 mas shift on
    catalogs that DO resolve is still measured and returned with the right sign."""
    rc, cfg = keyed_region
    _write_cat(tmp_path / 'master' / 'f212n_master.fits', seed=1)
    # src is the same stars displaced by +20 mas in RA (coordinate, no cosdec)
    # and -10 mas in Dec; the function returns the NEGATION (the shift to ADD).
    rng = np.random.default_rng(1)
    n = 400
    ra = 266.5 + rng.uniform(-0.01, 0.01, n)
    dec = -28.7 + rng.uniform(-0.01, 0.01, n)
    t = Table({'skycoord': SkyCoord(ra=ra + 0.020 / 3600.0,
                                    dec=dec - 0.010 / 3600.0, unit='deg'),
               'flux': np.full(n, 1000.0)})
    p = tmp_path / 'field' / 'catalogs' / \
        'f115w_merged_indivexp_merged_x_m2_dao_basic_vetted.fits'
    p.parent.mkdir(parents=True, exist_ok=True)
    t.write(p, overwrite=True)
    add_ra, add_de = bvo.crosstie_offset('f115w', rc)
    # returned in arcsec, negated (the shift to ADD)
    assert add_ra == pytest.approx(-0.020, abs=1e-4)
    assert add_de == pytest.approx(+0.010, abs=1e-4)


def test_too_few_pairs_still_warns_and_returns_zero(keyed_region, tmp_path, capsys):
    """A measurement that RAN and declined is untouched: still (0,0) + WARN."""
    rc, cfg = keyed_region
    _write_cat(tmp_path / 'master' / 'f212n_master.fits', n=5, seed=2)
    _write_cat(tmp_path / 'field' / 'catalogs' /
               'f115w_merged_indivexp_merged_x_m2_dao_basic_vetted.fits',
               n=5, seed=3)
    assert bvo.crosstie_offset('f115w', rc) == (0.0, 0.0)
    assert 'candidate pairs -> APPLYING 0 (WARN)' in capsys.readouterr().out
