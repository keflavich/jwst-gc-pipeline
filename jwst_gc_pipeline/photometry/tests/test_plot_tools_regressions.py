"""Regression tests for plotting/plot_tools.py helpers.

Pins ``_filter_to_wavelength``: the filter/color name -> effective wavelength
map was previously a 5-clause nested ternary copy-pasted ~16 times (in 3
divergent variants).  These tests lock the special-case wavelengths and the
generic 'F<NNN>X' -> NNN/100 um fallback so the consolidated helper can't drift.
"""
import pytest
import astropy.units as u

from jwst_gc_pipeline.plotting import plot_tools as P


@pytest.mark.parametrize('name,um', [
    ('410m405', 4.10),
    ('405m410', 4.05),
    ('182m187', 1.82),
    ('187m182', 1.87),
    ('Hmag', 1.634),
    ('Ksmag', 2.143527),
])
def test_special_names(name, um):
    assert P._filter_to_wavelength(name) == um * u.um


@pytest.mark.parametrize('name,um', [
    ('F410M', 4.10),
    ('f405n', 4.05),
    ('F200W', 2.00),
    ('f1130w', 11.30),
    ('F2550W', 25.50),
])
def test_generic_jwst_filter_fallback(name, um):
    assert P._filter_to_wavelength(name) == um * u.um
