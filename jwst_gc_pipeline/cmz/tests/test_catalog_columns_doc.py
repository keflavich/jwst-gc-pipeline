"""Field pages link the catalog column descriptions, and the file covers the
columns the released catalogs carry."""
import importlib.util
import os
import re

import pytest

_REL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'scripts', 'release'))

#: Column names of the catalogs on the release pages as of v1.9-2026.10, with
#: the filter replaced by {filter}.  The per-filter vetted catalogs carry the
#: same names without the _{filter} suffix, which the file explains once.
RELEASED_COLUMNS = """
skycoord_ref.ra skycoord_ref.dec skycoord_ref_filtername
sep_{filter} id_{filter} skycoord_{filter}.ra skycoord_{filter}.dec mask_{filter}
flux_{filter} flux_err_{filter} flux_err_prop_{filter} flux_init_{filter}
flux_jy_{filter} eflux_jy_{filter} mag_ab_{filter} emag_ab_{filter}
mag_vega_{filter} qfit_{filter} cfit_{filter} flags_{filter}
group_size_{filter} local_bkg_{filter} iter_found_{filter}
dra_{filter} ddec_{filter} std_ra_{filter} std_dec_{filter}
nmatch_{filter} nmatch_good_{filter}
is_saturated_{filter} replaced_saturated_{filter} satstar_match_sep_{filter}
satstar_gate_rejected_{filter} satclip_corrected_{filter}
satclip_corr_mag_{filter} near_saturated_{filter}_{filter}
satstar_nframes_{filter} satstar_nmeas_{filter}
satstar_std_ra_{filter} satstar_std_dec_{filter}
prominence_{filter} peak_sb_{filter} local_emission_snr_{filter}
sky_clean_{filter} mean_modelsub_bkg_{filter} mean_modelsub_bkg_err_{filter}
mean_modelsub_bkg_std_{filter} modelsub_bkg_nframes_{filter}
modelsub_bkg_rms_{filter} modelsub_bkg_npix_{filter}
forced_refit_frac_{filter} forced_refit_nframes_{filter}
independently_detected_{filter} n_filt_independent forced_filled_{filter}
forced_snr_{filter}
flux_jy_410m405 mag_ab_410m405 mag_vega_410m405
flux_jy_405m410 mag_ab_405m410 mag_vega_405m410
flux_jy_182m187 mag_ab_182m187 mag_vega_182m187
flux_jy_187m182 mag_ab_187m182 mag_vega_187m182
joint_tile also_in_tiles
source_id_union seed_filter_origin n_filters detected_{filter}
fluxerr_{filter}
""".split()


@pytest.fixture(scope='module')
def mw():
    import sys
    sys.path.insert(0, _REL)
    spec = importlib.util.spec_from_file_location(
        'make_webpage', os.path.join(_REL, 'make_webpage.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_released_column_is_described(mw):
    text = mw.CATALOG_COLUMNS_SRC.read_text()
    named = set(re.findall(r'[A-Za-z0-9_{}.]+', text))
    missing = [c for c in RELEASED_COLUMNS if c not in named]
    assert not missing, f'catalog_columns.txt does not describe {missing}'


def test_file_is_plain_ascii(mw):
    # served as text/plain with no charset; keep it readable everywhere
    mw.CATALOG_COLUMNS_SRC.read_bytes().decode('ascii')


def _manifest(field, files):
    return {'field': field, 'version': 'v1.7-2026.09', 'group': None,
            'release_path': f'/releases/v1.7/{field}',
            'built': '2026-09-17T12:00:00', 'mode': 'copy',
            'globus_collection_id': 'x',
            'globus_https_base': 'https://example.invalid', 'files': files}


def _catalog(field):
    return {'category': 'catalog', 'kind': 'catalog_full', 'filter': None,
            'iteration': 'm8', 'size_bytes': 1,
            'dest': f'{field}/catalogs/basic_merged_m8.fits',
            'url': 'https://example.invalid/basic_merged_m8.fits'}


def test_page_with_catalogs_links_the_descriptions(mw):
    page = mw.render_field_page('brick', _manifest('brick', [_catalog('brick')]), '')
    assert f"href='{mw.CATALOG_COLUMNS_NAME}'" in page


def test_page_without_catalogs_does_not(mw):
    page = mw.render_field_page('m92', _manifest('m92', []), '')
    assert mw.CATALOG_COLUMNS_NAME not in page


def test_flags_entry_lists_every_photutils_bit(mw):
    # released catalogs carry bits up to 2048 (brick v1.0 F200W has flags=2049)
    text = mw.CATALOG_COLUMNS_SRC.read_text()
    entry = text.split('  flags_{filter}\n', 1)[1].split('\n  group_size_', 1)[0]
    listed = {int(b) for b in re.findall(r'^\s+(\d+) = ', entry, re.M)}
    assert listed == {2 ** k for k in range(12)}
