"""OPTIONAL Java drivers for CDS Hipsgen.jar / Hipsgen-cat.jar.

The primary, pure-Python path is ``cmz.hips`` (incremental mono HiPS + two-color
derivation) — prefer it. This module is a convenience for two things pure-Python
does not yet cover, when Java is available:
  * ``build_catalog_hips`` — a progressive CATALOG HiPS (Hipsgen-cat) for Aladin
    (there is no pure-Python catalog-HiPS generator; the pure-Python catalog path
    is HATS via ``cmz.hats_export`` for analysis + a flat catalog for download);
  * ``build_mono_hips``/``build_rgb_hips`` — an alternative image-HiPS builder if
    you specifically want CDS Hipsgen's tiling/Allsky.

Both are Java tools. Point ``HIPSGEN_JAR`` / ``HIPSGENCAT_JAR`` at the jars (or
pass ``jar=``); :func:`_java` checks java + jar and raises a clear error if
absent. This module shells out; it does not reimplement HiPS.
"""
import importlib.util
import os
import shutil
import subprocess


def _java(jar_env, jar=None):
    if shutil.which('java') is None:
        raise RuntimeError("Hipsgen needs Java on PATH ('java' not found).")
    jar = jar or os.environ.get(jar_env)
    if not jar or not os.path.exists(jar):
        raise RuntimeError(
            f"Hipsgen jar not found: set ${jar_env} (or pass jar=) to the "
            f"Hipsgen jar path. Download from https://aladin.cds.unistra.fr/hips/")
    return jar


def _run(cmd):
    print('+ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _patch_hips_landing_page(out_dir):
    """Rewrite ``out_dir/index.html``'s CDS landing-template ``<script>`` tag
    to the self-hosted-first fallback chain (see
    ``scripts/release/patch_hips_landing_page.py``).

    Hipsgen's own ``INDEX`` action is what writes ``index.html``; this is a
    no-op if that file is not there (``RGB``-only run) or if the patcher
    script is not present alongside this checkout (e.g. a package installed
    without its sibling ``scripts/`` directory) -- never raises for either,
    since a stale landing page is cosmetic, not a release blocker.
    """
    index_html = os.path.join(out_dir, 'index.html')
    if not os.path.exists(index_html):
        return
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    patcher_path = os.path.join(repo_root, 'scripts', 'release',
                                'patch_hips_landing_page.py')
    if not os.path.exists(patcher_path):
        print(f'(skipping landing-page patch: {patcher_path} not found)',
              flush=True)
        return
    spec = importlib.util.spec_from_file_location(
        'patch_hips_landing_page', patcher_path)
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    status = patcher.patch_file(index_html)
    print(f'landing page: {status}\t{index_html}', flush=True)


def build_mono_hips(in_dir, out_dir, order=13, frame='galactic', jar=None,
                    extra=()):
    """Build/UPDATE a mono image HiPS from the FITS mosaics in ``in_dir``.

    Re-running with new files in ``in_dir`` incrementally updates ``out_dir``
    (Hipsgen only rewrites affected tiles).  Runs INDEX + TILES + a coverage MOC.
    """
    jar = _java('HIPSGEN_JAR', jar)
    base = ['java', '-jar', jar, f'in={in_dir}', f'out={out_dir}',
            f'order={order}', f'hips_frame={frame}']
    _run(base + ['INDEX', 'TILES', 'MOC'] + list(extra))
    _patch_hips_landing_page(out_dir)
    return out_dir


def build_rgb_hips(out_dir, red_hips, green_hips, blue_hips, jar=None, extra=()):
    """Build an RGB HiPS from three mono HiPS (Hipsgen RGB).

    For the two-color scheme, pass the interpolated green HiPS as ``green_hips``
    (or derive color with ``cmz.hips.derive_two_color_hips`` instead, which needs
    no Java).
    """
    jar = _java('HIPSGEN_JAR', jar)
    cmd = ['java', '-jar', jar, f'out={out_dir}',
           f'red={red_hips}', f'green={green_hips}', f'blue={blue_hips}', 'RGB']
    _run(cmd + list(extra))
    _patch_hips_landing_page(out_dir)
    return out_dir


def build_catalog_hips(in_catalog, out_dir, ra_col='ra', dec_col='dec',
                       jar=None, extra=()):
    """Build a progressive CATALOG HiPS (Hipsgen-cat.jar) for Aladin display.

    ``in_catalog`` is a flat table Hipsgen-cat can read (CSV/TSV/FITS/VOTable).
    The output HiPS catalog is what Aladin Lite/Desktop render as a zoomable
    source layer.
    """
    jar = _java('HIPSGENCAT_JAR', jar)
    cmd = ['java', '-jar', jar, f'-cat={in_catalog}', f'-out={out_dir}',
           f'-ra={ra_col}', f'-dec={dec_col}']
    _run(cmd + list(extra))
    return out_dir
