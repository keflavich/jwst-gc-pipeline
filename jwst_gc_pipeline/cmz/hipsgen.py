"""Thin driver for CDS Hipsgen.jar (image HiPS) and Hipsgen-cat.jar (catalog HiPS).

The recommended substrate for a GROWING survey: Hipsgen builds a mono HiPS from
FITS mosaics with NATIVE incremental tiling (re-run over an index that includes
new files; only affected tiles + their ancestors rewrite), plus RGB HiPS from
2-3 mono HiPS, a coverage MOC, and Allsky previews.  Hipsgen-cat builds a
progressive catalog HiPS that Aladin renders zoom-by-zoom.

Both are Java tools.  Point ``HIPSGEN_JAR`` / ``HIPSGENCAT_JAR`` at the jars (or
pass ``jar=``); :func:`_java` checks java + jar and raises a clear error if
absent.  This module shells out; it does not reimplement HiPS.
"""
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
