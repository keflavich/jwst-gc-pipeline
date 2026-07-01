#!/usr/bin/env python
"""Retarget BROKEN file references in sickle CARTA snippets to their current
on-disk equivalent.  Only rewrites references whose file no longer exists (e.g.
the swept non-'_group_' residual/model i2d); leaves working references intact.
Backs up each modified snippet to <name>.json.bak.  Dry-run unless --commit."""
import glob, json, os, re, shutil, sys

ROOT = '/orange/adamginsburg'
SNIPS = '/home/adamginsburg/.carta/config/snippets'
COMMIT = '--commit' in sys.argv

_appendfile = re.compile(r'appendFile\("([^"]+)"\)')
_addcat = re.compile(r'_addCatalog\("([^"]+)",\s*"([^"]+)"')


def _newest(pat):
    g = sorted(glob.glob(pat), key=os.path.getmtime)
    return g[-1] if g else None


def remap_image(relpath):
    """relpath is ROOT-relative.  Return a replacement rel-path if the file is
    missing and a current equivalent exists, else None (keep as-is)."""
    absp = os.path.join(ROOT, relpath)
    if os.path.exists(absp):
        return None
    d = os.path.dirname(absp)
    bn = os.path.basename(absp)
    m = re.search(r'clear-([a-z0-9]+)-([a-z0-9]+)_', bn)
    if not m:
        return None
    filt, mod = m.group(1), m.group(2)
    for kind in ('mergedcat_residual_i2d', 'mergedcat_model_i2d',
                 'mergedcat_residual_smoothed_bg_i2d'):
        if bn.endswith(kind + '.fits'):
            # prefer the newest _group_ product for this filter+module+kind
            rep = (_newest(f'{d}/*-{filt}-{mod}_*group*_{kind}.fits')
                   or _newest(f'{d}/*-{filt}-{mod}_*_{kind}.fits'))
            return os.path.relpath(rep, ROOT) if rep else None
    if bn.endswith('_data_i2d.fits'):
        rep = _newest(f'{d}/*-{filt}-{mod}_data_i2d.fits')
        return os.path.relpath(rep, ROOT) if rep else None
    return None


def remap_catalog(d_rel, fname):
    absdir = os.path.join(ROOT, d_rel.strip('/'))
    if os.path.exists(os.path.join(absdir, fname)):
        return None
    # per-filter merged catalog: <filt>_<mod>_indivexp_merged_..._dao_basic.fits
    m = re.match(r'([a-z0-9]+)_([a-z0-9]+)_indivexp_merged', fname)
    if m:
        cands = [c for c in glob.glob(
                 f'{absdir}/{m.group(1)}_{m.group(2)}_indivexp_merged_*_dao_basic.fits')
                 if not c.endswith(('_allcols.fits', '_vetted.fits'))]
        if cands:
            return os.path.basename(max(cands, key=lambda p: (
                int(re.search(r'_m(\d)_', p).group(1)) if re.search(r'_m(\d)_', p) else 0)))
    return None


def fix(path):
    s = json.load(open(path))
    code = s.get('code', '')
    changes = []

    def _img(mo):
        rel = mo.group(1)
        rep = remap_image(rel)
        if rep and rep != rel:
            changes.append((rel, rep))
            return f'appendFile("{rep}")'
        return mo.group(0)

    def _cat(mo):
        d, f = mo.group(1), mo.group(2)
        rep = remap_catalog(d, f)
        if rep and rep != f:
            changes.append((f, rep))
            return mo.group(0).replace(f'"{f}"', f'"{rep}"', 1)
        return mo.group(0)

    code2 = _appendfile.sub(_img, code)
    code2 = _addcat.sub(_cat, code2)
    if changes:
        print(f"\n{os.path.basename(path)}: {len(changes)} retargets")
        for a, b in changes:
            print(f"    {os.path.basename(a)}  ->  {os.path.basename(b)}")
        if COMMIT:
            shutil.copy(path, path + '.bak')
            s['code'] = code2
            json.dump(s, open(path, 'w'), indent=4)
    return len(changes)


def main():
    tot = 0
    for p in sorted(glob.glob(f'{SNIPS}/sickle*.json')):
        tot += fix(p)
    print(f"\n=== {'FIXED' if COMMIT else 'WOULD FIX'} {tot} references "
          f"across sickle snippets ===")
    if not COMMIT:
        print("(dry-run; --commit to write, backs up .bak)")


if __name__ == '__main__':
    main()
