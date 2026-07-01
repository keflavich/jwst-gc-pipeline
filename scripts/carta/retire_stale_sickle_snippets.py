#!/usr/bin/env python
"""Retire stale-pipeline sickle CARTA snippets (those referencing i2d products
that no longer exist -- legacy iter2/iter3/basic residuals, missing MIRI
brightsky mosaics) into snippets/_stale/.  A snippet is retired only if MOST
of its i2d references are broken (>=50%), so partially-current ones are flagged
not moved.  Reversible (plain move).  Dry-run unless --commit."""
import glob, json, os, re, shutil, sys

ROOT = '/orange/adamginsburg'
SNIPS = '/home/adamginsburg/.carta/config/snippets'
STALE = f'{SNIPS}/_stale'
COMMIT = '--commit' in sys.argv
_img = re.compile(r'appendFile\("([^"]+\.fits)"\)')


def scan(path):
    code = json.load(open(path)).get('code', '')
    refs = _img.findall(code)
    miss = [r for r in refs if not os.path.exists(os.path.join(ROOT, r))]
    return len(refs), len(miss)


def main():
    retire, flag = [], []
    for p in sorted(glob.glob(f'{SNIPS}/sickle*.json')):
        ntot, nmiss = scan(p)
        if nmiss == 0:
            continue
        frac = nmiss / max(ntot, 1)
        (retire if frac >= 0.5 else flag).append((p, ntot, nmiss, frac))
    print("=== RETIRE (>=50% broken -> stale-pipeline) ===")
    for p, nt, nm, fr in retire:
        print(f"  {os.path.basename(p):45s} {nm}/{nt} broken ({fr:.0%})")
    print("\n=== FLAG ONLY (partially broken, left in place) ===")
    for p, nt, nm, fr in flag:
        print(f"  {os.path.basename(p):45s} {nm}/{nt} broken ({fr:.0%})")
    print(f"\n=== {'MOVING' if COMMIT else 'WOULD MOVE'} {len(retire)} snippets to {STALE} ===")
    if COMMIT and retire:
        os.makedirs(STALE, exist_ok=True)
        for p, *_ in retire:
            shutil.move(p, os.path.join(STALE, os.path.basename(p)))
    elif not COMMIT:
        print("(dry-run; --commit to move)")


if __name__ == '__main__':
    main()
