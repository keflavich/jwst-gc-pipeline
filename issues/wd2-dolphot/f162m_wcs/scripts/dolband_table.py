"""m [arcsec] of (our band - dolphot per-band catalog) per detector, from
datascale_dolband_<DOLBAND>.txt.  m = |(tr-, curl+)| / (2*31) * 206265."""
import re

DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
print('| ours | dolphot | ' + ' | '.join(d.upper()[3:] for d in DETS) + ' |')
print('|---|---|' + '---|' * len(DETS))
for dol in ('F150W', 'F115W', 'F182M'):
    txt = open(f'datascale_dolband_{dol}.txt').read()
    for blk in re.split(r'\n### ', txt)[1:]:
        band = blk.split()[0].upper()
        m = {}
        for line in blk.splitlines():
            r = re.match(r'(nrc[ab]\d)\s.*\(tr- ([+-][\d.]+), curl\+ ([+-][\d.]+)\)', line)
            if r:
                a, b = float(r.group(2)), float(r.group(3))
                m[r.group(1)] = (a * a + b * b) ** 0.5 / 62 * 206265
        print(f'| {band} | {dol} | ' + ' | '.join(f'{m[d]:.1f}' if d in m else '–' for d in DETS) + ' |')
