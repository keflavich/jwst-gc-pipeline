
### F150W: R(F)=ZF/crf forms (14 flux bins, unsaturated stars, N=3583; F = crf 5x5 PSF amplitude in MJy/sr)
| image/fit | form | R_inf +- | c [MJy/sr] or (a, b) | chi2/dof |
|---|---|---|---|---|
| ZF (linearised by jwst), annulus bkg | (a) R_inf(1-c/F) | 0.9501 +- 0.0019 | c=8.2 +- 1.0 | 16.3/12 |
| ZF (linearised by jwst), annulus bkg | (b) power law | 0.9515 +- 0.0047 | a=0.010, b=0.93 | 16.2/11 |
| ZF, linearity removed, annulus bkg | (a) R_inf(1-c/F) | 0.9494 +- 0.0019 | c=8.2 +- 1.0 | 14.5/12 |
| ZF, linearity removed, annulus bkg | (b) power law | 0.9510 +- 0.0048 | a=0.010, b=0.91 | 14.4/11 |
| ZF, free 7x7 constant (both images) | (a) R_inf(1-c/F) | 0.9553 +- 0.0019 | c=3.6 +- 0.9 | 17.3/12 |
| ZF, free 7x7 constant (both images) | (b) power law | 1.1354 +- 1.6676 | a=0.163, b=0.05 | 15.6/11 |
R bins, std vs free-constant: F=53: 0.756; F=72: 0.853; F=89: 0.876; F=107: 0.886; F=137: 0.865; F=187: 0.910; F=254: 0.935; F=375: 0.934; F=511: 0.932; F=692: 0.936; F=921: 0.933; F=1207: 0.942; F=1614: 0.947; F=2099: 0.951  ||  free: F=35: 0.845; F=56: 0.861; F=71: 0.918; F=89: 0.929; F=113: 0.930; F=154: 0.934; F=219: 0.949; F=324: 0.956; F=470: 0.948; F=635: 0.942; F=872: 0.942; F=1144: 0.950; F=1576: 0.956; F=2080: 0.960
median free-constant fitted in 7x7 (MJy/sr): ZF -0.539, crf +3.663, g0 -0.333; ZF const minus R*crf const (stars with F>200): -4.193
linearity factor (linearised ZF / uncorrected ZF) on unsat stars by F: F=60: 1.0001; F=98: 1.0001; F=186: 1.0000; F=425: 1.0003; F=871: 1.0006; F=1724: 1.0012

F150W saturated stars, ZF-usable & dolphot-matched: N=706 stars; R_inf(a)=0.9501, c=8.2; R_inf(b)=0.9515; R_inf(free,a)=0.9553, c=3.6; R_inf(unlinearised)=0.9494; R_top(A)=0.9472
| method | Delta precap | Delta capped |
|---|---|---|
| A: top-bin R held | -0.001 +- 0.002 | +0.031 +- 0.001 |
| B: power-law extrap (v1 fit) | -0.002 +- 0.002 | +0.029 +- 0.001 |
| C: g0 chain | -0.017 +- 0.002 | +0.011 +- 0.001 |
| D1: R_inf(1-c/F) inverted | -0.002 +- 0.002 | +0.029 +- 0.001 |
| D2: R_inf const (form a) | -0.004 +- 0.002 | +0.028 +- 0.001 |
| Db: R_inf of power law | -0.006 +- 0.002 | +0.026 +- 0.001 |
| D3: free-const fits on ZF and crf | -0.004 +- 0.002 | +0.026 +- 0.001 |
| U: linearity removed (R_inf unlin) | -0.011 +- 0.002 | +0.018 +- 0.001 |
| dolphot dm_sat-dm_unsat | -0.084 +- 0.002 | -0.062 +- 0.001 |
by dolphot mag (N, D2 precap, D2 capped, D1 precap, dolphot precap, dolphot capped, linearity factor on F_zf):
  14-15: N=10  -0.002 +- 0.026  +0.028 +- 0.009  -0.002 +- 0.026  -0.102 +- 0.018  -0.072 +- 0.002  lin=1.0331 (-0.0353 mag)
  15-16: N=83  +0.004 +- 0.008  +0.038 +- 0.003  +0.004 +- 0.008  -0.110 +- 0.007  -0.077 +- 0.002  lin=1.0218 (-0.0234 mag)
  16-18: N=476  -0.005 +- 0.002  +0.027 +- 0.001  -0.005 +- 0.002  -0.091 +- 0.002  -0.066 +- 0.001  lin=1.0054 (-0.0059 mag)

### F162M: R(F)=ZF/crf forms (14 flux bins, unsaturated stars, N=3584; F = crf 5x5 PSF amplitude in MJy/sr)
| image/fit | form | R_inf +- | c [MJy/sr] or (a, b) | chi2/dof |
|---|---|---|---|---|
| ZF (linearised by jwst), annulus bkg | (a) R_inf(1-c/F) | 0.9499 +- 0.0018 | c=14.5 +- 1.7 | 19.5/12 |
| ZF (linearised by jwst), annulus bkg | (b) power law | 0.9571 +- 0.0063 | a=0.025, b=0.73 | 16.6/11 |
| ZF, linearity removed, annulus bkg | (a) R_inf(1-c/F) | 0.9497 +- 0.0018 | c=14.5 +- 1.7 | 17.1/12 |
| ZF, linearity removed, annulus bkg | (b) power law | 0.9536 +- 0.0049 | a=0.021, b=0.83 | 16.0/11 |
| ZF, free 7x7 constant (both images) | (a) R_inf(1-c/F) | 0.9529 +- 0.0020 | c=6.1 +- 1.8 | 7.8/12 |
| ZF, free 7x7 constant (both images) | (b) power law | 0.9569 +- 0.0070 | a=0.012, b=0.69 | 7.0/11 |
R bins, std vs free-constant: F=53: 0.640; F=73: 0.820; F=89: 0.884; F=112: 0.862; F=143: 0.858; F=204: 0.832; F=306: 0.880; F=472: 0.935; F=760: 0.930; F=1149: 0.933; F=1677: 0.939; F=2426: 0.944; F=3261: 0.948; F=4446: 0.949  ||  free: F=39: 0.940; F=58: 0.826; F=74: 0.924; F=92: 0.905; F=122: 0.892; F=167: 0.888; F=258: 0.914; F=413: 0.946; F=690: 0.953; F=1067: 0.943; F=1550: 0.947; F=2309: 0.948; F=3190: 0.953; F=4418: 0.954
median free-constant fitted in 7x7 (MJy/sr): ZF -1.127, crf +4.040, g0 -0.906; ZF const minus R*crf const (stars with F>200): -5.357
linearity factor (linearised ZF / uncorrected ZF) on unsat stars by F: F=59: 1.0000; F=100: 1.0003; F=204: 1.0000; F=574: 0.9998; F=1548: 1.0004; F=3617: 1.0011

F162M saturated stars, ZF-usable & dolphot-matched: N=716 stars; R_inf(a)=0.9499, c=14.5; R_inf(b)=0.9571; R_inf(free,a)=0.9529, c=6.1; R_inf(unlinearised)=0.9497; R_top(A)=0.9478
| method | Delta precap | Delta capped |
|---|---|---|
| A: top-bin R held | -0.001 +- 0.002 | +0.025 +- 0.001 |
| B: power-law extrap (v1 fit) | -0.006 +- 0.002 | +0.019 +- 0.001 |
| C: g0 chain | -0.010 +- 0.002 | +0.014 +- 0.001 |
| D1: R_inf(1-c/F) inverted | -0.001 +- 0.002 | +0.024 +- 0.001 |
| D2: R_inf const (form a) | -0.003 +- 0.002 | +0.023 +- 0.001 |
| Db: R_inf of power law | -0.011 +- 0.002 | +0.015 +- 0.001 |
| D3: free-const fits on ZF and crf | -0.001 +- 0.002 | +0.025 +- 0.001 |
| U: linearity removed (R_inf unlin) | -0.009 +- 0.002 | +0.015 +- 0.001 |
| dolphot dm_sat-dm_unsat | -0.046 +- 0.002 | -0.029 +- 0.001 |
by dolphot mag (N, D2 precap, D2 capped, D1 precap, dolphot precap, dolphot capped, linearity factor on F_zf):
  13-14: N=8  -0.008 +- 0.015  +0.033 +- 0.004  -0.007 +- 0.015  -0.086 +- 0.019  -0.047 +- 0.003  lin=1.0390 (-0.0416 mag)
  14-15: N=71  -0.008 +- 0.007  +0.041 +- 0.002  -0.008 +- 0.007  -0.085 +- 0.006  -0.046 +- 0.002  lin=1.0225 (-0.0242 mag)
  15-16: N=143  +0.002 +- 0.004  +0.030 +- 0.002  +0.002 +- 0.004  -0.072 +- 0.003  -0.046 +- 0.001  lin=1.0083 (-0.0090 mag)
  16-18: N=494  -0.004 +- 0.002  +0.015 +- 0.002  -0.002 +- 0.002  -0.037 +- 0.001  -0.021 +- 0.001  lin=1.0024 (-0.0027 mag)

### F200W: R(F)=ZF/crf forms (14 flux bins, unsaturated stars, N=3583; F = crf 5x5 PSF amplitude in MJy/sr)
| image/fit | form | R_inf +- | c [MJy/sr] or (a, b) | chi2/dof |
|---|---|---|---|---|
| ZF (linearised by jwst), annulus bkg | (a) R_inf(1-c/F) | 0.9514 +- 0.0017 | c=9.7 +- 0.9 | 25.2/12 |
| ZF (linearised by jwst), annulus bkg | (b) power law | 0.9570 +- 0.0049 | a=0.016, b=0.79 | 23.2/11 |
| ZF, linearity removed, annulus bkg | (a) R_inf(1-c/F) | 0.9504 +- 0.0017 | c=9.5 +- 0.9 | 24.0/12 |
| ZF, linearity removed, annulus bkg | (b) power law | 0.9570 +- 0.0053 | a=0.018, b=0.76 | 21.5/11 |
| ZF, free 7x7 constant (both images) | (a) R_inf(1-c/F) | 0.9557 +- 0.0018 | c=3.1 +- 0.9 | 16.5/12 |
| ZF, free 7x7 constant (both images) | (b) power law | 1.1167 +- 1.6205 | a=0.149, b=0.05 | 15.5/11 |
R bins, std vs free-constant: F=26: 0.523; F=43: 0.744; F=59: 0.840; F=76: 0.874; F=96: 0.840; F=126: 0.892; F=171: 0.916; F=241: 0.885; F=367: 0.924; F=533: 0.940; F=800: 0.928; F=1185: 0.943; F=1610: 0.950; F=2293: 0.949  ||  free: F=15: 0.589; F=27: 0.917; F=39: 0.811; F=54: 0.946; F=72: 0.918; F=95: 0.907; F=130: 0.967; F=187: 0.947; F=291: 0.944; F=450: 0.947; F=689: 0.940; F=1073: 0.950; F=1510: 0.954; F=2256: 0.958
median free-constant fitted in 7x7 (MJy/sr): ZF -0.295, crf +8.868, g0 -0.011; ZF const minus R*crf const (stars with F>200): -8.071
linearity factor (linearised ZF / uncorrected ZF) on unsat stars by F: F=25: 0.9999; F=62: 1.0000; F=115: 1.0005; F=256: 1.0003; F=705: 1.0004; F=1745: 1.0011

F200W saturated stars, ZF-usable & dolphot-matched: N=288 stars; R_inf(a)=0.9514, c=9.7; R_inf(b)=0.9570; R_inf(free,a)=0.9557, c=3.1; R_inf(unlinearised)=0.9504; R_top(A)=0.9478
| method | Delta precap | Delta capped |
|---|---|---|
| A: top-bin R held | +0.013 +- 0.003 | +0.051 +- 0.001 |
| B: power-law extrap (v1 fit) | -0.000 +- 0.003 | +0.037 +- 0.001 |
| C: g0 chain | -0.018 +- 0.003 | +0.019 +- 0.001 |
| D1: R_inf(1-c/F) inverted | +0.009 +- 0.003 | +0.047 +- 0.001 |
| D2: R_inf const (form a) | +0.009 +- 0.003 | +0.046 +- 0.001 |
| Db: R_inf of power law | +0.002 +- 0.003 | +0.040 +- 0.001 |
| D3: free-const fits on ZF and crf | +0.008 +- 0.003 | +0.046 +- 0.001 |
| U: linearity removed (R_inf unlin) | -0.007 +- 0.003 | +0.029 +- 0.001 |
| dolphot dm_sat-dm_unsat | -0.050 +- 0.002 | -0.016 +- 0.001 |
by dolphot mag (N, D2 precap, D2 capped, D1 precap, dolphot precap, dolphot capped, linearity factor on F_zf):
  14-15: N=79  +0.013 +- 0.007  +0.054 +- 0.001  +0.014 +- 0.007  -0.040 +- 0.005  -0.008 +- 0.003  lin=1.0255 (-0.0273 mag)
  15-16: N=195  +0.006 +- 0.003  +0.043 +- 0.001  +0.007 +- 0.003  -0.054 +- 0.003  -0.020 +- 0.001  lin=1.0126 (-0.0136 mag)
  16-18: N=14  +0.010 +- 0.008  +0.032 +- 0.008  +0.011 +- 0.008  -0.031 +- 0.005  -0.013 +- 0.007  lin=1.0054 (-0.0059 mag)

### F277W: R(F)=ZF/crf forms (14 flux bins, unsaturated stars, N=3583; F = crf 5x5 PSF amplitude in MJy/sr)
| image/fit | form | R_inf +- | c [MJy/sr] or (a, b) | chi2/dof |
|---|---|---|---|---|
| ZF (linearised by jwst), annulus bkg | (a) R_inf(1-c/F) | 0.9953 +- 0.0021 | c=3.1 +- 0.3 | 28.0/12 |
| ZF (linearised by jwst), annulus bkg | (b) power law | 1.1740 +- 0.4455 | a=0.137, b=0.12 | 15.3/11 |
| ZF, linearity removed, annulus bkg | (a) R_inf(1-c/F) | 0.9942 +- 0.0021 | c=3.1 +- 0.3 | 24.9/12 |
| ZF, linearity removed, annulus bkg | (b) power law | 1.1055 +- 0.2003 | a=0.087, b=0.18 | 14.2/11 |
| ZF, free 7x7 constant (both images) | (a) R_inf(1-c/F) | 0.9989 +- 0.0022 | c=2.1 +- 0.2 | 46.5/12 |
| ZF, free 7x7 constant (both images) | (b) power law | 1.3720 +- 2.6587 | a=0.259, b=0.05 | 24.9/11 |
R bins, std vs free-constant: F=42: 0.949; F=53: 0.943; F=61: 0.943; F=69: 0.947; F=78: 0.961; F=91: 0.958; F=103: 0.954; F=116: 0.956; F=138: 0.966; F=167: 0.978; F=207: 0.984; F=254: 0.980; F=306: 0.990; F=382: 0.990  ||  free: F=34: 0.950; F=47: 0.985; F=57: 0.977; F=65: 0.958; F=75: 0.962; F=86: 0.971; F=100: 0.972; F=113: 0.972; F=135: 0.971; F=163: 0.984; F=202: 0.992; F=253: 0.990; F=304: 1.000; F=379: 1.001
median free-constant fitted in 7x7 (MJy/sr): ZF -0.121, crf +4.457, g0 -0.074; ZF const minus R*crf const (stars with F>200): -4.380
linearity factor (linearised ZF / uncorrected ZF) on unsat stars by F: F=49: 1.0005; F=69: 1.0005; F=94: 1.0005; F=129: 1.0009; F=205: 1.0011; F=326: 1.0018

F277W saturated stars, ZF-usable & dolphot-matched: N=759 stars; R_inf(a)=0.9953, c=3.1; R_inf(b)=1.1740; R_inf(free,a)=0.9989, c=2.1; R_inf(unlinearised)=0.9942; R_top(A)=0.9873
| method | Delta precap | Delta capped |
|---|---|---|
| A: top-bin R held | +0.007 +- 0.001 | +0.012 +- 0.001 |
| B: power-law extrap (v1 fit) | -0.027 +- 0.001 | -0.024 +- 0.002 |
| C: g0 chain | -0.016 +- 0.001 | -0.013 +- 0.001 |
| D1: R_inf(1-c/F) inverted | +0.002 +- 0.001 | +0.006 +- 0.001 |
| D2: R_inf const (form a) | -0.002 +- 0.001 | +0.003 +- 0.001 |
| Db: R_inf of power law | -0.181 +- 0.001 | -0.176 +- 0.001 |
| D3: free-const fits on ZF and crf | +0.005 +- 0.001 | +0.009 +- 0.001 |
| U: linearity removed (R_inf unlin) | -0.014 +- 0.001 | -0.010 +- 0.002 |
| dolphot dm_sat-dm_unsat | +0.003 +- 0.002 | +0.007 +- 0.002 |
by dolphot mag (N, D2 precap, D2 capped, D1 precap, dolphot precap, dolphot capped, linearity factor on F_zf):
  14-15: N=78  -0.003 +- 0.003  +0.004 +- 0.002  -0.003 +- 0.003  -0.002 +- 0.005  +0.001 +- 0.005  lin=1.0516 (-0.0546 mag)
  15-16: N=137  -0.007 +- 0.003  -0.006 +- 0.003  -0.006 +- 0.003  -0.005 +- 0.003  -0.003 +- 0.003  lin=1.0170 (-0.0183 mag)
  16-18: N=543  -0.000 +- 0.002  +0.006 +- 0.002  +0.004 +- 0.002  +0.008 +- 0.002  +0.014 +- 0.003  lin=1.0048 (-0.0052 mag)

### F300M: R(F)=ZF/crf forms (14 flux bins, unsaturated stars, N=3583; F = crf 5x5 PSF amplitude in MJy/sr)
| image/fit | form | R_inf +- | c [MJy/sr] or (a, b) | chi2/dof |
|---|---|---|---|---|
| ZF (linearised by jwst), annulus bkg | (a) R_inf(1-c/F) | 0.9830 +- 0.0019 | c=3.7 +- 0.4 | 29.8/12 |
| ZF (linearised by jwst), annulus bkg | (b) power law | 1.3228 +- 1.8738 | a=0.254, b=0.05 | 13.1/11 |
| ZF, linearity removed, annulus bkg | (a) R_inf(1-c/F) | 0.9821 +- 0.0019 | c=3.8 +- 0.4 | 26.8/12 |
| ZF, linearity removed, annulus bkg | (b) power law | 1.3199 +- 1.8703 | a=0.253, b=0.05 | 13.7/11 |
| ZF, free 7x7 constant (both images) | (a) R_inf(1-c/F) | 0.9910 +- 0.0020 | c=3.2 +- 0.4 | 32.3/12 |
| ZF, free 7x7 constant (both images) | (b) power law | 1.3271 +- 1.9110 | a=0.249, b=0.05 | 8.6/11 |
R bins, std vs free-constant: F=50: 0.897; F=63: 0.948; F=75: 0.941; F=87: 0.961; F=103: 0.946; F=122: 0.944; F=155: 0.954; F=196: 0.963; F=244: 0.960; F=300: 0.968; F=375: 0.969; F=508: 0.971; F=678: 0.981; F=880: 0.988  ||  free: F=44: 0.941; F=61: 0.963; F=73: 0.953; F=85: 0.971; F=102: 0.959; F=122: 0.956; F=154: 0.959; F=195: 0.970; F=241: 0.968; F=298: 0.978; F=375: 0.977; F=502: 0.984; F=666: 0.990; F=872: 0.998
median free-constant fitted in 7x7 (MJy/sr): ZF -0.266, crf +4.128, g0 -0.180; ZF const minus R*crf const (stars with F>200): -4.458
linearity factor (linearised ZF / uncorrected ZF) on unsat stars by F: F=59: 1.0006; F=86: 1.0001; F=130: 1.0005; F=226: 1.0005; F=373: 1.0008; F=744: 1.0015

F300M saturated stars, ZF-usable & dolphot-matched: N=657 stars; R_inf(a)=0.9830, c=3.7; R_inf(b)=1.3228; R_inf(free,a)=0.9910, c=3.2; R_inf(unlinearised)=0.9821; R_top(A)=0.9828
| method | Delta precap | Delta capped |
|---|---|---|
| A: top-bin R held | +0.002 +- 0.001 | +0.009 +- 0.001 |
| B: power-law extrap (v1 fit) | -0.024 +- 0.002 | -0.017 +- 0.002 |
| C: g0 chain | -0.021 +- 0.002 | -0.015 +- 0.001 |
| D1: R_inf(1-c/F) inverted | +0.003 +- 0.001 | +0.010 +- 0.001 |
| D2: R_inf const (form a) | +0.001 +- 0.001 | +0.009 +- 0.001 |
| Db: R_inf of power law | -0.321 +- 0.001 | -0.314 +- 0.001 |
| D3: free-const fits on ZF and crf | +0.002 +- 0.001 | +0.010 +- 0.001 |
| U: linearity removed (R_inf unlin) | -0.011 +- 0.002 | -0.003 +- 0.002 |
| dolphot dm_sat-dm_unsat | +0.004 +- 0.002 | +0.014 +- 0.002 |
by dolphot mag (N, D2 precap, D2 capped, D1 precap, dolphot precap, dolphot capped, linearity factor on F_zf):
  13-14: N=58  +0.002 +- 0.003  +0.012 +- 0.002  +0.003 +- 0.003  -0.022 +- 0.006  -0.014 +- 0.006  lin=1.0410 (-0.0436 mag)
  14-15: N=142  -0.009 +- 0.003  +0.000 +- 0.002  -0.009 +- 0.003  -0.024 +- 0.004  -0.015 +- 0.004  lin=1.0153 (-0.0165 mag)
  15-16: N=273  +0.004 +- 0.002  +0.009 +- 0.002  +0.006 +- 0.002  +0.014 +- 0.003  +0.018 +- 0.003  lin=1.0058 (-0.0063 mag)
  16-18: N=184  +0.005 +- 0.003  +0.016 +- 0.003  +0.008 +- 0.003  +0.030 +- 0.004  +0.040 +- 0.004  lin=1.0023 (-0.0025 mag)

### F405N: R(F)=ZF/crf forms (14 flux bins, unsaturated stars, N=3584; F = crf 5x5 PSF amplitude in MJy/sr)
| image/fit | form | R_inf +- | c [MJy/sr] or (a, b) | chi2/dof |
|---|---|---|---|---|
| ZF (linearised by jwst), annulus bkg | (a) R_inf(1-c/F) | 0.9658 +- 0.0047 | c=20.0 +- 5.0 | 5.4/12 |
| ZF (linearised by jwst), annulus bkg | (b) power law | 0.9739 +- 0.0131 | a=0.034, b=0.76 | 4.6/11 |
| ZF, linearity removed, annulus bkg | (a) R_inf(1-c/F) | 0.9646 +- 0.0048 | c=19.2 +- 5.0 | 7.8/12 |
| ZF, linearity removed, annulus bkg | (b) power law | 0.9753 +- 0.0156 | a=0.037, b=0.70 | 6.7/11 |
| ZF, free 7x7 constant (both images) | (a) R_inf(1-c/F) | 0.9762 +- 0.0049 | c=9.8 +- 4.9 | 11.1/12 |
| ZF, free 7x7 constant (both images) | (b) power law | 0.9719 +- 0.0043 | a=0.000, b=2.01 | 9.9/11 |
R bins, std vs free-constant: F=22: 0.284; F=32: -0.069; F=43: 0.659; F=56: 0.931; F=78: 0.713; F=106: 0.900; F=147: 0.827; F=206: 0.869; F=290: 0.907; F=500: 0.886; F=791: 0.925; F=1276: 0.954; F=2243: 0.959; F=4631: 0.962  ||  free: F=17: 0.432; F=21: -0.668; F=26: 0.498; F=32: 0.397; F=42: 0.807; F=58: 1.234; F=89: 1.038; F=131: 0.868; F=205: 0.915; F=323: 1.010; F=585: 0.929; F=1067: 0.955; F=1905: 0.979; F=4083: 0.972
median free-constant fitted in 7x7 (MJy/sr): ZF -0.541, crf +44.619, g0 -0.197; ZF const minus R*crf const (stars with F>200): -53.154
linearity factor (linearised ZF / uncorrected ZF) on unsat stars by F: F=6: 0.9989; F=15: 0.9945; F=28: 0.9982; F=54: 1.0013; F=180: 1.0015; F=1452: 1.0006

F405N saturated stars, ZF-usable & dolphot-matched: N=19 stars; R_inf(a)=0.9658, c=20.0; R_inf(b)=0.9739; R_inf(free,a)=0.9762, c=9.8; R_inf(unlinearised)=0.9646; R_top(A)=0.9609
| method | Delta precap | Delta capped |
|---|---|---|
| A: top-bin R held | +0.007 +- 0.007 | +0.028 +- 0.008 |
| B: power-law extrap (v1 fit) | -0.006 +- 0.007 | +0.018 +- 0.008 |
| C: g0 chain | -- | -- |
| D1: R_inf(1-c/F) inverted | +0.002 +- 0.007 | +0.024 +- 0.008 |
| D2: R_inf const (form a) | +0.002 +- 0.007 | +0.023 +- 0.008 |
| Db: R_inf of power law | -0.007 +- 0.007 | +0.013 +- 0.008 |
| D3: free-const fits on ZF and crf | -0.008 +- 0.011 | +0.020 +- 0.011 |
| U: linearity removed (R_inf unlin) | -0.006 +- 0.010 | +0.007 +- 0.009 |
| dolphot dm_sat-dm_unsat | +0.040 +- 0.014 | +0.051 +- 0.009 |
by dolphot mag (N, D2 precap, D2 capped, D1 precap, dolphot precap, dolphot capped, linearity factor on F_zf):
  8-13: N=16  +0.004 +- 0.007  +0.029 +- 0.009  +0.005 +- 0.007  +0.047 +- 0.012  +0.056 +- 0.006  lin=1.0045 (-0.0049 mag)
  13-14: N=3  -0.044 +- 0.054  -0.044 +- 0.054  -0.042 +- 0.054  +0.022 +- 0.004  +0.022 +- 0.004  lin=1.0017 (-0.0018 mag)
