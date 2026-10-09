# #1148 note A: rim and dilation-buffer pixels rewritten, measured curve (off) vs header rate (on)

wd2 exposure 1, tree_main2 inputs, PR branch 353ac7df, NIRCAM_SATSTAR_RECOVERED_CAP=1, sat_dilate=3, infl_tol=0.10. Script: rimcount.py; log: rimcount.log.

| frame | rim SAT off / on | buffer off / on | buffer only-on / only-off | SAT rim value on/off (median) | buffer-only-on: data / R_hdr g0 (median) |
|---|---|---|---|---|---|
| F150W nrcb1 | 33 / 33 | 30 / 51 | 21 / 0 | 0.9849 | 1.109 |
| F150W nrcb3 | 100 / 100 | 86 / 1245 | 1159 / 0 | 0.9330 | 1.122 |
| F200W nrcb1 | 82 / 82 | 60 / 112 | 52 / 0 | 0.9789 | 1.111 |
| F200W nrcb3 | 85 / 85 | 135 / 1828 | 1693 / 0 | 0.9319 | 1.123 |
| F250M nrcblong | 565 / 565 | 124 / 78 | 0 / 46 | 1.0063 | nan |
| F300M nrcblong | 770 / 770 | 164 / 143 | 4 / 25 | 0.9991 | 1.100 |

Logged R_header / measured bright-end R lines:

    [zeroframe R header] rim rewritten with R_header=0.08889 (PHOTMJSR / t(group 0)); measured bright-end R=0.08821 is 0.992x that
    [zeroframe R header] rim rewritten with R_header=0.08759 (PHOTMJSR / t(group 0)); measured bright-end R=0.09139 is 1.04x that
    [zeroframe R header] rim rewritten with R_header=0.07444 (PHOTMJSR / t(group 0)); measured bright-end R=0.07426 is 0.998x that
    [zeroframe R header] rim rewritten with R_header=0.07242 (PHOTMJSR / t(group 0)); measured bright-end R=0.07567 is 1.04x that
    [zeroframe R header] rim rewritten with R_header=0.06203 (PHOTMJSR / t(group 0)); measured bright-end R=0.059 is 0.951x that
    [zeroframe R header] rim rewritten with R_header=0.03595 (PHOTMJSR / t(group 0)); measured bright-end R=0.03454 is 0.961x that
