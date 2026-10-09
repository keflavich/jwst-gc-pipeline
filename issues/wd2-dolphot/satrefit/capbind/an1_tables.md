### F250M: binding pixel of the H cap (rows where cap_H x rcor < a_H)

| dolphot mag | rows | binds | frac replaced (ff) | frac g0 SAT flag | frac g0 SAT/DNU flag | median offset px | median g0 DN | median first-frame DN | median g0/ceiling | median src DN / FW | median data/model(a_H) | median data/model(a_H+h0+bgfree) | frac binding = model-peak pixel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12.3-13.0 | 104 | 83 (80%) | 0.95 | 0.94 | 0.95 | 0.53 | 42607 | 39891 | 1.02 | 0.85 | 0.868 | 0.905 | 0.61 |
| 13.0-13.5 | 105 | 73 (70%) | 1.00 | 0.77 | 1.00 | 0.40 | 41800 | 34190 | 1.00 | 0.73 | 0.940 | 0.959 | 0.97 |
| 13.5-14.0 | 203 | 92 (45%) | 1.00 | 0.03 | 1.00 | 0.41 | 38230 | 20008 | 0.91 | 0.43 | 0.962 | 0.991 | 0.93 |
| 14.0-15.0 | 723 | 322 (45%) | 0.64 | 0.00 | 0.64 | 0.42 | 23284 | 9878 | 0.56 | 0.28 | 0.969 | 0.992 | 0.91 |
| 15.0-16.0 | 1353 | 278 (21%) | 0.01 | 0.00 | 0.01 | 0.39 | 11384 | 4522 | 0.27 | 0.24 | 0.973 | 0.990 | 0.91 |
| 16.0-17.0 | 699 | 53 (8%) | 0.00 | 0.00 | 0.00 | 0.36 | 5314 | 2167 | 0.13 | 0.11 | 0.959 | 0.978 | 0.96 |

F250M binding pixels, 12.3-13 mag, split by pixel type

| type | n | median src DN / FW | median data/model(a_H) | median offset | median g0/ceiling |
|---|---|---|---|---|---|
| replaced (first frame x k) | 79 | 0.85 | 0.868 | 0.50 | 1.02 |
| direct g0 | 4 | 0.16 | 0.913 | 2.36 | 0.17 |

F250M: capcore2 design-b style exclusion (replaced pixels with first frame > t x max first frame treated as unmeasured, inside the flux bound only), rows binding in 12.3-13 mag

| t | rows | frac with binding pixel excluded | median cap_new/cap_old over rows with exclusion | median cap_new/cap_old all | frac cap_new < a_H | frac of rows whose new binding pixel is replaced | median src DN / FW of new binding pixel | median data/model(a_H) of new binding pixel |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 83 | 0.75 | 1.036 | 1.000 | 0.98 | 0.95 | 0.41 | 0.863 |
| 0.8 | 83 | 0.63 | 1.052 | 1.000 | 0.99 | 0.95 | 0.61 | 0.894 |

### F300M: binding pixel of the H cap (rows where cap_H x rcor < a_H)

| dolphot mag | rows | binds | frac replaced (ff) | frac g0 SAT flag | frac g0 SAT/DNU flag | median offset px | median g0 DN | median first-frame DN | median g0/ceiling | median src DN / FW | median data/model(a_H) | median data/model(a_H+h0+bgfree) | frac binding = model-peak pixel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12.3-13.0 | 72 | 64 (89%) | 1.00 | 0.98 | 1.00 | 0.50 | 42255 | 40506 | 1.01 | 0.84 | 0.872 | 0.891 | 0.62 |
| 13.0-13.5 | 98 | 73 (74%) | 1.00 | 0.81 | 1.00 | 0.41 | 42191 | 34246 | 1.01 | 0.71 | 0.938 | 0.958 | 0.88 |
| 13.5-14.0 | 264 | 141 (53%) | 1.00 | 0.01 | 1.00 | 0.41 | 39310 | 20919 | 0.94 | 0.44 | 0.973 | 1.000 | 0.91 |
| 14.0-15.0 | 876 | 392 (45%) | 0.69 | 0.00 | 0.69 | 0.41 | 23537 | 9982 | 0.56 | 0.29 | 0.974 | 0.998 | 0.91 |
| 15.0-16.0 | 1443 | 310 (21%) | 0.00 | 0.00 | 0.00 | 0.39 | 11674 | 4678 | 0.28 | 0.25 | 0.980 | 0.996 | 0.91 |
| 16.0-17.0 | 603 | 35 (6%) | 0.00 | 0.00 | 0.00 | 0.39 | 5940 | 2287 | 0.14 | 0.13 | 0.963 | 0.978 | 0.83 |

F300M binding pixels, 12.3-13 mag, split by pixel type

| type | n | median src DN / FW | median data/model(a_H) | median offset | median g0/ceiling |
|---|---|---|---|---|---|
| replaced (first frame x k) | 64 | 0.84 | 0.872 | 0.50 | 1.01 |

F300M: capcore2 design-b style exclusion (replaced pixels with first frame > t x max first frame treated as unmeasured, inside the flux bound only), rows binding in 12.3-13 mag

| t | rows | frac with binding pixel excluded | median cap_new/cap_old over rows with exclusion | median cap_new/cap_old all | frac cap_new < a_H | frac of rows whose new binding pixel is replaced | median src DN / FW of new binding pixel | median data/model(a_H) of new binding pixel |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 64 | 0.77 | 1.019 | 1.000 | 0.88 | 1.00 | 0.44 | 0.841 |
| 0.8 | 64 | 0.62 | 1.052 | 1.000 | 0.89 | 1.00 | 0.66 | 0.886 |

