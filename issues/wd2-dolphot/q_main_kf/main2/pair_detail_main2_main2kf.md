## main2kf vs main2
### shared values moving by > 0.1 mag (both arms matched to the dolphot star, both finite)
| band | moved | median dolphot mag of moved | main2kf closer | main2 closer | median abs(dm) main2 -> main2kf |
|---|---|---|---|---|---|
| F115W | 127 | 19.1 | 33 | 18 | 0.181 -> 0.082 |
| F150W | 215 | 21.8 | 52 | 41 | 0.610 -> 0.634 |
| F162M | 180 | 21.2 | 54 | 36 | 0.533 -> 0.288 |
| F182M | 57 | 21.5 | 21 | 21 | 0.981 -> 1.718 |
| F200W | 78 | 20.5 | 30 | 14 | 0.208 -> 0.161 |
| F250M | 382 | 15.7 | 123 | 100 | 0.096 -> 0.072 |
| F277W | 515 | 16.3 | 253 | 87 | 0.118 -> 0.041 |
| F300M | 409 | 15.5 | 144 | 80 | 0.109 -> 0.060 |
| F335M | 174 | 20.0 | 5 | 15 | 0.446 -> 0.713 |
| F410M | 138 | 14.4 | 25 | 14 | 0.136 -> 0.060 |
| F164N | 58 | 21.7 | 14 | 11 | 0.303 -> 0.266 |
| F187N | 56 | 17.1 | 6 | 9 | 0.249 -> 0.198 |
| F212N | 51 | 21.3 | 12 | 6 | 0.174 -> 0.181 |
| F323N | 122 | 18.0 | 11 | 15 | 0.347 -> 0.518 |
| F405N | 125 | 17.4 | 11 | 11 | 0.158 -> 0.167 |
| F466N | 119 | 16.9 | 8 | 6 | 0.106 -> 0.123 |

### rows: main2 52319, main2kf 52124; new in main2kf (no main2 row within 0.05") 969; gone 1164
new rows matched to a dolphot star: 28; unmatched: 941, of which spike_artifact 340
| new unmatched rows, not spike-flagged | N |
|---|---|
| finite mag in 1-1 of 10 broad/medium bands | 6 |
| finite mag in 2-3 of 10 broad/medium bands | 43 |
| finite mag in 4-6 of 10 broad/medium bands | 206 |
| finite mag in 7-10 of 10 broad/medium bands | 343 |
| no finite mag in any of the 10 | 3 |
| F200W (Vega) of new unmatched unflagged rows | N |
|---|---|
| 0-21 | 59 |
| 21-23 | 120 |
| 23-25 | 132 |
| 25-40 | 109 |
| no F200W | 181 |
reference: main2 unmatched unflagged rows 20242; finite in >= 4 of 10 bands: 18372 (0.91); main2kf new unmatched unflagged rows finite in >= 4: 549 (0.91)
reference: main2 dolphot-matched rows finite in >= 4 of 10 bands: 15536 of 15622
