## main2kfpk vs main2kf
### shared values moving by > 0.1 mag (both arms matched to the dolphot star, both finite)
| band | moved | median dolphot mag of moved | main2kfpk closer | main2kf closer | median abs(dm) main2kf -> main2kfpk |
|---|---|---|---|---|---|
| F115W | 33 | 18.4 | 13 | 10 | 0.265 -> 0.252 |
| F150W | 37 | 17.5 | 16 | 18 | 0.734 -> 0.738 |
| F162M | 21 | 16.7 | 7 | 12 | 0.179 -> 0.427 |
| F182M | 20 | 16.7 | 9 | 9 | 0.129 -> 0.304 |
| F200W | 24 | 17.1 | 9 | 9 | 0.224 -> 0.194 |
| F250M | 117 | 16.3 | 35 | 21 | 0.285 -> 0.123 |
| F277W | 243 | 17.4 | 98 | 46 | 0.261 -> 0.105 |
| F300M | 134 | 16.4 | 35 | 32 | 0.229 -> 0.183 |
| F335M | 177 | 16.4 | 37 | 20 | 0.258 -> 0.128 |
| F410M | 144 | 15.7 | 20 | 22 | 0.120 -> 0.159 |
| F164N | 11 | 18.1 | 5 | 6 | 0.229 -> 0.375 |
| F187N | 9 | 16.2 | 1 | 5 | 0.021 -> 0.544 |
| F212N | 11 | 16.8 | 1 | 7 | 0.026 -> 0.316 |
| F323N | 22 | 14.2 | 1 | 1 | 0.097 -> 0.083 |
| F405N | 18 | 12.8 | 2 | 1 | 0.202 -> 0.061 |
| F466N | 16 | 12.4 | 0 | 1 | 0.041 -> 3.139 |

### rows: main2kf 52124, main2kfpk 52166; new in main2kfpk (no main2kf row within 0.05") 191; gone 151
new rows matched to a dolphot star: 8; unmatched: 183, of which spike_artifact 62
| new unmatched rows, not spike-flagged | N |
|---|---|
| finite mag in 1-1 of 10 broad/medium bands | 3 |
| finite mag in 2-3 of 10 broad/medium bands | 8 |
| finite mag in 4-6 of 10 broad/medium bands | 40 |
| finite mag in 7-10 of 10 broad/medium bands | 70 |
| no finite mag in any of the 10 | 0 |
| F200W (Vega) of new unmatched unflagged rows | N |
|---|---|
| 0-21 | 32 |
| 21-23 | 15 |
| 23-25 | 17 |
| 25-40 | 9 |
| no F200W | 48 |
reference: main2kf unmatched unflagged rows 19857; finite in >= 4 of 10 bands: 18001 (0.91); main2kfpk new unmatched unflagged rows finite in >= 4: 110 (0.91)
reference: main2kf dolphot-matched rows finite in >= 4 of 10 bands: 15531 of 15612
