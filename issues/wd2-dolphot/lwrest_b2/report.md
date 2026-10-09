# b2 / c diagnosis (arm tree_main2kfpk)

Reference: dolphot ecsv wd2_nircam_wf_mf_nf. dm = ours - dolphot - ZP; good = |dm|<0.3 within 0.08". ZP 277W 24.1289, 250M 24.3156, 300M 23.9704.
Replication of the merge: replicate_merge.py (reads repo code only). F277W pre-replacement rows 42253, matching the log.

## b2 mechanisms (per-frame rows exist, merged row missing)
| band | R2 | S | P | total |
|---|---|---|---|---|
| F277W | 6 | 1 | 0 | 7 |
| F250M | 15 | 1 | 0 | 16 |
| F300M | 14 | 4 | 1 | 19 |

- R2: replace_saturated second pass (merge_catalogs.py ~4248-4262, radius `_satstar_replace_radius()` 0.5") pairs the good merged row with an unmatched neighbouring satstar (0.15-0.49" away); overwrite at 4417-4419 moves the row to the satstar position and flux. Veto at 3997 triggers only if satflux < 0.8*catflux. Example F277W 5943: 17.8 mag row becomes a 12.59 mag satstar 0.123" away.
- S: union sigma clip (703-707, mad_std per axis, flux|ra|dec) masks all 4 frames, nmatch_good=0, position NaN, row dropped at 1789-1791 (F277W: 49 rejected, matches log). Band-wide all-masked rows 49/68/64; 41/55/50 sit on a dolphot star, 39/55/49 would be good if unclipped.
- P: F300M 3187, merged position 0.099" from dolphot (per-frame scatter 0.058-0.134"), outside 0.08" tolerance. Not a merge fault.

## c mechanisms (merged row exists, |dm|>=0.3)
| band | C2 (frames agree, offset 0.3-0.5) | C2b (frames scattered) | C3 | C6 split |
|---|---|---|---|---|
| F277W (19) | 10 | 8 (5 bright by >1 mag) | 1 | 0 |
| F250M (3) | 2 | 0 | 0 | 1 |
| F300M (3) | 3 | 0 | 0 | 0 |

Merge is faithful for most; fits are off in the frames (blend, wing, resbg). 4901 and 4985 have dolphot neighbours 1.4-1.8 mag brighter at 0.35-0.56". C6 (250M 3648): 1-frame fragment (16.58) at 0.035" beats 3-frame row (15.44, dolphot 15.39) at 0.051". Disjoint-frame fragment pairs <0.1": 86/49/49, only 3/2/0 on dolphot stars.

## Proposals (offline estimates, not full-chain A/B)
1. Second-pass veto: skip replacement when old row mag - satstar mag < 1.5-2.5 (and point-source qfit).
   - dmag<1.5: vetoes 23/19/22 pairs (11/15/16 wrong, 1/0/1 right); b2 fixed 5/6, 9/15, 10/14.
   - dmag<2.5: vetoes 49/66/48 (32/43/32 wrong, 3/7/6 right); b2 fixed 5/6, 13/15, 11/14.
   - qfit rules veto many more right pairs; worse.
2. Sigma clip fallback: when clip masks all frames (or N<=4), use unclipped mean. Rescues 49/68/64 rows, ~143 good stars.
3. P/C: no merge change justified. Flux-combine alternatives (median, max, brightest-half, ivar) move good count by <=~10 per band; max-frame loses ~20 net in F277W. Optional dedup of disjoint-frame fragments (C6) gains <=5.

## Open questions
- Veto can leave the satstar appended next to the retained old row (duplicates).
- Full-chain A/B needed.
- "other" bucket in second-pass study.
- C2 stars: possible dolphot vs our photometry offset (dm +0.3-0.5).

Figure: b2c_cutouts.png (make: fig_cutouts.py). Tables: mech_all.ecsv, c_detail.ecsv, c_blend.ecsv, second_pass_*.ecsv, rule_eval.ecsv, alt_combine_*.ecsv, fragments_*.ecsv.
