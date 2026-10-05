# S_seedtreas: seed-core DQ filter A/B on gc-treasury F480M (#1101 review follow-up)
- Frames: jw10678040001_02101_0000{1..6}_nrcalong_destreak_o040_crf.fits (gc-treasury F480M o040), symlinked with ramps.
- Arms: s0 = SATSTAR_SEED_CORE_DQ=0, s1 = default on. Same extended-emission NIRCam satstar env as R_rcal/run_frame_rc.py.
- Code: jwst-gc-pipeline-wt-seedfix at e4eac2c3 (read-only; do not edit that worktree while these jobs run).
- Jobs: jobids.txt (sb_st.sh). Outputs: tree_<arm>/F480M/pipeline/*_rctest_satstar_*.fits.
- Score: cross_exp.py (per-star position / flux scatter across the 6 dithered exposures; flags-16 rate).
  Dithering moves static bad pixels relative to each star, so a seed pulled by them scatters across exposures.
