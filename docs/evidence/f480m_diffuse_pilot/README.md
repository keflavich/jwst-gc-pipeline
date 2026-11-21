# Treasury F480M diffuse-emission pilot evidence

This records the 2026-11-20 pilot referenced by [the release proposal](../../reports/F480M_DIFFUSE_EMISSION_RELEASE_PROPOSAL.md). The injection and held-out overlap numbers here are new measurements. [The earlier rerun](https://github.com/keflavich/jwst-gc-pipeline/pull/1271) supplied the comparison mosaics and reports a different overlap statistic.

## Lineage audit

[lineage.json](lineage.json) records the selected residual tiles and the CRF names from their per-frame ledgers. The on-disk survey recipe used `EXCLUDE = ('o138', 'o139')`. It selected 66 tiles and 784 ledger records. The two excluded tiles had 12 records each, which accounts for the difference from the complete 68-tile, 808-record inventory checked in review. [Review with full-inventory check](https://github.com/keflavich/jwst-gc-pipeline/pull/1357#pullrequestreview-5998142645)

## Paired injections

[measure.py](measure.py) reproduces the transfer measurement. The per-observation outputs list the six NRCBLONG calibrated exposures and corresponding aligned CRFs used for their GWCS and masks, with coverage, the maximum interpolation check error, and all template metrics. [o065](o065.json) · [o072](o072.json) · [o079](o079.json) · [Across-observation summaries](injection_summary.json)

The templates have a peak amplitude of 0.6 MJy/sr and are defined in a common tangent plane. For each filter window the script computes the difference between the two filtered realizations and projects it using the fixed WCS and mask. The no-filter injection, projected on the same grid, is the unit-response reference. The pilot uses equal exposure weights, does no stellar fitting, and scores the shared area covered by at least three exposures. The numerical values in the proposal are medians across the three observation files.

To rerun on HiPerGator from the repository root, set `PYTHONPATH` to a checkout of the pipeline at [fdcab712](https://github.com/keflavich/jwst-gc-pipeline/commit/fdcab712dd78bf7895d0fc8643548737841be8eb). This pins the row estimator and GWCS wrapper used in the pilot. The original run used NumPy 2.4.6, SciPy 1.17.0, and Astropy 7.2.0. The FITS inputs are the archived `woff/gc-treasury/F480M/pipeline` files under the original pilot directory. Set `F480M_PILOT_ROOT` if that directory is staged elsewhere.

```bash
PYTHONPATH=/path/to/pipeline-at-fdcab712 python docs/evidence/f480m_diffuse_pilot/measure.py /tmp/f480m-injection --skip-audit
```

## Held-out overlap edges

[overlap_input.json](overlap_input.json) captures the block medians and block counts from the three archived `survey_i2d_*.json` tables used in the fit. The source tables were produced using bilinear, NaN-propagating reprojection and 64 × 64 pixel blocks on archived image mosaics before stellar subtraction. The analysis uses the 129 edges with at least five valid blocks in all three variants. It fits each tested edge from the other edges and scores the 122 whose endpoints remain connected. [Original i2d overlap method](https://github.com/keflavich/jwst-gc-pipeline/pull/1271)

- `prod` denotes the archived production mosaics with destreak.
- `woff` denotes the scratch image mosaics with destreak off.
- `wjoint` denotes the scratch mosaics made with destreak off after subtracting a joint detector-row model. That prototype solved row offsets per amplifier and exposure from differences of overlapping same-module frames, with a detector-noise spectral prior.

Run [overlap.py](overlap.py) with NumPy and SciPy to regenerate [the per-edge predictions and scores](overlap.json). This replay uses the checked-in overlap measurements and has no FITS input dependency.

```bash
python docs/evidence/f480m_diffuse_pilot/overlap.py /tmp/f480m-overlap.json
```

*Prepared by GPT-6.2 Sol (Codex).*
