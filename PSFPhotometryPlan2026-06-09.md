PSF Photometry Plan 2026-06-09

Following Matt Hosek's approach, we'll change the PSF Photometry pipeline.

The old path for iter1 is:
 * Satstar finding/subtraction
 * BasicPSFPhotometry(finder=daofind)
 * De-duplicate
 * Saturated star wing rejection
 * IterativePSFPhotometry(finder=daofind)
 * dedup, satstar reject
(then iter2, iter3, iter4 were done outside the loop)

The new path will be:
 * Satstar finding/subtraction
 * seed catalog = daofind
 * saturated wing rejection
 * BasicPSFPhotometry (=iter1)
 * De-duplicate
 * seed catalog = daofind(residual) + previous catalog
 * Saturated star wing rejection & deduplication
 * BasicPSFPhotometry (=iter2)

Merge individual frames into joint i2d image
 * seed_catalog = daofind(i2d) + previous catalog
 * Saturated star wing rejection & deduplication
 * BasicPSFPhotometry (=iter3)
 * seed_catalog = daofind(residual i2d) + previous catalog
 * Saturated star wing rejection & deduplication
 * BasicPSFPhotometry (=iter4)
 * filter catalog w/qfit rejection [needs investigation: how do we reject fake stars that are definitely extended emission?  probably need a combination of qfit + local noise + local background]
 * background map = smooth residual i2d merged image, subtract reprojected smoothed background map from individual frames
 * seed catalog = daofind(background-subtracted map) + previous catalog
 * saturated star wing rejection & deduplication
 * BasicPSFPhotometry _on background-subtracted image_ (=iter5)
 * recompute background map and compare to previous background map
 * seed catalog = daofind(background map) + previous catalog
 * saturated star wing rejection & deduplication
 * BasicPSFPhotometry _on latest background-subtracted image_ (=iter6)

Then we do the cross-band merging:
 * new seed = cross-filter merge catalog with stringent requirements (good qfit, at least 2 filters with separation <10 mas and s/n > 5 each) + previous catalog
 * saturated star wing rejection & deduplication
 * BasicPSFPhotometry _on latest background-subtracted image_ (=iter7)
