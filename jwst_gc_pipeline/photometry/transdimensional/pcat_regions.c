/*
 * Region bookkeeping helpers for the jwst_gc_pipeline PCAT sampler
 * (jwst_gc_pipeline.photometry.transdimensional.pcat_sampler).
 *
 * Every image pixel carries an integer label: the index of the active
 * (same-parity) region whose margin-extended box contains the pixel, or -1.
 * The extended boxes of same-parity regions are disjoint when 2*margin <=
 * regsize, so a pixel has at most one label.
 */
#include <stddef.h>

/* out[l] += sum over pixels with label l of w*(dm^2 - 2*resid*dm)
 * (the chi^2 change of region l when its model changes by dm).
 * out[nlab] collects sum |dm|*w over pixels with label -1 and w > 0 (leak). */
void pcat_dchi2_regions(int npix, const float* dm, const float* resid,
                        const float* w, const int* label, int nlab, double* out)
{
    int p, l;
    for (l = 0; l <= nlab; l++) out[l] = 0.;
    for (p = 0; p < npix; p++) {
        float d = dm[p];
        if (d == 0.f) continue;
        l = label[p];
        if (l >= 0) out[l] += (double)w[p] * ((double)d * d - 2.0 * (double)resid[p] * d);
        else if (w[p] > 0.f) out[nlab] += (double)w[p] * (d > 0 ? d : -d);
    }
}

/* resid -= dm and model += dm for pixels whose label is accepted. */
void pcat_apply_accept(int npix, const float* dm, const int* label,
                       const int* accept, float* resid, float* model)
{
    int p, l;
    for (p = 0; p < npix; p++) {
        float d = dm[p];
        if (d == 0.f) continue;
        l = label[p];
        if (l >= 0 && accept[l]) {
            resid[p] -= d;
            model[p] += d;
        }
    }
}

/* Full-image chi^2 change for a global (background) proposal. */
double pcat_dchi2_global(int npix, const float* dm, const float* resid, const float* w)
{
    int p;
    double s = 0.;
    for (p = 0; p < npix; p++) {
        double d = dm[p];
        s += (double)w[p] * (d * d - 2.0 * (double)resid[p] * d);
    }
    return s;
}
