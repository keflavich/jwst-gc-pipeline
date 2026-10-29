/*
 * PCAT likelihood / model kernel.
 *
 * Vendored from PCAT-DE (Richard Feder, MIT License, see LICENSE_PCAT-DE.txt),
 * file pcat-de/pcat-lion-openblas.c at commit 033aa43
 * (https://github.com/RichardFeder/pcat-de), which descends from the Lion
 * kernel of Portillo et al. (2017) and Daylan et al.
 *
 * Changes relative to upstream (also listed in
 * outputs/pcat_cutout/runs/pcat/setup/PATCHES.md):
 *   P1  include <cblas.h> (or <mkl_cblas.h> with -DPCAT_USE_MKL, or a local
 *       LP64 prototype with -DPCAT_DECLARE_CBLAS); upstream commented the
 *       include out and called cblas_sgemm undeclared.
 *   P2  drop #include "i_malloc.h" (Intel-only header, unused).
 *   P3  the phonion hash table is heap-allocated (malloc/free); upstream used a
 *       stack VLA int hash[NY*NX] that overflows the stack for large images.
 *   P4  -DPCAT_NO_CBLAS builds a BLAS-free variant with the plain triple loop of
 *       upstream blas.c (clib_eval_modl), accumulated in float.
 *   P5  max/min macros renamed pcat_max/pcat_min to avoid clashes with system
 *       headers; logic unchanged.
 * The function signatures and the arithmetic are unchanged.
 */
#include <stdlib.h>
#include <stdbool.h>
#ifndef PCAT_NO_CBLAS
#if defined(PCAT_USE_MKL)
#include <mkl_cblas.h>
#elif defined(PCAT_DECLARE_CBLAS)
/* P1b: minimal LP64 CBLAS prototype (the conda env's cblas.h is the ILP64
 * openblas64_ header while libcblas.so is the LP64 OpenBLAS build) */
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                 const float alpha, const float *A, const int lda, const float *B,
                 const int ldb, const float beta, float *C, const int ldc);
#else
#include <cblas.h>
#endif
#endif

#define pcat_max(a,b) \
    ({ __typeof__ (a) _a = (a);    \
       __typeof__ (b) _b = (b);    \
       _a > _b ? _a : _b; })
#define pcat_min(a,b) \
    ({ __typeof__ (a) _a = (a);    \
       __typeof__ (b) _b = (b);    \
       _a < _b ? _a : _b; })

void pcat_imag_acpt(int NX, int NY, float* image, float* image_acpt, int* reg_acpt, int regsize, int margin, int offsetx, int offsety) {
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++) {
        y0 = pcat_max(j*regsize-offsety-margin, 0);
        y1 = pcat_min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++) {
                x0 = pcat_max(i*regsize-offsetx-margin, 0);
                x1 = pcat_min((i+1)*regsize-offsetx+margin, NX);
                if (reg_acpt[j*NREGX+i] > 0) {
                    for (jj=y0 ; jj<y1; jj++)
                     for (ii=x0 ; ii<x1; ii++)
                        image_acpt[jj*NX+ii] = image[jj*NX+ii];
                }
        }
    }
}

void pcat_like_eval(int NX, int NY, float* image, float* ref, float* weight, double* diff2, int regsize, int margin, int offsetx, int offsety) {
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++) {
        y0 = pcat_max(j*regsize-offsety-margin, 0);
        y1 = pcat_min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++) {
                x0 = pcat_max(i*regsize-offsetx-margin, 0);
                x1 = pcat_min((i+1)*regsize-offsetx+margin, NX);
                diff2[j*NREGX+i] = 0.;
                for (jj=y0 ; jj<y1; jj++)
                 for (ii=x0 ; ii<x1; ii++)
                    diff2[j*NREGX+i] += (image[jj*NX+ii]-ref[jj*NX+ii])*(image[jj*NX+ii]-ref[jj*NX+ii]) * weight[jj*NX+ii];
        }
    }
}

void pcat_model_eval(int NX, int NY, int nstar, int nc, int k, float* A, float* B, float* C, int* x,
	int* y, float* image, float* ref, float* weight, double* diff2, int regsize, int margin,
	int offsetx, int offsety)
{
    int      i,imax,j,j2,jmax,rad,istar,xx,yy,i2;
    int n = nc*nc;
    rad = nc/2;

    // overwrite and shorten A matrix
    // save time if there are many sources per pixel
    int *hash = (int*) malloc(sizeof(int) * (size_t)NY * (size_t)NX);   /* P3 */
    for (i=0; i<NY*NX; i++) { hash[i] = -1; }
    int jstar = 0;
    for (istar = 0; istar < nstar; istar++)
    {
        xx = x[istar];
        yy = y[istar];
        int idx = yy*NX+xx;
        if (hash[idx] != -1) {
            for (i=0; i<k; i++) { A[hash[idx]*k+i] += A[istar*k+i]; }
        }
        else {
            hash[idx] = jstar;
            for (i=0; i<k; i++) { A[jstar*k+i] = A[istar*k+i]; }
            x[jstar] = x[istar];
            y[jstar] = y[istar];
            jstar++;
        }
    }
    free(hash);
    nstar = jstar;

    //  matrix multiplication
#ifndef PCAT_NO_CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        nstar, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    {   /* P4: upstream blas.c triple loop */
        int p, c;
        for (p = 0; p < nstar; p++) {
            float *crow = C + (size_t)p*n;
            for (i = 0; i < n; i++) crow[i] = 0.f;
            for (c = 0; c < k; c++) {
                float a = A[p*k+c];
                const float *brow = B + (size_t)c*n;
                for (i = 0; i < n; i++) crow[i] += a * brow[i];
            }
        }
    }
#endif

    //  loop over stars, insert psfs into image
    for (istar = 0 ; istar < nstar ; istar++)
    {
	xx = x[istar];
	yy = y[istar];
	imax = pcat_min(xx+rad,NX-1);
	jmax = pcat_min(yy+rad,NY-1);
	for (j = pcat_max(yy-rad,0), j2 = (istar*nc+j-yy+rad)*nc ; j <= jmax ; j++, j2+=nc)
	    for (i = pcat_max(xx-rad,0), i2 = i-xx+rad ; i <= imax ; i++, i2++)
		image[j*NX+i] += C[i2+j2];
    }

    pcat_like_eval(NX, NY, image, ref, weight, diff2, regsize, margin, offsetx, offsety);
}
