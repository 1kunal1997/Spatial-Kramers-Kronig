/* Fused C kernels for the O(N) legs of the DiHiTI passivity pipeline.
 *
 * Baseline (tmm_helper.ht_derivative, and the standard dihiti() used in
 * cycle1/cycle2 benchmarks):
 *
 *   u   = np.gradient(e_re, xx)                      # O(N), 1 pass, allocates
 *   v   = np.imag(scipy.signal.hilbert(u))            # O(N log N) FFT -- left alone
 *   e   = cumulative_trapezoid(v, xx, initial=0)      # O(N), 1 pass, allocates
 *   e  -= np.linspace(e[0], e[-1], N)                 # O(N), 2 more passes+allocs (linspace, subtract)
 *   verdict = e[interior].min()                       # O(N), 1 more pass+alloc (boolean mask + min)
 *
 * numpy pays ~5 separate array-length passes and ~4 temporary allocations
 * for the non-FFT part. These kernels fuse it to 2 passes and 0 temporaries
 * (aside from the caller-supplied output buffer), assuming a UNIFORM grid
 * (dx constant) -- true for every grid used in this project (np.linspace).
 *
 * grad_fast   : replicates np.gradient(u, x) on a uniform grid exactly
 *               (2nd-order central interior, 1st-order one-sided edges --
 *               numpy's default edge_order=1).
 * post_full   : cumtrapz + de-tilt, writing the full e_im array (used only
 *               for correctness verification against the numpy baseline).
 * post_min    : cumtrapz + de-tilt + interior-min, FUSED, no full array
 *               materialized -- this is the fast path for a real verdict.
 */
#include <math.h>
#include <stddef.h>

void grad_fast(const double *u, double dx, long N, double *out) {
    if (N < 2) { if (N == 1) out[0] = 0.0; return; }
    out[0] = (u[1] - u[0]) / dx;
    out[N-1] = (u[N-1] - u[N-2]) / dx;
    double inv2dx = 1.0 / (2.0 * dx);
    for (long i = 1; i < N - 1; i++) {
        out[i] = (u[i+1] - u[i-1]) * inv2dx;
    }
}

/* cumulative_trapezoid(v, x, initial=0) then subtract linspace(e[0],e[-1],N),
 * writing the full de-tilted array to e_im_out. */
void post_full(const double *v, double dx, long N, double *e_im_out) {
    if (N == 0) return;
    double acc = 0.0;
    e_im_out[0] = 0.0;
    double half_dx = 0.5 * dx;
    for (long i = 1; i < N; i++) {
        acc += half_dx * (v[i-1] + v[i]);
        e_im_out[i] = acc;
    }
    double total = e_im_out[N-1];   /* e[0] == 0, so ramp is total*i/(N-1) */
    if (N > 1) {
        double slope = total / (double)(N - 1);
        for (long i = 0; i < N; i++) {
            e_im_out[i] -= slope * (double)i;
        }
    }
}

/* Same computation as post_full, but instead of writing the full array it
 * tracks the running minimum over the interior index window [lo, hi]
 * (inclusive) and returns ONLY that scalar. This is the real fast path:
 * it needs two passes over v (one forward-accumulate for the cumtrapz,
 * one for the de-tilt+min) but allocates nothing beyond two doubles. */
double post_min(const double *v, double dx, long N, long lo, long hi,
                 double *e_last_out /* optional: total pre-detilt sum, may be NULL */) {
    if (N == 0) return 0.0;
    /* Pass 1: forward cumulative trapezoid, kept only as running scalar
     * plus we still need e_im[i] values in pass 2 -- since post-detilt
     * value at i needs e_im[i] (a prefix sum), we cannot avoid storing
     * SOME O(N) intermediate. We reuse e_last_out's caller buffer? No --
     * to keep this truly allocation-free we recompute the prefix sum in
     * pass 2 by re-accumulating, which trades one FMA-loop for one malloc.
     * On modern hardware a fused re-accumulate is cheaper than an
     * allocate+free of a 64KB-8MB buffer, so we do that. */
    double total = 0.0;
    double half_dx = 0.5 * dx;
    for (long i = 1; i < N; i++) {
        total += half_dx * (v[i-1] + v[i]);
    }
    if (e_last_out) *e_last_out = total;
    double slope = (N > 1) ? total / (double)(N - 1) : 0.0;

    double acc = 0.0;
    double minval = INFINITY;
    for (long i = 1; i < N; i++) {
        acc += half_dx * (v[i-1] + v[i]);
        if (i >= lo && i <= hi) {
            double detilted = acc - slope * (double)i;
            if (detilted < minval) minval = detilted;
        }
    }
    /* i=0 case: e_im[0]=0, detilted = 0 - 0 = 0 */
    if (0 >= lo && 0 <= hi && 0.0 < minval) minval = 0.0;
    return minval;
}

/* Fully fused top-to-bottom verdict, EXCLUDING the FFT/Hilbert step which
 * the caller must still do in scipy: takes e_re and the Hilbert-transformed
 * derivative v = imag(hilbert(grad(e_re))), and does grad+post in one call
 * chain to minimize Python round-trips. Kept as two calls (grad_fast then
 * post_min) at the call site since the FFT must happen between them; this
 * function exists only to document that no further fusion is possible
 * without also reimplementing the FFT (explicitly out of scope per the
 * task brief). */
