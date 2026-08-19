"""Diagnostic: decompose baseline wall-clock into its four stages (gradient,
FFT/hilbert, cumtrapz, detilt+mask+min) to explain WHY the end-to-end
compiled speedup plateaus well under 3x even though the O(N) legs alone
see a much larger win. This is an Amdahl's-law check, not a new candidate.
"""
import sys, os, time, json
import numpy as np

_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
import tmm_helper as tmm_h
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import dihiti_fast as df

OUT = os.path.dirname(os.path.abspath(__file__))


def timeit_ms(fn, reps=101):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
        ts.append(t1 - t0)
    return float(np.median(ts) * 1e3)


def diagnose(N, L=2.5, reps=101):
    x = np.linspace(-L, L, N)
    e_re = tmm_h.logistic(x, 4.0, 1.7)
    dx = x[1] - x[0]
    lo, hi = df._interior_bounds(N, 0.95)

    t_grad_np = timeit_ms(lambda: np.gradient(e_re, x), reps)
    u = np.gradient(e_re, x)
    t_hilbert = timeit_ms(lambda: hilbert(u), reps)
    v = np.imag(hilbert(u))
    t_cumtrapz = timeit_ms(lambda: cumulative_trapezoid(v, x, initial=0.0), reps)
    e = cumulative_trapezoid(v, x, initial=0.0)
    t_detilt_min = timeit_ms(
        lambda: (e - np.linspace(e[0], e[-1], N))[np.abs(x) <= 0.95 * L].min(), reps)

    t_grad_c = timeit_ms(lambda: df.grad_fast(e_re, dx), reps)
    t_post_c = timeit_ms(lambda: df.post_min(v, dx, lo, hi), reps)

    numpy_ON = t_grad_np + t_cumtrapz + t_detilt_min
    numpy_total = numpy_ON + t_hilbert
    c_ON = t_grad_c + t_post_c
    c_total = c_ON + t_hilbert  # FFT untouched, shared by both

    amdahl_ceiling = numpy_total / t_hilbert  # speedup if O(N) legs were FREE (t->0)

    return dict(N=N, reps=reps,
                t_grad_np_ms=t_grad_np, t_hilbert_ms=t_hilbert,
                t_cumtrapz_ms=t_cumtrapz, t_detilt_min_ms=t_detilt_min,
                t_grad_c_ms=t_grad_c, t_post_c_ms=t_post_c,
                numpy_ON_legs_ms=numpy_ON, numpy_total_ms=numpy_total,
                c_ON_legs_ms=c_ON, c_total_ms=c_total,
                ON_legs_speedup=numpy_ON / c_ON,
                end_to_end_speedup=numpy_total / c_total,
                fft_fraction_of_baseline=t_hilbert / numpy_total,
                amdahl_ceiling_speedup=amdahl_ceiling)


if __name__ == "__main__":
    results = [diagnose(N) for N in (2**12, 2**14, 2**16, 60000)]
    for r in results:
        print(f"N={r['N']:>7}  numpy_total={r['numpy_total_ms']:.4f}ms  "
              f"(grad={r['t_grad_np_ms']:.4f} fft={r['t_hilbert_ms']:.4f} "
              f"cumtrapz={r['t_cumtrapz_ms']:.4f} detilt/min={r['t_detilt_min_ms']:.4f})  "
              f"c_total={r['c_total_ms']:.4f}ms  "
              f"ON-legs-speedup={r['ON_legs_speedup']:.3f}x  "
              f"end-to-end={r['end_to_end_speedup']:.3f}x  "
              f"FFT%={100*r['fft_fraction_of_baseline']:.1f}  "
              f"Amdahl-ceiling(FFT free O(N))={r['amdahl_ceiling_speedup']:.3f}x")
    with open(os.path.join(OUT, "amdahl.json"), "w") as f:
        json.dump(results, f, indent=2)
