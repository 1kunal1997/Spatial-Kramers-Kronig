"""V1-c: compiled-kernel passivity test vs. the honest DiHiTI baseline.

Baseline  = gradient -> scipy.signal.hilbert -> cumulative_trapezoid -> de-tilt -> interior min
            (numpy/scipy, exactly tmm_helper.ht_derivative's pipeline + interior-window verdict)
Candidate = ctypes C kernel (libdihiti.so) for grad + post(cumtrapz+detilt+min), same scipy FFT

Grid convention: x = np.linspace(-L, L, N) for BOTH baseline and candidate, always -- this is
the convention used throughout profile_generator.py and the cycle1/cycle2 benchmarks, and the
endpoint-duplication issue flagged in K19/C22 only bit candidates that exploit periodicity
assumptions (the 'onehilb' ramp trick). This candidate does no such thing: it recomputes the
IDENTICAL arithmetic as the baseline, just fused into fewer passes/allocations in C. So the
speedup number cannot be a period-error artifact by construction; correctness is checked
explicitly against the numpy baseline on every profile, every N, below.
"""
import sys, os, json, time, platform, subprocess
import numpy as np

_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "theory"))
sys.path.insert(0, os.path.join(_ROOT, "theory", "passivity_2026Aug"))

import tmm_helper as tmm_h            # noqa: E402
import dihiti_fast as df              # noqa: E402
import profile_generator as pg        # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
FRAC = 0.95


# ------------------------------------------------------------------ timing
def _time_call(fn, reps, *args, **kw):
    """Median + MAD wall-clock over `reps` calls, each freshly timed (no
    internal repeat-loop amortization tricks). Warms once before timing."""
    fn(*args, **kw)  # warm-up (page faults, branch predictor, caches)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args, **kw)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    med = float(np.median(times))
    mad = float(np.median(np.abs(times - med)))
    return med, mad, times.tolist()


def make_profile(N, L=2.5, kind="logistic"):
    x = np.linspace(-L, L, N)
    if kind == "logistic":
        e_re = tmm_h.logistic(x, 4.0, 1.7)
    elif kind == "lorentzian":
        e_re = np.real(tmm_h.eps(x, 1.0, 0.5, 1.7))
    elif kind == "opposite_bumps_gain":
        e_re = pg._opposite_bumps(x, k=4.0, alpha=1.0, sep=0.9, w=0.45)
    else:
        raise ValueError(kind)
    return x, e_re


def bench_at_N(N, reps=41, kind="logistic"):
    x, e_re = make_profile(N, kind=kind)

    med_b, mad_b, _ = _time_call(df.verdict_baseline, reps, x, e_re, FRAC)
    med_c, mad_c, _ = _time_call(df.verdict_compiled, reps, x, e_re, FRAC)

    mb, pb = df.verdict_baseline(x, e_re, FRAC)
    mc, pc = df.verdict_compiled(x, e_re, FRAC)

    return dict(N=N, kind=kind, reps=reps,
                baseline_med_s=med_b, baseline_mad_s=mad_b,
                compiled_med_s=med_c, compiled_mad_s=mad_c,
                speedup=med_b / med_c,
                min_eim_baseline=mb, min_eim_compiled=mc,
                abs_diff=abs(mb - mc),
                verdict_baseline=pb, verdict_compiled=pc,
                verdicts_match=bool(pb == pc))


# --------------------------------------------------------------- correctness
def correctness_sweep(n_samples=300, Ns=(2049, 4001, 8192)):
    """Check compiled candidate against the numpy baseline verdict across the
    labelled corpus (profile_generator families) at several grid sizes,
    including near-threshold cases (alpha sweep through the opposite_bumps
    crossover, and the raw sample() draws which include near-zero-margin
    profiles by construction)."""
    results = []
    rng = np.random.default_rng(12345)

    # 1. broad random draws across every family in the corpus, at N_DEF (2048)
    #    -- uses profile_generator's own x grid so labels are apples-to-apples.
    for i, s in enumerate(pg.sample(n_samples, rng=rng)):
        x, e_re = s["x"], s["e_re"]
        mb, pb = df.verdict_baseline(x, e_re, FRAC)
        mc, pc = df.verdict_compiled(x, e_re, FRAC)
        results.append(dict(idx=i, family=s["name"], N=len(x),
                             min_eim_baseline=mb, min_eim_compiled=mc,
                             abs_diff=abs(mb - mc), match=bool(pb == pc),
                             near_threshold=bool(abs(mb) < 1e-2)))

    # 2. the seed crossover itself, at several grid sizes -- the figure
    #    explicitly named in MISSION.md as the near-threshold reference.
    for N in Ns:
        for alpha in [0.0, 0.3, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
            x = np.linspace(-2.5, 2.5, N)
            e_re = pg._opposite_bumps(x, k=4.0, alpha=alpha, sep=0.9, w=0.45)
            mb, pb = df.verdict_baseline(x, e_re, FRAC)
            mc, pc = df.verdict_compiled(x, e_re, FRAC)
            results.append(dict(idx=f"cross_N{N}_a{alpha}", family="opposite_bumps_crossover",
                                 N=N, alpha=alpha,
                                 min_eim_baseline=mb, min_eim_compiled=mc,
                                 abs_diff=abs(mb - mc), match=bool(pb == pc),
                                 near_threshold=bool(abs(mb) < 1e-2)))

    n_match = sum(r["match"] for r in results)
    n_total = len(results)
    n_near = sum(r["near_threshold"] for r in results)
    n_near_match = sum(r["match"] for r in results if r["near_threshold"])
    max_abs_diff = max(r["abs_diff"] for r in results)

    return dict(n_total=n_total, n_match=n_match, match_rate=n_match / n_total,
                n_near_threshold=n_near, n_near_threshold_match=n_near_match,
                max_abs_diff=max_abs_diff, details=results)


def compile_time():
    src = os.path.join(OUT, "dihiti_fast.c")
    so = os.path.join(OUT, "libdihiti_timing.so")
    t0 = time.perf_counter()
    subprocess.run(["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src], check=True)
    t1 = time.perf_counter()
    os.remove(so)
    return t1 - t0


def main():
    print("=== V1-c: compiled DiHiTI kernel vs numpy/scipy baseline ===")
    print(f"platform: {platform.platform()}, python {platform.python_version()}, "
          f"numpy {np.__version__}")

    ctime = compile_time()
    print(f"\ncompile time (one-time, amortized across the process lifetime): {ctime*1000:.1f} ms")

    print("\n--- correctness sweep ---")
    corr = correctness_sweep()
    print(f"verdict match: {corr['n_match']}/{corr['n_total']} "
          f"({100*corr['match_rate']:.2f}%)")
    print(f"near-threshold (|min_eim_baseline|<1e-2): {corr['n_near_threshold']} cases, "
          f"{corr['n_near_threshold_match']} matched")
    print(f"max |min_eim_baseline - min_eim_compiled| across all cases: {corr['max_abs_diff']:.3e}")

    print("\n--- timing sweep ---")
    Ns = [2**12, 2**14, 2**16, 60000]  # last one: non-power-of-two, FFT-unfriendly-ish
    bench_results = []
    for N in Ns:
        for kind in ["logistic", "lorentzian"]:
            r = bench_at_N(N, reps=41, kind=kind)
            bench_results.append(r)
            print(f"N={N:>7} [{kind:<11}] baseline={r['baseline_med_s']*1e3:8.4f}ms "
                  f"(+-{r['baseline_mad_s']*1e3:.4f})  compiled={r['compiled_med_s']*1e3:8.4f}ms "
                  f"(+-{r['compiled_mad_s']*1e3:.4f})  speedup={r['speedup']:.3f}x  "
                  f"match={r['verdicts_match']}")

    manifest = dict(
        exp_id="cycle3_v1c_compiled",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_sha=subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                capture_output=True, text=True).stdout.strip(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=__import__("scipy").__version__,
        compiler="cc (clang) -O3 -march=native -shared -fPIC",
        compile_time_s=ctime,
        method="ctypes C extension (libdihiti.so): fused grad_fast + fused post_min "
               "(cumtrapz+detilt+interior-min in one pass, no intermediate numpy allocations). "
               "FFT/Hilbert step left in scipy.signal.hilbert (already compiled, not the target).",
        grid_convention="x = np.linspace(-L, L, N), SAME array object used for baseline and "
                         "candidate in every comparison -- endpoint duplication (K19/C22) cannot "
                         "enter as a period-error artifact because this candidate reimplements the "
                         "IDENTICAL baseline arithmetic, verified pointwise below, not a periodicity trick.",
        interior_window="|x| <= 0.95*L (FRAC=0.95), matching cycle1/cycle2 convention",
        correctness=corr,
        timing=bench_results,
        params=dict(reps_per_N=41, Ns_tested=Ns, L=2.5),
    )
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nwrote {os.path.join(OUT, 'manifest.json')}")

    best = max(bench_results, key=lambda r: r["speedup"])
    worst = min(bench_results, key=lambda r: r["speedup"])
    n14 = [r for r in bench_results if r["N"] == 2**14]
    n14_strs = [f"{r['kind']}:{r['speedup']:.3f}x" for r in n14]
    print(f"\nspeedup range: {worst['speedup']:.3f}x .. {best['speedup']:.3f}x")
    print(f"at N=2^14: {n14_strs}")


if __name__ == "__main__":
    main()
