"""
Test script for TRA function unification with auto-coherence classification.

Phase 1: Baseline tests (before any changes to tmm_helper.py)
Phase 3: Validation tests (after implementation)
"""
import numpy as np
import tmm
import tmm_helper as tmm_h
from numpy import inf, pi, sin, sqrt

PYTHON_ENV = "/opt/miniconda3/envs/env/bin/python"

# ============================================================================
# Test A: Verify inc_tmm with all 'c' matches coh_tmm
# ============================================================================
def test_A_inc_equals_coh():
    print("=" * 70)
    print("TEST A: inc_tmm (all 'c') vs coh_tmm")
    print("=" * 70)

    n_list_raw, d_list_raw = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, 2.3, plot_flag=False)

    # Add semi-infinite air layers
    n_list = [1] + n_list_raw + [1]
    d_list = [inf] + d_list_raw + [inf]

    # c_list: first/last must be 'i', all interior 'c'
    c_list = ['i'] + ['c'] * len(n_list_raw) + ['i']

    max_diff_T, max_diff_R = 0, 0

    for pol in ['s', 'p']:
        for lamb in [2.0, 3.0, 5.0]:
            for angle in [0, 0.3, 0.7]:
                res_coh = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)
                res_inc = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)
                dT = abs(res_coh['T'] - res_inc['T'])
                dR = abs(res_coh['R'] - res_inc['R'])
                max_diff_T = max(max_diff_T, dT)
                max_diff_R = max(max_diff_R, dR)

    print(f"  Max |T_coh - T_inc|: {max_diff_T:.2e}")
    print(f"  Max |R_coh - R_inc|: {max_diff_R:.2e}")
    passed = max_diff_T < 1e-10 and max_diff_R < 1e-10
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed

# ============================================================================
# Test B: Thin sKK coating — baseline coherent results
# ============================================================================
def test_B_thin_skk_baseline():
    print("\n" + "=" * 70)
    print("TEST B: Thin sKK coating (coherent baseline)")
    print("=" * 70)

    n_list_raw, d_list_raw = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, 2.3, plot_flag=False)
    n_list = [1] + n_list_raw + [1]
    d_list = [inf] + d_list_raw + [inf]
    n_list_rev = n_list[::-1]
    d_list_rev = d_list[::-1]

    lambda_list = np.linspace(2, 5, 100)

    T_LR, R_LR, A_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
    T_RL, R_RL, A_RL = tmm_h.TRA_wavelength(n_list_rev, d_list_rev, lambda_list)

    # sKK asymmetry check
    asym = np.mean(np.abs(A_LR - A_RL))
    print(f"  Num layers: {len(n_list_raw)}")
    print(f"  Max layer thickness: {max(d_list_raw):.4f} um")
    print(f"  Mean |A_LR - A_RL|: {asym:.6f}")
    print(f"  sKK asymmetry present: {asym > 0.01}")

    # Report max optical thickness at shortest wavelength
    lamb_min = lambda_list[0]
    max_opt = max(np.real(n) * d / lamb_min for n, d in zip(n_list_raw, d_list_raw))
    print(f"  Max n*d/lambda (at lambda={lamb_min} um): {max_opt:.4f}")
    print(f"  -> With threshold=5, all layers classified as: {'c' if max_opt < 5 else 'SOME i!'}")

    return T_LR, R_LR, A_LR, T_RL, R_RL, A_RL

# ============================================================================
# Test C: Thick substrate — baseline incoherent results
# ============================================================================
def test_C_thick_substrate_baseline():
    print("\n" + "=" * 70)
    print("TEST C: Thick substrate (incoherent baseline)")
    print("=" * 70)

    n_list = [1, 1.7, 1]  # air / sapphire / air
    d_list = [inf, 5000, inf]  # 5mm substrate
    c_list = ['i', 'i', 'i']

    lambda_list = np.linspace(2, 5, 100)
    T, R, A = tmm_h.TRA_wavelength_inc(n_list, d_list, c_list, lambda_list)

    # Check smoothness (no FP fringes)
    T_diff = np.diff(T)
    smoothness = np.std(T_diff)
    print(f"  T range: [{T.min():.6f}, {T.max():.6f}]")
    print(f"  Smoothness (std of dT): {smoothness:.2e}")

    # What would coherent give? (should have fringes)
    T_coh, R_coh, A_coh = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
    T_coh_diff = np.diff(T_coh)
    smoothness_coh = np.std(T_coh_diff)
    print(f"  Coherent T smoothness: {smoothness_coh:.2e} (should be much larger)")

    # Optical thickness at lambda=2um
    opt_thick = 1.7 * 5000 / 2.0
    print(f"  Optical thickness n*d/lambda: {opt_thick:.1f}")

    return T, R, A

# ============================================================================
# Test D: Coating + substrate (mixed coherent/incoherent baseline)
# ============================================================================
def test_D_mixed_baseline():
    print("\n" + "=" * 70)
    print("TEST D: Coating + substrate (mixed baseline)")
    print("=" * 70)

    n_coat, d_coat = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, 2.3, plot_flag=False)

    # Build full stack: air | coating layers | 5mm sapphire | air
    n_list = [1] + n_coat + [1.7, 1]
    d_list = [inf] + d_coat + [5000, inf]

    # Manual c_list: semi-infinite 'i', coating 'c', substrate 'i', exit 'i'
    c_list = ['i'] + ['c'] * len(n_coat) + ['i', 'i']

    lambda_list = np.linspace(2, 5, 100)
    T, R, A = tmm_h.TRA_wavelength_inc(n_list, d_list, c_list, lambda_list)

    print(f"  Total layers: {len(n_list)}")
    print(f"  c_list: {c_list[:3]}...{c_list[-3:]}")
    print(f"  T mean: {T.mean():.6f}")
    print(f"  R mean: {R.mean():.6f}")

    return T, R, A, c_list

# ============================================================================
# Test E: Threshold sensitivity — does threshold=5 misclassify thin sKK layers?
# ============================================================================
def test_E_threshold_sensitivity():
    print("\n" + "=" * 70)
    print("TEST E: Threshold sensitivity on thin sKK coating")
    print("=" * 70)

    n_list_raw, d_list_raw = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, 2.3, plot_flag=False)
    n_list = [1] + n_list_raw + [1]
    d_list = [inf] + d_list_raw + [inf]

    lambda_list = np.linspace(2, 5, 100)
    lamb_min = lambda_list[0]

    # Show max optical thickness of interior layers
    opt_thicknesses = [np.real(n) * d / lamb_min for n, d in zip(n_list_raw, d_list_raw)]
    max_opt = max(opt_thicknesses)
    print(f"  Max n*d/lambda across all coating layers (lambda={lamb_min}): {max_opt:.4f}")
    print(f"  Threshold that would first misclassify: {max_opt:.4f}")

    # Test different thresholds
    for thresh in [0.01, 0.05, 1, 2, 5, 10, 50]:
        c_list = tmm_h._make_c_list(n_list, d_list, lamb_min, angle=0, threshold=thresh)
        n_incoherent = sum(1 for c in c_list[1:-1] if c == 'i')  # exclude first/last
        T, R, A = tmm_h.TRA_wavelength(n_list, d_list, lambda_list, threshold=thresh)
        T_rev, R_rev, A_rev = tmm_h.TRA_wavelength(n_list[::-1], d_list[::-1], lambda_list, threshold=thresh)
        asym = np.mean(np.abs(A - A_rev))
        print(f"  threshold={thresh:6.2f}: {n_incoherent:3d}/{len(n_list_raw)} layers 'i', "
              f"mean|A_LR-A_RL|={asym:.6f}, T_mean={T.mean():.6f}")

# ============================================================================
# Test F: Intermediate thickness — scaled-up coating
# ============================================================================
def test_F_intermediate():
    print("\n" + "=" * 70)
    print("TEST F: Intermediate thickness (scaled-up gam)")
    print("=" * 70)

    # gam=0.5 makes a ~200um wide profile (vs 0.01 = 4um)
    for gam in [0.01, 0.1, 0.5, 1.0]:
        n_list_raw, d_list_raw = tmm_h.generate_n_and_d_v6_symmetry(gam, 10, 2.3, plot_flag=False)
        n_list = [1] + n_list_raw + [1]
        d_list = [inf] + d_list_raw + [inf]

        total_thick = sum(d_list_raw)
        lamb = 2.0
        opt_thicknesses = [np.real(n) * d / lamb for n, d in zip(n_list_raw, d_list_raw)]
        max_opt = max(opt_thicknesses)
        c_list = tmm_h._make_c_list(n_list, d_list, lamb, angle=0, threshold=5)
        n_inc = sum(1 for c in c_list[1:-1] if c == 'i')

        print(f"  gam={gam}: total_d={total_thick:.1f}um, {len(n_list_raw)} layers, "
              f"max(n*d/lam)={max_opt:.2f}, {n_inc} incoherent at lambda={lamb}")

# ============================================================================
# Phase 3: Validate new unified functions against Phase 1 baselines
# ============================================================================
def test_phase3_validation():
    print("\n" + "=" * 70)
    print("PHASE 3: VALIDATION (unified functions)")
    print("=" * 70)

    # --- Test B': Thin sKK with unified function ---
    print("\n  B': Thin sKK coating (unified, should match Phase 1 baseline)")
    n_list_raw, d_list_raw = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, 2.3, plot_flag=False)
    n_list = [1] + n_list_raw + [1]
    d_list = [inf] + d_list_raw + [inf]
    lambda_list = np.linspace(2, 5, 100)

    # The new TRA_wavelength uses inc_tmm internally with auto c_list
    T_LR, R_LR, A_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
    T_RL, R_RL, A_RL = tmm_h.TRA_wavelength(n_list[::-1], d_list[::-1], lambda_list)

    asym = np.mean(np.abs(A_LR - A_RL))
    c_list_sample = tmm_h._make_c_list(n_list, d_list, 2.0)
    n_inc = sum(1 for c in c_list_sample[1:-1] if c == 'i')
    print(f"    Auto c_list at lambda=2: {n_inc} incoherent layers (should be 0)")
    print(f"    sKK asymmetry: {asym:.6f} (should match baseline ~0.057)")
    passed_B = n_inc == 0 and asym > 0.01
    print(f"    RESULT: {'PASS' if passed_B else 'FAIL'}")

    # --- Test C': Thick substrate with unified function ---
    print("\n  C': Thick substrate (unified, should auto-detect incoherent)")
    n_list_sub = [1, 1.7, 1]
    d_list_sub = [inf, 5000, inf]
    T_sub, R_sub, A_sub = tmm_h.TRA_wavelength(n_list_sub, d_list_sub, lambda_list)

    c_list_sub = tmm_h._make_c_list(n_list_sub, d_list_sub, 2.0)
    smoothness = np.std(np.diff(T_sub))
    print(f"    Auto c_list: {c_list_sub}")
    print(f"    Smoothness (std of dT): {smoothness:.2e} (should be ~0)")
    passed_C = c_list_sub == ['i', 'i', 'i'] and smoothness < 1e-10
    print(f"    RESULT: {'PASS' if passed_C else 'FAIL'}")

    # --- Test D': Mixed coating + substrate with unified function ---
    print("\n  D': Coating + substrate (unified, should match manual c_list)")
    n_coat, d_coat = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, 2.3, plot_flag=False)
    n_list_mix = [1] + n_coat + [1.7, 1]
    d_list_mix = [inf] + d_coat + [5000, inf]

    T_mix, R_mix, A_mix = tmm_h.TRA_wavelength(n_list_mix, d_list_mix, lambda_list)

    c_list_auto = tmm_h._make_c_list(n_list_mix, d_list_mix, 3.0)
    n_coat_inc = sum(1 for c in c_list_auto[1:1+len(n_coat)] if c == 'i')
    substrate_class = c_list_auto[-2]  # second to last (substrate)
    print(f"    Coating layers classified 'i': {n_coat_inc} (should be 0)")
    print(f"    Substrate classified: '{substrate_class}' (should be 'i')")
    print(f"    T mean: {T_mix.mean():.6f}")
    passed_D = n_coat_inc == 0 and substrate_class == 'i'
    print(f"    RESULT: {'PASS' if passed_D else 'FAIL'}")

    return passed_B and passed_C and passed_D


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    print("PHASE 3: POST-IMPLEMENTATION VALIDATION")
    print("=" * 70)

    passed_A = test_A_inc_equals_coh()
    test_B_thin_skk_baseline()
    test_C_thick_substrate_baseline()
    test_D_mixed_baseline()
    test_E_threshold_sensitivity()
    test_F_intermediate()
    passed_phase3 = test_phase3_validation()

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Test A (inc=coh):      {'PASS' if passed_A else 'FAIL'}")
    print(f"  Phase 3 validation:    {'PASS' if passed_phase3 else 'FAIL'}")
