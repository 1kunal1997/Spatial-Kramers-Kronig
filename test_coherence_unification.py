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
# Figures: 1-to-1 before/after comparison
# ============================================================================
def save_figures(figdir="theory/figures"):
    import os
    from plot_functions import plot_setup, plot, legend
    import colors
    os.makedirs(figdir, exist_ok=True)

    lambda_list = np.linspace(2, 5, 300)

    # ---- Case 1: Thin sKK coating (clearly coherent) ----
    # Parameters: A=10, gam=0.01, nb=2.3, delta=0.1
    # Semi-infinite layers use n_inf=nb to eliminate FP cavity at air/coating boundary
    nb = 2.3
    n_raw, d_raw = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, nb, plot_flag=False)
    n_list = [nb] + n_raw + [nb]
    d_list = [inf] + d_raw + [inf]

    # "BEFORE": old coh_tmm directly (what the old TRA_wavelength did)
    T_old = np.zeros_like(lambda_list)
    R_old = np.zeros_like(lambda_list)
    for j, lamb in enumerate(lambda_list):
        res = tmm.coh_tmm('s', n_list, d_list, 0, lamb)
        T_old[j] = res['T']
        R_old[j] = res['R']
    A_old = 1 - T_old - R_old

    # "AFTER": new unified TRA_wavelength (uses inc_tmm + auto c_list)
    T_new, R_new, A_new = tmm_h.TRA_wavelength(n_list, d_list, lambda_list, pol='s')

    # Figure 1a: T comparison
    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Transmittance',
                         title='Thin sKK Coating: T (A=10, $\\gamma$=0.01, nb=2.3, n$_{inf}$=nb)',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_list, T_old, label='T old (coh\\_tmm)', color=colors.blue, auto_scale=True)
    plot(fig, ax, lambda_list, T_new, '--', label='T new (unified)', color=colors.red, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_thin_coating_T.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_thin_coating_T.png")
    print(f"    max|T_old - T_new| = {np.max(np.abs(T_old - T_new)):.2e}")

    # Figure 1b: R comparison
    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Reflectance',
                         title='Thin sKK Coating: R (A=10, $\\gamma$=0.01, nb=2.3, n$_{inf}$=nb)',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_list, R_old, label='R old (coh\\_tmm)', color=colors.blue, auto_scale=True)
    plot(fig, ax, lambda_list, R_new, '--', label='R new (unified)', color=colors.red, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_thin_coating_R.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_thin_coating_R.png")
    print(f"    max|R_old - R_new| = {np.max(np.abs(R_old - R_new)):.2e}")

    # Figure 1c: A_LR vs A_RL (sKK asymmetry preserved)
    T_old_rev = np.zeros_like(lambda_list)
    R_old_rev = np.zeros_like(lambda_list)
    n_rev, d_rev = n_list[::-1], d_list[::-1]
    for j, lamb in enumerate(lambda_list):
        res = tmm.coh_tmm('s', n_rev, d_rev, 0, lamb)
        T_old_rev[j] = res['T']
        R_old_rev[j] = res['R']
    A_old_rev = 1 - T_old_rev - R_old_rev
    T_new_rev, R_new_rev, A_new_rev = tmm_h.TRA_wavelength(n_rev, d_rev, lambda_list, pol='s')

    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Absorptance',
                         title='Thin sKK: A$_{LR}$ vs A$_{RL}$ (A=10, $\\gamma$=0.01, nb=2.3, n$_{inf}$=nb)',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_list, A_old, label='A$_{LR}$ old', color=colors.blue, auto_scale=True)
    plot(fig, ax, lambda_list, A_new, '--', label='A$_{LR}$ new', color=colors.red, auto_scale=True)
    plot(fig, ax, lambda_list, A_old_rev, label='A$_{RL}$ old', color=colors.green, auto_scale=True)
    plot(fig, ax, lambda_list, A_new_rev, '--', label='A$_{RL}$ new', color=colors.orange, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_thin_coating_A_asymmetry.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_thin_coating_A_asymmetry.png")
    print(f"    max|A_LR_old - A_LR_new| = {np.max(np.abs(A_old - A_new)):.2e}")
    print(f"    max|A_RL_old - A_RL_new| = {np.max(np.abs(A_old_rev - A_new_rev)):.2e}")

    # ---- Case 2: Thick substrate (clearly incoherent) ----
    # 5mm sapphire (n=1.7), no absorption
    n_sub = [1, 1.7, 1]
    d_sub = [inf, 5000, inf]

    # "BEFORE" for thick: old TRA_inc with manual c_list=['i','i','i']
    # which called inc_tmm directly — replicate that
    T_old_sub = np.zeros_like(lambda_list)
    R_old_sub = np.zeros_like(lambda_list)
    for j, lamb in enumerate(lambda_list):
        res = tmm.inc_tmm('s', n_sub, d_sub, ['i','i','i'], 0, lamb)
        T_old_sub[j] = res['T']
        R_old_sub[j] = res['R']
    A_old_sub = 1 - T_old_sub - R_old_sub

    # "AFTER": new unified (should auto-detect incoherent)
    T_new_sub, R_new_sub, A_new_sub = tmm_h.TRA_wavelength(n_sub, d_sub, lambda_list, pol='s')

    # Figure 2: T comparison for thick substrate
    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Transmittance',
                         title='5mm Sapphire: T (old manual inc vs new auto)',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_list, T_old_sub, label='T old (manual c\\_list)', color=colors.blue, auto_scale=True)
    plot(fig, ax, lambda_list, T_new_sub, '--', label='T new (unified)', color=colors.red, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_thick_substrate_T.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_thick_substrate_T.png")
    print(f"    max|T_old - T_new| = {np.max(np.abs(T_old_sub - T_new_sub)):.2e}")

    # ---- Case 3: sKK coating + thick substrate (mixed) ----
    # This is the key demo: forced all-coherent produces FP fringes from substrate,
    # auto-coherence flips substrate to 'i' while keeping coating layers 'c'.
    n_coat, d_coat = tmm_h.generate_n_and_d_v6_symmetry(0.01, 10, 2.3, plot_flag=False)
    n_mix = [1] + n_coat + [1.7, 1]
    d_mix = [inf] + d_coat + [5000, inf]

    # Forced all-coherent (threshold=1e12 so nothing is ever 'i' except semi-inf)
    T_allc, R_allc, A_allc = tmm_h.TRA_wavelength(n_mix, d_mix, lambda_list, pol='s', threshold=1e12)

    # Auto-coherence (threshold=5): substrate should flip to 'i'
    T_auto, R_auto, A_auto = tmm_h.TRA_wavelength(n_mix, d_mix, lambda_list, pol='s', threshold=5)

    # Print c_list to confirm
    c_sample = tmm_h._make_c_list(n_mix, d_mix, 3.0, threshold=5)
    n_c = sum(1 for c in c_sample if c == 'c')
    n_i = sum(1 for c in c_sample if c == 'i')
    print(f"\n  Case 3: sKK coating + 5mm sapphire")
    print(f"    c_list: {n_c} coherent, {n_i} incoherent")
    print(f"    Substrate (second-to-last layer) classified: '{c_sample[-2]}'")

    # Figure 3a: T — forced coherent vs auto
    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Transmittance',
                         title='sKK Coating + 5mm Sapphire: T',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_list, T_allc, label='T forced all-coherent', color=colors.red, auto_scale=True)
    plot(fig, ax, lambda_list, T_auto, label='T auto-coherence', color=colors.blue, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_mixed_coating_substrate_T.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_mixed_coating_substrate_T.png")

    # Figure 3b: R — forced coherent vs auto
    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Reflectance',
                         title='sKK Coating + 5mm Sapphire: R',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_list, R_allc, label='R forced all-coherent', color=colors.red, auto_scale=True)
    plot(fig, ax, lambda_list, R_auto, label='R auto-coherence', color=colors.blue, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_mixed_coating_substrate_R.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_mixed_coating_substrate_R.png")

    # ---- Case 4: Intermediate — coarse sKK with thick high-contrast layers ----
    # A=50, gam=0.1, nb=2.3, delta=0.5 => 10 layers
    # Layers 0-2 and 8-9: thick (d=3-10um), n up to 3.1, n*d/lam=5-12 -> 'i'
    # Layers 3-7: thin sKK peak (d=0.06-0.12um), n up to 5.7 -> 'c'
    # Big n contrast at boundaries (3.1 -> 5.7) means FP fringes should be visible
    nb_int = 2.3
    n_int_raw, d_int_raw = tmm_h.generate_n_and_d_v6_symmetry(0.1, 50, nb_int, delta=0.5, plot_flag=False)
    n_int = [nb_int] + n_int_raw + [nb_int]
    d_int = [inf] + d_int_raw + [inf]

    lambda_int = np.linspace(2, 5, 2000)

    # "BEFORE" / naive: all-coherent via coh_tmm
    T_coh_int = np.zeros_like(lambda_int)
    R_coh_int = np.zeros_like(lambda_int)
    for j, lamb in enumerate(lambda_int):
        res = tmm.coh_tmm('s', n_int, d_int, 0, lamb)
        T_coh_int[j] = res['T']
        R_coh_int[j] = res['R']
    A_coh_int = 1 - T_coh_int - R_coh_int

    # "AFTER": auto-coherence flips thick outer layers to 'i'
    T_auto_int, R_auto_int, A_auto_int = tmm_h.TRA_wavelength(n_int, d_int, lambda_int, pol='s', threshold=5)

    # Report classification
    c_int = tmm_h._make_c_list(n_int, d_int, 2.0, threshold=5)
    n_c_int = sum(1 for c in c_int[1:-1] if c == 'c')
    n_i_int = sum(1 for c in c_int[1:-1] if c == 'i')
    print(f"\n  Case 4: Intermediate coating (A=50, gam=0.1, nb=2.3, delta=0.5)")
    print(f"    {len(n_int_raw)} layers, {n_c_int} coherent, {n_i_int} incoherent at lambda=2um")
    for i, (n, d) in enumerate(zip(n_int_raw, d_int_raw)):
        o = np.real(n) * d / 2.0
        tag = 'i' if o > 5 else 'c'
        print(f"      layer {i}: n={np.real(n):.2f}, d={d:.2f}um, n*d/lam={o:.1f} -> {tag}")

    # Figure 4a: T
    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Transmittance',
                         title='Intermediate: T (A=50, $\\gamma$=0.1, $\\delta$=0.5, n$_{inf}$=nb)',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_int, T_coh_int, label='T old (all coherent)', color=colors.red, auto_scale=True)
    plot(fig, ax, lambda_int, T_auto_int, label='T new (auto-coherence)', color=colors.blue, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_intermediate_T.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_intermediate_T.png")

    # Figure 4b: R
    fig, ax = plot_setup('Wavelength ($\\mu$m)', 'Reflectance',
                         title='Intermediate: R (A=50, $\\gamma$=0.1, $\\delta$=0.5, n$_{inf}$=nb)',
                         xlim=(2, 5), figsize=(6, 4), auto_scale=True)
    plot(fig, ax, lambda_int, R_coh_int, label='R old (all coherent)', color=colors.red, auto_scale=True)
    plot(fig, ax, lambda_int, R_auto_int, label='R new (auto-coherence)', color=colors.blue, auto_scale=True)
    legend(fig, ax, auto_scale=True)
    fig.savefig(os.path.join(figdir, "verify_intermediate_R.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {figdir}/verify_intermediate_R.png")


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
    print("FIGURES")
    print("=" * 70)
    save_figures()

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Test A (inc=coh):      {'PASS' if passed_A else 'FAIL'}")
    print(f"  Phase 3 validation:    {'PASS' if passed_phase3 else 'FAIL'}")
