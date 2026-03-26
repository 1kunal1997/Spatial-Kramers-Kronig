"""
Diagnostic script: Investigate numerical stability of sKK reflectance
for thin coatings in the thickness design space.

Question: Is the nonzero R_back for sub-wavelength sKK coatings physically
real or a numerical artifact of discretization / TMM?

Tests:
  1. Inspect discretized layer stacks for thin coatings
  2. Compare TMM result vs analytical Fresnel for single uniform thin film
  3. Test convergence with finer discretization (smaller delta)
  4. Check whether thin-coating R matches uniform-film Fresnel expectation
"""

import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import tmm

# ============================================================================
# Replicate core functions from skk_analysis_consolidated.py
# ============================================================================

def logistic_eps(x, k, nb, sx=1):
    return (nb**2 - 1) / (1 + np.exp(sx * k * x)) + 1

def ht_derivative(xx, e_re, pad_factor=8):
    N = len(e_re)
    u = np.gradient(e_re, xx)
    pad = pad_factor * N
    u_pad = np.pad(u, (pad, pad), mode='constant', constant_values=0)
    v_pad = np.imag(hilbert(u_pad))
    v = v_pad[pad:pad+N]
    e_im = cumulative_trapezoid(v, xx, initial=0)
    e_im -= np.linspace(e_im[0], e_im[-1], N)
    return e_im

def discretize_profile(xx, ee, delta=0.05):
    e_scale = np.max(np.abs(ee - ee[0]))
    if e_scale < 1e-12:
        e_scale = 1.0
    x_scale = xx[-1] - xx[0]
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        if ds > delta:
            xq.append(xx[k])
            eq.append(ee[k])
    xq.append(xx[-1])
    eq.append(ee[-1])
    xq, eq = np.array(xq), np.array(eq)
    d_list = np.diff(xq).tolist()
    e_list = (eq[:-1] + eq[1:]) / 2
    n_list = np.sqrt(e_list).tolist()
    return n_list, d_list

def load_sapphire_data():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    for ri_path in [
        os.path.join(base, 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat'),
        os.path.join(base, 'RI', 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat'),
    ]:
        if os.path.exists(ri_path):
            data = np.genfromtxt(ri_path)
            return data[50:351, 0], data[50:351, 2], data[50:351, 3]
    raise FileNotFoundError("Sapphire data not found")

def Rback_single_wavelength(n_coating, d_coating, n_sub, angle_deg, lam, pol):
    """Backside reflection at a single wavelength using inc_tmm."""
    angle = angle_deg * np.pi / 180
    n_t = list(n_coating)
    d_t = list(d_coating)
    c_t = ['c'] * len(n_coating)
    # Insert substrate (incoherent, 5000 μm thick)
    n_t.insert(0, n_sub)
    d_t.insert(0, 5000)
    c_t.insert(0, 'i')
    # Ambient on both sides
    n_t.insert(0, 1)
    n_t.append(1)
    d_t.insert(0, np.inf)
    d_t.append(np.inf)
    c_t.insert(0, 'i')
    c_t.append('i')

    th_f = tmm.snell(1, n_sub, angle)
    Rf = tmm.interface_R(pol, 1, n_sub, angle, th_f)
    res = tmm.inc_tmm(pol, n_t, d_t, c_t, angle, lam)
    Rb = res['R'] - Rf
    return Rb

def fresnel_thin_film_R(n_film, d_film, n_sub, angle_deg, lam, pol):
    """Exact coherent TMM for air | uniform thin film | substrate | air
    with incoherent substrate, returning R_back only.
    This is the analytical reference: what would a uniform film of given n, d do?
    """
    angle = angle_deg * np.pi / 180
    n_t = [1, n_sub, n_film, 1]
    d_t = [np.inf, 5000, d_film, np.inf]
    c_t = ['i', 'i', 'c', 'i']

    th_f = tmm.snell(1, n_sub, angle)
    Rf = tmm.interface_R(pol, 1, n_sub, angle, th_f)
    res = tmm.inc_tmm(pol, n_t, d_t, c_t, angle, lam)
    Rb = res['R'] - Rf
    return Rb

# ============================================================================
# Parameters
# ============================================================================
nb = 1.7
delta = 0.01
angle_test = 80
pol_test = 's'
lam_test = 3.0  # single wavelength for diagnostics

lamdata, ndata, kdata = load_sapphire_data()
idx_lam = np.argmin(np.abs(lamdata - lam_test))
n_sub = complex(ndata[idx_lam], kdata[idx_lam])

k_values = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100])

# ============================================================================
# TEST 1: Inspect layer stacks for each k_steep
# ============================================================================
print("=" * 80)
print("TEST 1: Layer stack inspection for each k_steep")
print("=" * 80)
print(f"{'k':>5s} {'thick':>8s} {'N_grid':>7s} {'N_layers':>9s} {'d_min':>8s} {'d_max':>8s} "
      f"{'n_min':>10s} {'n_max':>10s} {'<n_re>':>8s}")

for k_val in k_values:
    dx_k = 1 / (100 * k_val)
    xmin_k = -20 / k_val
    xmax_k = -xmin_k
    nx_k = 1 + int(np.floor((xmax_k - xmin_k) / dx_k))
    xx_k = np.linspace(xmin_k, xmax_k, nx_k)
    e_re_k = logistic_eps(xx_k, k_val, nb)
    e_im_k = ht_derivative(xx_k, e_re_k)
    ee_k = e_re_k + 1j * e_im_k

    nc, dc = discretize_profile(xx_k, ee_k, delta=delta)
    thickness = 2 * xmax_k
    n_arr = np.array(nc)
    d_arr = np.array(dc)

    print(f"{k_val:5d} {thickness:8.3f} {nx_k:7d} {len(nc):9d} {d_arr.min():8.5f} {d_arr.max():8.5f} "
          f"{n_arr.real.min():10.5f} {n_arr.real.max():10.5f} {np.mean(n_arr.real):8.5f}")

# ============================================================================
# TEST 2: TMM vs Fresnel for uniform thin film (sKK coating vs avg-n film)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: sKK TMM R_back vs uniform thin film Fresnel R_back (λ=3 μm)")
print("=" * 80)
print(f"{'k':>5s} {'thick':>8s} {'R_sKK':>10s} {'R_GRIN':>10s} {'R_uniform':>10s} "
      f"{'R_no_coat':>10s} {'n_avg':>10s}")

# No-coating reference (bare substrate back surface)
R_bare = fresnel_thin_film_R(1.0+0j, 0.0, n_sub, angle_test, lam_test, pol_test)

for k_val in k_values:
    dx_k = 1 / (100 * k_val)
    xmin_k = -20 / k_val
    xmax_k = -xmin_k
    nx_k = 1 + int(np.floor((xmax_k - xmin_k) / dx_k))
    xx_k = np.linspace(xmin_k, xmax_k, nx_k)
    e_re_k = logistic_eps(xx_k, k_val, nb)
    e_im_k = ht_derivative(xx_k, e_re_k)

    thickness = 2 * xmax_k

    # sKK coating
    nc_skk, dc_skk = discretize_profile(xx_k, e_re_k + 1j * e_im_k, delta=delta)
    R_skk = Rback_single_wavelength(nc_skk, dc_skk, n_sub, angle_test, lam_test, pol_test)

    # GRIN (real only)
    nc_grin, dc_grin = discretize_profile(xx_k, e_re_k + 0j, delta=delta)
    R_grin = Rback_single_wavelength(nc_grin, dc_grin, n_sub, angle_test, lam_test, pol_test)

    # Uniform thin film with average n
    n_avg = np.mean(np.sqrt(e_re_k + 1j * e_im_k))
    R_uniform = fresnel_thin_film_R(n_avg, thickness, n_sub, angle_test, lam_test, pol_test)

    print(f"{k_val:5d} {thickness:8.3f} {R_skk:10.6f} {R_grin:10.6f} {R_uniform:10.6f} "
          f"{R_bare:10.6f} {n_avg:10.5f}")

# ============================================================================
# TEST 3: Convergence test — does finer discretization (smaller delta) change R?
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Discretization convergence (varying delta) at λ=3 μm")
print("=" * 80)

delta_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
thin_k_values = [50, 75, 100]  # thinnest coatings

for k_val in thin_k_values:
    dx_k = 1 / (100 * k_val)
    xmin_k = -20 / k_val
    xmax_k = -xmin_k
    nx_k = 1 + int(np.floor((xmax_k - xmin_k) / dx_k))
    xx_k = np.linspace(xmin_k, xmax_k, nx_k)
    e_re_k = logistic_eps(xx_k, k_val, nb)
    e_im_k = ht_derivative(xx_k, e_re_k)
    ee_k = e_re_k + 1j * e_im_k
    thickness = 2 * xmax_k

    print(f"\n  k={k_val}, thickness={thickness:.3f} μm, N_grid={nx_k}")
    print(f"  {'delta':>8s} {'N_layers':>9s} {'R_back':>10s}")

    for dval in delta_values:
        nc, dc = discretize_profile(xx_k, ee_k, delta=dval)
        R = Rback_single_wavelength(nc, dc, n_sub, angle_test, lam_test, pol_test)
        print(f"  {dval:8.4f} {len(nc):9d} {R:10.6f}")

# ============================================================================
# TEST 4: Dense grid convergence — does finer spatial grid (more points) help?
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Spatial grid convergence (varying number of grid points)")
print("=" * 80)

for k_val in [75, 100]:
    xmin_k = -20 / k_val
    xmax_k = -xmin_k
    thickness = 2 * xmax_k

    print(f"\n  k={k_val}, thickness={thickness:.3f} μm")
    print(f"  {'grid_factor':>12s} {'N_grid':>7s} {'N_layers':>9s} {'R_sKK':>10s} {'R_GRIN':>10s}")

    for gf in [1, 2, 5, 10, 20]:
        dx_k = 1 / (100 * k_val * gf)
        nx_k = 1 + int(np.floor((xmax_k - xmin_k) / dx_k))
        xx_k = np.linspace(xmin_k, xmax_k, nx_k)
        e_re_k = logistic_eps(xx_k, k_val, nb)
        e_im_k = ht_derivative(xx_k, e_re_k)

        nc_skk, dc_skk = discretize_profile(xx_k, e_re_k + 1j * e_im_k, delta=delta)
        R_skk = Rback_single_wavelength(nc_skk, dc_skk, n_sub, angle_test, lam_test, pol_test)

        nc_grin, dc_grin = discretize_profile(xx_k, e_re_k + 0j, delta=delta)
        R_grin = Rback_single_wavelength(nc_grin, dc_grin, n_sub, angle_test, lam_test, pol_test)

        print(f"  {gf:12d} {nx_k:7d} {len(nc_skk):9d} {R_skk:10.6f} {R_grin:10.6f}")

# ============================================================================
# TEST 5: Profile quality check — ε'' quality for thin coatings
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: sKK profile quality for thin vs thick coatings")
print("=" * 80)
print(f"{'k':>5s} {'thick':>8s} {'max|ε_im|':>10s} {'ε_im_left':>10s} {'ε_im_right':>10s} "
      f"{'ε_re range':>11s} {'ratio_im/re':>12s}")

for k_val in k_values:
    dx_k = 1 / (100 * k_val)
    xmin_k = -20 / k_val
    xmax_k = -xmin_k
    nx_k = 1 + int(np.floor((xmax_k - xmin_k) / dx_k))
    xx_k = np.linspace(xmin_k, xmax_k, nx_k)
    e_re_k = logistic_eps(xx_k, k_val, nb)
    e_im_k = ht_derivative(xx_k, e_re_k)

    thickness = 2 * xmax_k
    e_re_range = e_re_k.max() - e_re_k.min()
    max_eim = np.max(np.abs(e_im_k))
    ratio = max_eim / e_re_range if e_re_range > 0 else 0

    print(f"{k_val:5d} {thickness:8.3f} {max_eim:10.5f} {e_im_k[0]:10.5f} {e_im_k[-1]:10.5f} "
          f"{e_re_range:11.5f} {ratio:12.5f}")

# ============================================================================
# TEST 6: Direct check — is the sKK R_back for thin coatings close to GRIN?
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: Wavelength-averaged R_back comparison (same as in thickness sweep)")
print("=" * 80)
print(f"{'k':>5s} {'thick':>8s} {'R_sKK':>10s} {'R_GRIN':>10s} {'R_bulk':>10s} "
      f"{'sKK/GRIN':>10s} {'sKK benefit':>12s}")

# Bulk reference (wavelength-averaged)
from scipy.signal.windows import tukey as _tw  # just to avoid re-import issues
angle = angle_test * np.pi / 180
R_bulk_arr = np.zeros(len(lamdata))
for i, wl in enumerate(lamdata):
    n_s = complex(ndata[i], kdata[i])
    n_b = [1, n_s, 1]
    d_b = [np.inf, 5000, np.inf]
    c_b = ['i', 'i', 'i']
    th_f = tmm.snell(1, n_s, angle)
    Rf = tmm.interface_R(pol_test, 1, n_s, angle, th_f)
    res = tmm.inc_tmm(pol_test, n_b, d_b, c_b, angle, wl)
    R_bulk_arr[i] = res['R'] - Rf
R_bulk_avg = np.trapezoid(R_bulk_arr, lamdata) / (lamdata[-1] - lamdata[0])

for k_val in k_values:
    dx_k = 1 / (100 * k_val)
    xmin_k = -20 / k_val
    xmax_k = -xmin_k
    nx_k = 1 + int(np.floor((xmax_k - xmin_k) / dx_k))
    xx_k = np.linspace(xmin_k, xmax_k, nx_k)
    e_re_k = logistic_eps(xx_k, k_val, nb)
    e_im_k = ht_derivative(xx_k, e_re_k)
    thickness = 2 * xmax_k

    # sKK
    nc_skk, dc_skk = discretize_profile(xx_k, e_re_k + 1j * e_im_k, delta=delta)
    R_skk_arr = np.zeros(len(lamdata))
    for i, wl in enumerate(lamdata):
        n_s = complex(ndata[i], kdata[i])
        R_skk_arr[i] = Rback_single_wavelength(nc_skk, dc_skk, n_s, angle_test, wl, pol_test)
    R_skk_avg = np.trapezoid(R_skk_arr, lamdata) / (lamdata[-1] - lamdata[0])

    # GRIN
    nc_grin, dc_grin = discretize_profile(xx_k, e_re_k + 0j, delta=delta)
    R_grin_arr = np.zeros(len(lamdata))
    for i, wl in enumerate(lamdata):
        n_s = complex(ndata[i], kdata[i])
        R_grin_arr[i] = Rback_single_wavelength(nc_grin, dc_grin, n_s, angle_test, wl, pol_test)
    R_grin_avg = np.trapezoid(R_grin_arr, lamdata) / (lamdata[-1] - lamdata[0])

    ratio = R_skk_avg / R_grin_avg if R_grin_avg > 1e-10 else float('nan')
    benefit = R_grin_avg - R_skk_avg

    print(f"{k_val:5d} {thickness:8.3f} {R_skk_avg:10.6f} {R_grin_avg:10.6f} {R_bulk_avg:10.6f} "
          f"{ratio:10.4f} {benefit:12.6f}")

# ============================================================================
# TEST 7: Physical limit check — what does Fresnel give for sub-λ films?
# ============================================================================
print("\n" + "=" * 80)
print("TEST 7: Expected Fresnel R_back for uniform films (n=nb) of various thickness")
print("        This shows what R_back *should* be for sub-λ coatings regardless of profile")
print("=" * 80)
print(f"{'d (μm)':>8s} {'R_back (λ=3)':>12s}")

for d_val in [0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]:
    R = fresnel_thin_film_R(complex(nb, 0), d_val, n_sub, angle_test, lam_test, pol_test)
    print(f"{d_val:8.2f} {R:12.6f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key question: Is the nonzero sKK R_back for thin coatings a numerical artifact?

Check the results above:
- TEST 1: If thin coatings have very few layers, discretization may be inadequate.
- TEST 2: If sKK R_back ≈ uniform film R_back for thin coatings, the wave can't
          resolve the spatial profile (physically expected behavior).
- TEST 3: If R_back changes significantly with delta, it's a discretization artifact.
- TEST 4: If R_back changes with grid density, it's a spatial resolution artifact.
- TEST 5: If ε'' is noisy or has large boundary values for thin coatings,
          the HT is poorly converged.
- TEST 6: If sKK ≈ GRIN for thin coatings, the loss profile isn't helping.
- TEST 7: Baseline Fresnel R_back for uniform thin films on sapphire.
""")
