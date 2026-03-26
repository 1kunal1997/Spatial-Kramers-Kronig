"""
Bruggeman Mixture Search (single wavelength; default 3 µm)

- Loads n,k for a set of materials from refractiveindex.info using the `refractiveindex` package
- Computes effective index via the Bruggeman model (with branch stitching for physicality)
- Searches all 2-material mixtures over a volume-fraction grid to match target (n,k) points
- Outputs a CSV ranking candidate pairs for each target, and (optionally) plots

USAGE
-----
    python bruggeman_mixture_search.py --wl_um 3.0 --grid 1001 --top 10 --csv results.csv --plot

Notes
-----
- Wavelength is treated in microns and converted to nm for the library.
- You can edit the `MATERIALS` list and `PAGE_LOOKUP` mapping to match your inventory.
- Targets include thickness 'd' but it is not used in the matching (optical thickness design happens later).
"""
#%% ##############################################################################################################

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import itertools
import math
import sys
import csv
from typing import Dict, Tuple, List

import numpy as np
from plot_functions import plot_setup, plot, legend
import colors

try:
    from refractiveindex import RefractiveIndexMaterial
except Exception as e:
    print("ERROR: The 'refractiveindex' package is required. Install with:\n"
          "    pip install refractiveindex\n")
    raise

# ---------------------- Inputs you can customize ----------------------

# Candidate materials (keep only those your fab team can access)
MATERIALS = ["AlN","Al2O3","Al","BaF2","Bi","B4C","CdSe","C","Cr","Cu","Ge2Sb2Te5","Ge","Au","HfO2","In","Ir","Fe2O3","Fe","Pb","LiNbO3","MgF2","MgO","Mg","Mn","Mo","Ni","Nb2O5","Nb","Pd","Pt","Rh","Se","SiC","SiO2","SiO","Si3N4","Si","Ag","Ta2O5","Ta","Sn","TiO2","TiN","Ti","W","Y2O3","ZnSe","ZnS","ZnTe","Zn","ZrO2","Zr","ZnO"]

# -------------------- Oxidation-safe filtering --------------------

# Metals prone to oxidation — skip these completely
DISALLOWED = {
    "Al", "Cr", "Cu", "Fe", "In", "Ir", "Mg", "Mn", "Mo",
    "Ni", "Nb", "Pd", "Pt", "Rh", "Ta", "Sn", "Ti", "W",
    "Zn", "Zr", "Ag", "Pb", "Au", "Bi", "TiN"
}

# Filter out unsafe materials
MATERIALS = [m for m in MATERIALS if m not in DISALLOWED]

# Define lossy materials you’re still allowing
# (for reference only — no need to force them into pairing logic)
LOSSY = ["C", "B4C", "Ge2Sb2Te5", "Fe2O3", "ZnO"]

print(len(MATERIALS))

# Map (material -> dataset page/author) for refractiveindex.info
PAGE_LOOKUP = {
    "AlN":"Kischkat",
    "Al2O3":"Kischkat",
    "Al":"Rakic",
    "BaF2":"Malitson",
    "SrTiO3":"Dodge",
    "BaTiO3":"Wemple",
    "BiFeO3":"Kummar",
    "Bi2Se3":"Ermolaev",
    "Bi2Te3":"Ermolaev",
    "Bi4Ti3O12":"Simon",
    "Bi":"Hagemann",
    "B4C":"Larruquert",
    "BN":"Grundinin",
    "B":"Fernandez-Perea",
    "CdSe":"Lisitsa-o",
    "CdS":"Treharne",
    "CdTe":"Marple",
    "Ca":"Mathewson",
    "C":"Djurisic-o",
    "CeF3":"Rodriguez-de_Marcos",
    "Ce":"Fernandez-Perea",
    "Cr":"Rakic-BB",
    "Co":"Werner",
    "Cu":"Babar",
    "Er2O3":"Adair",
    "Er":"Larruquert",
    "Eu":"Fernandez-Perea",
    "Ge2Sb2Te5":"Frantz-crystal",
    "Ge":"Amotchkina",
    "Au":"Rakic-BB",
    "HfO2":"Franta",
    "Hf":"Windt",
    "Ho":"Fernandez-Perea",
    "In2O3-SnO2":"Minenkov-glass",
    "In":"Golovashkin-295K",
    "Ir":"Schmitt-ALD",
    "Fe2O3":"Querry-o",
    "Fe":"Querry",
    "PbTe":"Weiting-300K",
    "PbTiO3":"Singh",
    "Pb":"Ordal",
    "LiNbO3":"Zelmon-o",
    "MgF2":"Franta",
    "MgO":"Stephens",
    "Mg":"Hagemann",
    "Mn":"Querry",
    "MoS2":"Ermolaev-o",
    "MoO3":"Lajaunie-beta",
    "Mo":"Ordal",
    "Ni-Fe":"Tikuisis_bare150nm",
    "Ni":"Rakic-BB",
    "Nb2O5":"Franta",
    "Nb":"Golovashkin-293K",
    "Pd":"Rakic-BB",
    "Pt":"Rakic-BB",
    "Re":"Windt",
    "Rh":"Weaver",
    "Ru":"Windt",
    "Se":"Campel-o",
    "SiC":"Wang-4H-o",
    "SiO2":"Franta",
    "SiO":"Hass",
    "Si3N4":"Luke",
    "Si":"Franta",
    "Ag":"Rakic-BB",
    "SrTiO3":"Dodge",
    "Sr":"Rodríguez-de_Marcos",
    "Ta2O5":"Franta",
    "Ta":"Ordal",
    "Tm":"Vidal-Dasilva",
    "Sn":"Golovashkin-293K",
    "TiC":"Pfluger",
    "TiO2":"Franta",
    "TiN":"Beliaev-sputtering",
    "Ti":"Rakic-BB",
    "WS2":"Vyshnevyy",
    "WO3":"Kulikova",
    "W":"Rakic-BB",
    "VC":"Pfluger",
    "V":"Werner",
    "Yb2O3":"Medenbach",
    "Yb":"Larruquert",
    "Y2O3":"Nigara",
    "ZnO":"Querry",
    "ZnSe":"Amotchkina",
    "ZnS":"Amotchkina",
    "ZnTe":"Li",
    "Zn":"Querry",
    "ZrO2":"Wood",
    "Zr":"Querry",
}

# Target (n,k,d) triplets at the design wavelength (kramers-kronig profile samples)

'''
TARGETS = [
    (2.31, 0.0, 0.58),
    (2.31, 0.0, 0.8 ),
    (2.32, 0.0, 0.8 ),
    (2.33, 0.0, 0.8 ),
    (2.37, 0.01,0.72),
    (2.5,  0.04,0.19),
    (2.67, 0.15,0.05),
    (2.78, 0.3, 0.02),
    (2.83, 0.48,0.02),
    (2.81, 0.65,0.01),
    (2.73, 0.82,0.01),
    (2.59, 0.95,0.01),
    (2.41, 1.02,0.01),
    (2.2,  1.02,0.01),
    (1.99, 0.92,0.01),
    (1.83, 0.74,0.02),
    (1.78, 0.47,0.02),
    (1.88, 0.21,0.05),
    (2.08, 0.05,0.19),
    (2.23, 0.01,0.72),
    (2.27, 0.0, 0.8 ),
    (2.28, 0.0, 0.8 ),
    (2.29, 0.0, 0.8 ),
    (2.29, 0.0, 0.58),
]
'''

TARGETS = [
    (2.28, 0.0, 3.9),
    (2.8, 0.45, 0.08 ),
    (2.48, 1.01, 0.04 ),
    (1.77, 0.7, 0.08 ),
    (2.26, 0.01, 3.9),
]


#%% ###############################################################################################

# ---------------------- Bruggeman and utilities ----------------------
def bruggeman_eps(e1: complex, e2: complex, f2: np.ndarray) -> np.ndarray:
    """
    Bruggeman effective permittivity for two-component mixture.
    e1: permittivity of component 1
    e2: permittivity of component 2
    f2: volume fraction of component 2 (numpy array)
    Returns eps_eff (numpy array), using stitched physical branch (Im >= 0) and continuity.
    """
    f2 = np.asarray(f2)
    f1 = 1.0 - f2

    Hb = e1*(3.0*f1 - 1.0) + e2*(3.0*f2 - 1.0)
    rad = np.sqrt(Hb*Hb + 8.0*e1*e2)

    eps_plus  = (Hb + rad) / 4.0
    eps_minus = (Hb - rad) / 4.0

    # Stitch branch: choose Im>=0 and minimal jump from previous step
    eps_eff = np.zeros_like(eps_plus, dtype=complex)
    prev = None
    for i in range(eps_eff.size):
        candidates = [eps_plus[i], eps_minus[i]]
        # Filter by positive imaginary part (passive)
        physical = [e for e in candidates if np.imag(e) >= 0]
        if not physical:
            # if both have negative Im (rare), choose the one with larger Im
            physical = [max(candidates, key=lambda z: np.imag(z))]
        if prev is None:
            # start from the physically valid with larger Im to avoid branch-cut glitches
            choice = max(physical, key=lambda z: np.imag(z))
        else:
            # continuity: smallest change from previous
            choice = min(physical, key=lambda z: abs(z - prev))
        eps_eff[i] = choice
        prev = choice
    return eps_eff

def bruggeman_eps_with_air(e1, e2, f2, f_air=0.05):
    """
    Effective permittivity with porosity (air fraction).
    e1, e2: complex permittivities of the two materials.
    f2: volume fraction of material 2 (can be array).
    f_air: fraction of air (void), 0.0–0.1 typically.
    Returns: complex eps_eff (same shape as f2)
    """
    e_air = 1.0 + 0j

    # normalize so that solids occupy (1 - f_air)
    f2_eff = (1 - f_air) * np.asarray(f2)
    f1_eff = (1 - f_air) * (1 - np.asarray(f2))

    # first mix the two solids
    eps_solid = bruggeman_eps(e1, e2, f2_eff / (f1_eff + f2_eff))

    # then mix that solid mixture with air
    eps_eff = bruggeman_eps(eps_solid, e_air, f_air)
    return eps_eff


def nk_to_eps(nk: complex) -> complex:
    return nk * nk

def eps_to_nk(eps: complex) -> complex:
    # Principal sqrt; Im(n) (i.e., k) should be >= 0 by construction from eps
    return np.sqrt(eps)


def load_material_nk(wl_nm: float, materials: List[str]) -> Dict[str, complex]:
    """
    Load complex refractive index (n + i k) for each material at wl_nm (nm).
    Materials that lack k (or fail) are skipped.
    """
    out = {}
    for m in materials:
        page = PAGE_LOOKUP.get(m)
        if page is None:
            continue
        try:
            rim = RefractiveIndexMaterial(shelf="main", book=m, page=page)
            n = rim.get_refractive_index(wl_nm)
            try:
                k = rim.get_extinction_coefficient(wl_nm) or 0.0
            except Exception:
                k = 0.0
            
            if n is None:
                continue
            nk = complex(n, k)
            out[m] = nk
        except Exception as e:
            # Skip materials that fail
            continue
    return out


def match_targets(material_nk: Dict[str, complex],
                  targets: List[Tuple[float, float, float]],
                  grid: int = 1001,
                  top: int = 8) -> List[Dict]:
    """
    For each target (n,k,d), search all material pairs and fractions to find best matches.
    Returns a list of result rows (dicts) for CSV/printing.
    """
    names = sorted(material_nk.keys())
    fracs = np.linspace(0.0, 1.0, grid)

    rows = []
    for t_idx, (nt, kt, dt) in enumerate(targets):
        target = complex(nt, kt)
        best = []  # list of tuples (error, result_dict)

        for a, b in itertools.combinations(names, 2):
            nk1 = material_nk[a]
            nk2 = material_nk[b]
            e1 = nk_to_eps(nk1)
            e2 = nk_to_eps(nk2)

            #eps_eff = bruggeman_eps(e1, e2, fracs)
            eps_eff = bruggeman_eps_with_air(e1, e2, fracs, f_air=0.05)
            nk_eff = eps_to_nk(eps_eff)  # complex n = n + i k

            # Compute error over the grid
            dn = np.real(nk_eff) - nt
            dk = np.imag(nk_eff) - kt
            err = np.sqrt(dn*dn + dk*dk)
            i = int(np.argmin(err))
            e = float(err[i])

            row = {
                "target_index": t_idx,
                "target_n": nt,
                "target_k": kt,
                "target_d": dt,
                "A": a,
                "B": b,
                "f_B": float(fracs[i]),  # fraction of material B
                "n_eff": float(np.real(nk_eff[i])),
                "k_eff": float(np.imag(nk_eff[i])),
                "error": e,
            }

            # Keep only top N
            if len(best) < top:
                best.append((e, row))
                best.sort(key=lambda x: x[0])
            else:
                if e < best[-1][0]:
                    best[-1] = (e, row)
                    best.sort(key=lambda x: x[0])

        # Append best rows for this target
        rows.extend([r for (_, r) in best])
    return rows

#%% ###################################################################################################################

wl_um = 3.0
grid = 1001
top = 8
csv_file = os.path.join(_PROJECT_ROOT, 'mixture_results.csv')
wl_nm = wl_um * 1000.0
print(f"Loading materials at {wl_um:.3f} µm ({wl_nm:.1f} nm)...")
material_nk = load_material_nk(wl_nm, MATERIALS)
print(len(material_nk))

#%% ##############################################################
# --- Single-material pre-check ----------------------------------
pure_tol_n = 0.1   # allowable deviation in n
pure_tol_k = 0.1   # allowable deviation in k

pure_candidates = []

print("Checking for single-material matches within tolerance...")
for ti, (n_t, k_t, d_t) in enumerate(TARGETS):
    best_err = np.inf
    best_entry = None
    for name, nk in material_nk.items():
        n, k = np.real(nk), np.imag(nk)
        err_n = abs(n - n_t)
        err_k = abs(k - k_t)
        if err_n <= pure_tol_n and err_k <= pure_tol_k:
            total_err = math.hypot(err_n, err_k)
            if total_err < best_err:
                best_err = total_err
                best_entry = {
                    "target_index": ti,
                    "target_n": n_t,
                    "target_k": k_t,
                    "target_d": d_t,
                    "A": name,      # single material
                    "B": "-",       # no mixture partner
                    "f_B": 0.0,
                    "n_eff": n,
                    "k_eff": k,
                    "error": total_err,
                }
    if best_entry:
        pure_candidates.append(best_entry)

print(f"Found {len(pure_candidates)} single-material candidates within tolerance.")

if not material_nk:
    print("No materials were successfully loaded with both n and k at this wavelength.")
    sys.exit(1)

print(f"Loaded {len(material_nk)} materials with n & k.")
rows = match_targets(material_nk, TARGETS, grid=grid, top=top)

# merge single-material hits into the main result list,
# replacing mixtures for those targets
targets_with_pure = {r["target_index"] for r in pure_candidates}
rows = [
    r for r in rows if r["target_index"] not in targets_with_pure
] + pure_candidates

rows = sorted(rows, key=lambda r: r["target_index"])

print(f"Computed candidates. Writing CSV: {csv_file}")

# Write CSV
fieldnames = ["target_index","target_n","target_k","target_d","A","B","f_B","n_eff","k_eff","error"]
with open(csv_file, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("Done.")

#%% #################################################################################################

import matplotlib.pyplot as plt
# For each target, plot the best (lowest error) candidate curve
names = sorted(material_nk.keys())
fracs = np.linspace(0.0, 1.0, grid)

# Pick best per target from rows
best_by_target = {}
for r in rows:
    ti = r["target_index"]
    if ti not in best_by_target or r["error"] < best_by_target[ti]["error"]:
        best_by_target[ti] = r

xlabel = "Fraction of Material B"
ylabel = "Refractive Index"
xlim = (0.0, 1.0)
auto_scale = True
nk_eff_list = []
nk_target_list = []
d_list = []
for ti, r in best_by_target.items():

    n_target = TARGETS[ti][0]
    k_target = TARGETS[ti][1]
    d = TARGETS[ti][2]
    a, b = r["A"], r["B"]
    concB = np.round(r["f_B"],2)
    concA = np.round(1 - concB,2)
    n_eff = np.round(r["n_eff"],2)
    k_eff = np.round(r["k_eff"],2)

    d_list.append(d)
    nk_target_list.append(n_target + 1j*k_target)
    nk_eff_list.append(n_eff + 1j*k_eff)

    # Single material – just plot as flat line
    if b == "-":
        title = f"Layer {ti}: n={n_target}, k={k_target}, d={d}$\mu m$ \n Single material: {a} (n={n_eff})"
        fig, ax = plot_setup(
        xlabel, ylabel, title=title,
        xlim=xlim, figsize=(5,4),auto_scale=auto_scale)   

        ax.axhline(n_eff, color=colors.blue, label="n")
        ax.axhline(k_eff, color=colors.red, label="k")
        ax.axhline(n_target, ls="--", color=colors.blue, label="target n")
        ax.axhline(k_target, ls="--", color=colors.red, label="target k")
        legend(fig,ax,auto_scale=auto_scale)
        continue

    nk1 = material_nk[a]; nk2 = material_nk[b]
    e1 = nk_to_eps(nk1); e2 = nk_to_eps(nk2)
    eps_eff = bruggeman_eps(e1, e2, fracs)
    nk_eff = eps_to_nk(eps_eff)

    #title = f"Layer {ti}: , n={n_target}, k={k_target}, d={d}$\mu m$ \n Mixture: {concA}*{a} + {concB}*{b} (n={n_eff}, k={k_eff})"
    title = f"Layer {ti}: {concA}*{a} + {concB}*{b} with 5% porosity"
    fig, ax = plot_setup(
    xlabel, ylabel, title=title,
    xlim=xlim, figsize=(5,4),auto_scale=auto_scale) 

    plot(fig,ax, fracs, np.real(nk_eff), color=colors.blue, label="n$_{eff}$", auto_scale=auto_scale)
    plot(fig,ax, fracs, np.imag(nk_eff), color=colors.red, label="k$_{eff}$", auto_scale=auto_scale)
    ax.scatter(concB, n_eff, color=colors.blue, marker="x", label=f"n$_{{best}}$ = {n_eff}", zorder=3)
    ax.scatter(concB, k_eff, color=colors.red, marker="x", label=f"k$_{{best}}$ = {k_eff}", zorder=3)
    ax.axhline(n_target, ls="--", color=colors.blue, label=f"n$_{{target}}$ = {n_target}")
    ax.axhline(k_target, ls="--", color=colors.red, label=f"k$_{{target}}$ = {k_target}")

    legend(fig,ax,auto_scale=auto_scale)
plt.show()

# %%

np.savetxt(os.path.join(_PROJECT_ROOT, 'nk_eff_graphite_only.txt'), nk_eff_list)
np.savetxt(os.path.join(_PROJECT_ROOT, 'd_list_graphite_only.txt'), d_list)


# %%

target_n_list = [nk.real for nk in nk_target_list]
target_k_list = [nk.imag for nk in nk_target_list]
n_eff_list = [nk.real for nk in nk_eff_list]
k_eff_list = [nk.imag for nk in nk_eff_list]
#%%
# Example: use cumulative sum of layer thicknesses for position axis
total_thickness = np.sum(d_list)
x_edges = np.concatenate(([0], np.cumsum(d_list))) - total_thickness / 2
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
zoomed = False

# --- Re(n) plot ---
xlabel = 'x ($\mu$m)'; ylabel = 'Re(n)'
title = f''
if (zoomed):
    xlim = (x_edges[0]/10,x_edges[-1]/10)
else:
    xlim = (x_edges[0],x_edges[-1])
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim)

ax.stairs(target_n_list, x_edges, linewidth=1.5, baseline=target_n_list[0], label='Target n', color=colors.red, linestyle='--')
ax.stairs(n_eff_list, x_edges, linewidth=2, baseline=n_eff_list[0], label='Effective n', color=colors.blue)
plot(fig,ax, x_centers, n_eff_list, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)

# --- Im(n) plot ---
k_max   = max(np.max(target_k_list), np.max(k_eff_list))
ylabel = 'Im(n)'
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim, ylim=(-k_max/20,k_max*(1+1/20)))

ax.stairs(target_k_list, x_edges, linewidth=1.5, baseline=0, label='target', color=colors.red, linestyle='--')
ax.stairs(k_eff_list, x_edges, linewidth=2, baseline=0, label='effective', color=colors.blue)
plot(fig,ax, x_centers, k_eff_list, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)

legend(fig, ax, auto_scale=True)

# %%
wls = np.linspace(2,5,300)
nk = np.zeros_like(wls, dtype=complex)
page = PAGE_LOOKUP.get("ZnS")
rim = RefractiveIndexMaterial(shelf="main", book="ZnS", page=page)
for i, wl in enumerate(wls*1000):
    #print(i)
    n = rim.get_refractive_index(wl)
    k = rim.get_extinction_coefficient(wl)
    nk[i] = complex(n, k)
    #print(graphite_nk[i])

print(wls[99])
print(nk[99])

data = np.column_stack([wls, np.real(nk), np.imag(nk)])
np.savetxt(os.path.join(_PROJECT_ROOT, "RI", "ZnS_nk_2-5um.txt"), data, header="wl[um] n k", fmt="%.6e")
# %%
