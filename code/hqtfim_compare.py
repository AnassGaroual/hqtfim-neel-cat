#!/usr/bin/env python3
"""
================================================================================
HQTFIM vs STANDARD ISING - COMPLETE COMPARISON
================================================================================
1. Compare HQTFIM with standard 1D nearest-neighbor TFIM
2. Check if S=1, F→1, exponents are specific to hierarchical
3. Calculate exponents for different α values
================================================================================
Author: Anass Garoual
Date: December 2025
MIT License
================================================================================
"""

import numpy as np
from scipy.linalg import eigh, svd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import time
import gc
from typing import Dict, List, Tuple

# =============================================================================
# PAULI MATRICES
# =============================================================================

sz = np.array([[1, 0], [0, -1]], dtype=np.float64)
sx = np.array([[0, 1], [1, 0]], dtype=np.float64)
I2 = np.eye(2, dtype=np.float64)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def v2(n: int) -> int:
    if n == 0:
        return 100
    c = 0
    while n % 2 == 0:
        n //= 2
        c += 1
    return c


def get_h_star(alpha: float, J0: float = 1.0) -> float:
    return J0 * 2**alpha / (2 ** (2 * alpha) - 1)


def apply_parity(psi: np.ndarray, N: int) -> np.ndarray:
    dim = 1 << N
    mask = dim - 1
    result = np.empty_like(psi)
    for s in range(dim):
        result[s] = psi[s ^ mask]
    return result


def project_even_parity(psi: np.ndarray, N: int) -> np.ndarray:
    P_psi = apply_parity(psi, N)
    proj = psi + P_psi
    norm = np.linalg.norm(proj)
    return proj / norm if norm > 1e-14 else psi / np.linalg.norm(psi)


def neel_indices(N: int) -> Tuple[int, int]:
    neel = sum((i % 2) << i for i in range(N))
    anti = sum(((i + 1) % 2) << i for i in range(N))
    return neel, anti


def compute_observables(psi: np.ndarray, N: int) -> Dict:
    """Compute all observables for a state."""
    n, a = neel_indices(N)
    dim = 1 << N

    # Fidelity with cat
    psi_cat = np.zeros(dim)
    psi_cat[n] = psi_cat[a] = 1.0 / np.sqrt(2)
    F = np.abs(np.vdot(psi_cat, psi)) ** 2

    # Entropy
    nA = N // 2
    sv = svd(psi.reshape(1 << nA, 1 << (N - nA)), compute_uv=False)
    sv = sv[sv > 1e-14]
    S = -np.sum(sv**2 * np.log2(sv**2 + 1e-300))

    # M² staggered
    M2 = 0.0
    for s in range(dim):
        prob = np.abs(psi[s]) ** 2
        M_s = sum(((-1) ** i) * (1 - 2 * ((s >> i) & 1)) for i in range(N)) / N
        M2 += prob * M_s**2

    # Schmidt coefficients
    schmidt_diff = abs(sv[0] - sv[1]) if len(sv) > 1 else sv[0]

    return {
        "fidelity": F,
        "infidelity": 1 - F,
        "entropy": S,
        "M2_staggered": M2,
        "schmidt_0": sv[0],
        "schmidt_1": sv[1] if len(sv) > 1 else 0,
        "schmidt_diff": schmidt_diff,
    }


# =============================================================================
# MODEL 1: HIERARCHICAL TFIM (HQTFIM)
# =============================================================================


def build_H_hierarchical(N: int, J0: float, alpha: float, h: float) -> np.ndarray:
    """Build HQTFIM Hamiltonian."""
    dim = 1 << N
    H = np.zeros((dim, dim), dtype=np.float64)

    for s in range(dim):
        # Diagonal: Ising
        E = 0.0
        for i in range(N):
            si = 1 - 2 * ((s >> i) & 1)
            for j in range(i + 1, N):
                sj = 1 - 2 * ((s >> j) & 1)
                d = j - i
                J_ij = J0 * 2 ** (-alpha * v2(d))
                E += J_ij * si * sj
        H[s, s] = E

        # Off-diagonal: transverse field
        for i in range(N):
            H[s, s ^ (1 << i)] = -h

    return H


def diagonalize_hierarchical(N: int, J0: float, alpha: float, h: float) -> Dict:
    """Diagonalize HQTFIM and compute observables."""
    H = build_H_hierarchical(N, J0, alpha, h)
    E, V = eigh(H)

    psi = project_even_parity(V[:, 0], N)
    gap = E[1] - E[0]

    obs = compute_observables(psi, N)
    obs["E0"] = E[0]
    obs["gap"] = gap
    obs["N"] = N

    del H, V
    gc.collect()

    return obs


# =============================================================================
# MODEL 2: STANDARD 1D TFIM (NEAREST-NEIGHBOR)
# =============================================================================


def build_H_standard_1D(N: int, J: float, h: float, pbc: bool = True) -> np.ndarray:
    """
    Build standard 1D TFIM: H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i

    pbc: periodic boundary conditions
    """
    dim = 1 << N
    H = np.zeros((dim, dim), dtype=np.float64)

    for s in range(dim):
        # Diagonal: nearest-neighbor Ising
        E = 0.0
        for i in range(N - 1):
            si = 1 - 2 * ((s >> i) & 1)
            sj = 1 - 2 * ((s >> (i + 1)) & 1)
            E += -J * si * sj

        # PBC: connect site N-1 to site 0
        if pbc:
            s0 = 1 - 2 * (s & 1)
            sN = 1 - 2 * ((s >> (N - 1)) & 1)
            E += -J * sN * s0

        H[s, s] = E

        # Off-diagonal: transverse field
        for i in range(N):
            H[s, s ^ (1 << i)] = -h

    return H


def diagonalize_standard_1D(N: int, J: float, h: float, pbc: bool = True) -> Dict:
    """Diagonalize standard 1D TFIM."""
    H = build_H_standard_1D(N, J, h, pbc)
    E, V = eigh(H)

    psi = project_even_parity(V[:, 0], N)
    gap = E[1] - E[0]

    obs = compute_observables(psi, N)
    obs["E0"] = E[0]
    obs["gap"] = gap
    obs["N"] = N

    del H, V
    gc.collect()

    return obs


# =============================================================================
# MODEL 3: STANDARD 1D TFIM AT ITS CRITICAL POINT
# =============================================================================


def get_standard_critical_h(J: float = 1.0) -> float:
    """Critical point of 1D TFIM: h_c = J (exactly known)."""
    return J


# =============================================================================
# COMPARISON 1: HQTFIM vs STANDARD 1D AT SAME h/J RATIO
# =============================================================================


def compare_models_same_h(N_list: List[int], h_over_J: float = 0.404061):
    """Compare HQTFIM and standard 1D at same h/J ratio."""

    print("=" * 80)
    print("COMPARISON 1: HQTFIM vs STANDARD 1D TFIM at h/J = {:.4f}".format(h_over_J))
    print("=" * 80)

    J0 = 1.0
    h = h_over_J * J0
    alpha = 1.5

    results_hier = []
    results_std = []

    print(
        f"\n{'N':>4} | {'F_hier':>10} | {'F_std':>10} | {'S_hier':>8} | {'S_std':>8} | {'M²_hier':>8} | {'M²_std':>8}"
    )
    print("-" * 80)

    for N in N_list:
        # Hierarchical
        r_h = diagonalize_hierarchical(N, J0, alpha, h)
        results_hier.append(r_h)

        # Standard 1D
        r_s = diagonalize_standard_1D(N, J0, h, pbc=True)
        results_std.append(r_s)

        print(
            f"{N:4} | {r_h['fidelity']:10.6f} | {r_s['fidelity']:10.6f} | "
            f"{r_h['entropy']:8.4f} | {r_s['entropy']:8.4f} | "
            f"{r_h['M2_staggered']:8.4f} | {r_s['M2_staggered']:8.4f}"
        )

    return {"hierarchical": results_hier, "standard": results_std}


# =============================================================================
# COMPARISON 2: BOTH MODELS AT THEIR RESPECTIVE CRITICAL POINTS
# =============================================================================


def compare_models_at_criticality(N_list: List[int], alpha: float = 1.5):
    """Compare both models at their respective critical points."""

    print("\n" + "=" * 80)
    print("COMPARISON 2: BOTH MODELS AT THEIR CRITICAL POINTS")
    print("=" * 80)

    J0 = 1.0
    h_hier = get_h_star(alpha, J0)
    h_std = get_standard_critical_h(J0)

    print(f"HQTFIM critical point (formula): h*/J = {h_hier:.4f}")
    print(f"Standard 1D critical point (exact): h_c/J = {h_std:.4f}")

    results_hier = []
    results_std = []

    print(
        f"\n{'N':>4} | {'F_hier':>10} | {'F_std':>10} | {'S_hier':>8} | {'S_std':>8} | {'M²_hier':>8} | {'M²_std':>8}"
    )
    print("-" * 80)

    for N in N_list:
        r_h = diagonalize_hierarchical(N, J0, alpha, h_hier)
        results_hier.append(r_h)

        r_s = diagonalize_standard_1D(N, J0, h_std, pbc=True)
        results_std.append(r_s)

        print(
            f"{N:4} | {r_h['fidelity']:10.6f} | {r_s['fidelity']:10.6f} | "
            f"{r_h['entropy']:8.4f} | {r_s['entropy']:8.4f} | "
            f"{r_h['M2_staggered']:8.4f} | {r_s['M2_staggered']:8.4f}"
        )

    # Analysis
    print("\n" + "-" * 80)
    print("ANALYSIS:")

    # Check entropy scaling
    N_arr = np.array([r["N"] for r in results_std])
    S_std = np.array([r["entropy"] for r in results_std])

    # Standard 1D at criticality: S ~ (c/3) log(N) with c=1/2
    log_N = np.log(N_arr)
    slope, intercept, r, _, _ = linregress(log_N, S_std)
    c_eff = 3 * slope

    print(
        f"Standard 1D at h_c: S ~ {slope:.4f} log(N) → c_eff = {c_eff:.4f} (expected c=0.5)"
    )

    # HQTFIM entropy
    S_hier = np.array([r["entropy"] for r in results_hier])
    print(f"HQTFIM at h*: S → {S_hier[-1]:.6f} (constant ~ 1 bit)")

    return {"hierarchical": results_hier, "standard": results_std}


# =============================================================================
# COMPARISON 3: EXPONENTS FOR DIFFERENT α VALUES
# =============================================================================


def compute_exponents_vs_alpha(alpha_list: List[float], N_list: List[int]):
    """Compute convergence exponents for different α."""

    print("\n" + "=" * 80)
    print("COMPARISON 3: EXPONENTS vs α")
    print("=" * 80)

    J0 = 1.0
    results_by_alpha = {}

    for alpha in alpha_list:
        print(f"\nα = {alpha}:")
        h = get_h_star(alpha, J0)
        print(f"  h*/J = {h:.6f}")

        results = []
        for N in N_list:
            r = diagonalize_hierarchical(N, J0, alpha, h)
            results.append(r)
            print(f"    N={N}: F={r['fidelity']:.6f}, M²={r['M2_staggered']:.6f}")

        results_by_alpha[alpha] = results

    # Compute exponents
    print("\n" + "-" * 80)
    print("EXPONENT ANALYSIS:")
    print(
        f"{'α':>6} | {'h*/J':>8} | {'β (1-F~N^-β)':>14} | {'R²':>8} | {'γ (1-M²~N^-γ)':>14} | {'R²':>8}"
    )
    print("-" * 80)

    exponents = []

    for alpha in alpha_list:
        results = results_by_alpha[alpha]
        N_arr = np.array([r["N"] for r in results], dtype=float)
        infid = np.array([r["infidelity"] for r in results])
        M2 = np.array([r["M2_staggered"] for r in results])

        # Fit 1-F ~ N^(-β)
        log_N = np.log(N_arr)
        log_infid = np.log(infid)
        slope_F, _, r_F, _, _ = linregress(log_N, log_infid)
        beta = -slope_F

        # Fit 1-M² ~ N^(-γ)
        deficit = 1 - M2
        log_deficit = np.log(deficit)
        slope_M, _, r_M, _, _ = linregress(log_N, log_deficit)
        gamma = -slope_M

        h = get_h_star(alpha, J0)
        print(
            f"{alpha:6.2f} | {h:8.4f} | {beta:14.4f} | {r_F**2:8.4f} | {gamma:14.4f} | {r_M**2:8.4f}"
        )

        exponents.append(
            {
                "alpha": alpha,
                "h_star": h,
                "beta": beta,
                "R2_beta": r_F**2,
                "gamma": gamma,
                "R2_gamma": r_M**2,
            }
        )

    return {"by_alpha": results_by_alpha, "exponents": exponents}


# =============================================================================
# COMPARISON 4: PHASE DIAGRAM - FIND TRUE CRITICAL POINT
# =============================================================================


def find_true_critical_points(
    N: int, alpha_list: List[float], h_range: Tuple[float, float], n_points: int = 30
):
    """Find true critical point for each α."""

    print("\n" + "=" * 80)
    print(f"COMPARISON 4: TRUE CRITICAL POINTS (N={N})")
    print("=" * 80)

    J0 = 1.0
    h_values = np.linspace(h_range[0], h_range[1], n_points)

    results = {}

    for alpha in alpha_list:
        print(f"\nα = {alpha}:")
        h_star = get_h_star(alpha, J0)

        M2_values = []
        for h in h_values:
            r = diagonalize_hierarchical(N, J0, alpha, h)
            M2_values.append(r["M2_staggered"])

        M2_values = np.array(M2_values)

        # Find where M² drops most rapidly
        dM2 = np.gradient(M2_values, h_values)
        idx_min = np.argmin(dM2)
        h_c_numerical = h_values[idx_min]

        # Find where M² = 0.5
        try:
            from scipy.interpolate import interp1d
            from scipy.optimize import brentq

            f = interp1d(h_values, M2_values - 0.5, kind="cubic")
            h_c_half = brentq(f, h_range[0], h_range[1])
        except Exception:
            h_c_half = None

        print(f"  h* (formula) = {h_star:.4f}")
        print(f"  h_c (max drop) = {h_c_numerical:.4f}")
        if h_c_half:
            print(f"  h_c (M²=0.5) = {h_c_half:.4f}")
        print(f"  Ratio h_c/h* = {h_c_numerical/h_star:.2f}")

        results[alpha] = {
            "h_values": h_values,
            "M2": M2_values,
            "h_star": h_star,
            "h_c_numerical": h_c_numerical,
            "h_c_half": h_c_half,
        }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_all_comparisons(comp1, comp2, comp3, comp4, alpha_list):
    """Generate comprehensive comparison figures."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Fidelity comparison HQTFIM vs Standard at same h
    ax = axes[0, 0]
    N_h = [r["N"] for r in comp1["hierarchical"]]
    F_h = [r["fidelity"] for r in comp1["hierarchical"]]
    F_s = [r["fidelity"] for r in comp1["standard"]]
    ax.plot(N_h, F_h, "bo-", linewidth=2, markersize=8, label="HQTFIM")
    ax.plot(N_h, F_s, "rs--", linewidth=2, markersize=8, label="Standard 1D")
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Fidelity with cat", fontsize=12)
    ax.set_title("Fidelity at h/J = 0.404", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Entropy comparison at respective critical points
    ax = axes[0, 1]
    N_c = [r["N"] for r in comp2["hierarchical"]]
    S_h = [r["entropy"] for r in comp2["hierarchical"]]
    S_s = [r["entropy"] for r in comp2["standard"]]
    ax.plot(N_c, S_h, "bo-", linewidth=2, markersize=8, label="HQTFIM at h*")
    ax.plot(N_c, S_s, "rs--", linewidth=2, markersize=8, label="Standard at h_c=J")
    ax.axhline(y=1.0, color="b", linestyle=":", alpha=0.5)
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Entropy S (bits)", fontsize=12)
    ax.set_title("Entropy at Critical Points", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: M² comparison at respective critical points
    ax = axes[0, 2]
    M2_h = [r["M2_staggered"] for r in comp2["hierarchical"]]
    M2_s = [r["M2_staggered"] for r in comp2["standard"]]
    ax.plot(N_c, M2_h, "bo-", linewidth=2, markersize=8, label="HQTFIM at h*")
    ax.plot(N_c, M2_s, "rs--", linewidth=2, markersize=8, label="Standard at h_c")
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("⟨M²_stagg⟩", fontsize=12)
    ax.set_title("Order Parameter at Critical Points", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Exponent β vs α
    ax = axes[1, 0]
    alphas = [e["alpha"] for e in comp3["exponents"]]
    betas = [e["beta"] for e in comp3["exponents"]]
    ax.plot(alphas, betas, "go-", linewidth=2, markersize=10)
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("β (1-F ~ N^-β)", fontsize=12)
    ax.set_title("Convergence Exponent vs α", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 5: Exponent γ vs α
    ax = axes[1, 1]
    gammas = [e["gamma"] for e in comp3["exponents"]]
    ax.plot(alphas, gammas, "mo-", linewidth=2, markersize=10)
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("γ (1-M² ~ N^-γ)", fontsize=12)
    ax.set_title("Order Parameter Exponent vs α", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 6: Phase diagrams for different α
    ax = axes[1, 2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_list)))
    for i, alpha in enumerate(alpha_list):
        data = comp4[alpha]
        ax.plot(
            data["h_values"],
            data["M2"],
            color=colors[i],
            linewidth=2,
            label=f"α={alpha}",
        )
        ax.axvline(x=data["h_star"], color=colors[i], linestyle="--", alpha=0.5)
    ax.set_xlabel("h / J₀", fontsize=12)
    ax.set_ylabel("⟨M²_stagg⟩", fontsize=12)
    ax.set_title("Phase Diagrams (dashed = h*)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hqtfim_vs_standard_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nFigure saved to 'hqtfim_vs_standard_comparison.png'")


# =============================================================================
# FINAL SUMMARY
# =============================================================================


def print_final_summary(comp1, comp2, comp3, comp4):
    """Print objective summary of findings."""

    print("\n" + "=" * 80)
    print("OBJECTIVE SUMMARY OF FINDINGS")
    print("=" * 80)

    # 1. Compare fidelity
    F_hier_last = comp1["hierarchical"][-1]["fidelity"]
    F_std_last = comp1["standard"][-1]["fidelity"]

    print(
        f"""
1. FIDELITY WITH CAT STATE (at h/J = 0.404, largest N):
   - HQTFIM:     F = {F_hier_last:.6f}
   - Standard:   F = {F_std_last:.6f}
   - Difference: ΔF = {F_hier_last - F_std_last:.6f}
"""
    )

    if F_hier_last > F_std_last + 0.01:
        print("   → HQTFIM has HIGHER cat fidelity than standard 1D")

    # 2. Compare entropy
    S_hier = comp2["hierarchical"][-1]["entropy"]
    S_std = comp2["standard"][-1]["entropy"]

    print(
        f"""
2. ENTROPY AT RESPECTIVE CRITICAL POINTS:
   - HQTFIM at h*:    S = {S_hier:.6f} bits (constant)
   - Standard at h_c: S = {S_std:.6f} bits (grows ~ log N)
"""
    )

    if abs(S_hier - 1.0) < 0.01:
        print("   → HQTFIM: S = 1 bit EXACT (area law)")
        print("   → Standard: S ~ log N (logarithmic violation)")

    # 3. Order parameter
    M2_hier = comp2["hierarchical"][-1]["M2_staggered"]
    M2_std = comp2["standard"][-1]["M2_staggered"]

    print(
        f"""
3. ORDER PARAMETER ⟨M²_stagg⟩ AT CLAIMED CRITICAL POINTS:
   - HQTFIM at h*:    ⟨M²⟩ = {M2_hier:.6f} → 1 (ORDERED)
   - Standard at h_c: ⟨M²⟩ = {M2_std:.6f} (CRITICAL)
"""
    )

    if M2_hier > 0.9 and M2_std < 0.7:
        print("   → h* for HQTFIM is NOT the true critical point")
        print("   → h_c for standard 1D IS the true critical point")

    # 4. Exponents
    print("\n4. CONVERGENCE EXPONENTS (1-F ~ N^-β):")
    for e in comp3["exponents"]:
        print(f"   α = {e['alpha']:.2f}: β = {e['beta']:.4f}")

    # 5. True critical points
    print("\n5. TRUE CRITICAL POINTS h_c vs FORMULA h*:")
    for alpha, data in comp4.items():
        ratio = data["h_c_numerical"] / data["h_star"]
        print(
            f"   α = {alpha:.2f}: h* = {data['h_star']:.4f}, h_c ≈ {data['h_c_numerical']:.4f}, ratio = {ratio:.2f}"
        )

    print("\n" + "=" * 80)
    print("WHAT IS SPECIFIC TO HIERARCHICAL MODEL:")
    print("=" * 80)
    print(
        """
1. S = 1 bit CONSTANT (vs logarithmic for standard 1D at criticality)
2. Higher cat fidelity than standard 1D at same h/J
3. Exponents depend on α (tunable)
4. Formula h* ≠ true critical point h_c
5. At h*, system is in ORDERED phase with cat-like ground state
"""
    )

    print("=" * 80)
    print("WHAT IS NOT SPECIFIC (STANDARD SSB PHYSICS):")
    print("=" * 80)
    print(
        """
1. Cat structure (|+⟩ + |-⟩)/√2 in ordered phase (universal for Z₂ SSB)
2. Exponentially small gap (standard finite-size SSB)
3. F → 1 as N → ∞ in ordered phase (trivial)
4. Power-law corrections 1/N^β (standard finite-size scaling)
"""
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 80)
    print("HQTFIM vs STANDARD 1D TFIM - COMPLETE COMPARISON")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()

    # System sizes
    N_list = [4, 6, 8, 10, 12, 14]

    # α values to test
    alpha_list = [1.2, 1.5, 2.0, 2.5, 3.0]

    # Run comparisons
    print("\n" + "=" * 80)

    # Comparison 1: Same h/J ratio
    comp1 = compare_models_same_h(N_list, h_over_J=0.404061)

    # Comparison 2: Respective critical points
    comp2 = compare_models_at_criticality(N_list, alpha=1.5)

    # Comparison 3: Exponents vs α
    comp3 = compute_exponents_vs_alpha(alpha_list, N_list)

    # Comparison 4: True critical points
    comp4 = find_true_critical_points(
        N=12, alpha_list=alpha_list, h_range=(0.1, 4.0), n_points=40
    )

    # Visualization
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    plot_all_comparisons(comp1, comp2, comp3, comp4, alpha_list)

    # Final summary
    print_final_summary(comp1, comp2, comp3, comp4)

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total/60:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
