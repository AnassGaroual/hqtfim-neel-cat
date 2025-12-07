"""
hqtfim-neel-cat: tools for comparing the hierarchical TFIM (HQTFIM)
with the standard 1D transverse-field Ising model.

Public API:
- Hamiltonian builders (hierarchical and standard 1D)
- Diagonalization helpers
- Observable computation
- Comparison and plotting utilities used in the paper.
"""

from .hqtfim_compare import (
    v2,
    get_h_star,
    apply_parity,
    project_even_parity,
    neel_indices,
    compute_observables,
    build_H_hierarchical,
    diagonalize_hierarchical,
    build_H_standard_1D,
    diagonalize_standard_1D,
    get_standard_critical_h,
    compare_models_same_h,
    compare_models_at_criticality,
    compute_exponents_vs_alpha,
    find_true_critical_points,
    plot_all_comparisons,
    print_final_summary,
)

__all__ = [
    "v2",
    "get_h_star",
    "apply_parity",
    "project_even_parity",
    "neel_indices",
    "compute_observables",
    "build_H_hierarchical",
    "diagonalize_hierarchical",
    "build_H_standard_1D",
    "diagonalize_standard_1D",
    "get_standard_critical_h",
    "compare_models_same_h",
    "compare_models_at_criticality",
    "compute_exponents_vs_alpha",
    "find_true_critical_points",
    "plot_all_comparisons",
    "print_final_summary",
]

__version__ = "1.0.0"
