"""Fault-tolerant quantum computing (FTQC) cost models and classical operation baselines.

Provides analytical Clifford+T resource proxies for:
  - Discrete QROM lookup tables (Babbush et al. 2018, unary iteration)
  - Continuous rotation synthesis (Ross & Selinger 2016, gridsynth) for QuadraticForm arithmetic
  - Common downstream oracle stages (Draper adders, comparators, and phase kick)
  - Classical bit-operation baselines for polynomial evaluation

Note on Pedagogical Model Scope:
  These models provide an analytical teaching proxy based on literature scaling
  (Ross & Selinger 2016 typical-case 3*log2(1/eps) asymptotic and Babbush et al.
  2018 unary iteration), not compiler-synthesized exact gate layouts. They illustrate
  the qualitative lesson that continuous rotation synthesis overhead can invert
  the NISQ advantage at small-to-medium N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FTQCResourceVector:
    """Multi-dimensional fault-tolerant Clifford+T resource estimate.

    Attributes:
        n: Index register width.
        m: Value register width.
        eps: Target synthesis precision / error budget (0.0 for exact discrete QROM).
        t_count_nominal: Primary analytical T-count proxy.
        continuous_rotations: Number of arbitrary non-Clifford phase rotations synthesized.
        discrete_toffolis: Number of discrete Toffoli gates.
        logical_qubits: Primary register width for the stage/circuit.
        ancilla_qubits: Clean or borrowable ancilla count for the stage.
        stage_scope: Scope of the resource vector ('value_loader' or 'full_distance_oracle').
        model_notes: Description of compilation assumptions.
    """

    n: int
    m: int
    eps: float
    t_count_nominal: int
    continuous_rotations: int
    discrete_toffolis: int
    logical_qubits: int
    ancilla_qubits: int
    stage_scope: str = "value_loader"
    model_notes: str = "Ross-Selinger gridsynth typical scaling proxy"


def classical_bit_ops_per_eval(n: int, m: int) -> int:
    """Estimated classical bit operations to compute (A*i^2 + B*i + C) mod 2^m.

    n-bit index i squared: ~ n^2 bit mults + n adders.
    Multiplication by A: ~ n * m bit ops.
    Linear term B*i + C: ~ m bit ops.
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    return n**2 + n * m + m


def ftqc_rotation_t_count(eps_rot: float) -> int:
    """Analytical T-count proxy to synthesize an arbitrary single-qubit Z-rotation to precision eps_rot.

    Uses the typical-case asymptotic for Clifford+T single-qubit rotation synthesis
    (Ross & Selinger 2016, arXiv:1403.2975):
        T(R_z, eps_rot) ~ ceil(3 * log2(1 / eps_rot))

    Args:
        eps_rot: Target synthesis precision in (0, 1).
    """
    if not (0.0 < eps_rot < 1.0):
        raise ValueError(f"eps_rot must be in (0, 1), got {eps_rot}")
    return math.ceil(3 * math.log2(1.0 / eps_rot))


def ftqc_coherent_loader_t_count(
    n: int, m: int, eps: float = 1e-6, ccphase_decomp: str = "unassisted"
) -> int:
    """Fault-tolerant T-count proxy for coherent QuadraticForm arithmetic loader.

    Accounting & Decomposition Derivation (Analytical Teaching Proxy):
      - In QuadraticForm phase coupling:
        * Linear terms: n * m controlled-phase (CP) gates -> 3 R_z rotations each = 3nm R_z.
        * Diagonal quadratic terms (j = k): n * m CP gates -> 3 R_z rotations each = 3nm R_z.
          (Note: In binary arithmetic x_j^2 = x_j, so compilers / QuadraticFormGate merge
          linear and diagonal terms into nm CP gates. This unmerged proxy bills them separately
          as a conservative analytical upper-bound proxy).
        * Off-diagonal quadratic terms (j < k): n(n-1)/2 * m doubly-controlled phase (CCPhase) gates.
          Under standard unassisted Clifford+T decomposition: 7 R_z rotations each = 7/2 * n(n-1)m R_z
          (or 4 R_z each with 1 clean ancilla).
        * Result register QFT + IQFT: 2 * m(m-1)/2 = m(m-1) CP gates -> 3 R_z rotations each = 3m(m-1) R_z.
      - With standard reversible uncomputation (2x), total non-Clifford rotations:
          Unassisted (default):
            K_total = 2 * [6nm + 7/2 * n(n-1)m + 3m(m-1)] = 12nm + 7n(n-1)m + 6m(m-1)
          Ancilla-assisted (1 ancilla):
            K_total = 2 * [6nm + 2n(n-1)m + 3m(m-1)] = 12nm + 4n(n-1)m + 6m(m-1)
      - Each rotation is synthesized to precision eps_rot = eps / K_total via
        optimal Ross-Selinger gridsynth proxy: ceil(3 * log2(K_total / eps)).

    Sensitivity & Scenarios:
      Treating all rotations as continuous arbitrary angles provides an unmerged analytical proxy.
      Compiling low-order dyadic QFT/quadratic angles (CZ, CS, T) reduces continuous synthesis
      overhead and shifts the crossover point leftward.
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    if ccphase_decomp == "unassisted":
        cc_mult = 7
    elif ccphase_decomp in ("ancilla_assisted", "ancilla"):
        cc_mult = 4
    else:
        raise ValueError(
            f"Unknown ccphase_decomp: '{ccphase_decomp}'. Expected 'unassisted' or 'ancilla_assisted'."
        )

    k_total = 12 * n * m + cc_mult * n * (n - 1) * m + 6 * m * (m - 1)
    if k_total == 0:
        return 0
    cost_per_rot = math.ceil(3 * math.log2(k_total / eps))
    return k_total * cost_per_rot


def ftqc_coherent_loader_resources(
    n: int, m: int, eps: float = 1e-6, ccphase_decomp: str = "unassisted"
) -> FTQCResourceVector:
    """Multi-dimensional FTQC resource vector for the coherent arithmetic value loader stage."""
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    if ccphase_decomp == "unassisted":
        cc_mult = 7
        ancilla_q = 0
    elif ccphase_decomp in ("ancilla_assisted", "ancilla"):
        cc_mult = 4
        ancilla_q = 1
    else:
        raise ValueError(
            f"Unknown ccphase_decomp: '{ccphase_decomp}'. Expected 'unassisted' or 'ancilla_assisted'."
        )

    k_total = 12 * n * m + cc_mult * n * (n - 1) * m + 6 * m * (m - 1)
    cost_per_rot = math.ceil(3 * math.log2(k_total / eps)) if k_total > 0 else 0
    t_nominal = k_total * cost_per_rot

    # Value loader stage scope: index register n + value register m
    logical_q = n + m

    return FTQCResourceVector(
        n=n,
        m=m,
        eps=eps,
        t_count_nominal=t_nominal,
        continuous_rotations=k_total,
        discrete_toffolis=0,
        logical_qubits=logical_q,
        ancilla_qubits=ancilla_q,
        stage_scope="value_loader",
        model_notes=f"Coherent QuadraticForm ({ccphase_decomp} CCPhase) with Ross-Selinger proxy",
    )


def ftqc_qrom_loader_t_count(n: int, m: int, uncompute_mode: str = "reversible") -> int:
    """Fault-tolerant T-count for discrete QROM loader via unary iteration (Babbush et al. 2018).

    Accounting & Assumptions:
      - Selection tree over N = 2^n elements requires N - 1 Toffoli gates (independent of m).
      - Each Toffoli gate is compiled to 4 T gates via measurement-assisted / catalyst state
        distillation (Jones 2013, Gidney 2018; standard unitary Clifford+T is 7 T) with zero
        rotation synthesis error (eps = 0).
      - Under standard reversible uncomputation, Toffoli count doubles:
          T_total = 2 * 4 * (N - 1) = 8 * (2^n - 1)
      - Under one-way / measurement-assisted uncomputation: 4 * (2^n - 1).
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    big_n = 2**n
    if uncompute_mode == "reversible":
        return 8 * (big_n - 1)
    elif uncompute_mode in ("one_way", "measurement"):
        return 4 * (big_n - 1)
    else:
        raise ValueError(
            f"Unknown uncompute_mode: '{uncompute_mode}'. Expected 'reversible', 'one_way', or 'measurement'."
        )


def ftqc_qrom_loader_resources(
    n: int, m: int, uncompute_mode: str = "reversible"
) -> FTQCResourceVector:
    """Multi-dimensional FTQC resource vector for the discrete QROM value loader stage."""
    t_nominal = ftqc_qrom_loader_t_count(n, m, uncompute_mode=uncompute_mode)
    big_n = 2**n
    toffolis = 2 * (big_n - 1) if uncompute_mode == "reversible" else (big_n - 1)

    # QROM loader stage scope: index register n + value register m, plus 1 clean unary ancilla
    logical_q = n + m
    ancilla_q = 1

    return FTQCResourceVector(
        n=n,
        m=m,
        eps=0.0,
        t_count_nominal=t_nominal,
        continuous_rotations=0,
        discrete_toffolis=toffolis,
        logical_qubits=logical_q,
        ancilla_qubits=ancilla_q,
        stage_scope="value_loader",
        model_notes="Discrete QROM unary iteration (Babbush et al. 2018, 4T catalyst Toffoli)",
    )


def ftqc_common_pipeline_t_count(m: int, eps: float = 1e-6) -> int:
    """Fault-tolerant T-count proxy for the common downstream oracle pipeline:
    add_constant(-t), absolute_value(m), IntegerComparator(m), Z(flag), and uncomputation.

    Accounting:
      - add_constant on m+1 qubits: QFT/IQFT (2 * m(m+1)/2 CP = 3m(m+1) R_z) + (m+1) P (1 R_z each) = (3m+1)(m+1) R_z.
      - absolute_value on m qubits: QFT/IQFT (2 * m(m-1)/2 CP = 3m(m-1) R_z) + m CP (3m R_z) = 3m^2 R_z.
      - With compute + uncompute, total rotations: K_common = 2 * [(3m+1)(m+1) + 3m^2] = 2 * (6m^2 + 4m + 1) = 12m^2 + 8m + 2.
      - Rotation synthesis cost: K_common * ceil(3 * log2(K_common / eps)).
      - IntegerComparator on m qubits: ripple-carry comparator with 2*(m-1) Toffolis.
        With compute + uncompute: 4*(m-1) Toffolis = 16*(m-1) T gates (0 if m=1).
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got m={m}")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    k_common = 12 * m**2 + 8 * m + 2
    t_rot = k_common * math.ceil(3 * math.log2(k_common / eps))
    t_toffoli = 16 * max(0, m - 1)
    return t_rot + t_toffoli


def ftqc_full_oracle_resources(
    n: int,
    m: int,
    loader_type: str = "coherent",
    eps: float = 1e-6,
    uncompute_mode: str = "reversible",
    ccphase_decomp: str = "unassisted",
) -> FTQCResourceVector:
    """Multi-dimensional FTQC resource vector for the complete distance oracle."""
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    # Allocate error budget equally between loader and downstream pipeline
    eps_stage = eps / 2.0
    t_downstream = ftqc_common_pipeline_t_count(m, eps=eps_stage)
    k_downstream = 12 * m**2 + 8 * m + 2
    toffoli_downstream = 4 * max(0, m - 1)
    comp_ancillas = max(0, m - 1)

    if loader_type == "coherent":
        t_load = ftqc_coherent_loader_t_count(n, m, eps=eps_stage, ccphase_decomp=ccphase_decomp)
        cc_mult = 7 if ccphase_decomp == "unassisted" else 4
        k_load = 12 * n * m + cc_mult * n * (n - 1) * m + 6 * m * (m - 1)
        toffoli_load = 0
        loader_ancillas = 0 if ccphase_decomp == "unassisted" else 1
        notes = f"Complete distance oracle with coherent arithmetic loader ({ccphase_decomp} CCPhase)"
    elif loader_type == "qrom":
        t_load = ftqc_qrom_loader_t_count(n, m, uncompute_mode=uncompute_mode)
        k_load = 0
        big_n = 2**n
        toffoli_load = 2 * (big_n - 1) if uncompute_mode == "reversible" else (big_n - 1)
        loader_ancillas = 1
        notes = "Complete distance oracle with discrete QROM unary iteration loader"
    else:
        raise ValueError(f"Unknown loader_type: '{loader_type}'. Expected 'coherent' or 'qrom'.")

    t_total = t_load + t_downstream
    total_rotations = k_load + k_downstream
    total_toffolis = toffoli_load + toffoli_downstream
    logical_q = n + 2 * m + 1
    ancilla_q = loader_ancillas + comp_ancillas

    return FTQCResourceVector(
        n=n,
        m=m,
        eps=eps,
        t_count_nominal=t_total,
        continuous_rotations=total_rotations,
        discrete_toffolis=total_toffolis,
        logical_qubits=logical_q,
        ancilla_qubits=ancilla_q,
        stage_scope="full_distance_oracle",
        model_notes=notes,
    )

