"""NISQ gate complexity modeling and record computation.

Provides unified, typed data models and computation routines for NISQ gate counts,
Grover and Dürr–Høyer iteration accounting, and classical complexity baselines.
Acts as the single source of truth for both CLI scaling tables (scaling.py)
and visual figures (closest_search/plotting.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from qiskit import transpile

from .circuits import diffuser, distance_oracle, qrom_distance_oracle
from .ftqc import classical_bit_ops_per_eval
from .search import f_values


@dataclass(frozen=True)
class NISQScalingRecord:
    """Computed NISQ gate scaling metrics for a single index dimension n."""

    n: int
    m: int
    big_n: int
    oracle_gates: int
    cnot_gates: int
    diff_gates: int
    iter_gates: int
    grover_iters: int
    grover_total_q_gates: int
    dh_expected_queries: float
    dh_total_q_gates: int
    c_blackbox_bit_ops: int
    c_blackbox_cpu_ops: int
    grover_oracle_calls: int = 0
    grover_diffuser_calls: int = 0
    dh_expected_quantum_queries: float = 0.0
    dh_expected_classical_verifications: int = 0
    verification_queries: int = 1

    @property
    def oracle_str(self) -> str:
        """Formatted gate count with CNOT breakdown."""
        return f"{self.oracle_gates:,} ({self.cnot_gates:,})"


@dataclass(frozen=True)
class PeriodicScalingRecord:
    """Classical period-reduction metrics for fixed value register m."""

    n: int
    m: int
    big_n: int
    mod_fixed: int
    c_bb_evals: int
    c_st_evals: int
    c_bb_ops: int
    c_st_ops: int


@dataclass(frozen=True)
class QROMComparisonRecord:
    """Gate complexity comparison between explicit QROM lookup and coherent arithmetic."""

    n: int
    m: int
    big_n: int
    qrom_gates: int
    comp_gates: int
    ratio: float


def optimal_grover_iterations(big_n: int, marked_count: int) -> int:
    """Exact optimal Grover iteration count for finite search spaces.

    Evaluates the exact success probability P(R) = sin^2((2R+1) * theta / 2)
    where theta = 2 * asin(sqrt(k / N)) for discrete non-negative integer candidate
    rotations around the continuous optimum R* = (pi - theta) / (2*theta),
    returning argmax_R P(R).

    When k > N/2, doing 0 iterations (R=0) is optimal (success probability k/N > 0.5),
    preventing destructive state over-rotation.
    """
    if big_n < 1 or marked_count < 1 or marked_count > big_n:
        raise ValueError(f"Invalid inputs: big_n={big_n}, marked_count={marked_count}")
    if marked_count == big_n:
        return 0

    theta = 2.0 * math.asin(math.sqrt(marked_count / big_n))
    r_star = (math.pi - theta) / (2.0 * theta)
    cand_floor = max(0, math.floor(r_star))
    cand_ceil = max(0, math.ceil(r_star))
    candidates = {0, cand_floor, cand_ceil}

    def success_prob(r: int) -> float:
        return math.sin((2 * r + 1) * theta / 2.0) ** 2

    return max(candidates, key=success_prob)


def compute_nisq_scaling_records(
    max_n: int = 12,
    a: int = 2,
    b: int = 3,
    c: int = 1,
    target: int = 6,
    threshold: int = 1,
    basis: Sequence[str] = ("u", "cx"),
    seed_transpiler: int = 42,
) -> list[NISQScalingRecord]:
    """Compute NISQ gate complexity records across n in [2, max_n].

    Args:
        max_n: Maximum number of index qubits (N = 2^n).
        a, b, c: Polynomial coefficients for f(i) = (a*i^2 + b*i + c) mod 2^m.
        target: Query target value t.
        threshold: Proximity threshold (k=1 unique match when m = n+1).
        basis: Target transpiler basis gates (default: ['u', 'cx']).
        seed_transpiler: Random seed for deterministic transpilation.

    Returns:
        List of NISQScalingRecord dataclasses for n = 2..max_n.
    """
    if max_n < 2:
        raise ValueError(f"max_n must be at least 2, got {max_n}")

    records: list[NISQScalingRecord] = []
    basis_list = list(basis)

    for n in range(2, max_n + 1):
        m = n + 1
        big_n = 2**n

        # Explicitly count marked states satisfying the proximity threshold condition
        values = f_values(n, m, a, b, c)
        marked_count = sum(1 for v in values if abs(v - target) < threshold)
        if marked_count == 0:
            raise ValueError(
                f"No states satisfy |f(i) - {target}| < {threshold} on [0, {big_n}) for m={m}; "
                "single-run Grover count undefined."
            )

        oracle = distance_oracle(n, m, a, b, c, target, threshold)
        t_oracle = transpile(
            oracle, basis_gates=basis_list, optimization_level=1, seed_transpiler=seed_transpiler
        )
        oracle_ops = t_oracle.count_ops()
        oracle_gates = sum(oracle_ops.values())
        cnot_gates = oracle_ops.get("cx", 0)

        t_diff = transpile(
            diffuser(n), basis_gates=basis_list, optimization_level=1, seed_transpiler=seed_transpiler
        )
        diff_gates = sum(t_diff.count_ops().values())

        iter_gates = oracle_gates + diff_gates
        grover_iters = optimal_grover_iterations(big_n, marked_count)
        grover_total_q_gates = grover_iters * iter_gates

        dh_expected_queries = (45.0 / 4.0) * math.sqrt(big_n) + 0.7 * (n**2)
        dh_total_q_gates = round(dh_expected_queries * iter_gates)

        eval_bit_ops = classical_bit_ops_per_eval(n, m)
        c_blackbox_bit_ops = big_n * eval_bit_ops
        c_blackbox_cpu_ops = big_n * 3

        records.append(
            NISQScalingRecord(
                n=n,
                m=m,
                big_n=big_n,
                oracle_gates=oracle_gates,
                cnot_gates=cnot_gates,
                diff_gates=diff_gates,
                iter_gates=iter_gates,
                grover_iters=grover_iters,
                grover_total_q_gates=grover_total_q_gates,
                dh_expected_queries=dh_expected_queries,
                dh_total_q_gates=dh_total_q_gates,
                c_blackbox_bit_ops=c_blackbox_bit_ops,
                c_blackbox_cpu_ops=c_blackbox_cpu_ops,
                grover_oracle_calls=grover_iters,
                grover_diffuser_calls=grover_iters,
                dh_expected_quantum_queries=dh_expected_queries,
                dh_expected_classical_verifications=max(1, math.ceil(math.log2(big_n))),
                verification_queries=1,
            )
        )

    return records


def compute_periodic_scaling_records(
    max_n: int = 8,
    m_fixed: int = 4,
    a: int = 2,
    b: int = 3,
    c: int = 1,
) -> list[PeriodicScalingRecord]:
    """Compute classical evaluation complexity under modular periodicity f(i + 2^m) = f(i).

    Args:
        max_n: Maximum number of index qubits.
        m_fixed: Fixed value register width.
        a, b, c: Polynomial coefficients.

    Returns:
        List of PeriodicScalingRecord dataclasses for n in [4, min(8, max_n)], or [] if max_n < 4.
    """
    if max_n < 1:
        raise ValueError(f"max_n must be positive, got {max_n}")
    if m_fixed < 1:
        raise ValueError(f"m_fixed must be positive, got {m_fixed}")

    records: list[PeriodicScalingRecord] = []
    if max_n < 4:
        return records

    mod_fixed = 2**m_fixed

    for n in range(4, min(9, max_n + 1)):
        big_n = 2**n
        eval_ops = classical_bit_ops_per_eval(n, m_fixed)

        c_bb_evals = big_n
        c_st_evals = min(big_n, mod_fixed)

        c_bb_ops = c_bb_evals * eval_ops
        c_st_ops = c_st_evals * eval_ops

        records.append(
            PeriodicScalingRecord(
                n=n,
                m=m_fixed,
                big_n=big_n,
                mod_fixed=mod_fixed,
                c_bb_evals=c_bb_evals,
                c_st_evals=c_st_evals,
                c_bb_ops=c_bb_ops,
                c_st_ops=c_st_ops,
            )
        )

    return records


def compute_qrom_comparison_records(
    max_n: int = 6,
    a: int = 2,
    b: int = 3,
    c: int = 1,
    target: int = 6,
    threshold: int = 1,
    basis: Sequence[str] = ("u", "cx"),
    seed_transpiler: int = 42,
) -> list[QROMComparisonRecord]:
    """Compute empirical gate counts comparing QROM lookup vs coherent arithmetic oracle.

    Args:
        max_n: Maximum index qubits to test (recommended <= 6 due to QROM scaling).
        a, b, c: Polynomial coefficients.
        target: Query target.
        threshold: Match threshold.
        basis: Target transpiler basis gates.
        seed_transpiler: Deterministic transpilation seed.

    Returns:
        List of QROMComparisonRecord dataclasses for n = 2..min(6, max_n).
    """
    if max_n < 2:
        raise ValueError(f"max_n must be at least 2, got {max_n}")

    records: list[QROMComparisonRecord] = []
    basis_list = list(basis)

    for n_q in range(2, min(7, max_n + 1)):
        m_q = n_q + 1
        big_n_q = 2**n_q
        tbl = f_values(n_q, m_q, a, b, c)

        qrom_qc = qrom_distance_oracle(n_q, m_q, tbl, target, threshold)
        t_qrom = transpile(
            qrom_qc, basis_gates=basis_list, optimization_level=1, seed_transpiler=seed_transpiler
        )
        qrom_gates = sum(t_qrom.count_ops().values())

        comp_qc = distance_oracle(n_q, m_q, a, b, c, target, threshold)
        t_comp = transpile(
            comp_qc, basis_gates=basis_list, optimization_level=1, seed_transpiler=seed_transpiler
        )
        comp_gates = sum(t_comp.count_ops().values())

        ratio = qrom_gates / max(1, comp_gates)

        records.append(
            QROMComparisonRecord(
                n=n_q,
                m=m_q,
                big_n=big_n_q,
                qrom_gates=qrom_gates,
                comp_gates=comp_gates,
                ratio=ratio,
            )
        )

    return records
