"""Classical driver: Durr-Hoyer minimum finding over |f(i) - target|.

The quantum subroutine is Grover search with an unknown number of marked
items, handled with the BBHT (Boyer-Brassard-Hoyer-Tapp) exponential
schedule.  Each Grover iteration is one oracle query; verifying a measured
candidate costs one more query.  Dürr & Høyer (1996, arXiv:quant-ph/9607014)
show that this adaptive search loop finds the minimum with high probability
in Θ(sqrt(N)) expected queries (at most (45/4)sqrt(N) + (7/10)log²N ≈ 11.25 sqrt(N)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import transpile

if TYPE_CHECKING:
    from qiskit_aer import AerSimulator

from .circuits import grover_circuit, projected_statevector_bytes, total_oracle_qubits


def f_values(n: int, m: int, a: int, b: int, c: int) -> list[int]:
    """Classical table of f(i) = (a*i^2 + b*i + c) mod 2^m for all N = 2^n indices.

    Used for classical verification, reporting, and as input to the baseline
    QROM comparison oracle (``qrom_distance_oracle``). The computed quantum
    oracle (``distance_oracle``) computes values coherently and never touches
    this table.
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if n > 20:
        raise ValueError(f"n={n} exceeds table generation limit (2^n > 10^6 elements)")
    return [(a * i * i + b * i + c) % 2**m for i in range(2**n)]


def distance(n: int, m: int, a: int, b: int, c: int, target: int, i: int) -> int:
    """|f(i) - target| -- one classical evaluation of the black box."""
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not 0 <= target < 2**m:
        raise ValueError(f"target must be in [0, {2**m}), got {target}")
    return abs((a * i * i + b * i + c) % 2**m - target)


def classical_closest(
    n: int, m: int, a: int, b: int, c: int, target: int
) -> tuple[int, list[int]]:
    """Exhaustive scan (black-box baseline): returns (min distance, all argmin indices).

    Costs N = 2^n function evaluations, treating f as an opaque black box.
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not 0 <= target < 2**m:
        raise ValueError(f"target must be in [0, {2**m}), got {target}")
    if n > 20:
        raise ValueError(f"n={n} exceeds exhaustive scan limit (2^n > 10^6 elements)")
    dists = [distance(n, m, a, b, c, target, i) for i in range(2**n)]
    best = min(dists)
    return best, [i for i, d in enumerate(dists) if d == best]


def classical_structured_closest(
    n: int, m: int, a: int, b: int, c: int, target: int
) -> tuple[int, list[int], int]:
    """Period-reduction classical baseline for f(i) = (a*i^2 + b*i + c) mod 2^m.

    Exploits modular periodicity: f(i + 2^m) = f(i) mod 2^m. When N = 2^n > 2^m, scanning
    the first 2^m indices evaluates all unique function values in O(min(N, 2^m)) evaluations.

    Returns:
        (min_distance, argmin_indices_in_[0, 2^n), evaluations_performed)
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not 0 <= target < 2**m:
        raise ValueError(f"target must be in [0, {2**m}), got {target}")
    if n > 20:
        raise ValueError(f"n={n} exceeds index expansion limit (2^n > 10^6 elements)")
    eval_count = min(2**n, 2**m)
    period_dists = [distance(n, m, a, b, c, target, i) for i in range(eval_count)]
    best = min(period_dists)

    best_in_period = [i for i, d in enumerate(period_dists) if d == best]
    if n <= m:
        all_best = best_in_period
    else:
        all_best = []
        period_size = 2**m
        for base in best_in_period:
            idx = base
            while idx < 2**n:
                all_best.append(idx)
                idx += period_size
        all_best.sort()

    return best, all_best, eval_count


@dataclass(frozen=True)
class ClassicalAlgebraicRecord:
    """Detailed operational metrics for classical algebraic closest-value search.

    Attributes:
        min_distance: Absolute difference |f(i) - target| discovered.
        argmin_indices: All optimal indices in [0, 2^n).
        delta_layers_tested: Number of proximity layers delta = 0, 1, 2, ... explored.
        congruence_evaluations: Total modular congruence tests performed.
        hensel_branches_explored: Total lifting tree branch nodes visited.
        estimated_bit_ops: Derived software bit-level operations across all lifting steps.
    """

    min_distance: int
    argmin_indices: list[int]
    delta_layers_tested: int
    congruence_evaluations: int
    hensel_branches_explored: int
    estimated_bit_ops: int

    def __iter__(self):
        """Allow backward-compatible unpacking: dist, indices, layers = record."""
        return iter((self.min_distance, self.argmin_indices, self.delta_layers_tested))


def _solve_quadratic_congruence_mod_2m_detailed(
    a: int, b: int, c_prime: int, m: int
) -> tuple[list[int], int, int, int]:
    """Find all roots x in [0, 2^m) satisfying a*x^2 + b*x + c_prime = 0 mod 2^m.

    Uses iterative 2-adic / Hensel-style branch lifting from k=1..m.

    Returns:
        (roots, congruence_evaluations, branches_explored, bit_operations)
    """
    mod = 2**m
    evals = 0
    branches = 0
    bit_ops = 0

    # Base solutions mod 2
    current_solutions: list[int] = []
    for r in (0, 1):
        evals += 1
        bit_ops += 4  # minimal mod 2 evaluation
        if (a * r * r + b * r + c_prime) % 2 == 0:
            current_solutions.append(r)
            branches += 1

    for k in range(1, m):
        next_mod = 2 ** (k + 1)
        next_solutions: list[int] = []
        step = 2**k
        step_bit_cost = (k + 1) * (2 * k + 5)
        for r in current_solutions:
            for mult in (0, 1):
                cand = r + mult * step
                evals += 1
                branches += 1
                bit_ops += step_bit_cost
                if (a * cand * cand + b * cand + c_prime) % next_mod == 0:
                    next_solutions.append(cand)
        current_solutions = next_solutions
        if not current_solutions:
            break

    roots = sorted(set(r % mod for r in current_solutions))
    return roots, evals, branches, bit_ops


def _solve_quadratic_congruence_mod_2m(
    a: int, b: int, c_prime: int, m: int
) -> list[int]:
    """Find all roots x in [0, 2^m) satisfying a*x^2 + b*x + c_prime = 0 mod 2^m."""
    roots, _, _, _ = _solve_quadratic_congruence_mod_2m_detailed(a, b, c_prime, m)
    return roots


def classical_algebraic_closest(
    n: int, m: int, a: int, b: int, c: int, target: int
) -> ClassicalAlgebraicRecord:
    """Algebraic closest-value solver exploiting modular root-finding and Hensel lifting.

    Instead of scanning the 2^n index space, tests candidate values v = target +/- delta
    (for delta = 0, 1, 2, ...) and solves the quadratic congruence
    a*i^2 + b*i + (c - v) = 0 mod 2^m via 2-adic branch lifting.

    Complexity & Worst-Case Bounds:
      - Best/Typical Case: Target value is in the image of f(i), found in delta=0 or small delta
        in O(poly(m)) operations.
      - Worst-Case Layer Bound: When target is far from the image of f(i), may test up to 2^m
        delta layers.
      - Degenerate Branching Bound: For non-coprime coefficients (e.g., a = 0 mod 2), derivative
        degeneracy mod 2 can cause multi-way Hensel branching up to O(2^{m/2}) nodes.
      - Operation Metric: estimated_bit_ops is a derived software bit-level heuristic model
        tracking modular arithmetic cost per branch node.

    Returns:
        ClassicalAlgebraicRecord (unpacks as (min_distance, argmin_indices, layers_tested))
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not 0 <= target < 2**m:
        raise ValueError(f"target must be in [0, {2**m}), got {target}")
    if n > 20:
        raise ValueError(f"n={n} exceeds index expansion limit (2^n > 10^6 elements)")

    mod = 2**m
    big_n = 2**n
    layers_tested = 0
    total_evals = 0
    total_branches = 0
    total_bit_ops = 0

    for delta in range(mod):
        layers_tested += 1
        candidate_values = []
        if (target - delta) >= 0:
            candidate_values.append((target - delta) % mod)
        if delta > 0 and (target + delta) < mod:
            candidate_values.append((target + delta) % mod)

        all_roots_period: set[int] = set()
        for v in candidate_values:
            c_prime = (c - v) % mod
            roots, evals, branches, bit_ops = _solve_quadratic_congruence_mod_2m_detailed(a, b, c_prime, m)
            total_evals += evals
            total_branches += branches
            total_bit_ops += bit_ops
            all_roots_period.update(roots)

        if all_roots_period:
            # Expand period roots to full n-bit index range [0, 2^n)
            all_best: list[int] = []
            period_size = mod
            for base in all_roots_period:
                if base < big_n:
                    idx = base
                    while idx < big_n:
                        all_best.append(idx)
                        idx += period_size
            if all_best:
                all_best.sort()
                return ClassicalAlgebraicRecord(
                    min_distance=delta,
                    argmin_indices=all_best,
                    delta_layers_tested=layers_tested,
                    congruence_evaluations=total_evals,
                    hensel_branches_explored=total_branches,
                    estimated_bit_ops=total_bit_ops,
                )

    # Fallback to period reduction if no roots found within mod
    best, all_best_fallback, _ = classical_structured_closest(n, m, a, b, c, target)
    return ClassicalAlgebraicRecord(
        min_distance=best,
        argmin_indices=all_best_fallback,
        delta_layers_tested=layers_tested,
        congruence_evaluations=total_evals,
        hensel_branches_explored=total_branches,
        estimated_bit_ops=total_bit_ops,
    )


@dataclass(frozen=True)
class Round:
    threshold: int
    grover_iterations: int
    measured_index: int
    measured_distance: int
    improved: bool


@dataclass
class SearchResult:
    best_index: int
    best_distance: int
    oracle_queries: int
    rounds: list[Round] = field(default_factory=list)

    @property
    def threshold_history(self) -> list[int]:
        hist = []
        for r in self.rounds:
            if r.improved:
                hist.append(r.measured_distance)
        return hist


def _run_grover(
    n: int,
    m: int,
    a: int,
    b: int,
    c: int,
    target: int,
    threshold: int,
    iterations: int,
    simulator: AerSimulator,
    seed: int,
    circuit_cache: dict[tuple[int, int], Any] | None = None,
) -> int:
    """Execute one Grover circuit on the simulator, returning the measured index."""
    cache_key = (threshold, iterations)
    if circuit_cache is not None and cache_key in circuit_cache:
        tqc = circuit_cache[cache_key]
    else:
        qc = grover_circuit(n, m, a, b, c, target, threshold, iterations)
        tqc = transpile(qc, simulator)
        if circuit_cache is not None:
            circuit_cache[cache_key] = tqc

    counts = simulator.run(tqc, shots=1, seed_simulator=seed).result().get_counts()
    bitstring = next(iter(counts))
    return int(bitstring, 2)


def closest_value_search(
    n: int,
    m: int,
    a: int,
    b: int,
    c: int,
    target: int,
    rng: np.random.Generator | None = None,
    max_queries: int | None = None,
    simulator: AerSimulator | None = None,
    max_sim_qubits: int = 26,
    force_unbounded: bool = False,
) -> SearchResult:
    """Find argmin_i |f(i) - target| via Durr-Hoyer + BBHT on a simulator.

    Query accounting: the initial random sample costs 1, each Grover run
    costs (iterations + 1) -- one oracle call per iteration plus one to
    verify the measured candidate.  Analytical expected query complexity is
    Θ(sqrt(N)) (at most (45/4)sqrt(N) + (7/10)log²N ≈ 11.25 sqrt(N) queries;
    Dürr & Høyer 1996).  ``max_queries`` defaults to 15*sqrt(N) + 10 as an
    empirical simulation safety ceiling.

    Safety Guard:
      Enforces a total circuit qubit ceiling of ``max_sim_qubits`` (default 26, ~1 GB RAM)
      based on total allocated oracle qubits (n + 2m + 1), protecting against OOM crashes.
      Pass ``force_unbounded=True`` to bypass for high-memory environments.
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not 0 <= target < 2**m:
        raise ValueError(f"target must be in [0, {2**m}), got {target}")

    total_q = total_oracle_qubits(n, m)
    if total_q > max_sim_qubits and not force_unbounded:
        mem_mb = projected_statevector_bytes(n, m) / (1024 * 1024)
        raise ValueError(
            f"Total simulated circuit qubits ({total_q} = {n} idx + {2*m+1} aux) "
            f"exceeds default safety ceiling of {max_sim_qubits} qubits (~{mem_mb:.1f} MB statevector). "
            f"Pass force_unbounded=True to override if your environment has sufficient RAM."
        )

    if max_queries is not None and max_queries < 1:
        raise ValueError(f"max_queries must be at least 1, got {max_queries}")

    if rng is None:
        rng = np.random.default_rng()
    if simulator is None:
        try:
            from qiskit_aer import AerSimulator
        except ImportError as err:
            raise ImportError(
                "qiskit-aer is required for simulator execution. "
                "Install with `pip install 'durr-hoyer-quantum-search[sim]'`"
            ) from err
        simulator = AerSimulator()

    big_n = 2**n
    if max_queries is None:
        # Durr-Hoyer style budget: c * sqrt(N) with headroom for small N.
        max_queries = math.ceil(15 * math.sqrt(big_n)) + 10

    best_index = int(rng.integers(big_n))
    best_distance = distance(n, m, a, b, c, target, best_index)
    queries = 1
    rounds: list[Round] = []
    circuit_cache: dict[tuple[int, int], Any] = {}

    lam = 6 / 5  # BBHT growth factor
    bbht_m = 1.0

    while best_distance > 0 and queries < max_queries:
        remaining = max_queries - queries
        if remaining <= 0:
            break
        # In BBHT (Boyer et al. 1998), choose integer j uniformly from {0, ..., ceil(m)-1}.
        upper_bound = max(1, math.ceil(bbht_m)) - 1
        max_allowed_iterations = min(upper_bound, remaining - 1)

        iterations = int(rng.integers(0, max_allowed_iterations + 1)) if max_allowed_iterations >= 0 else 0
        if iterations == 0:
            # Zero Grover iterations = uniform random sample; still 1 verify query.
            measured = int(rng.integers(big_n))
        else:
            measured = _run_grover(
                n,
                m,
                a,
                b,
                c,
                target,
                best_distance,
                iterations,
                simulator,
                seed=int(rng.integers(2**31)),
                circuit_cache=circuit_cache,
            )
        queries += iterations + 1

        d = distance(n, m, a, b, c, target, measured)
        improved = d < best_distance
        rounds.append(Round(best_distance, iterations, measured, d, improved))

        if improved:
            best_index, best_distance = measured, d
            bbht_m = 1.0  # marked set changed: restart the BBHT schedule
        else:
            bbht_m = min(lam * bbht_m, math.sqrt(big_n))

    return SearchResult(best_index, best_distance, queries, rounds)
