"""Tests for the closest-value quantum search toy model.

Statevector-level checks of every circuit building block (cheap at toy
sizes), plus randomized end-to-end runs of the full Durr-Hoyer loop.
"""

import dataclasses
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from closest_search import (
    FTQCResourceVector,
    NISQScalingRecord,
    PeriodicScalingRecord,
    QROMComparisonRecord,
    Round,
    SearchResult,
    absolute_value,
    add_constant,
    classical_algebraic_closest,
    classical_bit_ops_per_eval,
    classical_closest,
    classical_structured_closest,
    closest_value_search,
    compute_nisq_scaling_records,
    compute_periodic_scaling_records,
    compute_qrom_comparison_records,
    diffuser,
    distance,
    distance_oracle,
    f_values,
    ftqc_coherent_loader_resources,
    ftqc_coherent_loader_t_count,
    ftqc_common_pipeline_t_count,
    ftqc_full_oracle_resources,
    ftqc_qrom_loader_resources,
    ftqc_qrom_loader_t_count,
    ftqc_rotation_t_count,
    grover_circuit,
    optimal_grover_iterations,
    projected_statevector_bytes,
    qrom_distance_oracle,
    qrom_value_function,
    total_oracle_qubits,
    value_function,
)


def basis_input(num_qubits: int, value: int, num_value_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for b in range(num_value_bits):
        if (value >> b) & 1:
            qc.x(b)
    return qc


def measured_basis_state(qc: QuantumCircuit) -> int:
    """Assert the statevector is a single computational basis state, return it."""
    sv = Statevector(qc).data
    idx = int(np.argmax(np.abs(sv)))
    assert np.isclose(abs(sv[idx]), 1.0, atol=1e-9), "state is not a basis state"
    return idx


@pytest.mark.parametrize("a,b,c", [(2, 3, 1), (5, 0, 7), (0, 1, 0), (3, 3, 3)])
def test_value_function_computes_f_for_every_index(a, b, c):
    n, m = 3, 4
    for i in range(2**n):
        qc = basis_input(n + m, i, n)
        qc.compose(value_function(n, m, a, b, c), inplace=True)
        out = measured_basis_state(qc) >> n
        assert out == (a * i * i + b * i + c) % 2**m


@pytest.mark.parametrize("k", [1, 5, -3, -16, 13])
def test_add_constant_mod_2n(k):
    nq = 5
    for x in range(2**nq):
        qc = basis_input(nq, x, nq)
        qc.compose(add_constant(nq, k), inplace=True)
        assert measured_basis_state(qc) == (x + k) % 2**nq


def test_absolute_value_after_subtraction():
    """subtract t then absolute_value leaves |v - t| in the lower m bits."""
    m, t = 4, 6
    for v in range(2**m):
        qc = basis_input(m + 1, v, m)  # sign qubit starts 0
        qc.compose(add_constant(m + 1, -t), inplace=True)
        qc.compose(absolute_value(m), inplace=True)
        out = measured_basis_state(qc)
        assert out & (2**m - 1) == abs(v - t)


@pytest.mark.parametrize(
    "a,b,c,target,threshold",
    [(2, 3, 1, 6, 3), (1, 0, 0, 9, 4), (7, 2, 10, 7, 1), (0, 3, 2, 15, 16)],
)
def test_oracle_marks_exactly_the_close_indices(a, b, c, target, threshold):
    n, m = 3, 4
    marked = {
        i for i, v in enumerate(f_values(n, m, a, b, c)) if abs(v - target) < threshold
    }

    oracle = distance_oracle(n, m, a, b, c, target, threshold)
    qc = QuantumCircuit(oracle.num_qubits)
    qc.h(range(n))
    qc.compose(oracle, inplace=True)
    sv = Statevector(qc).data

    amp = 1 / math.sqrt(2**n)
    for idx, amplitude in enumerate(sv):
        i = idx & (2**n - 1)
        ancillas = idx >> n
        if ancillas != 0:
            assert abs(amplitude) < 1e-9, "ancillas not uncomputed"
            continue
        expected = -amp if i in marked else amp
        assert np.isclose(amplitude, expected, atol=1e-9), (i, amplitude)



def test_oracle_with_single_value_qubit():
    """Verify comparator behavior when m = 1 (no ancilla qubits allocated)."""
    n, m, a, b, c, target, threshold = 2, 1, 1, 0, 0, 0, 1
    marked = {
        i for i, v in enumerate(f_values(n, m, a, b, c)) if abs(v - target) < threshold
    }
    oracle = distance_oracle(n, m, a, b, c, target, threshold)
    qc = QuantumCircuit(oracle.num_qubits)
    qc.h(range(n))
    qc.compose(oracle, inplace=True)
    sv = Statevector(qc).data
    amp = 1 / math.sqrt(2**n)
    for idx, amplitude in enumerate(sv):
        i = idx & (2**n - 1)
        ancillas = idx >> n
        if ancillas != 0:
            assert abs(amplitude) < 1e-9, "ancillas not uncomputed"
            continue
        expected = -amp if i in marked else amp
        assert np.isclose(amplitude, expected, atol=1e-9), (i, amplitude)


def test_oracle_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=16, threshold=3)
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=-1, threshold=3)
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=5, threshold=0)
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=5, threshold=17)


def test_diffuser_is_inversion_about_mean():
    n = 3
    big_n = 2**n
    expected = 2 / big_n * np.ones((big_n, big_n)) - np.eye(big_n)
    from qiskit.quantum_info import Operator

    actual = Operator(diffuser(n)).data
    # Allow a global phase difference.
    ratio = actual[np.abs(expected) > 1e-12][0] / expected[np.abs(expected) > 1e-12][0]
    assert np.allclose(actual, ratio * expected, atol=1e-9)


def test_grover_amplifies_single_closest_index():
    """One marked item, optimal iteration count: success probability ~ 1."""
    n, m = 3, 4
    a, b, c, target = 2, 3, 1, 6  # f(1) = 6, unique exact match
    qc = grover_circuit(n, m, a, b, c, target, threshold=1, iterations=2, measure=False)
    probs = Statevector(qc).probabilities(range(n))
    assert probs[1] > 0.9


def test_grover_amplifies_ties_together():
    """Multiple equally-close indices are all amplified by the same oracle."""
    n, m = 3, 4
    a, b, c, target = 1, 12, 10, 7  # exact matches at i in {1, 3}
    best_d, best_set = classical_closest(n, m, a, b, c, target)
    assert best_d == 0 and best_set == [1, 3]

    qc = grover_circuit(
        n, m, a, b, c, target, threshold=1, iterations=1, measure=False
    )
    probs = Statevector(qc).probabilities(range(n))
    assert sum(probs[i] for i in best_set) > 0.99


def test_end_to_end_randomized():
    n, m = 3, 4
    rng = np.random.default_rng(123)
    hits = 0
    trials = 10
    for _ in range(trials):
        a, b, c = (int(rng.integers(0, 2**m)) for _ in range(3))
        target = int(rng.integers(0, 2**m))
        true_best, _ = classical_closest(n, m, a, b, c, target)
        result = closest_value_search(n, m, a, b, c, target, rng=rng)
        assert result.best_distance >= true_best  # never claims better than optimum
        hits += result.best_distance == true_best
    assert hits >= 8, f"only {hits}/{trials} runs found the true minimum"


def test_end_to_end_with_ties():
    n, m = 3, 4
    a, b, c, target = 1, 12, 10, 7  # exact matches at i = 1 and i = 3
    best_d, best_set = classical_closest(n, m, a, b, c, target)
    assert best_d == 0 and best_set == [1, 3]

    rng = np.random.default_rng(7)
    result = closest_value_search(n, m, a, b, c, target, rng=rng)
    assert result.best_distance == 0
    assert result.best_index in best_set


def test_query_count_accounting():
    n, m = 3, 4
    rng = np.random.default_rng(5)
    result = closest_value_search(n, m, 2, 3, 1, 6, rng=rng)
    expected = 1 + sum(r.grover_iterations + 1 for r in result.rounds)
    assert result.oracle_queries == expected


def test_classical_structured_closest_matches_exhaustive():
    """Verify structured solver matches black-box exhaustive results across parameters."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        n = int(rng.integers(2, 6))
        m = int(rng.integers(2, 5))
        a, b, c = (int(rng.integers(0, 2**m)) for _ in range(3))
        target = int(rng.integers(0, 2**m))

        ex_d, ex_set = classical_closest(n, m, a, b, c, target)
        st_d, st_set, st_evals = classical_structured_closest(n, m, a, b, c, target)

        assert ex_d == st_d
        assert ex_set == st_set
        assert st_evals == min(2**n, 2**m)


def test_end_to_end_nonzero_minimum():
    """Verify search succeeds when the closest value has distance > 0 (no exact match)."""
    n, m = 3, 4
    # All values mod 16: f(i) = (2i^2 + 4i + 2) mod 16 gives even values {2, 8, 2, 0, 2, 8, 2, 0}
    # Target 5 has closest values 2 (distance 3) or 8 (distance 3).
    a, b, c, target = 2, 4, 2, 5
    best_d, best_set = classical_closest(n, m, a, b, c, target)
    assert best_d > 0

    rng = np.random.default_rng(42)
    result = closest_value_search(n, m, a, b, c, target, rng=rng)
    assert result.best_distance == best_d
    assert result.best_index in best_set


def test_diffuser_n1():
    """Verify n=1 diffuser acts as inversion about the mean (Z gate)."""
    from qiskit.quantum_info import Operator
    actual = Operator(diffuser(1)).data
    expected = 2 / 2 * np.ones((2, 2)) - np.eye(2)
    non_zero = np.abs(expected) > 1e-12
    ratio = actual[non_zero][0] / expected[non_zero][0]
    assert np.allclose(actual, ratio * expected, atol=1e-9)



@pytest.mark.parametrize("n,m", [(2, 4), (4, 3), (2, 2)])
def test_value_function_varying_dimensions(n, m):
    """Verify value_function computes correctly for n != m."""
    a, b, c = 1, 2, 3
    for i in range(2**n):
        qc = basis_input(n + m, i, n)
        qc.compose(value_function(n, m, a, b, c), inplace=True)
        out = measured_basis_state(qc) >> n
        assert out == (a * i * i + b * i + c) % 2**m


def test_bbht_starts_with_zero_iterations():
    """Verify that on bbht_m = 1.0, the first round applies 0 Grover iterations."""
    n, m = 3, 4
    rng = np.random.default_rng(1)
    result = closest_value_search(n, m, 2, 3, 1, 6, rng=rng)
    assert len(result.rounds) > 0
    assert result.rounds[0].grover_iterations == 0


def test_qrom_value_function_computes_f():
    """Verify QROM value loader computes |i>|f(i)> on all basis states."""
    n, m, a, b, c = 3, 4, 2, 3, 1
    tbl = f_values(n, m, a, b, c)
    for i in range(2**n):
        qc = basis_input(n + m, i, n)
        qc.compose(qrom_value_function(n, m, tbl), inplace=True)
        out = measured_basis_state(qc) >> n
        assert out == tbl[i]


def test_qrom_distance_oracle_matches_computed_oracle():
    """Verify qrom_distance_oracle marks exactly the same indices with exact phase kick."""
    n, m, a, b, c, target, threshold = 3, 4, 2, 3, 1, 6, 3
    tbl = f_values(n, m, a, b, c)

    computed_oracle = distance_oracle(n, m, a, b, c, target, threshold)
    qrom_oracle = qrom_distance_oracle(n, m, tbl, target, threshold)

    qc_comp = QuantumCircuit(computed_oracle.num_qubits)
    qc_comp.h(range(n))
    qc_comp.compose(computed_oracle, inplace=True)
    sv_comp = Statevector(qc_comp).data

    qc_qrom = QuantumCircuit(qrom_oracle.num_qubits)
    qc_qrom.h(range(n))
    qc_qrom.compose(qrom_oracle, inplace=True)
    sv_qrom = Statevector(qc_qrom).data

    assert np.allclose(sv_comp, sv_qrom, atol=1e-9)




def test_n_greater_than_m_regime():
    """Test n = 5 > m = 3 where indices repeat period modulo 2^m."""
    n, m = 5, 3
    a, b, c, target = 2, 1, 3, 4
    best_d, best_set, _ = classical_structured_closest(n, m, a, b, c, target)

    rng = np.random.default_rng(10)
    result = closest_value_search(n, m, a, b, c, target, rng=rng)
    assert result.best_distance == best_d
    assert result.best_index in best_set


def test_budget_clamping_strict():
    """Verify oracle_queries never exceeds max_queries."""
    n, m = 4, 4
    rng = np.random.default_rng(99)
    for max_q in [1, 5, 10, 15]:
        result = closest_value_search(n, m, 2, 3, 1, 6, rng=rng, max_queries=max_q)
        assert result.oracle_queries <= max_q


def test_scaling_golden_gate_counts():
    """Verify transpiled gate scaling for published README rows through canonical nisq.py API."""
    # NISQ Golden Records (n = 2, 3, 4)
    nisq_records = compute_nisq_scaling_records(max_n=4, a=2, b=3, c=1, target=6, threshold=1)
    assert len(nisq_records) == 3

    # n = 2, m = 3
    r2 = nisq_records[0]
    assert r2.n == 2 and r2.m == 3 and r2.big_n == 4
    assert 400 <= r2.oracle_gates <= 650
    assert 180 <= r2.cnot_gates <= 300
    assert r2.diff_gates == 5
    assert r2.iter_gates == r2.oracle_gates + 5
    assert r2.c_blackbox_bit_ops == 52
    assert r2.c_blackbox_cpu_ops == 12

    # n = 3, m = 4
    r3 = nisq_records[1]
    assert r3.n == 3 and r3.m == 4 and r3.big_n == 8
    assert 750 <= r3.oracle_gates <= 1100
    assert 320 <= r3.cnot_gates <= 500
    assert 10 <= r3.diff_gates <= 30
    assert r3.c_blackbox_bit_ops == 200
    assert r3.c_blackbox_cpu_ops == 24

    # n = 4, m = 5
    r4 = nisq_records[2]
    assert r4.n == 4 and r4.m == 5 and r4.big_n == 16
    assert 1200 <= r4.oracle_gates <= 1800
    assert 550 <= r4.cnot_gates <= 850
    assert r4.c_blackbox_bit_ops == 656
    assert r4.c_blackbox_cpu_ops == 48

    # QROM Comparison Golden Records (n = 2, 3, 4)
    qrom_records = compute_qrom_comparison_records(max_n=4, a=2, b=3, c=1, target=6, threshold=1)
    assert len(qrom_records) == 3

    qr2, qr3, qr4 = qrom_records
    assert 450 <= qr2.qrom_gates <= 700
    assert 1200 <= qr3.qrom_gates <= 1900
    assert 3800 <= qr4.qrom_gates <= 6000

    # Invariants: Monotonic scaling & QROM overhead growth
    assert r2.oracle_gates < r3.oracle_gates < r4.oracle_gates
    assert qr2.qrom_gates < qr3.qrom_gates < qr4.qrom_gates
    assert qr4.ratio > qr2.ratio


def test_ftqc_rotation_t_count():
    with pytest.raises(ValueError):
        ftqc_rotation_t_count(0.0)
    with pytest.raises(ValueError):
        ftqc_rotation_t_count(-1e-4)
    with pytest.raises(ValueError):
        ftqc_rotation_t_count(1.0)
    with pytest.raises(ValueError):
        ftqc_rotation_t_count(2.0)

    # For eps_rot = 1e-6: log2(1e6) ~ 19.93156, ceil(3 * 19.93156) = 60
    assert ftqc_rotation_t_count(1e-6) == 60
    # For eps_rot = 1e-10: log2(1e10) ~ 33.21928, ceil(3 * 33.21928) = 100
    assert ftqc_rotation_t_count(1e-10) == 100


def test_ftqc_coherent_loader_t_count_properties():
    # Parameter validation
    with pytest.raises(ValueError):
        ftqc_coherent_loader_t_count(0, 3)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_t_count(2, 0)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_t_count(2, 3, eps=0.0)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_t_count(2, 3, eps=-1e-5)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_t_count(2, 3, eps=1.0)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_t_count(2, 3, eps=1.5)

    # Exact known values
    # n=2, m=3, eps=1e-6: K = 12(2)(3) + 7(2)(1)(3) + 6(3)(2) = 150, eps_rot = 1e-6/150, ceil(3*log2(1.5e8)) = 82 -> 12,300
    assert ftqc_coherent_loader_t_count(2, 3, eps=1e-6) == 12300
    # n=3, m=4, eps=1e-6: K = 12(3)(4) + 7(3)(2)(4) + 6(4)(3) = 384, eps_rot = 1e-6/384, ceil(3*log2(3.84e8)) = 86 -> 33,024
    assert ftqc_coherent_loader_t_count(3, 4, eps=1e-6) == 33024

    # Monotonicity in n
    assert ftqc_coherent_loader_t_count(3, 4, 1e-6) > ftqc_coherent_loader_t_count(2, 4, 1e-6)
    # Monotonicity in m
    assert ftqc_coherent_loader_t_count(3, 5, 1e-6) > ftqc_coherent_loader_t_count(3, 4, 1e-6)
    # Monotonicity in tighter error budget (smaller eps -> more T gates)
    assert ftqc_coherent_loader_t_count(3, 4, 1e-10) > ftqc_coherent_loader_t_count(3, 4, 1e-6)


def test_ftqc_qrom_loader_t_count_properties():
    # Parameter validation
    with pytest.raises(ValueError):
        ftqc_qrom_loader_t_count(0, 3)
    with pytest.raises(ValueError):
        ftqc_qrom_loader_t_count(2, 0)
    with pytest.raises(ValueError):
        ftqc_qrom_loader_t_count(2, 3, uncompute_mode="invalid_mode")

    # Exact known values for reversible mode (8*(2^n - 1))
    assert ftqc_qrom_loader_t_count(2, 3, "reversible") == 8 * (4 - 1)  # 24
    assert ftqc_qrom_loader_t_count(3, 4, "reversible") == 8 * (8 - 1)  # 56
    assert ftqc_qrom_loader_t_count(4, 5, "reversible") == 8 * (16 - 1)  # 120

    # One-way / measurement mode (4*(2^n - 1))
    assert ftqc_qrom_loader_t_count(2, 3, "one_way") == 4 * (4 - 1)  # 12
    assert ftqc_qrom_loader_t_count(3, 4, "measurement") == 4 * (8 - 1)  # 28


def test_ftqc_crossover_range():
    # Small N advantage: QROM is > 50x cheaper
    for n in [2, 3, 4, 5]:
        m = n + 1
        t_coh = ftqc_coherent_loader_t_count(n, m, eps=1e-6)
        t_qrom = ftqc_qrom_loader_t_count(n, m, "reversible")
        ratio = t_coh / t_qrom
        assert ratio > 50.0

    # Crossover occurs at n = 19 to 20 for eps = 1e-6
    # At n = 18, QROM is still cheaper (ratio > 1)
    t_coh_18 = ftqc_coherent_loader_t_count(18, 19, eps=1e-6)
    t_qrom_18 = ftqc_qrom_loader_t_count(18, 19, "reversible")
    assert t_coh_18 > t_qrom_18

    # At n = 20, Coherent arithmetic is cheaper (ratio < 1)
    t_coh_20 = ftqc_coherent_loader_t_count(20, 21, eps=1e-6)
    t_qrom_20 = ftqc_qrom_loader_t_count(20, 21, "reversible")
    assert t_coh_20 < t_qrom_20


def test_ftqc_common_pipeline_t_count():
    with pytest.raises(ValueError):
        ftqc_common_pipeline_t_count(0)
    with pytest.raises(ValueError):
        ftqc_common_pipeline_t_count(4, eps=0.0)
    with pytest.raises(ValueError):
        ftqc_common_pipeline_t_count(4, eps=1.0)

    # Monotonicity in m and 1/eps
    assert ftqc_common_pipeline_t_count(5, 1e-6) > ftqc_common_pipeline_t_count(4, 1e-6)
    assert ftqc_common_pipeline_t_count(4, 1e-10) > ftqc_common_pipeline_t_count(4, 1e-6)


def test_parameter_bounds_and_validations():
    """Verify input validation guards across circuit, search, and cost functions."""
    # Circuits parameter guards
    with pytest.raises(ValueError):
        value_function(0, 3, 1, 1, 1)
    with pytest.raises(ValueError):
        value_function(3, 0, 1, 1, 1)
    with pytest.raises(ValueError):
        add_constant(0, 5)
    with pytest.raises(ValueError):
        absolute_value(0)
    with pytest.raises(ValueError):
        diffuser(0)
    with pytest.raises(ValueError):
        grover_circuit(3, 4, 1, 1, 1, target=6, threshold=2, iterations=-1)
    with pytest.raises(ValueError):
        distance_oracle(0, 3, 1, 1, 1, target=2, threshold=1)
    with pytest.raises(ValueError):
        distance_oracle(3, 0, 1, 1, 1, target=0, threshold=1)
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=-1, threshold=1)
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=16, threshold=1)
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=2, threshold=0)
    with pytest.raises(ValueError):
        distance_oracle(3, 4, 1, 1, 1, target=2, threshold=17)

    # QROM table length and target guards
    with pytest.raises(ValueError):
        qrom_value_function(3, 4, [1, 2, 3])  # Needs 2^3 = 8
    with pytest.raises(ValueError):
        qrom_distance_oracle(0, 4, [0] * 8, target=2, threshold=1)
    with pytest.raises(ValueError):
        qrom_distance_oracle(3, 0, [0] * 8, target=0, threshold=1)
    with pytest.raises(ValueError):
        qrom_distance_oracle(3, 4, [1, 2, 3], target=2, threshold=1)
    with pytest.raises(ValueError):
        qrom_distance_oracle(3, 4, [0] * 8, target=-1, threshold=1)
    with pytest.raises(ValueError):
        qrom_distance_oracle(3, 4, [0] * 8, target=16, threshold=1)
    with pytest.raises(ValueError):
        qrom_distance_oracle(3, 4, [0] * 8, target=2, threshold=0)
    with pytest.raises(ValueError):
        qrom_distance_oracle(3, 4, [0] * 8, target=2, threshold=17)

    # Search parameter and target validation guards
    with pytest.raises(ValueError):
        distance(0, 4, 1, 1, 1, target=2, i=0)
    with pytest.raises(ValueError):
        distance(3, 0, 1, 1, 1, target=2, i=0)
    with pytest.raises(ValueError):
        distance(3, 4, 1, 1, 1, target=-1, i=0)
    with pytest.raises(ValueError):
        distance(3, 4, 1, 1, 1, target=16, i=0)
    with pytest.raises(ValueError):
        f_values(0, 4, 1, 1, 1)
    with pytest.raises(ValueError):
        f_values(25, 4, 1, 1, 1)
    with pytest.raises(ValueError):
        classical_closest(0, 4, 1, 1, 1, target=2)
    with pytest.raises(ValueError):
        classical_closest(25, 4, 1, 1, 1, target=2)
    with pytest.raises(ValueError):
        classical_closest(3, 4, 1, 1, 1, target=-1)
    with pytest.raises(ValueError):
        classical_closest(3, 4, 1, 1, 1, target=16)
    with pytest.raises(ValueError):
        classical_structured_closest(0, 4, 1, 1, 1, target=2)
    with pytest.raises(ValueError):
        classical_structured_closest(25, 4, 1, 1, 1, target=2)
    with pytest.raises(ValueError):
        classical_structured_closest(3, 4, 1, 1, 1, target=-1)
    with pytest.raises(ValueError):
        classical_structured_closest(3, 4, 1, 1, 1, target=16)
    with pytest.raises(ValueError):
        closest_value_search(0, 4, 1, 1, 1, target=2)
    with pytest.raises(ValueError, match="safety ceiling"):
        closest_value_search(25, 4, 1, 1, 1, target=2)
    with pytest.raises(ValueError):
        closest_value_search(3, 4, 1, 1, 1, target=-1)
    with pytest.raises(ValueError):
        closest_value_search(3, 4, 1, 1, 1, target=16)
    with pytest.raises(ValueError):
        closest_value_search(3, 4, 1, 1, 1, target=-1, max_queries=1)
    with pytest.raises(ValueError):
        closest_value_search(3, 4, 1, 1, 1, target=16, max_queries=1)
    with pytest.raises(ValueError):
        closest_value_search(3, 4, 1, 1, 1, target=2, max_queries=0)
    with pytest.raises(ValueError):
        closest_value_search(3, 4, 1, 1, 1, target=2, max_queries=-5)

    # Classical cost model
    with pytest.raises(ValueError):
        classical_bit_ops_per_eval(0, 4)
    with pytest.raises(ValueError):
        classical_bit_ops_per_eval(3, 0)
    assert classical_bit_ops_per_eval(3, 4) == 3**2 + 3 * 4 + 4

    # NISQ scaling computation guards
    with pytest.raises(ValueError):
        compute_nisq_scaling_records(max_n=1)
    with pytest.raises(ValueError):
        # target with no marked states raises ValueError
        compute_nisq_scaling_records(max_n=2, target=999, threshold=1)
    with pytest.raises(ValueError):
        compute_periodic_scaling_records(max_n=0)
    with pytest.raises(ValueError):
        compute_periodic_scaling_records(max_n=5, m_fixed=0)
    with pytest.raises(ValueError):
        compute_qrom_comparison_records(max_n=1)


def test_compute_nisq_scaling_records():
    records = compute_nisq_scaling_records(max_n=4)
    assert len(records) == 3  # n = 2, 3, 4
    for r in records:
        assert isinstance(r, NISQScalingRecord)
        assert r.m == r.n + 1
        assert r.big_n == 2**r.n
        assert r.oracle_gates > 0
        assert r.cnot_gates > 0
        assert r.diff_gates > 0
        assert r.iter_gates == r.oracle_gates + r.diff_gates
        assert r.grover_total_q_gates == r.grover_iters * r.iter_gates
        assert r.dh_total_q_gates == round(r.dh_expected_queries * r.iter_gates)
        assert r.c_blackbox_bit_ops > 0
        assert r.c_blackbox_cpu_ops == 3 * r.big_n
        assert f"{r.oracle_gates:,}" in r.oracle_str

    # Monotonic scaling invariants
    assert records[0].oracle_gates < records[1].oracle_gates < records[2].oracle_gates
    assert records[0].grover_total_q_gates < records[1].grover_total_q_gates < records[2].grover_total_q_gates


def test_compute_periodic_scaling_records():
    # Verify empty list returned for max_n < 4
    assert compute_periodic_scaling_records(max_n=2) == []
    assert compute_periodic_scaling_records(max_n=3) == []

    records = compute_periodic_scaling_records(max_n=6, m_fixed=4)
    assert len(records) == 3  # n = 4, 5, 6
    for pr in records:
        assert isinstance(pr, PeriodicScalingRecord)
        assert pr.m == 4
        assert pr.mod_fixed == 16
        assert pr.c_bb_evals == 2**pr.n
        assert pr.c_st_evals == 16
        assert pr.c_bb_ops >= pr.c_st_ops


def test_compute_qrom_comparison_records():
    records = compute_qrom_comparison_records(max_n=4)
    assert len(records) == 3  # n = 2, 3, 4
    for qr in records:
        assert isinstance(qr, QROMComparisonRecord)
        assert qr.qrom_gates > 0
        assert qr.comp_gates > 0
        assert qr.ratio == qr.qrom_gates / qr.comp_gates

    # Invariant: QROM overhead ratio strictly increases with n
    assert records[0].ratio < records[1].ratio < records[2].ratio


def test_total_oracle_qubits_and_memory():
    assert total_oracle_qubits(3, 1) == 3 + 3  # 6 qubits
    assert total_oracle_qubits(3, 4) == 3 + 2 * 4 + 1  # 12 qubits
    assert total_oracle_qubits(4, 5) == 4 + 2 * 5 + 1  # 15 qubits
    with pytest.raises(ValueError):
        total_oracle_qubits(0, 4)
    with pytest.raises(ValueError):
        total_oracle_qubits(4, 0)

    # Statevector memory: 2^Q * 16 bytes
    assert projected_statevector_bytes(3, 4) == (2**12) * 16  # 65,536 bytes


def test_search_total_qubit_guard():
    # n=10, m=10 -> Q = 10 + 20 + 1 = 31 > 26
    with pytest.raises(ValueError) as excinfo:
        closest_value_search(10, 10, 2, 3, 1, target=6)
    assert "exceeds default safety ceiling of 26 qubits" in str(excinfo.value)


def test_optimal_grover_iterations():
    # Sparse marked cases
    assert optimal_grover_iterations(4, 1) == 1
    assert optimal_grover_iterations(8, 1) == 2
    assert optimal_grover_iterations(16, 1) == 3
    assert optimal_grover_iterations(64, 1) == 6

    # Dense marked cases: k > N/2 must return R=0 to prevent destructive over-rotation
    assert optimal_grover_iterations(4, 3) == 0
    assert optimal_grover_iterations(8, 7) == 0
    assert optimal_grover_iterations(8, 5) == 0

    # Boundary cases
    assert optimal_grover_iterations(4, 2) in (0, 1)  # Equal success probability P=0.5
    assert optimal_grover_iterations(4, 4) == 0

    # Invalid input guards
    with pytest.raises(ValueError):
        optimal_grover_iterations(0, 1)
    with pytest.raises(ValueError):
        optimal_grover_iterations(4, 0)
    with pytest.raises(ValueError):
        optimal_grover_iterations(4, 5)


def test_ftqc_resource_vector():
    coh_res = ftqc_coherent_loader_resources(3, 4, eps=1e-6)
    assert isinstance(coh_res, FTQCResourceVector)
    assert coh_res.n == 3
    assert coh_res.m == 4
    assert coh_res.logical_qubits == 3 + 4  # 7 loader qubits
    assert coh_res.ancilla_qubits == 0
    # K_total = 12(3)(4) + 7(3)(2)(4) + 6(4)(3) = 144 + 168 + 72 = 384
    assert coh_res.continuous_rotations == 12 * 3 * 4 + 7 * 3 * 2 * 4 + 6 * 4 * 3  # 384
    assert coh_res.discrete_toffolis == 0
    assert coh_res.t_count_nominal == 384 * math.ceil(3 * math.log2(384 / 1e-6))  # 33024
    assert coh_res.stage_scope == "value_loader"

    # Ancilla-assisted sensitivity check (4 R_z per CCPhase)
    coh_res_anc = ftqc_coherent_loader_resources(3, 4, eps=1e-6, ccphase_decomp="ancilla_assisted")
    assert coh_res_anc.ancilla_qubits == 1
    assert coh_res_anc.continuous_rotations == 12 * 3 * 4 + 4 * 3 * 2 * 4 + 6 * 4 * 3  # 312
    assert coh_res_anc.t_count_nominal == 312 * math.ceil(3 * math.log2(312 / 1e-6))  # 26520

    qrom_res = ftqc_qrom_loader_resources(3, 4, uncompute_mode="reversible")
    assert isinstance(qrom_res, FTQCResourceVector)
    assert qrom_res.t_count_nominal == 56
    assert qrom_res.logical_qubits == 3 + 4  # 7 loader qubits
    assert qrom_res.ancilla_qubits == 1
    assert qrom_res.discrete_toffolis == 14
    assert qrom_res.continuous_rotations == 0
    assert qrom_res.stage_scope == "value_loader"

    full_oracle_coh = ftqc_full_oracle_resources(3, 4, loader_type="coherent", eps=1e-6)
    assert isinstance(full_oracle_coh, FTQCResourceVector)
    assert full_oracle_coh.logical_qubits == 3 + 2 * 4 + 1  # 12
    assert full_oracle_coh.ancilla_qubits == 3  # comparator ancillae
    assert full_oracle_coh.stage_scope == "full_distance_oracle"
    assert full_oracle_coh.continuous_rotations == 384 + (12 * 16 + 8 * 4 + 2)  # 384 + 226 = 610
    assert full_oracle_coh.discrete_toffolis == 12  # 4 * (m - 1) = 12

    full_oracle_qrom = ftqc_full_oracle_resources(3, 4, loader_type="qrom", uncompute_mode="reversible")
    assert isinstance(full_oracle_qrom, FTQCResourceVector)
    assert full_oracle_qrom.logical_qubits == 3 + 2 * 4 + 1  # 12
    assert full_oracle_qrom.ancilla_qubits == 1 + 3  # 1 QROM unary ancilla + 3 comparator ancillae
    assert full_oracle_qrom.stage_scope == "full_distance_oracle"
    assert full_oracle_qrom.continuous_rotations == 226
    assert full_oracle_qrom.discrete_toffolis == 14 + 12  # 26

    with pytest.raises(ValueError):
        ftqc_coherent_loader_resources(0, 4)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_resources(3, 0)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_resources(3, 4, eps=0.0)
    with pytest.raises(ValueError):
        ftqc_coherent_loader_resources(3, 4, ccphase_decomp="invalid")
    with pytest.raises(ValueError):
        ftqc_coherent_loader_t_count(3, 4, ccphase_decomp="invalid")
    with pytest.raises(ValueError):
        ftqc_full_oracle_resources(0, 4)
    with pytest.raises(ValueError):
        ftqc_full_oracle_resources(3, 0)
    with pytest.raises(ValueError):
        ftqc_full_oracle_resources(3, 4, eps=0.0)
    with pytest.raises(ValueError):
        ftqc_full_oracle_resources(3, 4, loader_type="unsupported")


def test_quadratic_form_gate_term_counts():
    """Verify term classification and rotation count formulas against analytical definitions."""
    # Test formula for multiple (n, m) tuples
    for n, m in [(2, 3), (3, 4), (4, 5), (5, 6)]:
        # Linear terms: n * m
        n_linear = n * m
        # Diagonal terms: n * m
        n_diag = n * m
        # Off-diagonal terms: n(n-1)/2 * m
        n_offdiag = n * (n - 1) // 2 * m
        # QFT pairs: m(m-1)
        n_qft = m * (m - 1)

        # Unassisted rotations per compute: 3*n_linear + 3*n_diag + 7*n_offdiag + 3*n_qft
        k_compute = 3 * n_linear + 3 * n_diag + 7 * n_offdiag + 3 * n_qft
        # 2x for compute + uncompute
        k_total = 2 * k_compute
        assert k_total == 12 * n * m + 7 * n * (n - 1) * m + 6 * m * (m - 1)

        # Downstream: add_constant + absolute_value (each with QFT/IQFT)
        # add_constant(m+1): (3m+1)(m+1) R_z; absolute_value(m): 3m^2 R_z
        k_downstream = 2 * ((3 * m + 1) * (m + 1) + 3 * m * m)
        assert k_downstream == 12 * m**2 + 8 * m + 2


@pytest.mark.parametrize(
    "n,m,a,b,c,target",
    [
        (3, 4, 2, 3, 1, 6),
        (4, 4, 1, 0, 0, 9),
        (3, 5, 3, 1, 7, 12),
        (4, 5, 2, 3, 1, 6),
    ],
)
def test_classical_algebraic_closest(n, m, a, b, c, target):
    bb_d, bb_set = classical_closest(n, m, a, b, c, target)
    alg_rec = classical_algebraic_closest(n, m, a, b, c, target)
    assert alg_rec.min_distance == bb_d
    assert alg_rec.argmin_indices == bb_set
    assert alg_rec.delta_layers_tested > 0

    # Test detailed fields
    assert alg_rec.congruence_evaluations > 0
    assert alg_rec.hensel_branches_explored > 0
    assert alg_rec.estimated_bit_ops > 0


def test_search_data_models():
    rnd = Round(
        threshold=5,
        grover_iterations=1,
        measured_index=2,
        measured_distance=1,
        improved=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rnd.threshold = 1  # type: ignore[misc]

    res = SearchResult(
        best_index=2,
        best_distance=1,
        oracle_queries=3,
        rounds=[rnd],
    )
    assert res.best_index == 2
    assert len(res.rounds) == 1
    assert res.threshold_history == [1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.best_index = 0  # type: ignore[misc]





