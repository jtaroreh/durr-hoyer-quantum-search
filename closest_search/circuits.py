"""Circuit builders for closest-value unstructured quantum search.

Every value is *computed* coherently from the index register with arithmetic
primitives, so a single oracle call costs O(poly(n, m)) gates -- polynomial in
log(N) -- rather than the O(N) gates of an explicit lookup-table (QROM) oracle.
This preserves Grover's asymptotic O(poly(log N) * sqrt(N)) gate scaling over
the O(N * sqrt(N)) gate scaling of tabular lookups.

Architectural Pipeline Separation:
---------------------------------
The search oracle is modularized into two distinct stages:
  1. Value Loader:
     - Coherent Arithmetic Loader (`value_function` / `QuadraticForm`): computes
       |i>|0> -> |i>|f(i)> via QFT phase arithmetic in O(n^2 * m) gates.
     - Tabular QROM Loader (`qrom_value_function`): loads precomputed values
       from a lookup table via Gray-code multi-controlled X traversal in O(m * 2^n) gates.
  2. Common Downstream Pipeline:
     - Constant subtraction (`add_constant(-t)`): Draper QFT adder mapping
       (val, sign) -> f(i) - t in (m+1)-bit two's complement.
     - Absolute value (`absolute_value(m)`): sign-controlled two's complement
       negation mapping val -> |f(i) - t|.
     - Threshold comparison (`IntegerComparator`): ripple-carry comparator setting
       flag <- (|f(i) - t| < threshold).
     - Phase kick: Z(flag) flips the phase of marked states.
     - Exact uncomputation: reverses all dirty registers back to |0>.

The "Quadratic Sweet Spot":
---------------------------
Quadratic polynomials f(i) = a*i^2 + b*i + c mod 2^m represent a uniquely favorable
"sweet spot" for coherent quantum arithmetic:
  - When expanding the integer index i = sum_j 2^j x_j, the square i^2 decomposes
    into pairwise products x_j x_k without carry propagation.
  - In QFT basis, each bit product x_j x_k couples directly to result qubit l
    via a 2-qubit controlled-phase (CP) rotation with angle 2*pi * a * 2^(j+k+l) / 2^m.
  - This requires ZERO intermediate carry ancillas and ZERO Toffoli multipliers.
  - In contrast, higher-degree polynomials (e.g. cubic i^3), non-polynomial functions
    (reciprocals 1/x, square roots), or cryptographic hash functions (SHA-256, AES)
    require general reversible arithmetic networks (ripple-carry adders, Wallace-tree
    multipliers) with O(m^2) Toffoli gates and substantial ancilla scratch space.
"""

from __future__ import annotations

import math

from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister

try:
    from qiskit.circuit.library import QuadraticFormGate
except ImportError:
    from qiskit.circuit.library import (
        QuadraticForm as QuadraticFormGate,  # type: ignore[attr-defined]
    )

try:
    from qiskit.circuit.library import QFTGate
except ImportError:
    from qiskit.circuit.library import QFT as QFTGate  # type: ignore[attr-defined]

try:
    from qiskit.circuit.library import IntegerComparatorGate
except ImportError:
    from qiskit.circuit.library import (
        IntegerComparator as IntegerComparatorGate,  # type: ignore[attr-defined]
    )


def value_function(n: int, m: int, a: int, b: int, c: int) -> QuantumCircuit:
    """|i>|0> -> |i>|f(i)>  with  f(i) = (a*i^2 + b*i + c) mod 2^m.

    Built on ``QuadraticFormGate`` (QFT phase arithmetic).  The polynomial in
    the integer i = sum_j 2^j x_j is expanded into a quadratic form over the
    index *bits* x_j:

        a*i^2 = sum_{j,k} a * 2^(j+k) x_j x_k,    b*i = sum_j b * 2^j x_j

    Gate cost is O(n^2 * m): one controlled(-controlled) phase per
    (bit-pair, result-qubit) combination, plus a QFT on the result register.
    Because quadratic terms depend only on pairwise bit products x_j x_k,
    this arithmetic requires zero carry ancillas and no Toffoli multiplier tree
    (the "quadratic sweet spot").
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    mod = 2**m
    quadratic = [[(a * 2 ** (j + k)) % mod for k in range(n)] for j in range(n)]
    linear = [(b * 2**j) % mod for j in range(n)]
    gate = QuadraticFormGate(
        num_result_qubits=m, quadratic=quadratic, linear=linear, offset=c % mod
    )
    idx = QuantumRegister(n, "idx")
    val = QuantumRegister(m, "val")
    qc = QuantumCircuit(idx, val, name=f"f(i)=({a}i²+{b}i+{c}) mod {mod}")
    qc.append(gate, [*idx, *val])
    return qc


def add_constant(num_qubits: int, k: int) -> QuantumCircuit:
    """Draper QFT constant adder: |x> -> |x + k mod 2^num_qubits>.

    Only single-qubit phase rotations between a QFT/inverse-QFT pair, so the
    cost is O(num_qubits^2) regardless of k.  Use a negative k to subtract.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be positive, got {num_qubits}")
    qc = QuantumCircuit(num_qubits, name=f"+{k}" if k >= 0 else str(k))
    qft = QFTGate(num_qubits)
    qc.append(qft, range(num_qubits))
    for j in range(num_qubits):
        qc.p(2 * math.pi * k * 2**j / 2**num_qubits, j)
    qc.append(qft.inverse(), range(num_qubits))
    return qc


def absolute_value(m: int) -> QuantumCircuit:
    """Map an (m+1)-qubit two's-complement register holding d to |d| in its
    lower m qubits, conditioned on the sign qubit (qubit m, the MSB).

    If d >= 0 the lower m bits already equal |d| and nothing happens.  If
    d < 0 they hold d mod 2^m = 2^m - |d|, so an m-bit two's-complement
    negation (bit-flip then +1 mod 2^m), controlled on the sign qubit,
    recovers |d|.  The sign qubit itself is left as-is; the surrounding
    oracle uncomputes it.  Valid whenever |d| < 2^m, which always holds here
    since d = f(i) - t with both operands in [0, 2^m).
    """
    if m < 1:
        raise ValueError(f"m must be positive, got {m}")
    qc = QuantumCircuit(m + 1, name="|d|")
    sign = m
    for j in range(m):
        qc.cx(sign, j)
    # Controlled +1 mod 2^m: an uncontrolled QFT sandwich around controlled
    # phases collapses to the identity when the control is |0>.
    qft = QFTGate(m)
    qc.append(qft, range(m))
    for j in range(m):
        qc.cp(2 * math.pi * 2**j / 2**m, sign, j)
    qc.append(qft.inverse(), range(m))
    return qc


def total_oracle_qubits(n: int, m: int) -> int:
    """Calculate the total number of simulated qubits allocated by the search oracle.

    Register breakdown:
      - idx: n qubits
      - val: m qubits
      - sign: 1 qubit
      - flag: 1 qubit
      - comparator ancillas: max(0, m - 1) qubits
    Total: n + 2m + 1 for m >= 2 (or n + 3 for m = 1).
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    anc_count = max(0, m - 1)
    return n + m + 2 + anc_count


def projected_statevector_bytes(n: int, m: int) -> int:
    """Projected memory footprint in bytes for dense statevector simulation.

    Assumes standard complex128 representation (16 bytes per statevector amplitude).
    """
    q = total_oracle_qubits(n, m)
    return (2**q) * 16


def _build_distance_oracle(
    load_circuit: QuantumCircuit,
    n: int,
    m: int,
    target: int,
    threshold: int,
    oracle_name: str,
) -> QuantumCircuit:
    """Shared downstream compute-phase-uncompute pipeline for distance oracles."""
    idx = QuantumRegister(n, "idx")
    val = QuantumRegister(m, "val")
    sign = QuantumRegister(1, "sign")
    flag = QuantumRegister(1, "flag")

    cmp_gate = IntegerComparatorGate(num_state_qubits=m, value=threshold, geq=False)

    if hasattr(cmp_gate, "num_ancillas") and cmp_gate.num_ancillas > 0:
        anc = AncillaRegister(cmp_gate.num_ancillas, "anc")
        qc = QuantumCircuit(idx, val, sign, flag, anc, name=oracle_name)
        cmp_qubits = [*val, *flag, *anc]
    else:
        qc = QuantumCircuit(idx, val, sign, flag, name=oracle_name)
        cmp_qubits = [*val, *flag]

    sub = add_constant(m + 1, -target)
    absv = absolute_value(m)

    qc.compose(load_circuit, [*idx, *val], inplace=True)
    qc.compose(sub, [*val, *sign], inplace=True)
    qc.compose(absv, [*val, *sign], inplace=True)
    qc.append(cmp_gate, cmp_qubits)

    qc.z(flag)

    qc.append(cmp_gate.inverse(), cmp_qubits)
    qc.compose(absv.inverse(), [*val, *sign], inplace=True)
    qc.compose(sub.inverse(), [*val, *sign], inplace=True)
    qc.compose(load_circuit.inverse(), [*idx, *val], inplace=True)
    return qc


def distance_oracle(
    n: int, m: int, a: int, b: int, c: int, target: int, threshold: int
) -> QuantumCircuit:
    """Grover phase oracle marking every index i with |f(i) - target| < threshold."""
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if not 0 <= target < 2**m:
        raise ValueError(f"target must be in [0, {2**m}), got {target}")
    if not 1 <= threshold <= 2**m:
        raise ValueError(f"threshold must be in [1, {2**m}], got {threshold}")

    load = value_function(n, m, a, b, c)
    return _build_distance_oracle(
        load_circuit=load,
        n=n,
        m=m,
        target=target,
        threshold=threshold,
        oracle_name=f"O(|f-{target}|<{threshold})",
    )


def diffuser(n: int) -> QuantumCircuit:
    """Grover diffusion operator: 2|s><s| - I  on n qubits.

    Standard reflection about the uniform superposition |s> = H^n |0>.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        # Phase flip on |1...1>: H on the last qubit converts MCX into MCP(pi)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(
    n: int,
    m: int,
    a: int,
    b: int,
    c: int,
    target: int,
    threshold: int,
    iterations: int,
    measure: bool = True,
) -> QuantumCircuit:
    """Full Grover search circuit: uniform superposition over indices, then
    ``iterations`` rounds of (distance oracle, diffuser), then measurement of
    the index register.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    oracle = distance_oracle(n, m, a, b, c, target, threshold)
    diff = diffuser(n)

    qc = QuantumCircuit(oracle.num_qubits, n if measure else 0, name=f"Grover(R={iterations})")
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle, range(oracle.num_qubits), inplace=True)
        qc.compose(diff, range(n), inplace=True)

    if measure:
        qc.measure(range(n), range(n))
    return qc


def qrom_value_function(n: int, m: int, f_table: list[int]) -> QuantumCircuit:
    """Tabular QROM lookup oracle: |i>|0> -> |i>|f(i)> via Gray-code multi-controlled X gates.

    Constructs an explicit lookup table storing precomputed classical values f(i).
    Uses Gray-code ordering (g_k = k ^ (k >> 1)) so that exactly ONE control bit
    changes state between consecutive lookup entries, eliminating multi-qubit
    X-decoding sweeps and minimizing gate depth overhead.

    Gate Complexity:
      Requires 2^n multi-controlled NOT gates across m target qubits (Theta(m * 2^n) gates).
      In contrast, coherent arithmetic (`value_function`) requires only O(n^2 * m) gates.
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    total_entries = 2**n
    if len(f_table) < total_entries:
        raise ValueError(f"f_table must contain at least 2^n={total_entries} entries, got {len(f_table)}")

    idx = QuantumRegister(n, "idx")
    val = QuantumRegister(m, "val")
    qc = QuantumCircuit(idx, val, name=f"QROM_Table(N={total_entries})")

    # Initial state: decode g_0 = 0 (all 0s -> flip all n index qubits)
    for b in range(n):
        qc.x(idx[b])

    prev_g = 0
    for k in range(total_entries):
        curr_g = k ^ (k >> 1)  # standard binary reflected Gray code
        diff = prev_g ^ curr_g

        if diff != 0:
            flip_bit = (diff & -diff).bit_length() - 1
            qc.x(idx[flip_bit])

        table_val = f_table[curr_g] % (2**m)
        if table_val > 0:
            for bit in range(m):
                if (table_val >> bit) & 1:
                    if n == 1:
                        qc.cx(idx[0], val[bit])
                    else:
                        qc.mcx(list(idx), val[bit])

        prev_g = curr_g

    # Uncompute Gray code index shift for the final state (g_{2^n-1})
    last_g = (total_entries - 1) ^ ((total_entries - 1) >> 1)
    for b in range(n):
        if not ((last_g >> b) & 1):
            qc.x(idx[b])

    return qc


def qrom_distance_oracle(
    n: int, m: int, f_table: list[int], target: int, threshold: int
) -> QuantumCircuit:
    """Grover phase oracle using an explicit QROM lookup table for value loading.

    Replaces value_function with qrom_value_function while retaining the identical
    subtraction, absolute value, integer comparison, and phase kick pipeline.
    Demonstrates the empirical gate explosion of lookup tables vs. coherent arithmetic.
    """
    if n < 1 or m < 1:
        raise ValueError(f"n and m must be positive integers, got n={n}, m={m}")
    if len(f_table) < 2**n:
        raise ValueError(f"f_table must contain at least 2^n={2**n} entries, got {len(f_table)}")
    if not 0 <= target < 2**m:
        raise ValueError(f"target must be in [0, {2**m}), got {target}")
    if not 1 <= threshold <= 2**m:
        raise ValueError(f"threshold must be in [1, {2**m}], got {threshold}")

    load = qrom_value_function(n, m, f_table)
    return _build_distance_oracle(
        load_circuit=load,
        n=n,
        m=m,
        target=target,
        threshold=threshold,
        oracle_name=f"QROM_O(|f-{target}|<{threshold})",
    )
