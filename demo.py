"""End-to-end demo of closest-value unstructured quantum search.

Searches for the index i whose value f(i) = (A*i^2 + B*i + C) mod 2^m is
closest to a query target, using Grover search inside the Dürr–Høyer
minimum-finding loop. Prints the value table, the threshold evolution
round by round, a final measurement histogram over the closest set, and a
comparison against the classical exhaustive scan.

Usage:
    python demo.py [--n 3] [--m 4] [--a 2] [--b 3] [--c 1] [--target 6] [--seed 7]
                   [--force-unbounded] [--plot] [--save-plots [DIR]]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile

from closest_search import (
    classical_algebraic_closest,
    classical_closest,
    classical_structured_closest,
    closest_value_search,
    f_values,
    grover_circuit,
    optimal_grover_iterations,
    projected_statevector_bytes,
    total_oracle_qubits,
)


def histogram_of_closest_set(
    n: int, m: int, a: int, b: int, c: int, target: int, best_distance: int,
    simulator: Any, shots: int = 2048, seed: int = 0,
) -> dict[int, int]:
    """Run one showcase Grover circuit marking dist <= best_distance with optimal
    iterations (for visualization after the search loop discovers the minimum).
    """
    values = f_values(n, m, a, b, c)
    n_marked = sum(1 for v in values if abs(v - target) <= best_distance)
    big_n = 2**n
    iterations = optimal_grover_iterations(big_n, n_marked)
    qc = grover_circuit(n, m, a, b, c, target, best_distance + 1, iterations)
    tqc = transpile(qc, simulator)
    counts = simulator.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()
    return {int(k, 2): v for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3, help="index qubits (N = 2^n objects)")
    p.add_argument("--m", type=int, default=4, help="value bits (values mod 2^m)")
    p.add_argument("--a", type=int, default=2)
    p.add_argument("--b", type=int, default=3)
    p.add_argument("--c", type=int, default=1)
    p.add_argument("--target", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force-unbounded", action="store_true", help="Bypass simulator total qubit safety check")
    p.add_argument("--plot", action="store_true", help="Display interactive Seaborn/Matplotlib figures")
    p.add_argument("--save-plots", type=str, nargs="?", const="figures", default=None,
                   help="Directory to save figures (default: figures/)")
    args = p.parse_args()

    n, m, a, b, c, target = args.n, args.m, args.a, args.b, args.c, args.target
    if n < 1 or m < 1:
        p.error("--n and --m must be positive integers (>= 1)")
    if target < 0 or target >= 2**m:
        p.error(f"--target must be an integer in range [0, {2**m - 1}] for m={m}, got {target}")

    total_q = total_oracle_qubits(n, m)
    mem_mb = projected_statevector_bytes(n, m) / (1024 * 1024)
    if total_q > 26 and not args.force_unbounded:
        p.error(
            f"Total simulated circuit qubits ({total_q} qubits = {n} idx + {2*m+1} aux) "
            f"exceeds safety limit of 26 qubits (~{mem_mb:.1f} MB statevector). "
            f"Pass --force-unbounded to override."
        )

    big_n, mod = 2**n, 2**m
    rng = np.random.default_rng(args.seed)

    try:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
    except ImportError:
        print(
            "Error: qiskit-aer is required to execute the quantum circuit simulation.\n"
            "Install simulation dependencies via:\n"
            "    pip install 'durr-hoyer-quantum-search[sim]'\n"
            "or:\n"
            "    pip install qiskit-aer",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("=" * 64)
    print("Closest-value unstructured quantum search (computed oracle)")
    print("=" * 64)
    print(f"f(i) = ({a}*i^2 + {b}*i + {c}) mod {mod},   N = {big_n} objects")
    print(f"target t = {target} | simulated circuit: {total_q} qubits (~{mem_mb:.2f} MB statevector)\n")

    values = f_values(n, m, a, b, c)
    print(f"{'i':>4} {'f(i)':>6} {'|f(i)-t|':>9}")
    for i, v in enumerate(values):
        print(f"{i:>4} {v:>6} {abs(v - target):>9}")

    print("\n--- Dürr–Høyer search ---")
    result = closest_value_search(
        n, m, a, b, c, target, rng=rng, simulator=simulator, force_unbounded=args.force_unbounded
    )
    print(f"{'round':>6} {'threshold':>10} {'grover its':>11} {'measured i':>11} "
          f"{'dist':>5}  improved")
    for r_i, r in enumerate(result.rounds):
        print(f"{r_i:>6} {r.threshold:>10} {r.grover_iterations:>11} "
              f"{r.measured_index:>11} {r.measured_distance:>5}  "
              f"{'yes' if r.improved else 'no'}")

    print(f"\nquantum answer          : index {result.best_index}, "
          f"f = {values[result.best_index]}, distance = {result.best_distance}")
    dh_bound = 11.25 * math.sqrt(big_n) + 0.7 * (n**2)
    sim_ceiling = math.ceil(15 * math.sqrt(big_n)) + 10
    print(f"oracle queries          : {result.oracle_queries} "
          f"(paper expected bound ≤ 11.25√N + 0.7 log²N ≈ {dh_bound:.1f}; "
          f"this run stops at distance 0 or {sim_ceiling} queries)")

    best_d, best_set = classical_closest(n, m, a, b, c, target)
    struct_d, struct_set, struct_evals = classical_structured_closest(n, m, a, b, c, target)
    alg_rec = classical_algebraic_closest(n, m, a, b, c, target)

    print(f"\nclassical black-box scan: distance = {best_d}, argmin set = {best_set} "
          f"({big_n} evaluations)")
    print(f"classical period solver : distance = {struct_d}, argmin set = {struct_set} "
          f"({struct_evals} evaluations via periodicity mod {mod})")
    print(f"classical algebraic     : distance = {alg_rec.min_distance}, argmin set = {alg_rec.argmin_indices} "
          f"({alg_rec.delta_layers_tested} layers, {alg_rec.congruence_evaluations} checks, "
          f"{alg_rec.hensel_branches_explored} branch visits, ~{alg_rec.estimated_bit_ops:,} bit-ops)")

    verdict = "CORRECT" if result.best_distance == best_d else "SUBOPTIMAL"
    print(f"verdict                 : {verdict}")

    showcase_dist = result.best_distance
    quantum_closest_set = [i for i, v in enumerate(values) if abs(v - target) <= showcase_dist]

    if verdict == "SUBOPTIMAL":
        print(f"\n--- Showcase histogram (amplification of quantum candidate basin: dist <= {showcase_dist}) ---")
    else:
        print("\n--- Showcase histogram (amplification demo after threshold discovery) ---")

    hist = histogram_of_closest_set(
        n, m, a, b, c, target, showcase_dist, simulator, seed=args.seed
    )
    shots = sum(hist.values())
    for i, cnt in hist.items():
        bar = "#" * round(40 * cnt / shots)
        if i in best_set:
            star = " <-- global optimum" if verdict == "SUBOPTIMAL" else " <-- closest"
        elif i in quantum_closest_set:
            star = " <-- quantum candidate"
        else:
            star = ""
        print(f"  i={i:<3} {cnt:>5} ({cnt / shots:5.1%}) {bar}{star}")

    if args.plot or args.save_plots:
        try:
            from closest_search.plotting import (
                plot_durr_hoyer_trajectory,
                plot_oracle_pipeline,
                plot_quantum_amplitudes,
            )
        except ImportError:
            print(
                "Error: matplotlib and seaborn are required for visualization.\n"
                "Install plotting dependencies via:\n"
                "    pip install 'durr-hoyer-quantum-search[plot]'\n"
                "or:\n"
                "    pip install matplotlib seaborn",
                file=sys.stderr,
            )
            raise SystemExit(1)

        save_dir = Path(args.save_plots) if args.save_plots else None
        pipeline_path = save_dir / "oracle_pipeline.png" if save_dir else None
        traj_path = save_dir / "durr_hoyer_trajectory.png" if save_dir else None
        amp_path = save_dir / "grover_amplitudes.png" if save_dir else None

        plot_oracle_pipeline(save_path=pipeline_path, show=args.plot)
        plot_durr_hoyer_trajectory(result, values, target, save_path=traj_path, show=args.plot)
        plot_quantum_amplitudes(
            hist,
            best_set=quantum_closest_set,
            target=target,
            n=n,
            m=m,
            global_optima_set=best_set,
            save_path=amp_path,
            show=args.plot,
        )


if __name__ == "__main__":
    main()

