"""Gate-cost scaling of the computed-function oracle vs. classical baselines.

Demonstrates how a computed oracle's gate count scales as O(poly(n, m)) = O(poly(log N))
rather than the O(N) gates required by a QROM lookup table.

This script evaluates three key educational regimes:
  1. Unstructured NISQ Regime (m = n + 1): Problem size N = 2^n with target proximity threshold
     producing a verified unique marked state (k=1) across index width n. Demonstrates asymptotic
     gate scaling vs. classical operation models across problem size N.
  2. Periodic NISQ Regime (fixed m = 4, n > m): N = 2^n > 2^m where modular periodicity
     f(i + 2^m) = f(i) mod 2^m allows a period-aware classical solver to evaluate only 2^m states.
  3. Fault-Tolerant Clifford+T Regime & Crossover Ranges: Evaluates analytical T-count scaling
     proxies for discrete QROM (unary iteration, Babbush et al. 2018) vs. continuous rotation synthesis
     (Ross & Selinger 2016 gridsynth) across synthesis error budgets eps in {10^-4, 10^-6, 10^-8, 10^-10}.

Usage:
    python scaling.py [--max-n 12] [--markdown]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from closest_search.ftqc import (
    classical_bit_ops_per_eval,
    ftqc_coherent_loader_t_count,
    ftqc_common_pipeline_t_count,
    ftqc_full_oracle_resources,
    ftqc_qrom_loader_t_count,
    ftqc_rotation_t_count,
)
from closest_search.nisq import (
    compute_nisq_scaling_records,
    compute_periodic_scaling_records,
    compute_qrom_comparison_records,
)

__all__ = [
    "classical_bit_ops_per_eval",
    "ftqc_coherent_loader_t_count",
    "ftqc_common_pipeline_t_count",
    "ftqc_qrom_loader_t_count",
    "ftqc_rotation_t_count",
    "main",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-n", type=int, default=12, help="Max index qubits n (N = 2^n)")
    p.add_argument("--markdown", action="store_true", help="Print GFM tables for README.md")
    p.add_argument("--plot", action="store_true", help="Display interactive Seaborn/Matplotlib figures")
    p.add_argument("--save-plots", type=str, nargs="?", const="figures", default=None,
                   help="Directory to save figures (default: figures/)")
    args = p.parse_args()

    if args.max_n < 2:
        p.error("--max-n must be at least 2")
    if args.max_n > 14 and not args.markdown:
        print(f"Warning: --max-n={args.max_n} involves >= 16k states; transpilation may be slow.\n")

    a, b, c, target, threshold = 2, 3, 1, 6, 1

    if args.markdown:
        print("### Regime 1: Unstructured Index Space ($m = n + 1, N \\le 2^m$)\n")
        print("Gate scaling across index qubits $n$ for $f(i) = (2i^2 + 3i + 1) \\bmod 2^{n+1}$ with target $t=6$, threshold $1$ (exact unique target $k=1$). Compares transpiled NISQ gates under both Single-Run Grover and Dürr–Høyer minimum finding against classical models:\n")
        print("| $n$ | $m$ | $N=2^n$ | Q-Oracle (CNOTs) | Q-Iter Gates | Single-Run Q-Gates (Optimal $R$) | Dürr–Høyer Q-Gates (11.25√N + 0.7log²N) | C-BlackBox Bit-Ops | C-BlackBox CPU-Ops |")
        print("| --: | --: | ------: | ----------------: | -----------: | --------------------------------: | ---------------------------------------: | -----------------: | ------------------: |")
    else:
        print("=" * 104)
        print("REGIME 1: Unstructured Index Space (m = n + 1, N <= 2^m)")
        print(f"Function: f(i) = ({a}i² + {b}i + {c}) mod 2^m,  |f(i) - {target}| < {threshold} (k=1 unique match)")
        print("=" * 104 + "\n")
        header = (
            f"{'n':>3} {'m':>3} {'N=2^n':>7} {'Q-Oracle (CNOTs)':>19} {'Q-Iter Gates':>14} "
            f"{'Grover Q-Gates':>16} {'Dürr-Høyer Q-Gates':>20} {'C-Bit-Ops':>12} {'C-CPU-Ops':>11}"
        )
        print(header)
        print("-" * len(header))

    nisq_records = compute_nisq_scaling_records(
        max_n=args.max_n, a=a, b=b, c=c, target=target, threshold=threshold
    )

    for r in nisq_records:
        if args.markdown:
            print(
                f"| {r.n} | {r.m} | {r.big_n:,} | {r.oracle_str} | {r.iter_gates:,} | "
                f"{r.grover_total_q_gates:,} | {r.dh_total_q_gates:,} | {r.c_blackbox_bit_ops:,} | {r.c_blackbox_cpu_ops:,} |"
            )
        else:
            print(
                f"{r.n:>3} {r.m:>3} {r.big_n:>7,d} {r.oracle_str:>19} {r.iter_gates:>14,} "
                f"{r.grover_total_q_gates:>16,} {r.dh_total_q_gates:>20,} {r.c_blackbox_bit_ops:>12,} {r.c_blackbox_cpu_ops:>11,}"
            )

    if args.markdown:
        print("\n*Notes on accounting:*")
        print("- **Single-Run Grover:** Ideal single-run using exact discrete candidate evaluation $R = \\arg\\max_R \\sin^2((2R+1)\\arcsin\\sqrt{k/N})$ over integers neighboring $R^* = (\\pi - \\theta)/(2\\theta)$ ($\\theta = 2\\arcsin\\sqrt{k/N}$), correctly yielding $R=1$ at $N=4, k=1$ and $R=0$ when $k > N/2$ without destructive over-rotation.")
        print("- **Dürr–Høyer (1996) Gate Envelope:** Proven expected total query complexity (45/4)√N + 0.7 log²N function evaluation calls across randomized rounds (Grover iterations + classical candidate verifications). The reported gate total models an all-quantum gate upper bound ($E[Q] \\times G_{\\text{iter}}$).")
        print("- **Classical CPU-Ops:** Modern word-RAM CPU evaluates (A*i² + B*i + C) mod 2^m in ~3 CPU instructions.")
        print("- **Classical Bit-Ops:** Software bit-level gate model (n² + nm + m).")
    else:
        print("-" * len(header))
        print("\n* Accounting Notes:")
        print("  - Single-Run Grover: Discrete candidate evaluation R = argmax_R sin^2((2R+1)theta/2) around (pi-theta)/(2theta) (k=1 match).")
        print("  - Dürr-Høyer (1996) Gate Envelope: (45/4)√N + 0.7 log²N expected total queries. Gate total models an all-quantum gate upper bound.")
        print("  - C-CPU-Ops: ~3 word instructions per evaluation on 64-bit CPU.")
        print("  - C-Bit-Ops: Elementary bit-level software model (n² + nm + m).")

    if args.markdown:
        print("\n### Regime 2: Periodic Index Space (Fixed $m = 4, n > m$)\n")
        print("When $N > 2^m$, modular periodicity $f(i + 2^m) \\equiv f(i) \\pmod{2^m}$ caps classical evaluations at $2^m = 16$:\n")
        print("| $n$ | $m$ | $N=2^n$ | $2^m$ | C-BlackBox Evals | C-Struct Evals | C-BlackBox Ops | C-Struct Ops |")
        print("| --: | --: | ------: | ----: | ---------------: | -------------: | -------------: | -----------: |")
    else:
        print("\n" + "=" * 104)
        print("REGIME 2: Periodic Index Space (Fixed m = 4, n > m)")
        print("Demonstrates period-aware classical reduction: min(2^n, 2^m) = 2^m evaluations.")
        print("=" * 104 + "\n")
        p_header = (
            f"{'n':>3} {'m':>3} {'N=2^n':>7} {'2^m':>5} {'C-BlackBox Evals':>18} "
            f"{'C-Struct Evals':>16} {'C-BlackBox Ops':>16} {'C-Struct Ops':>14}"
        )
        print(p_header)
        print("-" * len(p_header))

    periodic_records = compute_periodic_scaling_records(max_n=args.max_n, m_fixed=4, a=a, b=b, c=c)
    if not periodic_records:
        if args.markdown:
            print("*Regime 2 omitted: requires $n \\ge 4$ ($N > 2^m$).*")
        else:
            print("  (Regime 2 omitted: requires n >= 4)")
            print("-" * len(p_header))
    else:
        for pr in periodic_records:
            if args.markdown:
                print(
                    f"| {pr.n} | {pr.m} | {pr.big_n:,} | {pr.mod_fixed} | {pr.c_bb_evals:,} | "
                    f"{pr.c_st_evals:,} | {pr.c_bb_ops:,} | {pr.c_st_ops:,} |"
                )
            else:
                print(
                    f"{pr.n:>3} {pr.m:>3} {pr.big_n:>7,d} {pr.mod_fixed:>5} {pr.c_bb_evals:>18,} "
                    f"{pr.c_st_evals:>16,} {pr.c_bb_ops:>16,} {pr.c_st_ops:>14,}"
                )

        if not args.markdown:
            print("-" * len(p_header))

    # Empirical QROM vs. Computed Oracle Comparison
    if args.markdown:
        print("\n### Empirical QROM vs. Computed Oracle Comparison ($n \\le 6$)\n")
        print("Direct gate-level comparison between an explicit value-loading QROM oracle ($|i\\rangle|0\\rangle \\to |i\\rangle|f(i)\\rangle$, optimized with Gray-code traversal) and the coherent arithmetic oracle (`QuadraticForm`), both transpiled to elementary basis $\\{u, cx\\}$ under `optimization_level=1, seed_transpiler=42` using the identical subtraction, absolute value, and comparator pipeline:\n")
        print("| $n$ | $m$ | $N=2^n$ | QROM Oracle Gates | Computed Oracle Gates | Ratio (QROM / Computed) |")
        print("| --: | --: | ------: | -----------------: | --------------------: | -----------------------: |")
    else:
        print("\n" + "=" * 104)
        print("EMPIRICAL QROM vs. COMPUTED ORACLE COMPARISON (n <= 6)")
        print("Gray-code multi-controlled X QROM vs. coherent QFT phase arithmetic ({u, cx}, seed=42).")
        print("=" * 104 + "\n")
        q_header = (
            f"{'n':>3} {'m':>3} {'N=2^n':>7} {'QROM Oracle Gates':>20} "
            f"{'Computed Oracle Gates':>24} {'Ratio':>10}"
        )
        print(q_header)
        print("-" * len(q_header))

    qrom_records = compute_qrom_comparison_records(
        max_n=min(6, args.max_n), a=a, b=b, c=c, target=target, threshold=threshold
    )
    for qr in qrom_records:
        if args.markdown:
            print(
                f"| {qr.n} | {qr.m} | {qr.big_n:,} | {qr.qrom_gates:,} | {qr.comp_gates:,} | {qr.ratio:.2f}x |"
            )
        else:
            print(
                f"{qr.n:>3} {qr.m:>3} {qr.big_n:>7} {qr.qrom_gates:>20,} {qr.comp_gates:>24,} {qr.ratio:>9.2f}x"
            )

    if not args.markdown:
        print("-" * len(q_header))

    # Regime 3: Fault-Tolerant Clifford+T Scaling & Crossover Ranges
    ft_n_values = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20]

    # --- Table 3A: Value Loader Stage ---
    if args.markdown:
        print("\n### Regime 3A: Fault-Tolerant Clifford+T Value Loader Scaling ($T_{\\text{load}}$) & Crossover Ranges\n")
        print(
            "Fault-tolerant Clifford+T cost comparison proxy for the **Value Loader stage** between the discrete selection tree QROM loader "
            "($8(N-1)$ $T$-gates, $\\varepsilon=0$, Babbush et al. 2018 unary iteration) and the coherent arithmetic loader "
            "(`QuadraticForm`, modeled via Ross & Selinger 2016 gridsynth proxy with $CP \\to 3 R_z$ and $CCPhase \\to 7 R_z$ rotations) "
            "across synthesis error budgets $\\varepsilon$. "
            "Both loaders connect to the identical downstream subtraction/comparator pipeline ($T_{\\text{common}} = O(m^2)$):\n"
        )
        print(
            "| $n$ | $m$ | $N=2^n$ | QROM Loader $T$ ($8(N-1)$) | Coherent Loader $T$ ($\\\\varepsilon=10^{-4}$) | "
            "Coherent Loader $T$ ($\\\\varepsilon=10^{-6}$) | Coherent Loader $T$ ($\\\\varepsilon=10^{-8}$) | "
            "Coherent Loader $T$ ($\\\\varepsilon=10^{-10}$) | Loader Ratio ($\\\\varepsilon=10^{-6}$) |"
        )
        print(
            "| --: | --: | ------: | -------------------------: | -------------------------------------------: | "
            "-------------------------------------------: | -------------------------------------------: | "
            "--------------------------------------------: | -------------------------------------: |"
        )
    else:
        print("\n" + "=" * 104)
        print("REGIME 3A: Fault-Tolerant Clifford+T Value Loader Scaling Proxy (Babbush 2018 vs Ross-Selinger 2016)")
        print("Discrete QROM selection tree (8(N-1) T) vs. continuous rotation synthesis for QuadraticForm (7 Rz/CCPhase).")
        print("=" * 104 + "\n")
        ft_header = (
            f"{'n':>3} {'m':>3} {'N=2^n':>9} {'QROM Load T':>13} {'Coh (1e-4)':>12} "
            f"{'Coh (1e-6)':>12} {'Coh (1e-8)':>12} {'Coh (1e-10)':>13}   {'Advantage (1e-6)':<20}"
        )
        print(ft_header)
        print("-" * len(ft_header))

    for n_ft in ft_n_values:
        m_ft = n_ft + 1
        big_n_ft = 2**n_ft
        t_qrom_ft = ftqc_qrom_loader_t_count(n_ft, m_ft, uncompute_mode="reversible")
        t_coh_4 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-4)
        t_coh_6 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-6)
        t_coh_8 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-8)
        t_coh_10 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-10)

        ratio_6 = t_coh_6 / max(1, t_qrom_ft)
        if ratio_6 >= 1.0:
            ratio_str = f"QROM {ratio_6:.1f}x cheaper"
        else:
            ratio_str = f"Coh {1.0/ratio_6:.2f}x cheaper"

        if args.markdown:
            print(
                f"| {n_ft} | {m_ft} | {big_n_ft:,} | {t_qrom_ft:,} | {t_coh_4:,} | "
                f"{t_coh_6:,} | {t_coh_8:,} | {t_coh_10:,} | {ratio_str} |"
            )
        else:
            print(
                f"{n_ft:>3} {m_ft:>3} {big_n_ft:>9,} {t_qrom_ft:>13,} {t_coh_4:>12,} "
                f"{t_coh_6:>12,} {t_coh_8:>12,} {t_coh_10:>13,}   {ratio_str:<20}"
            )

    if not args.markdown:
        print("-" * len(ft_header))

    # --- Table 3B: Full Distance Oracle Stage ---
    if args.markdown:
        print("\n### Regime 3B: Fault-Tolerant Clifford+T Full Distance Oracle Comparison ($T_{\\text{oracle}}$)\n")
        print(
            "Fault-tolerant Clifford+T cost comparison for the **complete Distance Oracle** "
            "($\\text{Value Loader} + \\text{Draper Subtraction} + \\text{Absolute Value} + \\text{Comparator} + \\text{Phase Kick} + \\text{Uncomputation}$), "
            "allocating error budget $\\varepsilon_{\\text{stage}} = \\varepsilon / 2$ equally between value loading and downstream arithmetic:\n"
        )
        print(
            "| $n$ | $m$ | $N=2^n$ | QROM Full Oracle $T$ ($\\\\varepsilon=10^{-6}$) | Coh Full Oracle $T$ ($\\\\varepsilon=10^{-4}$) | "
            "Coh Full Oracle $T$ ($\\\\varepsilon=10^{-6}$) | Coh Full Oracle $T$ ($\\\\varepsilon=10^{-8}$) | "
            "Coh Full Oracle $T$ ($\\\\varepsilon=10^{-10}$) | Oracle Ratio ($\\\\varepsilon=10^{-6}$) |"
        )
        print(
            "| --: | --: | ------: | -------------------------------------------: | ------------------------------------------: | "
            "------------------------------------------: | ------------------------------------------: | "
            "-------------------------------------------: | -------------------------------------: |"
        )
    else:
        print("\n" + "=" * 104)
        print("REGIME 3B: Fault-Tolerant Clifford+T Full Distance Oracle Comparison (Value Loader + Downstream Pipeline)")
        print("Complete oracle T-count including Draper subtraction, absolute value, comparator, and exact uncomputation.")
        print("=" * 104 + "\n")
        ft_full_header = (
            f"{'n':>3} {'m':>3} {'N=2^n':>9} {'QROM Full (1e-6)':>18} {'Coh Full (1e-4)':>16} "
            f"{'Coh Full (1e-6)':>16} {'Coh Full (1e-8)':>16} {'Coh Full (1e-10)':>17}   {'Advantage (1e-6)':<20}"
        )
        print(ft_full_header)
        print("-" * len(ft_full_header))

    for n_ft in ft_n_values:
        m_ft = n_ft + 1
        big_n_ft = 2**n_ft
        res_qrom_6 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="qrom", eps=1e-6)
        res_coh_4 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-4)
        res_coh_6 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-6)
        res_coh_8 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-8)
        res_coh_10 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-10)

        t_qrom_full_6 = res_qrom_6.t_count_nominal
        t_coh_full_4 = res_coh_4.t_count_nominal
        t_coh_full_6 = res_coh_6.t_count_nominal
        t_coh_full_8 = res_coh_8.t_count_nominal
        t_coh_full_10 = res_coh_10.t_count_nominal

        ratio_full_6 = t_coh_full_6 / max(1, t_qrom_full_6)
        if ratio_full_6 >= 1.0:
            ratio_full_str = f"QROM {ratio_full_6:.1f}x cheaper"
        else:
            ratio_full_str = f"Coh {1.0/ratio_full_6:.2f}x cheaper"

        if args.markdown:
            print(
                f"| {n_ft} | {m_ft} | {big_n_ft:,} | {t_qrom_full_6:,} | {t_coh_full_4:,} | "
                f"{t_coh_full_6:,} | {t_coh_full_8:,} | {t_coh_full_10:,} | {ratio_full_str} |"
            )
        else:
            print(
                f"{n_ft:>3} {m_ft:>3} {big_n_ft:>9,} {t_qrom_full_6:>18,} {t_coh_full_4:>16,} "
                f"{t_coh_full_6:>16,} {t_coh_full_8:>16,} {t_coh_full_10:>17,}   {ratio_full_str:<20}"
            )

    if args.markdown:
        print("\n*Key Fault-Tolerant Insights & Model Caveats:*")
        print("- **Dual-Table Reporting:** Table 3A isolates the Value Loader stage ($T_{\\text{load}}$) where architectural divergence occurs; Table 3B captures the complete Distance Oracle ($T_{\\text{oracle}}$).")
        print("- **Analytical Teaching Proxy & Non-Clifford Gate Accounting:** Coherent rotations model an analytical proxy charging 3 $R_z$ per $CP$ and 7 $R_z$ per $CCPhase$ (unassisted), giving $K_{\\text{total}} = 12nm + 7n(n-1)m + 6m(m-1)$. Note that in binary, $x_j^2 = x_j$, so Qiskit's `QuadraticFormGate` merges diagonal quadratic terms with linear terms into $nm$ $CP$ gates; the unmerged closed form provides a clean, conservative analytical proxy for continuous rotation overhead.")
        print("- **Small-N QROM Advantage:** For small-to-medium $N \\le 2^{19}$, QROM is **10× to 100× cheaper** in $T$-count because discrete Toffoli selection trees require zero continuous rotation synthesis, whereas coherent arithmetic synthesizes thousands of non-Clifford rotations.")
        print("- **Analytical Gridsynth Proxy:** Coherent $T$-counts model an analytical proxy by treating all phase angles as arbitrary continuous $R_z$ rotations. Exact low-order dyadic QFT/quadratic angles ($CZ, \\text{Controlled-}S, T$) do not require full $\\varepsilon$-synthesis; compiling exact dyadics or using 1 clean ancilla ($4 R_z$ per $CCPhase$) reduces coherent $T$-count and shifts the crossover leftward.")
        print("- **Toffoli Synthesis Model:** QROM Toffoli cost ($4T$ compute, $8T$ reversible) assumes measurement-assisted / catalyst decomposition (Jones 2013, Gidney 2018); standard unitary Clifford+$T$ Toffoli is $7T$.")
        print("- **FTQC Crossover Range:** Under this analytical unmerged model, the asymptotic polynomial advantage of coherent arithmetic overcomes rotation synthesis overhead at $N \\approx 10^6 \\text{ to } 2 \\times 10^6$ ($n = 20\\text{--}21$).")
    else:
        print("-" * len(ft_full_header))
        print("\n* Key Fault-Tolerant Insights & Model Caveats:")
        print("  - Dual-Table Scope: Table 3A reports T_load; Table 3B reports complete distance oracle T_oracle.")
        print("  - Analytical Teaching Proxy: Coherent loader charges 3 Rz per CP and 7 Rz per CCPhase (K_total = 12nm + 7n(n-1)m + 6m(m-1)).")
        print("  - Small-N QROM Advantage: For N <= 2^19, QROM is substantially cheaper because discrete Toffolis require 0 synthesis error.")
        print("  - Analytical Proxy: Treating all non-Clifford rotations as arbitrary R_z is a proxy upper bound;")
        print("    exact dyadic QFT/quadratic angles (CZ, CS, T) or ancilla assistance would reduce coherent T-count.")
        print("  - Toffoli Model: 4T Toffoli assumes measurement-assisted decomposition (Jones 2013, Gidney 2018).")
        print("  - FTQC Crossover: Coherent arithmetic beats QROM at N ~ 10^6 to 2x10^6 (n = 20-21).")

    if not args.markdown:
        print("\n" + "=" * 104)
        print("PEDAGOGICAL & FAULT-TOLERANCE NOTES")
        print("=" * 104)
        print("1. Unit Mismatch Disclaimer:")
        print("   Comparing transpiled NISQ gates ({u, cx}) against classical bit/word operations")
        print("   illustrates algorithmic scaling, NOT physical runtime speedup. Quantum gate")
        print("   cycles on NISQ/FTQC hardware are vastly slower than classical CPU clock cycles.")
        print("\n2. Fault-Tolerant (FTQC) Accounting:")
        print("   In fault-tolerant architectures, non-Clifford rotations require magic state distillation")
        print("   and surface code QEC. We model T-factory synthesis costs via Ross-Selinger (2016) proxy")
        print("   and unary iteration Toffoli networks via Babbush et al. (2018).")
        print("=" * 104)

    if args.plot or args.save_plots:
        try:
            from closest_search.plotting import (
                plot_ftqc_crossover,
                plot_nisq_scaling,
                plot_oracle_pipeline,
                plot_qrom_vs_coherent_nisq,
            )
        except ImportError:
            print(
                "Error: matplotlib and seaborn are required for visualization.\n"
                "Install plotting dependencies via:\n"
                "    pip install 'durr-hoyer-quantum-search[plot]'\n"
                "or:\n"
                "    pip install matplotlib seaborn"
            )
            return

        save_dir = Path(args.save_plots) if args.save_plots else None
        pipeline_path = save_dir / "oracle_pipeline.png" if save_dir else None
        nisq_path = save_dir / "nisq_scaling.png" if save_dir else None
        qrom_path = save_dir / "qrom_vs_coherent_nisq.png" if save_dir else None
        ftqc_path = save_dir / "ftqc_crossover.png" if save_dir else None

        plot_oracle_pipeline(save_path=pipeline_path, show=args.plot)
        plot_nisq_scaling(
            max_n=args.max_n, save_path=nisq_path, show=args.plot, records=nisq_records
        )
        plot_qrom_vs_coherent_nisq(
            max_n=min(6, args.max_n), save_path=qrom_path, show=args.plot, records=qrom_records
        )
        plot_ftqc_crossover(save_path=ftqc_path, show=args.plot)


if __name__ == "__main__":
    main()




