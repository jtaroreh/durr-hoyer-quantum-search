"""Visualization routines for the closest-value quantum search demo.

Uses Seaborn and Matplotlib to create publication-quality figures illustrating:
  1. Dürr-Høyer search trajectory and dynamic threshold reduction.
  2. Quantum probability amplification and measurement histograms.
  3. NISQ gate complexity scaling vs. classical operation baselines.
  4. Empirical QROM vs. Coherent arithmetic gate explosion.
  5. Fault-tolerant Clifford+T crossover under Ross-Selinger rotation synthesis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import seaborn as sns

from .ftqc import (
    ftqc_coherent_loader_t_count,
    ftqc_qrom_loader_t_count,
)
from .nisq import (
    NISQScalingRecord,
    QROMComparisonRecord,
    compute_nisq_scaling_records,
    compute_qrom_comparison_records,
)

if TYPE_CHECKING:
    from .search import SearchResult


def setup_plot_theme() -> None:
    """Configure modern, publication-ready styling with Seaborn."""
    sns.set_theme(style="ticks", font_scale=1.1)
    plt.rcParams.update({
        "figure.autolayout": True,
        "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def _save_or_show(fig: plt.Figure, save_path: str | Path | None = None, show: bool = False) -> plt.Figure:
    """Helper to save figure to disk and/or display interactively."""
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {p}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_durr_hoyer_trajectory(
    result: SearchResult,
    f_table: list[int],
    target: int,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot the Dürr-Høyer minimum-finding search trajectory."""
    setup_plot_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    n_states = len(f_table)
    indices = list(range(n_states))
    distances = [abs(v - target) for v in f_table]
    min_dist = min(distances)
    best_indices = [i for i, d in enumerate(distances) if d == min_dist]

    # Panel 1: Search Space Landscape
    palette = ["#2ca02c" if i in best_indices else "#4a7bb0" for i in indices]
    ax1.bar(
        indices,
        distances,
        color=palette,
        edgecolor="black",
        linewidth=1.2,
        width=0.6,
        alpha=0.85,
        zorder=3,
    )
    ax1.axhline(
        min_dist,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label=f"Global Minimum Distance = {min_dist}",
        zorder=4,
    )

    for i in best_indices:
        ax1.annotate(
            f"Global Min\n(i={i}, d={min_dist})",
            (i, distances[i]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color="#1b5e20",
            arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.5),
        )

    ax1.set_title(f"Objective Landscape: $|f(i) - {target}|$ ($N={n_states}$)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Basis State Index $i$", fontsize=11)
    ax1.set_ylabel("Distance $|f(i) - t|$", fontsize=11)
    ax1.set_xticks(indices)
    max_d = max(distances) if distances else 1
    ax1.set_ylim(-0.04 * max_d, max_d * 1.28)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc")

    # Panel 2: Dürr-Høyer Threshold Ladder
    rounds = result.rounds
    if rounds:
        round_nums = list(range(len(rounds)))
        thresholds = [r.threshold for r in rounds]
        measured_dists = [r.measured_distance for r in rounds]
        improved_flags = [r.improved for r in rounds]

        ax2.step(
            round_nums,
            thresholds,
            where="post",
            color="#d62728",
            linewidth=2.2,
            label="Active Threshold $\\lambda_k$",
            zorder=3,
        )
        ax2.plot(
            round_nums,
            thresholds,
            "o",
            color="#d62728",
            markersize=6,
            zorder=4,
        )

        added_imp = False
        added_unimp = False
        for r_idx, (m_dist, imp) in enumerate(zip(measured_dists, improved_flags)):
            pt_color = "#2ca02c" if imp else "#7f7f7f"
            marker = "*" if imp else "x"
            sz = 10 if imp else 7
            lbl = None
            if imp and not added_imp:
                lbl = "Improved Sample"
                added_imp = True
            elif not imp and not added_unimp:
                lbl = "Unimproved Sample"
                added_unimp = True

            ax2.plot(
                r_idx,
                m_dist,
                marker=marker,
                color=pt_color,
                markersize=sz,
                markeredgewidth=2,
                label=lbl,
                zorder=5,
            )

        ax2.axhline(
            result.best_distance,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.8,
            label=f"Discovered Best Distance = {result.best_distance}",
            zorder=2,
        )

        ax2.set_title(f"Dürr–Høyer Threshold Trajectory ({result.oracle_queries} Oracle Queries)", fontsize=13, fontweight="bold", pad=10)
        ax2.set_xlabel("Search Round $k$", fontsize=11)
        ax2.set_ylabel("Threshold / Distance", fontsize=11)
        ax2.set_xticks(round_nums)
        y_max = max(max(thresholds), max(measured_dists), 1)
        ax2.set_ylim(-0.06 * y_max, y_max * 1.52)
        ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc")
    else:
        ax2.text(0.5, 0.5, "Single sample terminated immediately", ha="center", va="center", transform=ax2.transAxes)

    return _save_or_show(fig, save_path, show)


def plot_quantum_amplitudes(
    hist: dict[int, int],
    best_set: list[int] | set[int],
    target: int,
    n: int,
    global_optima_set: list[int] | set[int] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot Grover amplitude amplification measurement probability distribution.

    Args:
        hist: Measurement counts dictionary {basis_index: count}.
        best_set: Set of indices marked/amplified in this circuit run (e.g. quantum candidate basin).
        target: Query target value t.
        n: Number of index qubits.
        global_optima_set: Optional true global minimum indices. If provided and different from
            best_set, distinguishes between the amplified candidate basin and the true global optimum.
        save_path: Optional path to save figure.
        show: If True, display figure interactively.
    """
    setup_plot_theme()
    fig, ax = plt.subplots(figsize=(9, 5))

    big_n = 2**n
    all_indices = list(range(big_n))
    shots = sum(hist.values()) if hist else 1
    probs = [hist.get(i, 0) / shots for i in all_indices]

    marked_indices = set(best_set)
    global_indices = set(global_optima_set) if global_optima_set is not None else marked_indices
    is_suboptimal = marked_indices != global_indices

    colors = []
    for i in all_indices:
        is_marked = i in marked_indices
        is_global = i in global_indices

        if is_marked and is_global:
            colors.append("#2ca02c")
        elif is_marked:
            colors.append("#ff7f0e")
        elif is_global:
            colors.append("#d62728")
        else:
            colors.append("#4a7bb0")

    ax.bar(
        all_indices,
        probs,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        width=0.6,
        alpha=0.9,
        zorder=3,
    )

    uniform_p = 1.0 / big_n
    ax.axhline(
        uniform_p,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Uniform Superposition ($1/N = {uniform_p:.3f}$ / {uniform_p:.1%})",
        zorder=4,
    )

    for idx, p in enumerate(probs):
        is_marked = idx in marked_indices
        is_global = idx in global_indices

        if is_marked and is_global:
            label_text = f"{p:.1%}\n(Closest)"
            text_color = "#1b5e20"
        elif is_marked:
            label_text = f"{p:.1%}\n(Candidate)"
            text_color = "#b25900"
        elif is_global and is_suboptimal:
            label_text = f"{p:.1%}\n(Global Opt)"
            text_color = "#b71c1c"
        else:
            label_text = None

        if label_text is not None and p > 0.005:
            ax.annotate(
                label_text,
                (idx, p),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    title_desc = " [Suboptimal Basin]" if is_suboptimal else ""
    ax.set_title(
        f"Grover Probability Amplification{title_desc} ($N={big_n}$ states, target $t={target}$)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Basis State Index $|i\\rangle$", fontsize=11)
    ax.set_ylabel("Measurement Probability", fontsize=11)
    ax.set_xticks(all_indices)
    max_p = max(probs) if probs else 1.0
    ax.set_ylim(-0.02 * max_p, max_p * 1.28)

    # Place legend in the opposite quadrant of the dominant probability peaks
    peak_indices = [i for i, p in enumerate(probs) if p > max_p * 0.4]
    mean_peak = sum(peak_indices) / len(peak_indices) if peak_indices else 0
    legend_loc: Literal["upper right", "upper left"] = "upper right" if mean_peak < big_n / 2 else "upper left"
    ax.legend(loc=legend_loc, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc")

    return _save_or_show(fig, save_path, show)


def plot_nisq_scaling(
    max_n: int = 12,
    save_path: str | Path | None = None,
    show: bool = False,
    records: list[NISQScalingRecord] | None = None,
) -> plt.Figure:
    """Plot log-log NISQ gate complexity comparing quantum search vs. classical models."""
    setup_plot_theme()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    if records is None:
        records = compute_nisq_scaling_records(max_n=max_n)
    big_n_vals = [r.big_n for r in records]
    grover_gates_list = [r.grover_total_q_gates for r in records]
    dh_gates_list = [r.dh_total_q_gates for r in records]
    c_bit_ops_list = [r.c_blackbox_bit_ops for r in records]
    c_cpu_ops_list = [r.c_blackbox_cpu_ops for r in records]

    ax.loglog(
        big_n_vals,
        grover_gates_list,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        markersize=6,
        label=r"Single-Run Grover Quantum Gates (optimal finite $R$, $k=1$)",
        zorder=4,
    )
    ax.loglog(
        big_n_vals,
        dh_gates_list,
        "s--",
        color="#9467bd",
        linewidth=2.2,
        markersize=6,
        label=r"Dürr–Høyer Quantum Gates ($11.25\sqrt{N} + 0.7\log_2^2 N$ queries)",
        zorder=3,
    )
    ax.loglog(
        big_n_vals,
        c_bit_ops_list,
        "^:",
        color="#d62728",
        linewidth=2.2,
        markersize=6,
        label="Classical Black-Box Bit-Ops ($N(n^2+nm+m)$)",
        zorder=4,
    )
    ax.loglog(
        big_n_vals,
        c_cpu_ops_list,
        "d-.",
        color="#ff7f0e",
        linewidth=2.2,
        markersize=6,
        label="Classical CPU Operations ($3N$ instructions)",
        zorder=3,
    )

    # Annotate Single-Run Grover vs Classical Bit-Ops operation-count intersection (n=11, N=2048)
    rec_11 = next((r for r in records if r.n == 11), None)
    if rec_11 is not None:
        ax.plot(
            [rec_11.big_n],
            [rec_11.grover_total_q_gates],
            marker="*",
            markersize=13,
            color="#e65100",
            markeredgecolor="black",
            markeredgewidth=1.1,
            zorder=6,
        )
        ax.annotate(
            f"Operation-Count Intersection ($n=11, N={rec_11.big_n:,}$)\n"
            f"• Ideal Single-Run Grover: {rec_11.grover_total_q_gates:,} gates\n"
            f"• Classical Black-Box Bit-Ops: {rec_11.c_blackbox_bit_ops:,} ops\n"
            r"$\rightarrow$ Grover < Bit-Ops (yet still $\gg$ CPU Ops: 6,144)",
            xy=(rec_11.big_n, rec_11.grover_total_q_gates),
            xytext=(90, 4e7),
            arrowprops=dict(facecolor="#222222", shrink=0.1, width=1.1, headwidth=5.5),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", edgecolor="#fbc02d", lw=1.2, alpha=0.95),
            zorder=7,
        )

    ax.set_ylim(5, 4e8)
    ax.set_title("NISQ Gate Scaling: Quantum vs. Classical Search Models", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Search Space Size $N = 2^n$", fontsize=11)
    ax.set_ylabel("Total Gate / Operation Count", fontsize=11)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.2)

    return _save_or_show(fig, save_path, show)


def plot_qrom_vs_coherent_nisq(
    max_n: int = 6,
    save_path: str | Path | None = None,
    show: bool = False,
    records: list[QROMComparisonRecord] | None = None,
) -> plt.Figure:
    """Plot NISQ gate explosion in tabular QROM vs. polynomial scaling in coherent arithmetic."""
    setup_plot_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    if records is None:
        records = compute_qrom_comparison_records(max_n=max_n)
    n_vals = [r.n for r in records]
    qrom_gates_list = [r.qrom_gates for r in records]
    comp_gates_list = [r.comp_gates for r in records]
    ratios = [r.ratio for r in records]

    # Panel 1: Absolute Gate Counts (Semilog)
    ax1.semilogy(
        n_vals,
        qrom_gates_list,
        "o-",
        color="#d62728",
        linewidth=2.5,
        markersize=7,
        label="QROM Oracle (Gray-Code $MCX$, $\\Theta(m 2^n)$)",
    )
    ax1.semilogy(
        n_vals,
        comp_gates_list,
        "s-",
        color="#2ca02c",
        linewidth=2.5,
        markersize=7,
        label="Computed Oracle (`QuadraticForm`, $\\mathcal{O}(n^2 m)$)",
    )

    ax1.set_title("NISQ Gate Count: QROM vs. Computed Oracle", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Index Qubits $n$ ($N=2^n$)", fontsize=11)
    ax1.set_ylabel("Transpiled Gates ($\\{u, cx\\}$)", fontsize=11)
    ax1.set_xticks(n_vals)
    ax1.set_xticklabels([f"$n={n}$\n($N={2**n}$)" for n in n_vals])
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)

    # Panel 2: Ratio (QROM / Computed)
    bars = ax2.bar(
        [f"n={n}\n(N={2**n})" for n in n_vals],
        ratios,
        color=sns.color_palette("crest", len(n_vals)),
        edgecolor="black",
        linewidth=1.2,
        width=0.55,
        zorder=3,
    )
    for bar, r in zip(bars, ratios):
        ax2.annotate(
            f"{r:.2f}×",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#1b5e20" if r > 1 else "#333333",
        )

    ax2.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5, label="Parity Baseline (1.0×)", zorder=4)
    ax2.set_title("Overhead Ratio: QROM / Computed", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Problem Size ($n$ qubits)", fontsize=11)
    ax2.set_ylabel("Gate Count Ratio", fontsize=11)
    ax2.set_ylim(0, max(ratios) * 1.25 if ratios else 2)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)

    return _save_or_show(fig, save_path, show)


def plot_ftqc_crossover(
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot Fault-Tolerant Clifford+T Value Loader proxy costs and crossover ranges."""
    setup_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    n_vals = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20]
    big_n_vals = [2**n for n in n_vals]

    # QROM T-count: 8(2^n - 1)
    qrom_t = [ftqc_qrom_loader_t_count(n, n + 1, uncompute_mode="reversible") for n in n_vals]

    # Coherent Arithmetic Loader T-count at varying precision
    eps_list = [1e-4, 1e-6, 1e-8, 1e-10]
    eps_colors = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]

    ax.loglog(
        big_n_vals,
        qrom_t,
        "o-",
        color="#d62728",
        linewidth=2.8,
        markersize=7,
        label="QROM Loader: $8(N - 1)$ $T$ gates (Unary Iteration, $\\varepsilon=0$)",
        zorder=5,
    )

    for eps, col in zip(eps_list, eps_colors):
        coh_t = [ftqc_coherent_loader_t_count(n, n + 1, eps=eps) for n in n_vals]
        ax.loglog(
            big_n_vals,
            coh_t,
            "s--",
            color=col,
            linewidth=1.8,
            markersize=5,
            label=f"Coherent Loader Proxy ($\\{{\\varepsilon = 10^{{-{int(-math.log10(eps))}}}\\}}$)",
            zorder=4,
        )

    # Shaded multi-decade synthesis precision envelope (eps in [10^-8, 10^-4])
    coh_t_4 = [ftqc_coherent_loader_t_count(n, n + 1, eps=1e-4) for n in n_vals]
    coh_t_8 = [ftqc_coherent_loader_t_count(n, n + 1, eps=1e-8) for n in n_vals]
    ax.fill_between(
        big_n_vals,
        coh_t_4,
        coh_t_8,
        color="#1f77b4",
        alpha=0.12,
        label=r"Coherent Synthesis Range ($\varepsilon \in [10^{-8}, 10^{-4}]$)",
    )

    ax.axvspan(1048576, 2097152, color="#ffeb3b", alpha=0.35, label="Crossover Region ($N \\approx 10^6 - 2\\times 10^6$)")
    ax.axvline(1048576, color="#f57f17", linestyle=":", linewidth=1.5)
    ax.axvline(2097152, color="#f57f17", linestyle=":", linewidth=1.5)

    ax.set_ylim(8, 4e8)
    ax.set_xlim(2.5, 7e6)

    ax.annotate(
        "Crossover Range\n$n = 20 - 21$\n($N \\approx 1.05\\text{M} - 2.10\\text{M}$)",
        xy=(1500000, 1.2e7),
        xytext=(70000, 7e7),
        arrowprops=dict(facecolor="#263238", shrink=0.08, width=1.2, headwidth=6, edgecolor="#263238"),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff9c4", ec="#fbc02d", lw=1.3),
        zorder=6,
    )

    ax.set_title("Fault-Tolerant Clifford+$T$ Value Loader: Analytical Scaling Proxy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Search Space Size $N = 2^n$ ($m = n + 1$ bits)", fontsize=11)
    ax.set_ylabel("Total $T$-Gate Count ($T_{\\text{load}}$)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9)

    return _save_or_show(fig, save_path, show)


def plot_oracle_pipeline(
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a clean, publication-ready architectural schematic of the computed
    quantum distance oracle pipeline, showing register layout, coherent arithmetic
    stages, comparator, phase kick, and uncomputation.
    """
    setup_plot_theme()
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=300)
    ax.set_xlim(-1.2, 17.2)
    ax.set_ylim(-1.0, 5.7)
    ax.axis("off")

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Register wires
    wires = [
        ("idx (n)", r"|i\rangle", r"(-1)^{[|f(i)-t| < \tau]} |i\rangle", 4.0, "#0d47a1"),
        ("val (m)", r"|0\rangle^{\otimes m}", r"|0\rangle^{\otimes m}", 3.0, "#1b5e20"),
        ("sign (1)", r"|0\rangle", r"|0\rangle", 2.0, "#bf360c"),
        ("anc (m-1)", r"|0\rangle^{\otimes (m-1)}", r"|0\rangle^{\otimes (m-1)}", 1.0, "#4a148c"),
        ("flag (1)", r"|0\rangle", r"|0\rangle", 0.0, "#b71c1c"),
    ]

    for name, init_s, out_s, y, col in wires:
        ax.plot([0.5, 15.2], [y, y], color="#455a64", lw=1.6, zorder=1)
        # Left label
        ax.text(0.3, y, f"{name}\n${init_s}$", ha="right", va="center", fontsize=10, fontweight="bold", color=col)
        # Right label
        ax.text(15.4, y, f"${out_s}$", ha="left", va="center", fontsize=10, fontweight="bold", color=col)

    # 1. Compute f(i)
    ax.plot([2.3, 2.3], [4.0, 3.35], color="#1565c0", lw=2, zorder=2)
    ax.scatter([2.3], [4.0], color="#1565c0", s=70, zorder=3)
    rect1 = patches.FancyBboxPatch((1.4, 2.65), 1.8, 0.7, boxstyle="round,pad=0.08", fc="#e3f2fd", ec="#1565c0", lw=1.8, zorder=2)
    ax.add_patch(rect1)
    ax.text(2.3, 3.0, "Compute $f(i)$\nQuadraticForm\n$O(n^2 m)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0d47a1")

    # 2. Subtract target -t
    rect2 = patches.FancyBboxPatch((3.6, 1.65), 1.8, 1.7, boxstyle="round,pad=0.08", fc="#e8f5e9", ec="#2e7d32", lw=1.8, zorder=2)
    ax.add_patch(rect2)
    ax.text(4.5, 2.5, "Subtract Target\nDraper Adder ($-t$)\nmod $2^{m+1}$\n$O(m^2)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1b5e20")

    # 3. Absolute value
    ax.plot([6.7, 6.7], [2.0, 2.65], color="#e65100", lw=2, zorder=2)
    ax.scatter([6.7], [2.0], color="#e65100", s=70, zorder=3)
    rect3 = patches.FancyBboxPatch((5.8, 2.65), 1.8, 0.7, boxstyle="round,pad=0.08", fc="#fff3e0", ec="#e65100", lw=1.8, zorder=2)
    ax.add_patch(rect3)
    ax.text(6.7, 3.0, "Absolute Value\nTwo's Complement\n$|f(i) - t|$\n$O(m^2)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#bf360c")

    # 4. Comparator
    rect4 = patches.FancyBboxPatch((8.0, 0.7), 1.9, 2.6, boxstyle="round,pad=0.08", fc="#f3e5f5", ec="#6a1b9a", lw=1.8, zorder=2)
    ax.add_patch(rect4)
    ax.text(8.95, 2.0, "Comparator\nIntegerComparator\n$|f(i)-t| < \\tau$\n(uses $m-1$ ancillas)\n$O(m)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#4a148c")
    ax.plot([8.95, 8.95], [0.7, 0.0], color="#6a1b9a", lw=2, zorder=2)
    ax.scatter([8.95], [0.0], color="#6a1b9a", s=70, zorder=3)

    # 5. Phase Kick (Z)
    rect5 = patches.FancyBboxPatch((10.35, -0.35), 0.9, 0.7, boxstyle="round,pad=0.08", fc="#ffebee", ec="#c62828", lw=2.0, zorder=2)
    ax.add_patch(rect5)
    ax.text(10.8, 0.0, "$Z$\n$(-1)$", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#b71c1c")

    # 6. Exact Uncomputation
    rect6 = patches.FancyBboxPatch((11.7, -0.45), 3.3, 4.9, boxstyle="round,pad=0.12", fc="#f8f9fa", ec="#455a64", lw=1.8, ls="--", zorder=2)
    ax.add_patch(rect6)
    ax.text(13.35, 2.0, "Exact Uncomputation ($O^\\dagger$)\nReverses Stages 4 $\\to$ 1\n\n• $\\mathrm{Comparator}^\\dagger$\n• $\\mathrm{AbsVal}^\\dagger$\n• $\\mathrm{Add}(+t)$\n• $\\mathrm{QuadraticForm}^\\dagger$\n\nRestores aux/val to $|0\\rangle$", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#263238")

    # Stage Header Badges at the top
    stages = [
        ("1. Value Loader", 2.3, "#1565c0", "#e3f2fd"),
        ("2. Draper Adder", 4.5, "#2e7d32", "#e8f5e9"),
        ("3. Abs Value", 6.7, "#e65100", "#fff3e0"),
        ("4. Comparator", 8.95, "#6a1b9a", "#f3e5f5"),
        ("5. Phase Kick", 10.8, "#c62828", "#ffebee"),
        ("6. Uncompute Pipeline", 13.35, "#455a64", "#eceff1"),
    ]
    for title, x_pos, stroke, fill in stages:
        badge = patches.FancyBboxPatch((x_pos - 0.85, 4.8), 1.7, 0.45, boxstyle="round,pad=0.05", fc=fill, ec=stroke, lw=1.4, zorder=3)
        ax.add_patch(badge)
        ax.text(x_pos, 5.02, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=stroke)

    # Main Title
    fig.suptitle("Computed Quantum Distance Oracle: Coherent Arithmetic Pipeline", fontsize=13.5, fontweight="bold", y=0.98)

    return _save_or_show(fig, save_path, show)


