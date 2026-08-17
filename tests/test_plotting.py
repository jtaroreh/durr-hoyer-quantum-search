"""Unit tests for visualization and plotting routines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend for testing
import matplotlib.pyplot as plt

from closest_search import (
    SearchResult,
    plot_durr_hoyer_trajectory,
    plot_ftqc_crossover,
    plot_nisq_scaling,
    plot_oracle_pipeline,
    plot_qrom_vs_coherent_nisq,
    plot_quantum_amplitudes,
    setup_plot_theme,
)
from closest_search.search import Round


def test_setup_plot_theme():
    setup_plot_theme()
    assert plt.rcParams["figure.autolayout"] is True


def test_plot_durr_hoyer_trajectory(tmp_path: Path):
    values = [1, 6, 15, 12, 13, 2, 11, 8]
    target = 6
    rounds = [
        Round(threshold=10, grover_iterations=1, measured_index=2, measured_distance=9, improved=True),
        Round(threshold=9, grover_iterations=1, measured_index=4, measured_distance=7, improved=True),
        Round(threshold=7, grover_iterations=2, measured_index=4, measured_distance=7, improved=False),
        Round(threshold=7, grover_iterations=2, measured_index=1, measured_distance=0, improved=True),
    ]
    result = SearchResult(best_index=1, best_distance=0, oracle_queries=9, rounds=rounds)

    out_file = tmp_path / "trajectory.png"
    fig = plot_durr_hoyer_trajectory(result, values, target, save_path=out_file, show=False)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_quantum_amplitudes(tmp_path: Path):
    hist = {1: 1800, 3: 50, 5: 100, 7: 98}
    best_set = [1]
    n, m, target = 3, 4, 6

    out_file = tmp_path / "amplitudes.png"
    fig = plot_quantum_amplitudes(hist, best_set, target, n, m, save_path=out_file, show=False)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_quantum_amplitudes_suboptimal_basin(tmp_path: Path):
    # Candidate basin amplified: [1, 5], but true global optimum is only [1]
    hist = {1: 1000, 5: 900, 3: 50, 7: 50}
    candidate_set = [1, 5]
    global_set = [1]
    n, m, target = 3, 4, 6

    out_file = tmp_path / "amplitudes_suboptimal.png"
    fig = plot_quantum_amplitudes(
        hist=hist,
        best_set=candidate_set,
        target=target,
        n=n,
        m=m,
        global_optima_set=global_set,
        save_path=out_file,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0



def test_plot_nisq_scaling(tmp_path: Path):
    out_file = tmp_path / "nisq_scaling.png"
    fig = plot_nisq_scaling(max_n=3, save_path=out_file, show=False)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_qrom_vs_coherent_nisq(tmp_path: Path):
    out_file = tmp_path / "qrom_comp.png"
    fig = plot_qrom_vs_coherent_nisq(max_n=3, save_path=out_file, show=False)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_ftqc_crossover(tmp_path: Path):
    out_file = tmp_path / "ftqc_crossover.png"
    fig = plot_ftqc_crossover(save_path=out_file, show=False)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_oracle_pipeline(tmp_path: Path):
    out_file = tmp_path / "oracle_pipeline.png"
    fig = plot_oracle_pipeline(save_path=out_file, show=False)
    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

