"""Test that README.md tables match current reproducible scaling computations across all regimes."""

from pathlib import Path

from closest_search.ftqc import (
    ftqc_coherent_loader_t_count,
    ftqc_full_oracle_resources,
    ftqc_qrom_loader_t_count,
)
from closest_search.nisq import (
    compute_nisq_scaling_records,
    compute_periodic_scaling_records,
    compute_qrom_comparison_records,
)


def test_readme_table_synchronization():
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md not found"
    content = readme_path.read_text(encoding="utf-8")

    # 1. Verify Regime 1: NISQ Scaling Table entries (n = 2..12)
    nisq_records = compute_nisq_scaling_records(max_n=12, a=2, b=3, c=1, target=6, threshold=1)
    for r in nisq_records:
        expected_nisq_row = (
            f"| {r.n} | {r.m} | {r.big_n:,} | {r.oracle_str} | {r.iter_gates:,} | "
            f"{r.grover_total_q_gates:,} | {r.dh_total_q_gates:,} | "
            f"{r.c_blackbox_bit_ops:,} | {r.c_blackbox_cpu_ops:,} |"
        )
        assert expected_nisq_row in content, f"Missing NISQ row in README: {expected_nisq_row}"

    # 2. Verify Regime 2: Periodic Scaling Table entries (n = 4..8)
    periodic_records = compute_periodic_scaling_records(max_n=8, m_fixed=4, a=2, b=3, c=1)
    for pr in periodic_records:
        expected_periodic_row = (
            f"| {pr.n} | {pr.m} | {pr.big_n:,} | {pr.mod_fixed} | "
            f"{pr.c_bb_evals:,} | {pr.c_st_evals:,} | {pr.c_bb_ops:,} | {pr.c_st_ops:,} |"
        )
        assert expected_periodic_row in content, f"Missing Periodic row in README: {expected_periodic_row}"

    # 3. Verify Empirical QROM vs. Computed Oracle Table (n = 2..6)
    qrom_records = compute_qrom_comparison_records(max_n=6, a=2, b=3, c=1, target=6, threshold=1)
    for qr in qrom_records:
        expected_qrom_row = (
            f"| {qr.n} | {qr.m} | {qr.big_n:,} | {qr.qrom_gates:,} | {qr.comp_gates:,} | {qr.ratio:.2f}x |"
        )
        assert expected_qrom_row in content, f"Missing QROM row in README: {expected_qrom_row}"

    # 4. Verify Regime 3A: Fault-Tolerant Value Loader Comparison Table (n = 2..20)
    ft_n_values = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20]
    for n_ft in ft_n_values:
        m_ft = n_ft + 1
        big_n_ft = 2**n_ft
        t_qrom_ft = ftqc_qrom_loader_t_count(n_ft, m_ft, uncompute_mode="reversible")
        t_coh_4 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-4)
        t_coh_6 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-6)
        t_coh_8 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-8)
        t_coh_10 = ftqc_coherent_loader_t_count(n_ft, m_ft, eps=1e-10)

        ratio_6 = t_coh_6 / max(1, t_qrom_ft)
        ratio_str = f"QROM {ratio_6:.1f}x cheaper" if ratio_6 >= 1.0 else f"Coh {1.0/ratio_6:.2f}x cheaper"

        expected_ftqc_row = (
            f"| {n_ft} | {m_ft} | {big_n_ft:,} | {t_qrom_ft:,} | {t_coh_4:,} | "
            f"{t_coh_6:,} | {t_coh_8:,} | {t_coh_10:,} | {ratio_str} |"
        )
        assert expected_ftqc_row in content, f"Missing FTQC Loader row in README: {expected_ftqc_row}"

    # 5. Verify Regime 3B: Fault-Tolerant Full Distance Oracle Table (n = 2..20)
    for n_ft in ft_n_values:
        m_ft = n_ft + 1
        big_n_ft = 2**n_ft
        res_qrom_6 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="qrom", eps=1e-6)
        res_coh_4 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-4)
        res_coh_6 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-6)
        res_coh_8 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-8)
        res_coh_10 = ftqc_full_oracle_resources(n_ft, m_ft, loader_type="coherent", eps=1e-10)

        ratio_full_6 = res_coh_6.t_count_nominal / max(1, res_qrom_6.t_count_nominal)
        ratio_full_str = f"QROM {ratio_full_6:.1f}x cheaper" if ratio_full_6 >= 1.0 else f"Coh {1.0/ratio_full_6:.2f}x cheaper"

        expected_full_row = (
            f"| {n_ft} | {m_ft} | {big_n_ft:,} | {res_qrom_6.t_count_nominal:,} | {res_coh_4.t_count_nominal:,} | "
            f"{res_coh_6.t_count_nominal:,} | {res_coh_8.t_count_nominal:,} | {res_coh_10.t_count_nominal:,} | {ratio_full_str} |"
        )
        assert expected_full_row in content, f"Missing FTQC Full Oracle row in README: {expected_full_row}"
