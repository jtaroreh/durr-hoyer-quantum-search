"""Closest-value unstructured quantum search with a computed-function oracle.

Finds argmin_i |f(i) - target| for f(i) = (A*i^2 + B*i + C) mod 2^m using
Grover search wrapped in the Dürr–Høyer minimum-finding loop.
"""

from __future__ import annotations

from .circuits import (
    absolute_value,
    add_constant,
    diffuser,
    distance_oracle,
    grover_circuit,
    projected_statevector_bytes,
    qrom_distance_oracle,
    qrom_value_function,
    total_oracle_qubits,
    value_function,
)
from .ftqc import (
    FTQCResourceVector,
    classical_bit_ops_per_eval,
    ftqc_coherent_loader_resources,
    ftqc_coherent_loader_t_count,
    ftqc_common_pipeline_t_count,
    ftqc_full_oracle_resources,
    ftqc_qrom_loader_resources,
    ftqc_qrom_loader_t_count,
    ftqc_rotation_t_count,
)
from .nisq import (
    NISQScalingRecord,
    PeriodicScalingRecord,
    QROMComparisonRecord,
    compute_nisq_scaling_records,
    compute_periodic_scaling_records,
    compute_qrom_comparison_records,
    optimal_grover_iterations,
)
from .search import (
    ClassicalAlgebraicRecord,
    Round,
    SearchResult,
    classical_algebraic_closest,
    classical_closest,
    classical_structured_closest,
    closest_value_search,
    distance,
    f_values,
)

__version__ = "0.2.0"

__all__ = [
    "ClassicalAlgebraicRecord",
    "FTQCResourceVector",
    "NISQScalingRecord",
    "PeriodicScalingRecord",
    "QROMComparisonRecord",
    "Round",
    "SearchResult",
    "__version__",
    "absolute_value",
    "add_constant",
    "classical_algebraic_closest",
    "classical_bit_ops_per_eval",
    "classical_closest",
    "classical_structured_closest",
    "closest_value_search",
    "compute_nisq_scaling_records",
    "compute_periodic_scaling_records",
    "compute_qrom_comparison_records",
    "diffuser",
    "distance",
    "distance_oracle",
    "f_values",
    "ftqc_coherent_loader_resources",
    "ftqc_coherent_loader_t_count",
    "ftqc_common_pipeline_t_count",
    "ftqc_full_oracle_resources",
    "ftqc_qrom_loader_resources",
    "ftqc_qrom_loader_t_count",
    "ftqc_rotation_t_count",
    "grover_circuit",
    "optimal_grover_iterations",
    "projected_statevector_bytes",
    "qrom_distance_oracle",
    "qrom_value_function",
    "total_oracle_qubits",
    "value_function",
]
