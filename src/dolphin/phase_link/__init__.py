"""Package for phase linking stacks of SLCs.

Currently implements the eigenvalue-based maximum likelihood (EMI) algorithm from
[@Ansari2018EfficientPhaseEstimation], as well as the EVD based approach from
[@Fornaro2015CAESARApproachBased] and [@Mirzaee2023NonlinearPhaseLinking]
"""

from ._closure_phase import (
    compute_nearest_closure_phases,
    compute_two_hop_closure_phases,
)
from ._compress import compress
from ._core import PhaseLinkRuntimeError, run_phase_linking
from ._looks import (
    CrlbLooksMethod,
    effective_looks_fraction,
    estimate_effective_looks_fraction,
)
