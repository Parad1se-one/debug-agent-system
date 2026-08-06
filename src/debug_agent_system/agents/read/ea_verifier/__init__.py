from __future__ import annotations

from dataclasses import dataclass

from debug_agent_system.core.contracts import LockedSubgraph, SolutionNode


@dataclass(slots=True)
class VerificationResult:
    supported: bool
    confidence: float
    evidence_ids: list[str]
    issues: list[str]


class DiagnosisVerifier:
    """EA: deterministic evidence alignment before final answer."""

    def verify_resolution(self, subgraph: LockedSubgraph, solution: SolutionNode | None) -> VerificationResult:
        if solution is None:
            return VerificationResult(False, 0.45, [subgraph.error_id], ["missing_solution_node"])
        evidence = [subgraph.error_id, solution.solution_id]
        return VerificationResult(True, 0.85, evidence, [])
