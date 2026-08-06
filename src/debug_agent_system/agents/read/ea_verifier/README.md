# EA Diagnosis Verifier Agent

- id: `EA`
- type: Verifier
- owner: `src/debug_agent_system/agents/read/ea_verifier`
- responsibility: verify that a resolution is supported by the locked KG evidence before O0 returns resolved.
- entrypoint: `DiagnosisVerifier.verify_resolution(subgraph, solution)`.
- inputs:
  - `LockedSubgraph` for current diagnosis.
  - Optional `SolutionNode` selected by B-D when user marks a check solved.
- outputs:
  - `VerificationResult`: `supported`, `confidence`, `evidence_ids`, `issues`.
- failure_modes:
  - Missing solution -> `supported=false`, `confidence=0.45`, issue `missing_solution_node`.
  - Supported solution -> confidence currently deterministic `0.85`.
- observability:
  - O0 stores verification in `metadata.verification` and uses confidence in final response.
- non_goals:
  - Does not judge facts with LLM in current implementation.
  - Does not alter KG or choose escalation owner.
