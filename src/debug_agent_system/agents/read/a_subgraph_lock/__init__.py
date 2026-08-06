from __future__ import annotations

from debug_agent_system.core.contracts import Candidate, LockedSubgraph
from debug_agent_system.knowledge.store import KGStore


class SubgraphLockAgent:
    """A: lock generation/traversal to one Error causal subgraph."""

    def __init__(self, store: KGStore) -> None:
        self.store = store

    def lock(self, candidate: Candidate) -> LockedSubgraph:
        return self.store.load_locked_subgraph(candidate.error_id)
