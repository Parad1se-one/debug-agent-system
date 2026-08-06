from __future__ import annotations

from debug_agent_system.core.contracts import LockedSubgraph, SessionState

_DEFAULT_OWNER = "@工程师午（其他问题及无法分类问题）"
_OWNER_BY_CATEGORY = {
    "算法与程序调优": "@工程师丁（算法调试/程序调优）",
    "系统与软件异常": "@工程师乙（工控机/系统软件）",
    "硬件与运控": "@工程师丑（运控/硬件）",
}


class EscalationAgent:
    """O-ESC: choose owner and package evidence on dead end/low confidence."""

    def escalate(self, state: SessionState, subgraph: LockedSubgraph, failure_type: str) -> dict:
        target = subgraph.escalation_target or _OWNER_BY_CATEGORY.get(subgraph.category) or _DEFAULT_OWNER
        if not target.startswith("@") and target in {"algorithm_debug", "review_station"}:
            target = "@工程师丁（算法调试/程序调优）" if target == "algorithm_debug" else "@工程师乙（复判站/系统软件）"
        return {
            "escalation_target": target or _DEFAULT_OWNER,
            "failure_type": failure_type,
            "evidence_pack": {
                "session_id": state.session_id,
                "query": state.query,
                "top_error_id": subgraph.error_id,
                "top_error_label": subgraph.label,
                "checks_presented": list(state.checks_presented),
                "check_results": dict(state.check_results),
                "ruled_out": list(state.ruled_out),
            },
        }
