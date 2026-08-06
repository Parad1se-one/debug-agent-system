class LogPatternAgent:
    """D3: propose LogPattern candidates from repeated unmatched log signatures."""
    def propose(self, log_summary: dict) -> dict:
        return {"type": "LogPatternCandidate", "status": "pending_review", "log_summary": log_summary}
