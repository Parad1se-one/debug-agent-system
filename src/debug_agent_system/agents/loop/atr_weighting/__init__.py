class ATRWeightingAgent:
    """D2: compute occurrence/edge-weight proposals; never writes without approval."""
    def propose(self, feedback: dict) -> dict:
        return {"type": "ATRWeightProposal", "status": "pending_review", "feedback": feedback}
