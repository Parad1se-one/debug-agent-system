class DiagnosticFeedbackAgent:
    """D1: convert DiagnosticSession terminal traces into write-side candidates."""
    def build_candidate(self, transcript: dict) -> dict:
        return {"type": "DiagnosticFeedback", "status": "pending_review", "transcript": transcript}
