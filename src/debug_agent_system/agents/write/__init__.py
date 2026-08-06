from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ChatCollectAgent",
    "KnowledgeExtractionAgent",
    "ConflictResolutionAgent",
    "QualityGateAgent",
    "IncrementalIngestAgent",
    "ReviewQueueAgent",
    "ReviewContextAgent",
    "RawDocIngestAgent",
    "SectionCaseBundleAgent",
    "WriteSidePipeline",
]

_MODULE_BY_NAME = {
    "ChatCollectAgent": ".w1_chat_collect",
    "KnowledgeExtractionAgent": ".w2_extract",
    "ConflictResolutionAgent": ".w3_conflict",
    "QualityGateAgent": ".w4_quality_gate",
    "IncrementalIngestAgent": ".w5_incremental_ingest",
    "ReviewQueueAgent": ".w6_review_queue",
    "ReviewContextAgent": ".review_context",
    "RawDocIngestAgent": ".w9_raw_doc_ingest",
    "SectionCaseBundleAgent": ".w10_section_case_bundle",
    "WriteSidePipeline": ".pipeline",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
