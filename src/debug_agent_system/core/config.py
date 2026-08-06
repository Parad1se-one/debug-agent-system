"""Minimal config loader with no runtime dependency beyond stdlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Thresholds:
    graph_match_min_score: float = 4.0
    graph_match_min_margin: float = 0.75
    low_confidence: float = 0.35
    ask_info_max_required_items: int = 3


@dataclass(slots=True)
class RuntimeOptions:
    max_turns: int = 12
    destructive_action_requires_human_confirm: bool = True


@dataclass(slots=True)
class EvidenceGapOptions:
    enabled: bool = True
    max_rounds: int = 2
    max_resources: int = 12
    max_bytes_per_resource: int = 65536


@dataclass(slots=True)
class ReadLLMOptions:
    provider: str = "codex"
    enabled: bool = False
    answer_composer_enabled: bool = False
    model: str = "gpt-5.3-codex"
    base_url: str = ""
    timeout_seconds: int = 60
    max_tool_rounds: int = 3
    max_answer_documents: int = 8
    max_answer_chunks: int = 64
    max_answer_input_chars: int = 60000
    answer_fallback: str = "deterministic"


@dataclass(slots=True)
class IncidentRuntimeOptions:
    """Jira/diagnostic-package structured evidence side path."""

    enabled: bool = False
    shadow_mode: bool = True
    max_package_bytes: int = 512 * 1024 * 1024
    max_member_bytes: int = 32 * 1024 * 1024
    max_dump_member_bytes: int = 8 * 1024 * 1024 * 1024
    max_total_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_members: int = 5000
    max_nesting: int = 3
    max_compression_ratio: float = 200.0
    allow_dump_analysis: bool = False
    allow_ocr_analysis: bool = False


@dataclass(slots=True)
class KnowledgeOptions:
    store: str = "kg_v2_json"
    sag_snapshot_mode: bool = False
    kg_v2_root: Path = Path("data/kg_v2")
    kg_v2_sqlite_path: Path = Path("data/kg_v2_sag/debug_agent_v2.sqlite")
    sqlite_sag_path: Path = Path("data/kg_sag/debug_agent.sqlite")
    raw_root: Path = Path("data/raw/aoi_debug_agent_sources")
    w1_root: Path = Path("data/results/w1_full_20260703_061455")


@dataclass(slots=True)
class RetrievalOptions:
    sag_max_hops: int = 1
    sag_event_budget: int = 150
    trace_enabled: bool = True
    entity_stopwords_path: Path | None = None
    sag_degree_penalty: bool = True
    sag_max_entity_degree: int = 240
    sag_family_canonicalization: bool = True
    sag_llm_rerank: bool = False


@dataclass(slots=True)
class SystemConfig:
    root: Path
    kg_root: Path
    session_store: Path
    knowledge: KnowledgeOptions = field(default_factory=KnowledgeOptions)
    retrieval: RetrievalOptions = field(default_factory=RetrievalOptions)
    thresholds: Thresholds = field(default_factory=Thresholds)
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)
    evidence_gap: EvidenceGapOptions = field(default_factory=EvidenceGapOptions)
    read_llm: ReadLLMOptions = field(default_factory=ReadLLMOptions)
    incident_runtime: IncidentRuntimeOptions = field(
        default_factory=IncidentRuntimeOptions
    )


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip('"\'')


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value)
    return result


def load_config(path: str | Path | None = None) -> SystemConfig:
    cfg_path = Path(path or "config/debug_agent_system.yaml").resolve()
    data = _parse_simple_yaml(cfg_path) if cfg_path.exists() else {}
    root = cfg_path.parent.parent if cfg_path.parent.name == "config" else cfg_path.parent
    kg_root = Path(str(data.get("kg_root") or "data/kg"))
    session_store = Path(str(data.get("session_store") or "data/sessions"))
    if not kg_root.is_absolute():
        kg_root = root / kg_root
    if not session_store.is_absolute():
        session_store = root / session_store
    knowledge_raw = data.get("knowledge") or {}
    retrieval_raw = data.get("retrieval") or {}
    sqlite_sag_path = Path(str(knowledge_raw.get("sqlite_sag_path") or "data/kg_sag/debug_agent.sqlite"))
    kg_v2_root = Path(str(knowledge_raw.get("kg_v2_root") or "data/kg_v2"))
    kg_v2_sqlite_path = Path(str(knowledge_raw.get("kg_v2_sqlite_path") or "data/kg_v2_sag/debug_agent_v2.sqlite"))
    raw_root = Path(str(knowledge_raw.get("raw_root") or "data/raw/aoi_debug_agent_sources"))
    w1_root = Path(str(knowledge_raw.get("w1_root") or "data/results/w1_full_20260703_061455"))
    if not sqlite_sag_path.is_absolute():
        sqlite_sag_path = root / sqlite_sag_path
    if not kg_v2_root.is_absolute():
        kg_v2_root = root / kg_v2_root
    if not kg_v2_sqlite_path.is_absolute():
        kg_v2_sqlite_path = root / kg_v2_sqlite_path
    if not raw_root.is_absolute():
        raw_root = root / raw_root
    if not w1_root.is_absolute():
        w1_root = root / w1_root
    entity_stopwords_path_raw = str(retrieval_raw.get("entity_stopwords_path") or "").strip()
    entity_stopwords_path = Path(entity_stopwords_path_raw) if entity_stopwords_path_raw else None
    if entity_stopwords_path is not None and not entity_stopwords_path.is_absolute():
        entity_stopwords_path = root / entity_stopwords_path
    thresholds = data.get("thresholds") or {}
    runtime = data.get("runtime") or {}
    evidence_gap = data.get("evidence_gap") or {}
    read_llm = data.get("read_llm") or {}
    incident_runtime = data.get("incident_runtime") or {}
    return SystemConfig(
        root=root,
        kg_root=kg_root,
        session_store=session_store,
        knowledge=KnowledgeOptions(
            store=str(knowledge_raw.get("store") or "kg_v2_json"),
            sag_snapshot_mode=bool(knowledge_raw.get("sag_snapshot_mode", False)),
            kg_v2_root=kg_v2_root,
            kg_v2_sqlite_path=kg_v2_sqlite_path,
            sqlite_sag_path=sqlite_sag_path,
            raw_root=raw_root,
            w1_root=w1_root,
        ),
        retrieval=RetrievalOptions(
            sag_max_hops=int(retrieval_raw.get("sag_max_hops", 1)),
            sag_event_budget=int(retrieval_raw.get("sag_event_budget", 150)),
            trace_enabled=bool(retrieval_raw.get("trace_enabled", True)),
            entity_stopwords_path=entity_stopwords_path,
            sag_degree_penalty=bool(retrieval_raw.get("sag_degree_penalty", True)),
            sag_max_entity_degree=int(retrieval_raw.get("sag_max_entity_degree", 240)),
            sag_family_canonicalization=bool(retrieval_raw.get("sag_family_canonicalization", True)),
            sag_llm_rerank=bool(retrieval_raw.get("sag_llm_rerank", False)),
        ),
        thresholds=Thresholds(
            graph_match_min_score=float(thresholds.get("graph_match_min_score", 4.0)),
            graph_match_min_margin=float(thresholds.get("graph_match_min_margin", 0.75)),
            low_confidence=float(thresholds.get("low_confidence", 0.35)),
            ask_info_max_required_items=int(thresholds.get("ask_info_max_required_items", 3)),
        ),
        runtime=RuntimeOptions(
            max_turns=int(runtime.get("max_turns", 12)),
            destructive_action_requires_human_confirm=bool(runtime.get("destructive_action_requires_human_confirm", True)),
        ),
        evidence_gap=EvidenceGapOptions(
            enabled=bool(evidence_gap.get("enabled", True)),
            max_rounds=int(evidence_gap.get("max_rounds", 2)),
            max_resources=int(evidence_gap.get("max_resources", 12)),
            max_bytes_per_resource=int(evidence_gap.get("max_bytes_per_resource", 65536)),
        ),
        read_llm=ReadLLMOptions(
            provider=str(read_llm.get("provider") or "codex"),
            enabled=bool(read_llm.get("enabled", False)),
            answer_composer_enabled=bool(
                read_llm.get("answer_composer_enabled", False)
            ),
            model=str(read_llm.get("model") or "gpt-5.3-codex"),
            # An empty value deliberately delegates endpoint discovery to
            # OPENAI_BASE_URL in .env.local.  Never persist a private gateway
            # URL or credential in the checked-in runtime configuration.
            base_url=str(read_llm.get("base_url") or ""),
            timeout_seconds=int(read_llm.get("timeout_seconds", 60)),
            max_tool_rounds=int(read_llm.get("max_tool_rounds", 3)),
            max_answer_documents=int(
                read_llm.get("max_answer_documents", 8)
            ),
            max_answer_chunks=int(read_llm.get("max_answer_chunks", 64)),
            max_answer_input_chars=int(
                read_llm.get("max_answer_input_chars", 60000)
            ),
            answer_fallback=str(
                read_llm.get("answer_fallback") or "deterministic"
            ),
        ),
        incident_runtime=IncidentRuntimeOptions(
            enabled=bool(incident_runtime.get("enabled", False)),
            shadow_mode=bool(incident_runtime.get("shadow_mode", True)),
            max_package_bytes=int(
                incident_runtime.get("max_package_bytes", 512 * 1024 * 1024)
            ),
            max_member_bytes=int(
                incident_runtime.get("max_member_bytes", 32 * 1024 * 1024)
            ),
            max_dump_member_bytes=int(
                incident_runtime.get(
                    "max_dump_member_bytes", 8 * 1024 * 1024 * 1024
                )
            ),
            max_total_uncompressed_bytes=int(
                incident_runtime.get(
                    "max_total_uncompressed_bytes", 1024 * 1024 * 1024
                )
            ),
            max_members=int(incident_runtime.get("max_members", 5000)),
            max_nesting=int(incident_runtime.get("max_nesting", 3)),
            max_compression_ratio=float(
                incident_runtime.get("max_compression_ratio", 200.0)
            ),
            allow_dump_analysis=bool(
                incident_runtime.get("allow_dump_analysis", False)
            ),
            allow_ocr_analysis=bool(
                incident_runtime.get("allow_ocr_analysis", False)
            ),
        ),
    )
