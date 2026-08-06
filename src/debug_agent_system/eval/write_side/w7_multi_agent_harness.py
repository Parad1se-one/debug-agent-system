"""CLI runner for the W7 multi-agent shadow implementation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write import QualityGateAgent, WriteSidePipeline
from debug_agent_system.agents.write.w6_review_queue import ReviewQueueAgent
from debug_agent_system.agents.write.w7_trace.batch_orchestrator import (
    AtomicExtractor,
    W7BatchShadowOrchestrator,
)
from debug_agent_system.agents.write.w7_trace.batch_candidate import (
    build_w7_batch_typed_candidate,
)
from debug_agent_system.agents.write.w7_trace.contracts import canonical_hash
from debug_agent_system.agents.write.w7_trace.model_client import (
    DeepSeekDecisionModelClient,
)
from debug_agent_system.agents.write.w7_trace.orchestrator import (
    W7ShadowOrchestrator,
)
from debug_agent_system.knowledge.json_store import JsonKGStore
from debug_agent_system.knowledge_v2 import JsonKGV2Store


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or "=" not in value:
            continue
        key, raw = value.split("=", 1)
        os.environ.setdefault(key.strip(), raw.strip().strip("'\""))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    for item in sorted(
        (value for value in path.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(path).as_posix(),
    ):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _episodes(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    direct = value.get("episodes")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    output: list[dict[str, Any]] = []
    for session in value.get("sessions") or []:
        if not isinstance(session, dict):
            continue
        output.extend(
            item
            for item in session.get("episodes") or []
            if isinstance(item, dict)
        )
    return output


def _chat_scope_id(episode: dict[str, Any]) -> str:
    explicit = str(episode.get("chat_id") or "")
    if explicit:
        return explicit
    value = str(
        episode.get("source_thread_id")
        or episode.get("thread_id")
        or episode.get("episode_id")
        or ""
    )
    if value.startswith("oc_"):
        candidate = value.split(":", 1)[0]
        if "_20" in candidate:
            candidate = candidate.split("_20", 1)[0]
        return candidate
    return value


def _episode_batches(
    episodes: list[dict[str, Any]],
    *,
    scope: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    if scope == "all":
        return [("all", episodes)] if episodes else []
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for index, episode in enumerate(episodes, 1):
        if scope == "episode":
            key = str(
                episode.get("episode_id") or f"episode-{index}"
            )
        elif scope == "thread":
            key = str(
                episode.get("source_thread_id")
                or episode.get("thread_id")
                or episode.get("episode_id")
                or f"thread-{index}"
            )
        elif scope == "chat":
            key = _chat_scope_id(episode) or f"chat-{index}"
        else:
            raise ValueError(f"unsupported_batch_scope:{scope}")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(episode)
    return [(key, grouped[key]) for key in order]


def _score_w7_typed_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Apply both typed admission and KG-v2 semantic W4 contracts."""

    agent = QualityGateAgent()
    gate = agent.score_typed_candidate(candidate)
    semantic_gate = agent.score_v2_bundle({
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "objects": candidate.get("objects") or {},
        "relations": candidate.get("relations") or [],
        "schema_valid": bool(candidate.get("schema_valid")),
        "schema_issues": list(candidate.get("schema_issues") or []),
        "strategy": {},
        "source_text": str(
            candidate.get("raw_text") or candidate.get("text") or ""
        ),
        "source_message_ids": list(
            candidate.get("message_ids") or []
        ),
        "source_messages": list(
            (candidate.get("payload") or {}).get("source_messages")
            or []
        ),
    })
    gate["kg_v2_semantic_gate"] = semantic_gate
    if not semantic_gate.get("passed"):
        gate.update({
            "decision": (
                "route_review"
                if gate.get("decision") == "admit"
                else gate.get("decision")
            ),
            "materialize_allowed": False,
            "merge_allowed": False,
            "issues": sorted(set([
                *gate.get("issues", []),
                *semantic_gate.get("issues", []),
                "kg_v2_semantic_gate_failed",
            ])),
            "passed": gate.get("decision") != "reject",
        })
    return gate


def run(
    *,
    input_path: Path,
    out_dir: Path,
    client: DeepSeekDecisionModelClient | None,
    episode_ids: set[str] | None = None,
    limit: int = 0,
    batch_scope: str = "episode",
    atomic_extractor: AtomicExtractor | None = None,
    review_agent: ReviewQueueAgent | None = None,
    decision_workers: int = 1,
    w2_workers: int = 1,
) -> dict[str, Any]:
    value = json.loads(input_path.read_text(encoding="utf-8"))
    episodes = _episodes(value)
    if episode_ids:
        episodes = [
            episode
            for episode in episodes
            if str(episode.get("episode_id") or "") in episode_ids
        ]
    if limit > 0:
        episodes = episodes[:limit]
    batches = _episode_batches(episodes, scope=batch_scope)
    results: list[dict[str, Any]] = []
    for index, (batch_id, batch_episodes) in enumerate(batches, 1):
        run_key = canonical_hash({
            "batch_scope": batch_scope,
            "batch_id": batch_id,
            "episode_ids": [
                str(value.get("episode_id") or "")
                for value in batch_episodes
            ],
        })[:12]
        orchestrator = W7ShadowOrchestrator(
            client=client,
            checkpoint_root=out_dir / "checkpoints" / run_key,
            component_workers=max(1, int(decision_workers)),
        )
        result = W7BatchShadowOrchestrator(
            orchestrator,
            decision_workers=decision_workers,
            atomic_workers=max(1, int(w2_workers)),
        ).run(
            batch_id=batch_id,
            episodes=batch_episodes,
            atomic_extractor=atomic_extractor,
        )
        if atomic_extractor is not None:
            typed_candidate, typed_issues = (
                build_w7_batch_typed_candidate(result)
            )
            gate = _score_w7_typed_candidate(typed_candidate)
            result.update({
                "typed_candidate": typed_candidate,
                "typed_candidate_build_issues": typed_issues,
                "quality_gate": gate,
                "queue_written": False,
                "kg_mutated": False,
            })
            if review_agent is not None:
                result["w6_review_item"] = (
                    review_agent.build_w7_trace_review_item(
                        typed_candidate,
                        gate,
                        trace_review_payload=result[
                            "w6_trace_review_payload"
                        ],
                    )
                )
        result["run_key"] = run_key
        result_path = out_dir / "results" / f"{run_key}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append({
            "index": index,
            "batch_id": batch_id,
            "episode_ids": [
                str(value.get("episode_id") or "")
                for value in batch_episodes
            ],
            "run_key": run_key,
            "schema_valid": result["schema_valid"],
            "result": str(result_path),
        })
    manifest = {
        "schema_version": "w7.multi_agent_shadow_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": _file_sha256(input_path),
        "batch_scope": batch_scope,
        "deepseek_enabled": client is not None,
        "w2_atomic_extraction_enabled": atomic_extractor is not None,
        "decision_workers": max(1, int(decision_workers)),
        "w2_workers": max(1, int(w2_workers)),
        "promotion_allowed": False,
        "legacy_authoritative": True,
        "results": results,
        "summary": {
            "episodes": len(episodes),
            "batches": len(results),
            "schema_valid": sum(
                bool(item.get("schema_valid")) for item in results
            ),
        },
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="w7-multi-agent-harness")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--batch-scope",
        choices=["episode", "thread", "chat", "all"],
        default="episode",
    )
    parser.add_argument("--run-w2", action="store_true")
    parser.add_argument(
        "--w2-mode",
        choices=["native_v2", "prompt_first"],
        default="native_v2",
    )
    parser.add_argument("--w2-workers", type=int, default=1)
    parser.add_argument("--decision-workers", type=int, default=1)
    parser.add_argument("--kg-root", type=Path, default=Path("data/kg"))
    parser.add_argument(
        "--kg-v2-root", type=Path, default=Path("data/kg_v2")
    )
    args = parser.parse_args(argv)
    _load_env(args.env_file)
    state_before = {
        "kg_root": _tree_sha256(args.kg_root),
        "kg_v2_root": _tree_sha256(args.kg_v2_root),
    }
    client = None
    if args.deepseek:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            parser.error("missing DEEPSEEK_API_KEY")
        client = DeepSeekDecisionModelClient(api_key=api_key)
    atomic_extractor = None
    review_agent = None
    if args.run_w2:
        pipeline = WriteSidePipeline(
            JsonKGStore(args.kg_root),
            w2_mode=args.w2_mode,
            review_context_enabled=False,
        )

        def extract_atomic(manifest):
            return pipeline.extract_w7_atomic_cases(
                manifest,
                w2_mode=args.w2_mode,
                workers=1,
            )

        atomic_extractor = extract_atomic
        review_agent = ReviewQueueAgent(
            JsonKGV2Store(args.kg_v2_root)
        )
    manifest = run(
        input_path=args.input,
        out_dir=args.out_dir,
        client=client,
        episode_ids=set(args.episode_id),
        limit=max(0, int(args.limit)),
        batch_scope=args.batch_scope,
        atomic_extractor=atomic_extractor,
        review_agent=review_agent,
        decision_workers=max(1, int(args.decision_workers)),
        w2_workers=max(1, int(args.w2_workers)),
    )
    state_after = {
        "kg_root": _tree_sha256(args.kg_root),
        "kg_v2_root": _tree_sha256(args.kg_v2_root),
    }
    manifest["state_hashes"] = {
        "before": state_before,
        "after": state_after,
        "unchanged": state_before == state_after,
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(manifest)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if (
        not args.deepseek
        or manifest["summary"]["schema_valid"]
        == manifest["summary"]["batches"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
