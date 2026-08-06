"""Re-run a frozen Xing episode set without mutating the active KG.

The source artifact must contain the exact W1 episodes to replay.  This keeps
W2/W3/W4 changes comparable while W1/W7 segmentation work is evaluated by a
separate benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.non_sop_intake import compute_kg_v2_graph_hash
from debug_agent_system.agents.write.pipeline import WriteSidePipeline
from debug_agent_system.knowledge.json_store import JsonKGStore


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run(
    *,
    source_run: str | Path,
    out_dir: str | Path,
    legacy_kg_root: str | Path = "data/kg",
    kg_v2_root: str | Path = "data/kg_v2",
    workers: int = 4,
    expected_count: int = 0,
    expected_sha256: str = "",
) -> dict[str, Any]:
    source_path = Path(source_run)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    episodes = [item for item in source.get("episodes") or [] if isinstance(item, dict)]
    episode_hash = _canonical_hash(episodes)
    if expected_count and len(episodes) != expected_count:
        raise ValueError(f"frozen_episode_count_mismatch:{len(episodes)}!={expected_count}")
    if expected_sha256 and episode_hash != expected_sha256:
        raise ValueError(f"frozen_episode_hash_mismatch:{episode_hash}!={expected_sha256}")
    episode_ids = [str(item.get("episode_id") or "") for item in episodes]
    if not all(episode_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("frozen_episode_ids_missing_or_duplicate")

    kg_hash_before = compute_kg_v2_graph_hash(kg_v2_root)
    pipeline = WriteSidePipeline(
        JsonKGStore(legacy_kg_root),
        queue_dir=output / "review_queue_legacy",
        kg_v2_root=kg_v2_root,
        kg_v2_queue_dir=output / "review_queue",
        w2_mode="native_v2",
    )
    pipeline.w2.deepseek_enabled = False
    for episode in episodes:
        episode["_write_source_type"] = "chat"
    partial_path = output / "w2_candidates.partial.jsonl"
    progress_path = output / "pipeline_progress.json"
    candidates = pipeline._extract_w2_candidates(
        episodes,
        workers=max(1, workers),
        w2_mode="native_v2",
        source_type="chat",
        partial_candidates_path=partial_path,
    )
    result = pipeline._run_candidate_episode_pairs(
        episodes,
        candidates,
        apply_approved=False,
        emit_episodes=True,
        dry_run_merge=True,
        kg_mode="v2",
        summary_counts={"summaries": 0, "episodes": len(episodes)},
        progress_path=progress_path,
    )
    kg_hash_after = compute_kg_v2_graph_hash(kg_v2_root)
    if kg_hash_after != kg_hash_before:
        raise RuntimeError("dry_run_mutated_active_kg_v2")

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_head = ""
    manifest = {
        "schema_version": "debug_agent_system.xing_frozen_episode_dry_run.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(source_path),
        "frozen_episode_count": len(episodes),
        "frozen_episode_canonical_sha256": episode_hash,
        "ordered_episode_id_sha256": _canonical_hash(episode_ids),
        "episode_id_set_sha256": _canonical_hash(sorted(episode_ids)),
        "git_head": git_head,
        "w2_mode": "native_v2",
        "deepseek_enabled": False,
        "w2_workers": max(1, workers),
        "kg_mode": "v2",
        "dry_run_merge": True,
        "apply_approved": False,
        "active_kg_hash_before": kg_hash_before,
        "active_kg_hash_after": kg_hash_after,
        "active_kg_unchanged": True,
    }
    (output / "pipeline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "summary": result.get("summary") or {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--legacy-kg-root", default="data/kg")
    parser.add_argument("--kg-v2-root", default="data/kg_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--expected-sha256", default="")
    args = parser.parse_args()
    print(json.dumps(run(
        source_run=args.source_run,
        out_dir=args.out_dir,
        legacy_kg_root=args.legacy_kg_root,
        kg_v2_root=args.kg_v2_root,
        workers=args.workers,
        expected_count=args.expected_count,
        expected_sha256=args.expected_sha256,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
