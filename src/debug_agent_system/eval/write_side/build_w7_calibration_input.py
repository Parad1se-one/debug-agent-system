"""Freeze source-only episodes for reviewed W7 calibration sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from debug_agent_system.agents.write.w7_trace.contracts import canonical_hash


def build(
    *,
    annotations_path: Path,
    out_path: Path,
    limit: int = 5,
    include_excluded: bool = False,
) -> dict[str, Any]:
    annotations = json.loads(
        annotations_path.read_text(encoding="utf-8")
    )
    selected: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for session in annotations.get("sessions") or []:
        if not isinstance(session, dict):
            continue
        if not str(session.get("reviewer") or "").strip():
            continue
        verdict = str(session.get("session_verdict") or "")
        if not verdict or (
            verdict == "exclude" and not include_excluded
        ):
            continue
        context_ref = str(session.get("full_context_json") or "")
        context_path = annotations_path.parent / context_ref
        if not context_path.is_file():
            raise FileNotFoundError(
                f"missing calibration full context: {context_path}"
            )
        context = json.loads(context_path.read_text(encoding="utf-8"))
        source_episodes = [
            item for item in context.get("source_episodes") or []
            if isinstance(item, dict)
        ]
        if not source_episodes:
            raise ValueError(
                f"calibration source episodes missing: {context_path}"
            )
        session_episode_ids: list[str] = []
        for episode in source_episodes:
            episode_id = str(episode.get("episode_id") or "")
            if not episode_id or episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)
            session_episode_ids.append(episode_id)
            episodes.append(episode)
        selected.append({
            "thread_id": str(session.get("thread_id") or ""),
            "verdict": verdict,
            "full_context_json": context_ref,
            "episode_ids": session_episode_ids,
            "source_context_hash": canonical_hash(source_episodes),
        })
        if limit > 0 and len(selected) >= limit:
            break
    payload = {
        "schema_version": "w7.calibration_source_input.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotations_path": str(annotations_path),
        "selection_policy": {
            "reviewer_required": True,
            "include_excluded": include_excluded,
            "limit": max(0, int(limit)),
            "labels_excluded_from_model_input": True,
        },
        "sessions": selected,
        "episodes": episodes,
        "summary": {
            "sessions": len(selected),
            "episodes": len(episodes),
        },
    }
    payload["source_input_hash"] = canonical_hash({
        "sessions": selected,
        "episodes": episodes,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build-w7-calibration-input"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/results/w7_release_gate_payload_candidate_20260724/"
            "human_annotations.json"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args(argv)
    result = build(
        annotations_path=args.annotations,
        out_path=args.out,
        limit=max(0, int(args.limit)),
        include_excluded=bool(args.include_excluded),
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0 if result["summary"]["sessions"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
