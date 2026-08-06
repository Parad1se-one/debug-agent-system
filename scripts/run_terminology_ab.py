#!/usr/bin/env python3
"""Run the paired KG_v2+raw terminology A/B experiment.

Both arms are constructed from one immutable common argument set. The only
arm-specific value passed to the read pipeline is ``terminology_enabled``.
Without ``--execute`` this prints the frozen plan and makes no model calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from debug_agent_system.kg_raw_codex.pipeline import run
from debug_agent_system.kg_raw_codex.prompt import (
    SYSTEM_PROMPT_SHA256,
    SYSTEM_PROMPT_VERSION,
)


DEFAULT_SUITE = REPO_ROOT / "data/eval/terminology_ab_v1"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "data/results/terminology_ab_v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=False,
    )
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _dirty_patch_sha256() -> str:
    tracked = _git("diff", "--binary", "HEAD").encode("utf-8")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    digest = hashlib.sha256(tracked)
    for relative in sorted(filter(None, untracked.splitlines())):
        if relative == "data/results" or relative.startswith("data/results/"):
            continue
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _paths_fingerprint(paths: list[Path], *, content: bool) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix != ".pyc"
            )
    for path in sorted(set(files)):
        relative = path.relative_to(REPO_ROOT).as_posix()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if content:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _fingerprints(suite: Path) -> dict[str, Any]:
    files = {
        "cases": suite / "cases.jsonl",
        "experiment": suite / "experiment.json",
        "runtime_config": REPO_ROOT / "config/kg_v2_raw_codex.json",
        "source_manifest": (
            REPO_ROOT
            / "data/raw/aoi_debug_agent_sources/kg_v2_source_manifest.json"
        ),
        "terminology_manifest": (
            REPO_ROOT / "data/kg_v2/terminology/terminology_manifest.json"
        ),
        "kg_relations": REPO_ROOT / "data/kg_v2/relations/edges.json",
    }
    terminology = _load_json(files["terminology_manifest"])
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "workspace_dirty_patch_sha256_at_observation": (
            _dirty_patch_sha256()
        ),
        "runtime_code_sha256": _paths_fingerprint(
            [
                REPO_ROOT / "src/debug_agent_system/kg_raw_codex",
                REPO_ROOT / "src/debug_agent_system/knowledge_v2",
                REPO_ROOT
                / "src/debug_agent_system/adapters/codex_read/client.py",
            ],
            content=True,
        ),
        "corpus_tree_stat_sha256": _paths_fingerprint(
            [REPO_ROOT / "data/raw", REPO_ROOT / "data/kg_v2"],
            content=False,
        ),
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "terminology_revision": str(terminology.get("revision") or ""),
        "file_sha256": {
            name: _sha256(path) for name, path in files.items()
        },
    }


def _runtime_frozen_view(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "workspace_dirty_patch_sha256_at_observation"
    }


def _arm_order(case_id: str, seed: int) -> list[tuple[str, bool]]:
    order = [("A_control", False), ("B_treatment", True)]
    local_seed = int(
        hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()[:16], 16
    )
    random.Random(local_seed).shuffle(order)
    return order


def _relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _artifact_path(output: Path, case_id: str, arm: str) -> Path:
    return output / "artifacts" / case_id / f"{arm}.json"


def _failure_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.failure.json")


def _reusable(
    path: Path,
    *,
    query: str,
    enabled: bool,
    runtime: str,
    model: str,
    terminology_revision: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        value.get("query") == query
        and value.get("terminology_enabled") is enabled
        and value.get("model") == model
        and (value.get("runtime") or {}).get("engine") == runtime
        and (value.get("prompt") or {}).get("system_sha256")
        == SYSTEM_PROMPT_SHA256
        and (value.get("terminology_manifest") or {}).get("revision")
        == terminology_revision
        and (value.get("verification") or {}).get("passed") is True
        and str(value.get("answer") or "").strip()
    )


def _run_fingerprint(
    frozen: dict[str, Any], runtime: str, model: str
) -> str:
    encoded = json.dumps(
        {
            "frozen": _runtime_frozen_view(frozen),
            "runtime": runtime,
            "model": model,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime", default="responses_api")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually make paired model calls; otherwise print a dry run.",
    )
    args = parser.parse_args()

    suite = args.suite.resolve()
    experiment = _load_json(suite / "experiment.json")
    cases = _load_cases(suite / "cases.jsonl")
    selected = cases[max(args.start - 1, 0):]
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]
    frozen = _fingerprints(suite)
    run_fingerprint = _run_fingerprint(frozen, args.runtime, args.model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else DEFAULT_RESULTS_ROOT / f"{stamp}_{run_fingerprint[:12]}"
    )
    seed = int(experiment["paired_design"]["fixed_order_seed"])
    schedule = [
        {
            "case_id": case["id"],
            "query": case["query"],
            "arms": [arm for arm, _ in _arm_order(case["id"], seed)],
        }
        for case in selected
    ]
    manifest = {
        "schema_version": "debug_agent_system.terminology_ab_run.v1",
        "benchmark_id": experiment["benchmark_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "case_count": len(selected),
        "expected_model_calls": len(selected) * 2,
        "runtime": args.runtime,
        "model": args.model,
        "only_arm_variable": "terminology_enabled",
        "run_fingerprint": run_fingerprint,
        "frozen": frozen,
        "schedule": schedule,
        "output_dir": _relative(output),
    }
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    results: list[dict[str, Any]] = []
    common_run_args = {"runtime": args.runtime, "model": args.model}
    for case in selected:
        for arm, enabled in _arm_order(case["id"], seed):
            artifact = _artifact_path(output, case["id"], arm)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if args.resume and _reusable(
                artifact,
                query=case["query"],
                enabled=enabled,
                runtime=args.runtime,
                model=args.model,
                terminology_revision=frozen["terminology_revision"],
            ):
                status = "reused"
                error = ""
            else:
                try:
                    payload = run(
                        case["query"],
                        artifact,
                        **common_run_args,
                        terminology_enabled=enabled,
                    )
                except Exception as exc:  # noqa: BLE001 - A/B failure datum.
                    status = "failed"
                    error = f"{type(exc).__name__}:{exc}"
                    failure = {
                        "schema_version": (
                            "debug_agent_system.terminology_ab_failure.v1"
                        ),
                        "case_id": case["id"],
                        "query": case["query"],
                        "arm": arm,
                        "terminology_enabled": enabled,
                        "runtime": args.runtime,
                        "model": args.model,
                        "run_fingerprint": run_fingerprint,
                        "error": error,
                    }
                    _failure_path(artifact).write_text(
                        json.dumps(failure, ensure_ascii=False, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    payload["ab_case_id"] = case["id"]
                    payload["ab_arm"] = arm
                    payload["ab_run_fingerprint"] = run_fingerprint
                    artifact.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                    _failure_path(artifact).unlink(missing_ok=True)
                    status = "completed"
                    error = ""
            row = {
                "case_id": case["id"],
                "arm": arm,
                "terminology_enabled": enabled,
                "status": status,
                "artifact": _relative(artifact),
            }
            if error:
                row["error"] = error
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            manifest["results"] = results
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    final_frozen = _fingerprints(suite)
    if _runtime_frozen_view(final_frozen) != _runtime_frozen_view(frozen):
        raise RuntimeError("frozen_inputs_changed_during_ab_run")
    manifest["workspace_dirty_patch_sha256_at_completion"] = (
        final_frozen["workspace_dirty_patch_sha256_at_observation"]
    )
    manifest["results"] = results
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
