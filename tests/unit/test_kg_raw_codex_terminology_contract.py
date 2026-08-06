from __future__ import annotations

import json
from pathlib import Path

from debug_agent_system.kg_raw_codex.coverage import build_answer_scope
from debug_agent_system.kg_raw_codex.terminology_contract import (
    audit_terminology_search_contract,
    build_resolver_context,
    build_terminology_search_contract,
    load_terminology_manifest,
    terminology_governance_authority_errors,
    terminology_search_errors,
)


def _resolution() -> dict:
    return {
        "resolved_mentions": [{
            "surface_form": "旧称",
            "relation_types": ["colloquial_alias"],
            "concept": {
                "concept_id": "concept:test",
                "canonical_name": "规范名",
            },
        }],
        "retrieval_expansions": [
            {
                "text": "规范名",
                "authority": "approved_equivalence",
            },
            {
                "text": "关联线索",
                "authority": "search_hint",
                "source_surface_form": "旧称",
            },
        ],
        "ambiguous_mentions": [{
            "surface_form": "歧义词",
            "reason": "context_required",
            "required_context": ["subsystems"],
        }],
    }


def test_contract_requires_original_and_approved_canonical_search() -> None:
    contract = build_terminology_search_contract("查询旧称", _resolution())

    assert contract["required_search_groups"] == [{
        "source_surface_form": "旧称",
        "canonical_name": "规范名",
        "concept_id": "concept:test",
        "relation_types": ["colloquial_alias"],
        "required_terms": ["旧称", "规范名"],
        "obligation": "search_all",
        "reason": "approved_equivalence",
    }]
    assert contract["optional_expansions"] == [{
        "text": "关联线索",
        "authority": "search_hint",
        "source_surface_form": "旧称",
        "can_lock_variant": False,
    }]
    assert contract["unresolved_terms"][0]["can_lock_variant"] is False
    assert "entity_ontology.json" in " ".join(
        contract["runtime_authority_paths"]
    )
    assert "review_queue" in " ".join(
        contract["governance_only_paths"]
    )


def test_search_audit_rejects_one_sided_equivalence_search() -> None:
    contract = build_terminology_search_contract("查询旧称", _resolution())
    audit = audit_terminology_search_contract(contract, [{
        "type": "function_call",
        "name": "search_text",
        "status": "ok",
        "arguments": {"query": "旧称", "path_glob": "data/**/*"},
    }])

    assert audit["complete"] is False
    assert audit["missing_terms"] == ["规范名"]
    assert terminology_search_errors(audit) == [
        "terminology_required_search_missing:规范名"
    ]


def test_search_audit_accepts_responses_and_cli_searches() -> None:
    contract = build_terminology_search_contract("查询旧称", _resolution())
    audit = audit_terminology_search_contract(contract, [
        {
            "type": "function_call",
            "name": "search_text",
            "status": "ok",
            "arguments": {"query": "旧称", "path_glob": "data/**/*"},
        },
        {
            "type": "command_execution",
            "status": "completed",
            "command": "rg -n -- '规范名' data/raw data/kg_v2",
        },
    ])

    assert audit["complete"] is True
    assert audit["missing_terms"] == []
    assert audit["groups"][0]["complete"] is True


def test_resolver_context_uses_scope_without_guessing_entity_layers() -> None:
    scope = build_answer_scope("设备不能启动时，如何检查电源？")
    context = build_resolver_context(scope)

    operations = (*scope.context_operations, *scope.requested_operations)
    assert operations
    assert all(operation in context["phases"] for operation in operations)
    assert all(
        f"{operation}阶段" in context["phases"]
        for operation in operations
    )
    assert "equipment_types" not in context
    assert "subsystems" not in context


def test_manifest_loader_returns_reproducibility_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "terminology/terminology_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "schema_version": "report.v1",
        "terminology_version": "terms.v2",
        "revision": "abc123",
        "concept_count": 2,
        "expression_count": 3,
        "sense_count": 4,
        "ignored": "not copied",
    }), encoding="utf-8")

    assert load_terminology_manifest(tmp_path) == {
        "status": "loaded",
        "schema_version": "report.v1",
        "terminology_version": "terms.v2",
        "revision": "abc123",
        "concept_count": 2,
        "expression_count": 3,
        "sense_count": 4,
    }


def test_governance_material_cannot_close_facets_or_support_claims() -> None:
    contract = build_terminology_search_contract("查询旧称", _resolution())
    path = "data/kg_v2/review_queue/terminology_candidates.json"
    draft = {
        "answer_markdown": f"候选已确认。【来源：{path}】",
        "coverage_ledger": [{
            "facet_id": "query_task:generic",
            "status": "covered",
            "source_paths": [path],
        }],
    }

    assert terminology_governance_authority_errors(
        draft,
        contract,
    ) == [f"terminology_governance_material_used_as_evidence:{path}"]


def test_governance_material_is_rejected_inside_grouped_citations() -> None:
    contract = build_terminology_search_contract("查询旧称", _resolution())
    path = "data/kg_v2/terminology/noun_terminology_inventory.json"
    draft = {
        "answer_markdown": (
            "结论。【来源：data/raw/example.md；"
            f"{path}】"
        ),
        "coverage_ledger": [],
    }

    assert terminology_governance_authority_errors(draft, contract) == [
        f"terminology_governance_material_used_as_evidence:{path}"
    ]
