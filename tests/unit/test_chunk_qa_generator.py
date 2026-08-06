from __future__ import annotations

import json

from debug_agent_system.eval.read_side.chunk_qa_generator import (
    _codex_cli_usage,
    build_case_record,
    generate_one_chunk,
    validate_model_payload,
)


CHUNK = {
    "text": (
        "【SOP】模板创建失败\n\n设备首次安装 buddy 0.11.2~0.11.4，"
        "升级 0.14.x 后可能遇到权限问题。下载 useraccess_restore.bat，"
        "关闭 buddy，右键使用管理员身份运行后恢复。"
    ),
    "metadata": {
        "source": "SOP",
        "title": "模板创建失败",
        "section_num": "1.1.1",
    },
}


def _response(payload: dict) -> dict:
    return {
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": json.dumps(payload)}],
        }],
    }


class FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = iter(payloads)
        self.requests: list[dict] = []
        self.last_usage = {"total_tokens": 10}
        self.last_request_id = "request-1"

    def create(self, body: dict) -> dict:
        self.requests.append(body)
        return _response(next(self.payloads))


def _accepted_payload() -> dict:
    return {
        "decision": "accept",
        "rejection_reason": "",
        "query": (
            "Buddy 从 0.11.2 升级到 0.14.x 后无法正常启动并导致模板创建失败，"
            "现场应如何恢复？"
        ),
        "reference_answer": (
            "下载 useraccess_restore.bat，关闭 Buddy，右键以管理员身份运行该文件。"
        ),
        "evidence_excerpts": [
            "下载 useraccess_restore.bat，关闭 buddy，右键使用管理员身份运行后恢复。"
        ],
        "query_type": "procedure",
        "product": "AOI",
        "module": "Buddy/模板管理",
        "quality_notes": "问题条件和恢复步骤均由当前 chunk 直接支持。",
    }


def test_accept_requires_verbatim_evidence_and_nonduplicate_query():
    payload = _accepted_payload()

    assert validate_model_payload(
        payload,
        chunk=CHUNK,
        existing_queries=[],
    ) == []

    payload["evidence_excerpts"] = ["原文中不存在的恢复步骤"]
    errors = validate_model_payload(payload, chunk=CHUNK, existing_queries=[])
    assert "evidence_0_not_verbatim" in errors


def test_thin_generic_answer_and_unnatural_query_are_rejected():
    payload = _accepted_payload()
    payload["query"] = "D盘内存满时，该故障该按什么现场处理？"
    payload["reference_answer"] = "指导客户进行D盘内存清理操作。"

    errors = validate_model_payload(payload, chunk=CHUNK, existing_queries=[])

    assert "query_mentions_source_context" in errors
    assert "answer_too_thin" in errors


def test_generator_sends_exactly_one_chunk_and_builds_traceable_case():
    client = FakeClient([_accepted_payload(), _accepted_payload()])

    generated = generate_one_chunk(
        client,
        model="gpt-test",
        chunk=CHUNK,
        chunk_index=7,
        existing_queries=[],
    )
    case = build_case_record(generated, chunk=CHUNK, case_number=1)

    assert len(client.requests) == 2
    request_text = client.requests[0]["input"][0]["content"][0]["text"]
    assert json.loads(request_text)["chunk_text"] == CHUNK["text"]
    assert case["case_id"] == "chunk-qa-0001"
    assert case["source"]["chunk_index"] == 7
    assert case["source"]["chunk_text"] == CHUNK["text"]
    assert case["answer_gold"]["evidence_excerpts"]


def test_invalid_first_attempt_is_corrected_on_same_chunk():
    bad = _accepted_payload()
    bad["query"] = "模板创建失败怎么处理？"
    client = FakeClient([bad, _accepted_payload(), _accepted_payload()])

    generated = generate_one_chunk(
        client,
        model="gpt-test",
        chunk=CHUNK,
        chunk_index=0,
        existing_queries=[],
        max_attempts=2,
    )

    assert generated.payload["decision"] == "accept"
    assert len(client.requests) == 3
    correction = client.requests[1]["input"][0]["content"][0]["text"]
    assert "previous_validation_errors" in correction


def test_independent_review_can_reject_an_authored_candidate():
    rejected = {
        **_accepted_payload(),
        "decision": "reject",
        "rejection_reason": "处理动作过于泛化，无法构成高质量样本。",
        "query": "",
        "reference_answer": "",
        "evidence_excerpts": [],
    }
    client = FakeClient([_accepted_payload(), rejected])

    generated = generate_one_chunk(
        client,
        model="gpt-test",
        chunk=CHUNK,
        chunk_index=0,
        existing_queries=[],
    )

    assert generated.payload["decision"] == "reject"
    assert [item["stage"] for item in generated.audit["attempts"]] == [
        "author",
        "review",
    ]


def test_reject_must_not_emit_partial_case():
    payload = {
        "decision": "reject",
        "rejection_reason": "只有问题，没有可判分的处理答案。",
        "query": "",
        "reference_answer": "",
        "evidence_excerpts": [],
        "query_type": "diagnosis",
        "product": "",
        "module": "",
        "quality_notes": "证据不足。",
    }

    assert validate_model_payload(payload, chunk=CHUNK, existing_queries=[]) == []


def test_codex_cli_usage_ignores_non_json_events():
    stdout = "\n".join([
        "not json",
        json.dumps({"usage": {"input_tokens": 11, "output_tokens": 3}}),
    ])

    assert _codex_cli_usage(stdout) == {"input_tokens": 11, "output_tokens": 3}
