from __future__ import annotations

from debug_agent_system.agents.write.w6_review_queue import _evidence_pack, _typed_review_context


def test_w6_shows_w7_promoted_message_and_linked_jira():
    episode = {
        "evidence_message_ids": ["m1", "m2"],
        "fault_description_messages": [{"message_id": "m1", "text": "扫码后不进板。"}],
        "case_evidence_messages": [{
            "message_id": "m2",
            "text": "相关问题已提交 TEST-1234：外置扫码枪扫码后无法进板。",
            "raw_text": "同一条消息还列出了无关的 TEST-1234。",
        }],
        "extracted": {
            "linked_jira_evidence": [{"issue_key": "TEST-1234", "summary": "扫码后无法进板"}],
        },
    }

    pack = _evidence_pack(episode)

    promoted = next(item for item in pack["messages"] if item["message_id"] == "m2")
    assert promoted["role"] == "case_evidence_messages"
    assert "TEST-1234" in promoted["content_summary"]
    assert "TEST-1234" in promoted["raw_content_summary"]
    assert pack["linked_jira_evidence"][0]["issue_key"] == "TEST-1234"


def test_w6_exposes_trace_link_as_non_authoritative_review_context():
    episode = {
        "episode_id": "ep-2",
        "trace_group_id": "w7-trace:abc",
        "trace_phase_index": 2,
        "trace_phase_count": 3,
        "trace_relation_type": "continuation_of",
        "previous_trace_episode_id": "ep-1",
        "trace_link_strength": "weak",
        "trace_link_reasons": ["shared_distinctive_fault_signature"],
        "trace_link_candidates": [{
            "candidate_episode_id": "ep-1",
            "linked": True,
            "link_strength": "weak",
        }],
    }

    context = _typed_review_context({
        "source_type": "chat",
        "source_ref": {"episode_id": "ep-2"},
        "payload": {"episode": episode},
    })

    trace = context["w7_trace_context"]
    assert trace["trace_group_id"] == "w7-trace:abc"
    assert trace["phase_index"] == 2
    assert trace["link_strength"] == "weak"
    assert trace["evidence_sharing_allowed"] is False
    assert trace["outcome_sharing_allowed"] is False
