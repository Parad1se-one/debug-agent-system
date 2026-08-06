import json
import sqlite3
import tempfile
from pathlib import Path

from debug_agent_system.core.config import load_config
from debug_agent_system.knowledge.sqlite_sag import SqliteSAGStore, build_sqlite_sag
from debug_agent_system.runtime import DebugAgentSystem


CAD_QUERY = (
    "现场故障：1.2.2.1.1 CAD导入报错解析失败， 导入后尺寸过大，导入后没显示，"
    "疑似CAD导入解析失败；现象补充：CAD文件导入AOI软件时解析失败，或导入后元件尺寸异常偏大、"
    "超出板卡范围，或导入后界面无任何元件显示。。请给出排查步骤，不要自动执行高风险操作。"
)

_REAL_SAG_STORE: SqliteSAGStore | None = None


def _real_sag_store() -> SqliteSAGStore:
    global _REAL_SAG_STORE
    if _REAL_SAG_STORE is None:
        _REAL_SAG_STORE = SqliteSAGStore("data/kg_sag/debug_agent.sqlite", kg_root="data/kg")
    return _REAL_SAG_STORE


def test_default_config_uses_sag_and_json_config_is_explicit():
    default = load_config("config/debug_agent_system.yaml")
    json_cfg = load_config("config/debug_agent_system_json.yaml")
    sag = load_config("config/debug_agent_system_sag.yaml")
    assert default.knowledge.store == "sqlite_sag_v2"
    assert json_cfg.knowledge.store == "kg_v2_json"
    assert sag.knowledge.store == "sqlite_sag_v2"
    assert default.retrieval.sag_max_hops == 1
    assert default.retrieval.sag_event_budget == 150


def test_sqlite_sag_build_retrieves_cad_and_reconstructs_subgraph():
    with tempfile.TemporaryDirectory() as tmp:
        raw_root = Path(tmp) / "raw"
        chunks = raw_root / "chunks"
        chunks.mkdir(parents=True)
        (chunks / "debug_chunks.json").write_text(json.dumps([{
            "text": "【SOP】CAD导入报错解析失败，导入后尺寸过大，导入后没显示。检查编码格式、XY及角度位置、特殊符号和坐标数值。",
            "metadata": {"source": "SOP", "title": "CAD导入报错解析失败， 导入后尺寸过大，导入后没显示。"},
        }], ensure_ascii=False), encoding="utf-8")
        db_path = Path(tmp) / "debug_agent.sqlite"
        report = build_sqlite_sag(
            db_path,
            raw_root=raw_root,
            kg_root="data/kg",
            w1_root=None,
        )
        assert report["counts"]["events"] > 0
        assert report["counts"]["event_entities"] > 0
        assert report["old_id_coverage"] > 0

        store = SqliteSAGStore(db_path, kg_root="data/kg")
        assert any(item.get("error_id") == "err:cad-import-failure" for item in store.errors)
        candidates = store.search_errors(CAD_QUERY, limit=5)
        assert candidates
        assert candidates[0].error_id == "err:cad-import-failure"
        trace = store.last_retrieval_trace
        assert trace.get("candidate_paths")
        assert "query_entities" in trace

        subgraph = store.load_locked_subgraph("err:cad-import-failure")
        check_ids = {check.check_id for check in subgraph.checks}
        assert "check:cad-import-failure-step1" in check_ids
        assert subgraph.next_edges_by_check == {} or isinstance(subgraph.next_edges_by_check, dict)


def test_w1_partial_source_is_retrieval_only_not_executable():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "w1_full"
        w1 = root / "w1"
        w1.mkdir(parents=True)
        (w1 / "run_manifest.json").write_text(json.dumps({
            "run_id": "mini_w1",
            "source": "test",
            "counts": {"messages": 1, "episodes": 1},
        }, ensure_ascii=False), encoding="utf-8")
        (w1 / "messages.jsonl").write_text(json.dumps({
            "message_id": "m1",
            "thread_id": "t1",
            "create_time": "2026-07-03 00:00",
            "msg_type": "text",
            "text": "现场反馈 CAD导入失败，但是还没有解决结论。",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        (w1 / "thread_summaries.json").write_text("[]", encoding="utf-8")
        (w1 / "episodes.json").write_text(json.dumps([{
            "episode_id": "ep1",
            "thread_id": "t1",
            "completeness": "partial",
            "fault_description_messages": [{"message_id": "m1", "text": "CAD导入失败"}],
            "diagnostic_chain_messages": [],
            "resolution_messages": [],
            "evidence_message_ids": ["m1"],
            "extracted": {"symptom_raw": "CAD导入失败"},
        }], ensure_ascii=False), encoding="utf-8")
        db_path = Path(tmp) / "debug_agent.sqlite"
        report = build_sqlite_sag(db_path, raw_root=Path(tmp) / "missing_raw", kg_root=Path(tmp) / "missing_kg", w1_root=root)
        assert report["w1_counts"] == {"messages": 1, "episodes": 1, "complete": 0, "partial": 1}

        conn = sqlite3.connect(db_path)
        executable_d_links = conn.execute(
            """
            SELECT COUNT(*) FROM event_links
            WHERE source_tier = 'D' AND relation IN ('has_check', 'next', 'resolved_by', 'requires_info')
            """
        ).fetchone()[0]
        d_events = conn.execute("SELECT COUNT(*) FROM events WHERE source_tier = 'D' AND needs_review = 1").fetchone()[0]
        assert executable_d_links == 0
        assert d_events >= 1


def test_runtime_sag_trace_contains_only_native_variant_paths():
    system = DebugAgentSystem.from_config("config/debug_agent_system.yaml")
    out = system.start({"query": "2D相机拍摄失败，提示操作失败"})
    candidates = out["metadata"]["retrieval"]["candidates"]
    assert candidates
    assert candidates[0]["route"] == "sag_v2_native"
    assert candidates[0]["retrieval_paths"]
    assert all(path["variant_id"].startswith("variant:") for path in candidates[0]["retrieval_paths"])


def test_sag_query_entity_noise_and_high_degree_trace_are_auditable():
    store = _real_sag_store()
    candidates = store.search_errors("客户反馈设备运行中软件会自动退出，发生时间大约20:40，应该怎么排查？", limit=10)
    assert candidates
    trace = store.last_retrieval_trace
    summary = trace["summary"]
    filtered = summary.get("filtered_query_entities") or []
    roles = trace.get("retrieval_entity_roles") or []
    retrieval_entities = {str(row.get("normalized") or "") for row in roles}
    skip_reasons = {str(row.get("skip_reason") or "") for row in filtered}
    assert "generic_device_high_degree" in skip_reasons
    assert "weak_cjk_ngram" in skip_reasons
    assert "temporal_context" in skip_reasons
    assert "设备" not in retrieval_entities
    assert "客户反馈" not in retrieval_entities
    assert "自动退出" in retrieval_entities
    assert all(str(row.get("role")) not in {"noise", "action_tried", "temporal_context"} for row in roles)
    assert all(
        not (str(row.get("raw_role")) == "cjk_ngram" and str(row.get("role")) == "domain")
        for row in roles
    )


def test_sag_no_power_keeps_specific_device_and_symptom_entities():
    store = _real_sag_store()
    candidates = store.search_errors("工控机完全无通电反应，风扇不转，开机键指示灯不亮，键盘灯和鼠标灯也不亮，应该怎么排查？", limit=5)
    assert candidates[0].error_id == "err:industrial-pc-no-boot"
    roles = store.last_retrieval_trace.get("retrieval_entity_roles") or []
    normalized = {str(row.get("normalized") or "") for row in roles}
    assert "工控机" in normalized
    assert "风扇不转" in normalized or "无通电" in normalized
    assert store.last_retrieval_trace["summary"]["d_only_top_candidate"] is False


def test_sag_explicit_board_contact_outranks_generic_running_black_screen():
    store = _real_sag_store()
    candidates = store.search_errors(
        "工控机在潮湿、多粉尘、振动环境运行一段时间后时好时坏，像硬件自检、短路或黑屏类故障，"
        "怀疑接口或板卡氧化接触不良，怎么排查？",
        limit=5,
    )
    assert candidates[0].error_id == "err:pcie-board-not-detected-by-system"


def test_sag_explicit_cad_angle_mismatch_outranks_generic_auto_alignment_failure():
    store = _real_sag_store()
    candidates = store.search_errors(
        "若自动对齐仍无法将所有拼板对齐，手动逐个对齐，疑似CAD器件角度不匹配；"
        "编程时部分器件框的角度与器件真实角度不匹配。",
        limit=5,
    )
    assert candidates[0].error_id == "err:cad-angle-mismatch"


def test_sag_kg_v2_variant_maps_back_to_legacy_error_id():
    store = _real_sag_store()
    candidates = store.search_errors("Buddy安装报错，应该怎么排查？", limit=5)
    assert candidates
    assert candidates[0].error_id == "err:deploy-buddy-install-error"
    candidate_ids = [item.error_id for item in candidates]
    assert not any(item.startswith("errv2:") for item in candidate_ids)
    mapping = store.conn.execute(
        """
        SELECT confidence, needs_review
        FROM event_links
        WHERE from_event_id LIKE 'event:kgv2_error:%'
          AND to_event_id = 'event:error:err:deploy-buddy-install-error'
          AND relation = 'maps_to_error'
        ORDER BY confidence DESC
        LIMIT 1
        """
    ).fetchone()
    assert mapping is not None
    assert float(mapping["confidence"]) >= 0.8
    assert int(mapping["needs_review"]) == 0
