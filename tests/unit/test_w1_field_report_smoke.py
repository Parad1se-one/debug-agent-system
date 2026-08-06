from __future__ import annotations

from debug_agent_system.eval.write_side.w1_field_report_smoke import _select_field_report_messages


def _message(index: int, text: str, hour: int) -> dict:
    return {
        "message_id": f"m{index}",
        "thread_id": "thread-1",
        "create_time": f"2026-01-01 {hour:02d}:00",
        "sender": {"name": "fae"},
        "text": text,
    }


def test_adaptive_selection_keeps_full_session_past_old_plus_six_window() -> None:
    items = [_message(0, "现场情况反馈：1.相机拍摄失败", 9)]
    items.extend(_message(index, "排查过程消息", 9 + index) for index in range(1, 12))
    items.append(_message(12, "更换相机网络后恢复正常", 21))

    selected, stats = _select_field_report_messages(
        {"thread-1": items}, limit=5, selection_mode="adaptive", quiet_gap_hours=12
    )

    assert len(selected) == 13
    assert "m12" in selected
    assert stats["truncated_windows"] == 0
    assert stats["window_max"] == 13


def test_adaptive_selection_stops_at_next_anchor_and_starts_new_session() -> None:
    items = [
        _message(0, "现场情况反馈：1.相机拍摄失败", 9),
        _message(1, "相机排查中", 10),
        _message(2, "相机恢复正常", 11),
        _message(3, "现场情况反馈：1.工控机蓝屏", 12),
        _message(4, "收集 DMP", 13),
    ]

    selected, stats = _select_field_report_messages(
        {"thread-1": items}, limit=0, selection_mode="adaptive", quiet_gap_hours=12
    )

    assert len(selected) == 5
    assert stats["anchors_selected"] == 2
    assert stats["truncated_windows"] == 0
