import json
from pathlib import Path

from debug_agent_system.eval.write_side.sync_jira_offline_full import (
    _load_env_file,
    _secret_config,
    normalize_issue_bundle,
    repair_existing_bundles,
)


def test_load_env_file_handles_quotes_without_exposing_values(tmp_path: Path) -> None:
    path = tmp_path / ".env.local"
    path.write_text("# comment\nJIRA_BASE_URL='https://jira.example'\nJIRA_PAT=secret-token\n", encoding="utf-8")

    assert _load_env_file(path) == {
        "JIRA_BASE_URL": "https://jira.example",
        "JIRA_PAT": "secret-token",
    }


def test_secret_config_accepts_account_password_without_pat(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ".env.local"
    path.write_text(
        "JIRA_BASE_URL=https://jira.example\nJIRA_ACCOUNT=user@example.com\nJIRA_PWD=secret-password\n",
        encoding="utf-8",
    )
    for key in ("JIRA_BASE_URL", "JIRA_PAT", "JIRA_TOKEN", "JIRA_ACCOUNT", "JIRA_PWD"):
        monkeypatch.delenv(key, raising=False)

    assert _secret_config(path) == (
        "https://jira.example",
        "",
        "user@example.com",
        "secret-password",
    )


def test_normalize_issue_bundle_preserves_links_comments_attachments_and_history(tmp_path: Path) -> None:
    bundle = {
        "issue_key": "SMTAOITS-1234",
        "fetched_at": "2026-07-21T00:00:00+00:00",
        "issue": {
            "id": "4258",
            "key": "SMTAOITS-1234",
            "names": {"customfield_1": "Site"},
            "schema": {"customfield_1": {"type": "string"}},
            "fields": {
                "summary": "编程失败",
                "description": "导入坐标到90%后失败",
                "status": {"name": "已解决"},
                "resolution": {"name": "完成"},
                "assignee": {"displayName": "杜雷雷"},
                "reporter": {"name": "wangmengchao"},
                "issuetype": {"name": "故障报告"},
                "priority": {"name": "P2"},
                "project": {"key": "SMTAOITS", "name": "AOI现场"},
                "labels": ["component-library"],
                "fixVersions": [{"name": "v1.4.0"}],
                "versions": [{"name": "1.3.7"}],
                "components": [{"name": "SDK"}],
                "issuelinks": [
                    {
                        "id": "99",
                        "type": {"id": "10003", "name": "Relates", "outward": "relates to"},
                        "outwardIssue": {
                            "key": "SMTAOITS-1234",
                            "fields": {"summary": "拍摄失败", "status": {"name": "已解决"}},
                        },
                    }
                ],
                "attachment": [
                    {
                        "id": "7",
                        "filename": "log.zip",
                        "size": 123,
                        "mimeType": "application/zip",
                        "author": {"displayName": "王孟超"},
                        "content": "https://jira.example/attachment/7/log.zip",
                    }
                ],
            },
        },
        "comments": [
            {
                "id": "1",
                "author": {"displayName": "杜雷雷"},
                "created": "2026-06-04",
                "body": "器件库问题",
                "renderedBody": "<p>器件库问题</p>",
            }
        ],
        "changelog": [{"id": "h1", "items": [{"field": "status"}]}],
        "remote_links": [{"id": 1, "object": {"url": "https://gitlab.example/commit/39c8619f"}}],
        "worklogs": [],
        "endpoint_errors": {},
    }

    normalized = normalize_issue_bundle(bundle, tmp_path / "SMTAOITS-1234.json")

    assert normalized["key"] == "SMTAOITS-1234"
    assert normalized["fix_versions"] == ["v1.4.0"]
    assert normalized["affected_versions"] == ["1.3.7"]
    assert normalized["issue_links"][0] == {
        "id": "99",
        "type_id": "10003",
        "type_name": "Relates",
        "direction": "outward",
        "description": "relates to",
        "issue_key": "SMTAOITS-1234",
        "summary": "拍摄失败",
        "status": "已解决",
    }
    assert normalized["attachments"][0]["filename"] == "log.zip"
    assert normalized["comments"][0]["body"] == "器件库问题"
    assert normalized["changelog"][0]["id"] == "h1"
    assert normalized["remote_links"][0]["id"] == 1


def test_repair_existing_bundle_uses_complete_embedded_changelog(tmp_path: Path) -> None:
    root = tmp_path / "raw_full_v2"
    raw_path = root / "issues" / "SMTAOITS-1.json"
    detail_path = root / "fault_details" / "SMTAOITS-1.json"
    raw_path.parent.mkdir(parents=True)
    detail_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({
        "issue_key": "SMTAOITS-1",
        "issue": {
            "key": "SMTAOITS-1",
            "fields": {"summary": "test"},
            "changelog": {"total": 1, "histories": [{"id": "h1"}]},
        },
        "comments": [],
        "changelog": [],
        "endpoint_errors": {"changelog": "endpoint_not_available"},
    }), encoding="utf-8")
    detail_path.write_text("{}", encoding="utf-8")

    stats = repair_existing_bundles(root)
    repaired_raw = json.loads(raw_path.read_text(encoding="utf-8"))
    repaired_detail = json.loads(detail_path.read_text(encoding="utf-8"))

    assert stats == {"scanned": 1, "repaired": 1, "failed": 0}
    assert repaired_raw["changelog"] == [{"id": "h1"}]
    assert repaired_raw["endpoint_errors"] == {}
    assert repaired_detail["changelog"] == [{"id": "h1"}]
