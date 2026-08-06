from __future__ import annotations

import json
import tarfile
import tempfile
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from debug_agent_system.adapters.cli import main as cli_main
from debug_agent_system.agents import (
    DocumentParserAgent as PublicDocumentParserAgent,
    EvidenceContextParserAgent as PublicEvidenceContextParserAgent,
    EvidenceToolAgent as PublicEvidenceToolAgent,
    parse_attachment_evidence as public_parse_attachment_evidence,
    parse_document_evidence as public_parse_document_evidence,
    parse_dmp_evidence as public_parse_dmp_evidence,
    parse_evidence_context as public_parse_evidence_context,
    parse_image_evidence as public_parse_image_evidence,
    parse_jira_evidence as public_parse_jira_evidence,
    parse_log_package_evidence as public_parse_log_package_evidence,
    parse_proj_evidence as public_parse_proj_evidence,
)
from debug_agent_system.agents.tools import (
    AttachmentParserAgent,
    DocumentParserAgent,
    DmpParserAgent,
    EvidenceContextParserAgent,
    EvidenceToolAgent,
    ImageParserAgent,
    JiraParserAgent,
    LogPackageParserAgent,
    ProjParserAgent,
    parse_attachment_evidence,
    parse_document_evidence,
    parse_dmp_evidence,
    parse_evidence_context,
    parse_image_evidence,
    parse_jira_evidence,
    parse_log_package_evidence,
    parse_proj_evidence,
)


def _png_header(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00"


def _webp_vp8x_header(width: int, height: int) -> bytes:
    w = (width - 1).to_bytes(3, "little")
    h = (height - 1).to_bytes(3, "little")
    payload = b"\x00\x00\x00\x00" + w + h
    return b"RIFF" + (len(payload) + 12).to_bytes(4, "little") + b"WEBP" + b"VP8X" + len(payload).to_bytes(4, "little") + payload


def test_attachment_parser_classifies_project_log_and_image_metadata_only():
    parser = AttachmentParserAgent()
    project = parser.parse({"name": "recipe.proj", "path": "/tmp/recipe.proj", "kind": "file", "size": 12})
    log = parser.parse("DLOG_AOI_20260601.zip")
    image = parser.parse({"name": "capture.webp", "kind": "image"})
    assert project["evidence_role"] == "program_file"
    assert project["content_read"] is False
    assert log["evidence_role"] == "log_package"
    assert log["archive_extracted"] is False
    assert image["evidence_role"] == "sample_image"


def test_image_parser_reads_png_and_webp_header_metadata_only():
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "capture.png"
        webp = Path(tmp) / "middle.webp"
        png.write_bytes(_png_header(1920, 1080))
        webp.write_bytes(_webp_vp8x_header(640, 480))

        png_out = ImageParserAgent().parse(png)
        webp_out = EvidenceToolAgent().parse("image", str(webp))

        assert png_out["type"] == "ImageParseResult"
        assert png_out["image_format"] == "png"
        assert png_out["width"] == 1920
        assert png_out["height"] == 1080
        assert png_out["header_read"] is True
        assert png_out["pixels_read"] is False
        assert png_out["ocr_performed"] is False
        assert png_out["content_read"] is False
        assert webp_out["tool_entry"]["tool"] == "image"
        assert webp_out["image_format"] == "webp"
        assert webp_out["width"] == 640
        assert webp_out["height"] == 480


def test_attachment_parser_reads_bounded_text_preview_for_safe_text_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "现场记录.txt"
        path.write_text(
            "version=1.3.5\ncamera_ip=192.168.1.10\nERROR init camera failed 0x80070005\nJIRA=SMTAOITS-1234\n",
            encoding="utf-8",
        )
        out = AttachmentParserAgent().parse(path, max_preview_bytes=1024)
        assert out["evidence_role"] == "data_file"
        assert out["content_read"] is True
        assert out["text_preview_read"] is True
        assert out["archive_extracted"] is False
        assert out["link_fetched"] is False
        assert "1.3.5" in out["key_hints"]["versions"]
        assert "192.168.1.10" in out["key_hints"]["ip_addresses"]
        assert "0x80070005" in out["key_hints"]["error_codes"]
        assert "SMTAOITS-1234" in out["key_hints"]["jira_ids"]
        assert "startup" in out["key_hints"]["phase_hints"]


def test_attachment_parser_keeps_pdf_and_image_metadata_only():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "现场报告.pdf"
        png = Path(tmp) / "截图.png"
        pdf.write_bytes(b"%PDF-1.7 fake")
        png.write_bytes(b"not really image")
        pdf_out = AttachmentParserAgent().parse(pdf)
        png_out = AttachmentParserAgent().parse(png)
        assert pdf_out["evidence_role"] == "data_file"
        assert pdf_out["content_read"] is False
        assert pdf_out["safe_to_read_text_preview"] is False
        assert png_out["evidence_role"] == "sample_image"
        assert png_out["content_read"] is False


def test_document_parser_reads_pdf_bounded_metadata_without_ocr_or_rendering():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "现场报告.pdf"
        pdf.write_bytes(b"%PDF-1.7\n1 0 obj << /Type /Page >> endobj\nBT (AOI init failed) Tj ET\n%%EOF")

        out = DocumentParserAgent().parse(pdf, max_preview_bytes=128)

        assert out["type"] == "DocumentParseResult"
        assert out["document_format"] == "pdf"
        assert out["pdf_version"] == "1.7"
        assert out["page_count_hint"] == 1
        assert out["header_read"] is True
        assert out["text_preview_read"] is True
        assert out["archive_extracted"] is False
        assert out["macros_executed"] is False
        assert out["formulas_evaluated"] is False
        assert out["ocr_performed"] is False
        assert "AOI init failed" in out["text_preview"]


def test_document_parser_reads_ooxml_manifest_and_text_preview_without_extracting():
    with tempfile.TemporaryDirectory() as tmp:
        xlsx = Path(tmp) / "诊断记录.xlsx"
        with zipfile.ZipFile(xlsx, "w") as zf:
            zf.writestr("docProps/core.xml", "<cp:coreProperties><dc:title>初始化失败</dc:title></cp:coreProperties>")
            zf.writestr("xl/sharedStrings.xml", "<sst><si><t>DLOG 初始化阶段报错</t></si></sst>")

        out = EvidenceToolAgent().parse("document", str(xlsx), max_bytes=1024)

        assert out["tool_entry"]["tool"] == "document"
        assert out["document_format"] == "excel_ooxml"
        assert out["archive_manifest_read"] is True
        assert out["archive_extracted"] is False
        assert out["text_preview_read"] is True
        assert out["macros_executed"] is False
        assert out["formulas_evaluated"] is False
        assert "DLOG 初始化阶段报错" in out["text_preview"]


def test_document_parser_keeps_legacy_office_ole_header_metadata_only():
    with tempfile.TemporaryDirectory() as tmp:
        xls = Path(tmp) / "旧版记录.xls"
        xls.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 128)

        out = DocumentParserAgent().parse(xls)

        assert out["document_format"] == "excel_ole"
        assert out["header_read"] is True
        assert out["ole_compound_file"] is True
        assert out["text_preview_read"] is False
        assert out["archive_extracted"] is False
        assert out["macros_executed"] is False
        assert out["formulas_evaluated"] is False


def test_jira_parser_extracts_issue_keys_and_does_not_fetch():
    with tempfile.TemporaryDirectory() as tmp:
        out = JiraParserAgent(offline_root=Path(tmp)).parse("[SMTAOITS-1234] 1.3.5 客户02 设备报错“应用异常”，之后闪退 - Jira https://jira.example.com/browse/SMTAOITS-1234")
        assert out["issue_keys"] == ["SMTAOITS-1234"]
        assert out["urls"][0]["type"] == "jira"
        assert out["title_hints"] == ["1.3.5 客户02 设备报错“应用异常”，之后闪退"]
        assert out["version_hints"] == ["1.3.5"]
        assert out["site_hints"] == ["客户02"]
        assert out["issue_summaries"][0]["title"] == "1.3.5 客户02 设备报错“应用异常”，之后闪退"
        assert out["fetched"] is False
        assert out["status"] == "metadata_only"


def test_jira_parser_adds_local_offline_description_and_comments_as_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "fault_details").mkdir()
        (root / "fault_details" / "SMTAOITS-1234.json").write_text(json.dumps({
            "key": "SMTAOITS-1234",
            "summary": "1.3.5 客户02 CAD导入解析失败",
            "status": "开放",
            "resolution": "Unresolved",
            "assignee": "owner1",
            "reporter": "fae1",
            "description": "客户描述：CAD导入报错解析失败，导入后尺寸过大，导入后没显示。",
            "comments": [
                {"author": "eng1", "created": "2026-01-01T00:00:00.000+0800", "body": "需要提供CAD文件、软件版本和报错截图。"}
            ],
        }, ensure_ascii=False), encoding="utf-8")

        out = JiraParserAgent(offline_root=root).parse("https://jira.example.com/browse/SMTAOITS-1234")

        assert out["fetched"] is False
        assert out["status"] == "offline_detail_found"
        assert out["offline_detail_found"] is True
        assert out["offline_details"][0]["summary"] == "1.3.5 客户02 CAD导入解析失败"
        assert "CAD导入报错解析失败" in out["description_hints"][0]
        assert "需要提供CAD文件" in out["comment_hints"][0]


def test_jira_parser_exposes_full_v2_link_and_attachment_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "fault_details").mkdir()
        (root / "fault_details" / "SMTAOITS-1234.json").write_text(json.dumps({
            "schema_version": "debug_agent_system.jira_offline_full.v2",
            "key": "SMTAOITS-1234",
            "summary": "导入坐标时进度到90后不动",
            "issue_links": [{"issue_key": "SMTAOITS-1234", "direction": "inward"}],
            "attachments": [{"filename": "现场数据.zip", "size": 1024}],
            "remote_links": [{"id": 1}],
            "changelog": [{"id": "history-1"}],
            "worklogs": [],
            "comments": [],
        }, ensure_ascii=False), encoding="utf-8")

        out = JiraParserAgent(offline_root=root).parse("SMTAOITS-1234")
        detail = out["offline_details"][0]

        assert detail["linked_issue_keys"] == ["SMTAOITS-1234"]
        assert detail["attachments"][0]["filename"] == "现场数据.zip"
        assert detail["remote_links_count"] == 1
        assert detail["changelog_count"] == 1
        assert detail["schema_version"] == "debug_agent_system.jira_offline_full.v2"


def test_dmp_parser_reads_header_metadata_only():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.DMP"
        path.write_bytes(b"PAGEDU64" + b"\x00" * 128 + b"BugCheck 0x0000007E")

        out = DmpParserAgent().parse(path, max_header_bytes=256)

        assert out["type"] == "DmpParseResult"
        assert out["header_signature"] == "PAGEDU64"
        assert out["dump_kind"] == "windows_kernel_or_complete_dump_64_candidate"
        assert out["architecture_hint"] == "x64"
        assert out["windbg_ready"] is True
        assert out["debugger_executed"] is False
        assert out["full_content_read"] is False
        assert any("BugCheck" in item for item in out["bugcheck_hints"])


def test_log_package_parser_lists_7z_manifest_with_bsdtar_when_available():
    import shutil
    import subprocess

    bsdtar = shutil.which("bsdtar")
    if not bsdtar:
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = root / "MEMORY.DMP"
        payload.write_bytes(b"PAGEDU64" + b"\x00" * 64)
        archive = root / "dump.7z"
        subprocess.run([bsdtar, "-caf", str(archive), "-C", str(root), payload.name], check=True, capture_output=True)

        out = LogPackageParserAgent().parse(archive)

        assert out["archive_format"] == "7z"
        assert out["archive_listing_supported"] is True
        assert out["archive_extracted"] is False
        assert out["has_dmp"] is True
        assert any(entry["name"].endswith("MEMORY.DMP") for entry in out["entries"])


def test_proj_parser_reads_bounded_preview_without_execution_or_mutation():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "board.proj"
        path.write_text("Version=1.3.5 CameraIP=192.168.1.10 相机=Basler " + "x" * 120, encoding="utf-8")
        out = ProjParserAgent().parse(path, max_bytes=80)
        assert out["type"] == "ProjParseResult"
        assert out["content_read"] is True
        assert out["truncated"] is True
        assert out["executed"] is False
        assert out["mutated"] is False
        assert "1.3.5" in out["key_hints"]["versions"]
        assert "192.168.1.10" in out["key_hints"]["ip_addresses"]
        assert "192.168.1.10" not in out["key_hints"]["versions"]


def test_proj_parser_reads_tar_manifest_and_bounded_text_entries_without_extracting():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recipe.proj"
        meta = b'{"name":"BOARD_AOI_TEST","version":"2.0.0","pcb_type":"TOP","device":"T90"}'
        rev = b'{"app_version":"1.3.5","sdk_version":"0.99.0+abc","models":[{"type":"MODEL_TYPE_DETECTION_COMPONENT"}],"files":{"0.csv":"0.csv","image_rgb":"image_rgb.tpg"}}'
        csv = b'R1 1.23 4.56 90 PART_A'
        image = b'binary-image' * 100
        with tarfile.open(path, "w") as tf:
            for name, raw in [("meta.json", meta), ("rev.abc.json", rev), ("0.csv", csv), ("image_rgb.tpg", image)]:
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                import io
                tf.addfile(info, io.BytesIO(raw))
        out = ProjParserAgent().parse(path, max_bytes=4096)
        assert out["archive_format"] == "tar"
        assert out["archive_manifest_read"] is True
        assert out["archive_extracted"] is False
        assert out["executed"] is False
        assert out["mutated"] is False
        assert out["text_entry_preview_read"] is True
        assert "BOARD_AOI_TEST" in out["key_hints"]["project_names"]
        assert out["key_hints"]["app_versions"] == ["1.3.5"]
        assert "2.0.0" in out["key_hints"]["schema_versions"]
        assert "MODEL_TYPE_DETECTION_COMPONENT" in out["key_hints"]["model_types"]
        assert "component_table" in out["key_hints"]["file_roles"]
        assert out["key_hints"]["has_board_images"] is True
        image_entry = next(item for item in out["entries"] if item["name"] == "image_rgb.tpg")
        assert image_entry["text_preview_read"] is False
        assert "1.23" not in out["key_hints"]["versions"]


def test_log_package_parser_reads_zip_manifest_and_bounded_text_hints_without_extracting():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "DLOG_init.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("startup/init.log", "2026-06-01 ERROR camera init failed code 0x80070005\nBugCheck 0x0000007E")
            zf.writestr("crash/MEMORY.DMP", b"binary dump")
            zf.writestr("windows/system.evtx", b"event log")
        out = LogPackageParserAgent().parse(path)
        assert out["type"] == "LogPackageParseResult"
        assert out["status"] == "text_hints"
        assert out["archive_listing_supported"] is True
        assert out["archive_extracted"] is False
        assert out["content_read"] is False
        assert out["text_preview_read"] is True
        assert "0x80070005" in out["text_hints"]["error_codes"]
        assert "startup" in out["text_hints"]["phase_hints"]
        assert out["has_dmp"] is True
        assert out["has_evtx"] is True
        assert out["has_startup_log"] is True
        assert {item["source"] for item in out["entries"]} == {"zip_central_directory"}
        dmp_entry = next(item for item in out["entries"] if item["role"] == "memory_dump")
        assert dmp_entry["text_preview_read"] is False


def test_evidence_context_parser_uses_source_manifest_and_routes_files():
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "20260603_sample"
        raw_dir = sample / "raw"
        raw_dir.mkdir(parents=True)
        proj = raw_dir / "board.proj"
        proj.write_text("Version=1.3.5 CameraIP=192.168.1.10", encoding="utf-8")
        dmp = raw_dir / "MEMORY.DMP"
        dmp.write_bytes(b"PAGEDU64" + b"\x00" * 64 + b"BugCheck 0x00000139")
        zip_path = raw_dir / "DLOG.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("startup/init.log", "ERROR init failed 0x80070005")
            zf.writestr("dump/MEMORY.DMP", b"dump")
        (sample / "source_manifest.json").write_text(json.dumps({
            "sample_id": "20260603_sample",
            "chat_name": "测试群",
            "segment_id": "oc_test",
            "anchor_messages": ["请提供 DMP 和日志包"],
            "expected_tools": ["dmp", "log_package", "proj"],
            "files": [
                {"name": proj.name, "relative_path": "raw/board.proj", "size": proj.stat().st_size},
                {"name": dmp.name, "relative_path": "raw/MEMORY.DMP", "size": dmp.stat().st_size},
                {"name": zip_path.name, "relative_path": "raw/DLOG.zip", "size": zip_path.stat().st_size},
            ],
        }, ensure_ascii=False), encoding="utf-8")

        out = EvidenceContextParserAgent().parse_context(sample, max_bytes=512)
        context = out["contexts"][0]

        assert out["type"] == "EvidenceContextParseResult"
        assert out["context_count"] == 1
        assert out["tool_counts"]["proj_parse_results"] == 1
        assert out["tool_counts"]["dmp_parse_results"] == 1
        assert out["tool_counts"]["log_package_parse_results"] == 1
        assert context["source_context"]["chat_name"] == "测试群"
        assert "1.3.5" in context["summary_hints"]["versions"]
        assert "192.168.1.10" in context["summary_hints"]["ip_addresses"]
        assert context["summary_hints"]["has_dmp"] is True
        assert context["safety"]["archive_extracted"] is False
        assert context["safety"]["debugger_executed"] is False


def test_evidence_context_parser_reads_jira_offline_fault_details():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "jira_offline" / "raw" / "fault_details"
        root.mkdir(parents=True)
        issue = root / "SMTAOI-9.json"
        issue.write_text(json.dumps({
            "key": "SMTAOI-9",
            "summary": "1.3.5 客户02 CAD导入解析失败",
            "description": "客户描述：CAD导入后尺寸过大，导入后没显示。",
            "comments": [{"author": "eng", "body": "需要提供CAD文件和软件版本。"}],
        }, ensure_ascii=False), encoding="utf-8")

        out = parse_evidence_context(root.parent, limit=1)
        context = out["contexts"][0]

        assert context["context_id"] == "SMTAOI-9"
        assert out["tool_counts"]["jira_parse_results"] == 1
        jira = context["tool_evidence"]["jira_parse_results"][0]
        assert jira["status"] == "offline_detail_found"
        assert "CAD导入后尺寸过大" in jira["description_hints"][0]
        assert "SMTAOI-9" in context["summary_hints"]["issue_keys"]


def test_cli_parse_evidence_context_outputs_json():
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample"
        raw_dir = sample / "raw"
        raw_dir.mkdir(parents=True)
        log_path = raw_dir / "DLOG.zip"
        with zipfile.ZipFile(log_path, "w") as zf:
            zf.writestr("startup.log", "ERROR startup failed")
        (sample / "source_manifest.json").write_text(json.dumps({
            "sample_id": "sample",
            "files": [{"name": log_path.name, "relative_path": "raw/DLOG.zip"}],
        }), encoding="utf-8")

        buf = StringIO()
        with redirect_stdout(buf):
            assert cli_main(["parse-evidence-context", str(sample)]) == 0
        out = json.loads(buf.getvalue())
        assert out["type"] == "EvidenceContextParseResult"
        assert out["tool_counts"]["log_package_parse_results"] == 1


def test_cli_tool_entries_are_independent_and_json_outputs():
    buf = StringIO()
    with redirect_stdout(buf):
        assert cli_main(["parse-jira", "https://jira.example.com/browse/SMTAOITS-1234"]) == 0
    out = json.loads(buf.getvalue())
    assert out["type"] == "JiraParseResult"
    assert out["issue_keys"] == ["SMTAOITS-1234"]


def test_evidence_tool_agent_is_stable_entry_for_other_agents():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recipe.proj"
        path.write_text("Version=2.4.6\nCameraIP=10.1.2.3\n", encoding="utf-8")
        agent = EvidenceToolAgent()

        proj = agent.parse("proj", str(path), max_bytes=64)
        jira = agent.parse("jira", "https://jira.example.com/browse/SMTAOITS-1234")
        attachment = agent.parse("attachment", {"name": "DLOG_init.zip", "path": "/tmp/DLOG_init.zip"})
        inferred = agent.infer_and_parse("https://jira.example.com/browse/SMTAOITS-1234")

        assert proj["type"] == "ProjParseResult"
        assert proj["tool_entry"]["tool"] == "proj"
        assert proj["executed"] is False
        assert "2.4.6" in proj["key_hints"]["versions"]
        assert "10.1.2.3" in proj["key_hints"]["ip_addresses"]
        assert "10.1.2.3" not in proj["key_hints"]["versions"]
        assert jira["issue_keys"] == ["SMTAOITS-1234"]
        assert jira["tool_entry"]["tool"] == "jira"
        assert attachment["evidence_role"] == "log_package"
        assert attachment["archive_extracted"] is False
        assert inferred["issue_keys"] == ["SMTAOITS-1234"]


def test_evidence_tool_agent_routes_log_package_and_cli_accepts_it():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "DLOG_init.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("DLOG/startup.log", "not read by parser")
        out = EvidenceToolAgent().parse("log_package", str(path))
        assert out["tool_entry"]["tool"] == "log_package"
        assert out["has_startup_log"] is True
        buf = StringIO()
        with redirect_stdout(buf):
            assert cli_main(["parse-evidence", "log_package", str(path)]) == 0
        cli_out = json.loads(buf.getvalue())
        assert cli_out["type"] == "LogPackageParseResult"
        assert cli_out["tool_entry"]["tool"] == "log_package"


def test_evidence_tool_agent_routes_document_and_cli_accepts_it():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "现场报告.pdf"
        path.write_bytes(b"%PDF-1.4\n1 0 obj << /Type /Page >> endobj\n%%EOF")
        out = EvidenceToolAgent().infer_and_parse(str(path))
        assert out["tool_entry"]["tool"] == "document"
        assert out["document_format"] == "pdf"
        buf = StringIO()
        with redirect_stdout(buf):
            assert cli_main(["parse-evidence", "document", str(path)]) == 0
        cli_out = json.loads(buf.getvalue())
        assert cli_out["type"] == "DocumentParseResult"
        assert cli_out["tool_entry"]["tool"] == "document"


def test_evidence_tool_agent_degrades_unknown_tools_to_parse_failed():
    out = EvidenceToolAgent().parse("unknown", {"name": "x.bin"})
    assert out["type"] == "EvidenceToolError"
    assert out["status"] == "parse_failed"
    assert out["tool_entry"]["tool"] == "unknown"


def test_cli_parse_evidence_accepts_json_payload():
    buf = StringIO()
    payload = json.dumps({"name": "capture.png", "kind": "image"}, ensure_ascii=False)
    with redirect_stdout(buf):
        assert cli_main(["parse-evidence", "attachment", payload]) == 0
    out = json.loads(buf.getvalue())
    assert out["type"] == "AttachmentParseResult"
    assert out["evidence_role"] == "sample_image"
    assert out["tool_entry"]["tool"] == "attachment"


def test_public_tool_functions_are_stable_entries_for_other_agents():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "line.proj"
        path.write_text("Version=3.2.1\nCameraIP=172.16.1.9\n", encoding="utf-8")

        proj = parse_proj_evidence(path, max_bytes=128)
        jira = parse_jira_evidence({"url": "https://jira.example.com/browse/SMTAOITS-1234"})
        attachment = parse_attachment_evidence({"name": "现场截图.png", "kind": "image"})
        image_path = Path(tmp) / "capture.png"
        image_path.write_bytes(_png_header(320, 240))
        image = parse_image_evidence(image_path)
        document_path = Path(tmp) / "report.pdf"
        document_path.write_bytes(b"%PDF-1.7\n%%EOF")
        document = parse_document_evidence(document_path)
        dmp_path = Path(tmp) / "MEMORY.DMP"
        dmp_path.write_bytes(b"PAGEDU64" + b"\x00" * 32)
        dmp = parse_dmp_evidence(dmp_path)
        log_package = parse_log_package_evidence({"name": "MEMORY.DMP", "kind": "file", "bytes": "123"})

        assert proj["type"] == "ProjParseResult"
        assert proj["tool_entry"]["tool"] == "proj"
        assert proj["executed"] is False
        assert "3.2.1" in proj["key_hints"]["versions"]
        assert "172.16.1.9" in proj["key_hints"]["ip_addresses"]
        assert jira["type"] == "JiraParseResult"
        assert jira["tool_entry"]["tool"] == "jira"
        assert jira["issue_keys"] == ["SMTAOITS-1234"]
        assert jira["fetched"] is False
        assert attachment["type"] == "AttachmentParseResult"
        assert attachment["tool_entry"]["tool"] == "attachment"
        assert attachment["evidence_role"] == "sample_image"
        assert attachment["content_read"] is False
        assert image["tool_entry"]["tool"] == "image"
        assert image["width"] == 320
        assert image["height"] == 240
        assert document["tool_entry"]["tool"] == "document"
        assert document["document_format"] == "pdf"
        assert dmp["tool_entry"]["tool"] == "dmp"
        assert dmp["windbg_ready"] is True
        assert log_package["type"] == "LogPackageParseResult"
        assert log_package["tool_entry"]["tool"] == "log_package"
        assert log_package["has_dmp"] is True


def test_top_level_agents_package_exports_tool_entries_for_other_agents():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "line.proj"
        path.write_text("Version=6.7.8\nCameraIP=10.9.8.7\n", encoding="utf-8")

        proj = public_parse_proj_evidence(path)
        jira = public_parse_jira_evidence("https://jira.example.com/browse/SMTAOITS-1234")
        log_package = public_parse_log_package_evidence({"name": "system.evtx", "kind": "file"})
        attachment = public_parse_attachment_evidence({"name": "DLOG_startup.zip", "kind": "file"})
        image_path = Path(tmp) / "capture.png"
        image_path.write_bytes(_png_header(800, 600))
        image = public_parse_image_evidence(image_path)
        document_path = Path(tmp) / "report.pdf"
        document_path.write_bytes(b"%PDF-1.7\n%%EOF")
        document = public_parse_document_evidence(document_path)
        dmp_path = Path(tmp) / "MEMORY.DMP"
        dmp_path.write_bytes(b"PAGEDU64" + b"\x00" * 32)
        dmp = public_parse_dmp_evidence(dmp_path)
        context = public_parse_evidence_context(path)
        routed = PublicEvidenceToolAgent().parse("attachment", {"name": "现场截图.png", "kind": "image"})

        assert proj["tool_entry"]["tool"] == "proj"
        assert "6.7.8" in proj["key_hints"]["versions"]
        assert "10.9.8.7" in proj["key_hints"]["ip_addresses"]
        assert jira["issue_keys"] == ["SMTAOITS-1234"]
        assert log_package["tool_entry"]["tool"] == "log_package"
        assert log_package["has_evtx"] is True
        assert attachment["evidence_role"] == "log_package"
        assert image["tool_entry"]["tool"] == "image"
        assert image["width"] == 800
        assert PublicDocumentParserAgent().parse(document_path)["document_format"] == "pdf"
        assert document["tool_entry"]["tool"] == "document"
        assert dmp["tool_entry"]["tool"] == "dmp"
        assert dmp["windbg_ready"] is True
        assert context["tool_counts"]["proj_parse_results"] == 1
        assert PublicEvidenceContextParserAgent().parse_context(path)["tool_counts"]["proj_parse_results"] == 1
        assert routed["evidence_role"] == "sample_image"
