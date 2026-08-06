from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

from debug_agent_system.incident_runtime.artifacts import ArtifactIntake, ArtifactLimits
from debug_agent_system.incident_runtime.runtime import IncidentEvidenceRuntime
from debug_agent_system.incident_runtime.scope import parse_incident_scope
from debug_agent_system.knowledge_v2.read_model import V2Candidate


CUDA_LOG = """[2026-08-01 21:30:00,093] ACME.symv.RuntimeError [0x00004774] ERROR - SYMV(1.1.0, bebc9fb04a2bf1e37ac51bf9135a3a11c0867d99) symv: subsystem error (-217) Gpu API call, an illegal memory access was encountered)
in cv::cuda::GpuMat::upload, file C:\\GitLab-Runner\\builds\\opencv\\modules\\core\\src\\cuda\\gpu_mat.cu, line 240, trace:
0# 18b97e5 7ffbcb6797e5 symv::SYAllocator::setDefaultAllocator in symv
1# 18e8ef4 7ffbcb6a8ef4 symv::cuda_init in symv
2# 1ba15c8 7ffbcb9615c8 symv::template_match in symv
"""


class FakeReadModel:
    def __init__(self) -> None:
        self.last_retrieval = {"chunks": [], "paths": [], "trace": {}}
        self.by_type = {"SourceCase": {}}

    def search_variants(self, query: str, limit: int = 10):
        self.query = query
        return [
            V2Candidate(
                family_id="family:gpu",
                variant_id="variant:cuda-illegal-access",
                family_label="GPU/CUDA 异常",
                variant_label="CUDA 非法内存访问",
                score=18.0,
                matched_fields=["cuda", "illegal_memory_access"],
                evidence_ids=["kg-evidence:gpu"],
                supporting_chunks=[{"text": "CUDA illegal memory access"}],
            )
        ]

    def get(self, object_id: str):
        if object_id == "family:gpu":
            return {"label": "GPU/CUDA", "subsystem": "gpu_cuda"}
        if object_id == "variant:cuda-illegal-access":
            return {"label": "CUDA 非法内存访问", "summary": "GPU 异步任务报告非法内存访问"}
        return None

    def compile_plan(self, family_id: str, variant_id: str):
        raise KeyError(variant_id)

    def required_info(self, values):
        return []


def test_cuda_log_builds_source_closed_incident_pack() -> None:
    model = FakeReadModel()
    runtime = IncidentEvidenceRuntime(model)  # type: ignore[arg-type]

    result = runtime.analyze(CUDA_LOG, [])

    assert result.status == "analyzed"
    assert result.events
    assert "-217" in result.events[0].error_codes
    assert result.stack_traces
    assert any(
        frame.function == "symv::cuda_init"
        for trace in result.stack_traces
        for frame in trace.frames
    )
    assert result.environment.values["symv_version"] == ["1.1.0"]
    assert result.evidence_pack["schema_version"] == "debug_agent_system.incident_evidence_pack.v3"
    assert result.verification["passed"] is True
    assert "异常被观察到的位置，不自动等同于根因位置" in result.report
    assert result.observability["canonical_kg_mutated"] is False


def test_volatile_stack_addresses_do_not_enter_kg_query() -> None:
    model = FakeReadModel()
    runtime = IncidentEvidenceRuntime(model)  # type: ignore[arg-type]
    runtime.analyze(CUDA_LOG, [])

    assert "7ffbcb6797e5" not in model.query
    assert "bebc9fb04a2bf1e37ac51bf9135a3a11c0867d99" not in model.query
    assert "C:\\GitLab-Runner" not in model.query
    assert "illegal_memory_access" in model.query


def test_safe_archive_manifest_rejects_path_traversal() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("logs/app.log", "ERROR CUDA failure -217")
        archive.writestr("../../escape.log", "must not be accepted")
    intake = ArtifactIntake(ArtifactLimits(max_member_bytes=1024 * 1024))
    _, roots, _ = intake.create_case(
        "诊断压缩包",
        [{"resource_id": "zip:1", "name": "bundle.zip", "text": buffer.getvalue().decode("latin-1")}],
    )
    # Text resources are UTF-8 encoded by contract, so use an explicit temporary-free
    # binary root to exercise the archive traversal guard itself.
    roots[0].data = buffer.getvalue()
    roots[0].manifest.sha256 = "binary-test"
    members = list(intake.iter_members(roots[0]))

    accepted = [item for item in members if item.manifest.status == "available"]
    rejected = [item for item in members if item.manifest.status == "rejected"]
    assert any(item.manifest.archive_member == "logs/app.log" for item in accepted)
    assert any("unsafe_archive_member_path" in item.manifest.safety_flags for item in rejected)
    assert not any(item.data == b"must not be accepted" for item in accepted)


def test_archive_limits_are_visible_in_manifest_instead_of_silent_drop() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("one.log", "ERROR first")
        archive.writestr("two.log", "ERROR second")
    intake = ArtifactIntake(ArtifactLimits(max_members=1))
    _, roots, _ = intake.create_case(
        "诊断压缩包",
        [{"resource_id": "zip:limit", "name": "bundle.zip", "text": "placeholder"}],
    )
    roots[0].data = buffer.getvalue()
    members = list(intake.iter_members(roots[0]))

    assert any(
        "archive_member_count_limit" in item.manifest.safety_flags
        for item in members
        if item.manifest.status == "rejected"
    )


def test_no_stack_report_still_preserves_detection_root_boundary() -> None:
    model = FakeReadModel()
    runtime = IncidentEvidenceRuntime(model)  # type: ignore[arg-type]
    result = runtime.analyze(
        "GPU 报错",
        [{"resource_id": "log:1", "name": "app.log", "text": "ERROR GPU timeout -217"}],
    )

    assert result.stack_traces == []
    assert result.verification["passed"] is True
    assert "异常被观察到的位置，不自动等同于根因位置" in result.report


def test_query_reference_times_are_independent_and_infer_year_from_package() -> None:
    scope = parse_incident_scope(
        "设备闪退，参考时间：8月1日21：30，8月3日6：04",
        ["诊断数据_[20260801-20260803].zip"],
    )

    assert scope.time_semantics == "independent_points"
    assert [item.reference_time for item in scope.reference_windows] == [
        "2026-08-01T21:30:00",
        "2026-08-03T06:04:00",
    ]
    assert scope.reference_windows[0].start_time == "2026-08-01T21:28:00"
    assert scope.reference_windows[1].end_time == "2026-08-03T06:07:00"
    assert "reference_time_year_inferred_from_resource" in scope.warnings


def test_path_backed_zip_streams_only_query_time_windows(tmp_path: Path) -> None:
    package = tmp_path / "diagnostic_[20260801-20260803].zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "logs/app-2026-08-01.log",
            "\n".join([
                "[2026-08-01 20:00:00,000] ERROR unrelated -999",
                "[2026-08-01 21:30:00,093] ERROR SYMV(1.1.0) subsystem error (-217): illegal memory access, trace:",
                "0# 18b97e5 7ffbcb6797e5 symv::cuda_init in symv",
                "[2026-08-01 21:30:57,013] INFO ApplicationPid: 19540",
            ]),
        )
        archive.writestr(
            "logs/app-2026-08-03.log",
            "\n".join([
                "[2026-08-03 06:03:40,091] ERROR SYMV(1.1.0) subsystem error (-217): illegal memory access, trace:",
                "0# 18b97e5 7ffbcb6797e5 symv::cuda_init in symv",
                "[2026-08-03 06:04:46,583] INFO ApplicationPid: 14284",
                "[2026-08-03 08:00:00,000] ERROR unrelated -998",
            ]),
        )
        archive.writestr("logs/app-2026-08-02.log", "[2026-08-02 12:00:00,000] ERROR unrelated -997")

    runtime = IncidentEvidenceRuntime(FakeReadModel())  # type: ignore[arg-type]
    result = runtime.analyze(
        "设备正常检测时闪退，参考时间：8月1日21：30，8月3日6：04",
        [{"resource_id": "zip:path", "name": package.name, "path": str(package)}],
    )

    root = next(item for item in result.case.artifacts if item.name == package.name)
    windows = [
        item for item in result.case.artifacts
        if item.metadata.get("derived_by") == "query_time_window_stream"
    ]
    skipped = [item for item in result.case.artifacts if item.parser_state == "scope_skipped"]
    assert root.metadata["path_backed"] is True
    assert len(windows) == 2
    assert any(item.archive_member.endswith("2026-08-02.log") for item in skipped)
    assert not any("-999" in item.message or "-998" in item.message for item in result.events)
    assert sum(1 for item in result.events if "-217" in item.error_codes) == 2
    assert any(item["type"] == "failure_followed_by_process_start" for item in result.correlations)
    assert any(item["type"] == "repeated_failure_signature" for item in result.correlations)
    assert result.observability["time_scoped"] is True
    assert result.observability["time_window_artifact_count"] == 2
    assert any(link.line_start == 2 for link in result.evidence_links)


def test_info_records_do_not_promote_asset_names_or_timeout_fields_to_errors() -> None:
    runtime = IncidentEvidenceRuntime(FakeReadModel())  # type: ignore[arg-type]
    result = runtime.analyze(
        "检查参考时间附近异常",
        [{
            "resource_id": "log:info-noise",
            "name": "service.log",
            "text": "\n".join([
                '[2026-08-01 21:29:00,000] INFO GET /image_pack/0.C156.white.jpg 200',
                '[2026-08-01 21:29:01,000] DEBUG reset_timeout=30000',
                '[2026-08-01 21:29:02,000] DEBUG theta=0.2 dy=-76.51 widget=0x210ed1bae30',
                'async_hold_board_when_sn_recognition_failed:false',
                '[2026-08-01 21:30:00,000] ERROR error code: C156 device failed',
            ]),
        }],
    )

    assert len(result.events) == 1
    assert result.events[0].error_codes == ["C156"]
    assert "device failed" in result.events[0].message


def test_restart_correlation_prefers_specific_failure_nearest_restart() -> None:
    runtime = IncidentEvidenceRuntime(FakeReadModel())  # type: ignore[arg-type]
    result = runtime.analyze(
        "设备闪退",
        [{
            "resource_id": "log:restart",
            "name": "app.log",
            "text": "\n".join([
                "[2026-08-01 21:28:00,000] ERROR unrelated generic failure",
                "[2026-08-01 21:30:00,093] ERROR subsystem error (-217): illegal memory access in cv::cuda::GpuMat::upload",
                "[2026-08-01 21:30:00,132] ERROR illegal memory access in symv::AsyncAllocator::free",
                "[2026-08-01 21:30:57,013] INFO ApplicationPid: 19540",
            ]),
        }],
    )

    restart = next(
        item for item in result.correlations
        if item["type"] == "failure_followed_by_process_start"
    )
    assert restart["failure_timestamp"].startswith("2026-08-01T21:30:00.093")
    assert restart["signature"].startswith("-217|illegal_memory_access")


def test_exception_header_with_inline_trace_is_both_event_and_trace_evidence() -> None:
    runtime = IncidentEvidenceRuntime(FakeReadModel())  # type: ignore[arg-type]
    result = runtime.analyze(
        "GPU 闪退",
        [{
            "resource_id": "log:inline-trace",
            "name": "app.log",
            "text": "\n".join([
                "[2026-08-01 21:30:00,093] ERROR subsystem error (-217): illegal memory access in cv::cuda::GpuMat::upload, file C:\\opencv\\gpu_mat.cu, line 240, trace:",
                "0# 18b97e5 7ffbcb6797e5 symv::cuda_init in symv",
            ]),
        }],
    )

    assert any("-217" in event.error_codes for event in result.events)
    assert any(
        frame.function == "cv::cuda::GpuMat::upload"
        for trace in result.stack_traces
        for frame in trace.frames
    )
