from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import struct
import zipfile

from debug_agent_system.incident_runtime.artifacts import ArtifactContent, ArtifactIntake
from debug_agent_system.incident_runtime.contracts import ArtifactManifest, DiagnosticEvent, EnvironmentSnapshot
from debug_agent_system.incident_runtime.hypotheses import HypothesisRuntime
from debug_agent_system.incident_runtime.minidump import parse_minidump
from debug_agent_system.incident_runtime.scope import parse_incident_scope
from debug_agent_system.incident_runtime.windows_events import (
    align_utc_timestamp_to_scope,
    normalize_event_xml,
)


def test_dependency_free_minidump_parser_reads_exception_process_and_module() -> None:
    parsed = parse_minidump(_synthetic_minidump())

    assert parsed["process"]["process_id"] == 4242
    assert parsed["system"]["architecture"] == "amd64"
    assert parsed["system"]["os_version"] == "10.0.19044"
    assert parsed["exception"]["code"] == "0xC0000005"
    assert parsed["exception"]["name"] == "access_violation"
    assert parsed["exception"]["access_type"] == "write"
    assert parsed["exception"]["module"] == "driver.dll"
    assert parsed["modules"][0]["file_version"] == "32.0.15.6070"


def test_evtx_xml_normalizer_preserves_provider_event_and_nested_values() -> None:
    xml = """<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">
      <System><Provider Name="nvlddmkm"/><EventID>13</EventID><Level>2</Level>
      <TimeCreated SystemTime="2026-08-01T13:29:59.780685Z"/>
      <EventRecordID>99</EventRecordID><Channel>System</Channel><Computer>AOI-1</Computer></System>
      <EventData><Data>&lt;string&gt;\\Device\\Video8&lt;/string&gt;
      &lt;string&gt;Graphics Exception: MISSING_INLINE_DATA&lt;/string&gt;</Data></EventData>
    </Event>"""

    record = normalize_event_xml(xml, record_number=7)

    assert record["provider"] == "nvlddmkm"
    assert record["event_id"] == "13"
    assert record["severity"] == "ERROR"
    assert "Graphics Exception" in record["message"]


def test_binary_utc_timestamp_aligns_to_query_local_reference_window() -> None:
    scope = parse_incident_scope(
        "参考时间：8月1日21：30",
        ["diagnostic_[20260801-20260803].zip"],
    ).to_dict()

    aligned = align_utc_timestamp_to_scope("2026-08-01T13:30:00+00:00", scope)

    assert aligned["local_utc_offset_minutes"] == 480
    assert aligned["timestamp_local"] == "2026-08-01T21:30:00"


def test_scoped_archive_keeps_evtx_dump_and_static_environment_without_dated_names() -> None:
    archive_bytes = BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("logs/app-2026-08-01.log", "[2026-08-01 21:30:00] ERROR failure")
        archive.writestr("system.evtx", b"EVTX")
        archive.writestr("uuid.dmp", b"MDMP")
        archive.writestr("sysinfo.json", b'{"os":"Windows"}')
        archive.writestr("unrelated.bin", b"skip")
    parent = ArtifactContent(
        ArtifactManifest(
            artifact_id="artifact:zip",
            resource_id="zip:1",
            name="diagnostic_[20260801].zip",
            kind="log_package",
            sha256="root",
        ),
        archive_bytes.getvalue(),
    )
    scope = parse_incident_scope("参考时间：8月1日21：30", [parent.manifest.name])

    members = list(ArtifactIntake().iter_scoped_members(parent, scope))
    by_name = {item.manifest.name: item for item in members}

    assert by_name["system.evtx"].manifest.metadata["scope_bypass_reason"] == "internally_timestamped_binary"
    assert by_name["uuid.dmp"].manifest.metadata["scope_bypass_reason"] == "high_value_crash_artifact"
    assert by_name["sysinfo.json"].manifest.metadata["scope_bypass_reason"] == "static_environment_snapshot"
    assert by_name["unrelated.bin"].manifest.parser_state == "scope_skipped"


def test_cross_source_gpu_chain_updates_hypothesis_without_query_specific_rules() -> None:
    events = [
        _event("driver", "gpu_driver_exception", "gpu_driver", "evidence:driver"),
        _event("cuda", "illegal_memory_access", "gpu_cuda", "evidence:cuda"),
        _event("dump", "crash_dump_exception", "application_crash", "evidence:dump"),
    ]
    environment = EnvironmentSnapshot(values={"nvidia_driver_version": ["32.0.15.6070"]})

    hypothesis = HypothesisRuntime(_EmptyReadModel()).build(
        {"candidates": [], "anchors": []},
        environment,
        events=events,
        correlations=[],
    )[0]

    assert hypothesis.hypothesis_id == "hypothesis:cross-source-gpu-driver-reset-chain"
    assert hypothesis.status == "supported"
    assert hypothesis.suspected_component == "gpu_driver"
    assert set(hypothesis.support_evidence_ids) == {"evidence:driver", "evidence:cuda", "evidence:dump"}


class _EmptyReadModel:
    def get(self, object_id: str):
        return None


def _event(artifact: str, kind: str, component: str, evidence: str) -> DiagnosticEvent:
    return DiagnosticEvent(
        event_id=f"event:{artifact}",
        artifact_id=f"artifact:{artifact}",
        sequence=1,
        severity="ERROR",
        message=kind,
        event_kind=kind,
        component=component,
        evidence_ids=[evidence],
    )


def _synthetic_minidump() -> bytes:
    data = bytearray(2048)
    created = int(datetime(2026, 8, 1, 13, 30, tzinfo=timezone.utc).timestamp())
    struct.pack_into("<4sIIIIIQ", data, 0, b"MDMP", 0xA793, 4, 32, 0, created, 0)
    streams = [(15, 24, 128), (7, 56, 160), (4, 112, 256), (6, 168, 512)]
    for index, item in enumerate(streams):
        struct.pack_into("<III", data, 32 + index * 12, *item)

    struct.pack_into("<IIIIII", data, 128, 24, 3, 4242, created - 10, 2, 1)
    struct.pack_into("<HHHBBIIII", data, 160, 9, 6, 0, 8, 1, 10, 0, 19044, 2)
    struct.pack_into("<I", data, 184, 0)

    name = "C:\\Windows\\System32\\driver.dll".encode("utf-16-le")
    struct.pack_into("<I", data, 1000, len(name))
    data[1004:1004 + len(name)] = name
    struct.pack_into("<I", data, 256, 1)
    module = 260
    struct.pack_into("<QIIII", data, module, 0x10000000, 0x2000, 0, created, 1000)
    struct.pack_into("<IIII", data, module + 32, (32 << 16), (15 << 16) | 6070, 0, 0)

    struct.pack_into("<I", data, 512, 77)
    struct.pack_into("<IIQQII", data, 520, 0xC0000005, 0, 0, 0x10000123, 2, 0)
    struct.pack_into("<QQ", data, 552, 1, 0xDEADBEEF)
    return bytes(data)
