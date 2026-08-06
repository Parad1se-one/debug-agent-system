"""Bounded, dependency-free Windows minidump metadata parsing.

This module deliberately does not attempt symbol loading or stack unwinding.
It reads only documented minidump streams needed to bind an exception to a
process, architecture, timestamp and loaded module.  All offsets are checked
before use so untrusted diagnostic packages remain data, never executable
input.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import struct
from typing import Any


MINIDUMP_PARSER_VERSION = "windows-minidump-metadata-v1"

# 完整内核转储（kernel full dump）头部签名，用于识别非 Minidump 的 MEMORY.DMP。
_KERNEL_DUMP_SIGNATURES = {
    b"PAGEDU64": "kernel_full_dump_amd64",
    b"PAGEDUMP": "kernel_full_dump_x86",
    b"PAGEDU32": "kernel_full_dump_x86",
}
_KERNEL_DUMP_BUGCHECK_NAMES = {
    0x0000004E: "CRITICAL_STRUCTURE_CORRUPTION",
    0x000000EF: "CRITICAL_PROCESS_DIED",
    0x000000D1: "DRIVER_IRQL_NOT_LESS_OR_EQUAL",
    0x00000050: "PAGE_FAULT_IN_NONPAGED_AREA",
    0x0000007A: "KERNEL_DATA_INPAGE_ERROR",
    0x0000000A: "IRQL_NOT_LESS_OR_EQUAL",
    0x0000003B: "SYSTEM_SERVICE_EXCEPTION",
    0x0000001E: "KMODE_EXCEPTION_NOT_HANDLED",
    0x000000C2: "BAD_POOL_CALLER",
    0x0000007F: "UNEXPECTED_KERNEL_MODE_TRAP",
    0x00000139: "KERNEL_SECURITY_CHECK_FAILURE",
    0x00000116: "VIDEO_TDR_FAILURE",
}

_STREAM_NAMES = {
    3: "ThreadListStream",
    4: "ModuleListStream",
    6: "ExceptionStream",
    7: "SystemInfoStream",
    15: "MiscInfoStream",
    17: "ThreadInfoListStream",
}
_ARCHITECTURES = {
    0: "x86",
    5: "arm",
    6: "ia64",
    9: "amd64",
    12: "arm64",
}
_EXCEPTION_NAMES = {
    0x80000003: "breakpoint",
    0x80000004: "single_step",
    0xC0000005: "access_violation",
    0xC000001D: "illegal_instruction",
    0xC0000094: "integer_divide_by_zero",
    0xC00000FD: "stack_overflow",
    0xC0000409: "stack_buffer_overrun_or_fast_fail",
    0x40000015: "fatal_app_exit",
    0xE06D7363: "microsoft_cpp_exception",
}


class MinidumpFormatError(ValueError):
    """Raised when a byte sequence is not a bounded valid minidump."""


class _MinidumpReader:
    """Bounded random-access view over bytes or a file handle.

    Kernel dumps routinely exceed a few GB, so the parser never requires the
    whole file in memory.  It reads the header, the stream directory and only
    the small metadata streams needed to bind an exception to a module.
    """

    __slots__ = ("_data", "_handle", "_total_size")

    def __init__(self, data: bytes | None = None, path: str | None = None) -> None:
        if data is not None:
            self._data = data
            self._handle = None
            self._total_size = len(data)
        elif path is not None:
            self._data = None
            self._handle = open(path, "rb")
            self._total_size = os.path.getsize(path)
        else:
            raise ValueError("minidump_reader_requires_data_or_path")

    @property
    def total_size(self) -> int:
        return self._total_size

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > self._total_size:
            raise MinidumpFormatError("minidump_offset_out_of_bounds")
        if self._data is not None:
            return self._data[offset:offset + size]
        self._handle.seek(offset)
        chunk = self._handle.read(size)
        if len(chunk) != size:
            raise MinidumpFormatError("minidump_read_short")
        return chunk

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None


def parse_minidump(data: bytes, *, max_modules: int = 2048) -> dict[str, Any]:
    """Return stable structured metadata from a Windows minidump byte blob."""

    reader = _MinidumpReader(data=data)
    try:
        return _parse_minidump(reader, max_modules=max_modules)
    finally:
        reader.close()


def parse_minidump_file(
    path: str | os.PathLike[str],
    *,
    max_modules: int = 2048,
) -> dict[str, Any]:
    """Return stable structured metadata from a Windows minidump file.

    Only the header, stream directory and small metadata streams are read, so
    multi-GB kernel dumps can be inspected without loading them into memory.
    """

    reader = _MinidumpReader(path=str(path))
    try:
        return _parse_minidump(reader, max_modules=max_modules)
    finally:
        reader.close()


def parse_kernel_dump_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse a PAGEDU64 full-kernel-dump header without loading the dump.

    ``MEMORY.DMP`` produced by a Windows kernel bugcheck is a full physical
    memory dump with a ``PAGEDU64`` header, not a Minidump.  The header alone
    carries the bugcheck code, its four parameters, OS version and processor
    architecture, which is the highest-confidence root-cause anchor available
    for a blue-screen case.  The rest of the file is never read.
    """

    reader = _MinidumpReader(path=str(path))
    try:
        if reader.total_size < 256:
            raise MinidumpFormatError("kernel_dump_too_small")
        signature = reader.read(0, 8)
        format_name = _KERNEL_DUMP_SIGNATURES.get(signature)
        if format_name is None:
            raise MinidumpFormatError("invalid_kernel_dump_signature")
        major = _unpack("<I", reader, 0x0C)[0]
        minor = _unpack("<I", reader, 0x10)[0]
        machine_image_type = _unpack("<I", reader, 0x30)[0]
        processor_count = _unpack("<I", reader, 0x34)[0]
        bugcheck_code = _unpack("<I", reader, 0x38)[0]
        bugcheck_parameters = [
            _unpack("<Q", reader, 0x40 + index * 8)[0]
            for index in range(4)
        ]
        return {
            "schema_version": "debug_agent_system.windows_kernel_dump.v1",
            "parser_version": MINIDUMP_PARSER_VERSION,
            "valid": True,
            "format": format_name,
            "file_size": reader.total_size,
            "os_version": f"{major}.{minor >> 16}.{minor & 0xFFFF}",
            "processor_count": processor_count,
            "machine_image_type": f"0x{machine_image_type:04X}",
            "bugcheck": {
                "code": f"0x{bugcheck_code:08X}",
                "name": _KERNEL_DUMP_BUGCHECK_NAMES.get(
                    bugcheck_code, "unknown_bugcheck"
                ),
                "parameters": [
                    f"0x{item:016X}" for item in bugcheck_parameters
                ],
            },
            "limitations": [
                "no_symbol_loading",
                "kernel_full_dump_header_only",
                "crashing_process_requires_symbolized_analysis",
            ],
        }
    finally:
        reader.close()


def _parse_minidump(reader: _MinidumpReader, *, max_modules: int) -> dict[str, Any]:
    if reader.total_size < 32 or reader.read(0, 4) != b"MDMP":
        raise MinidumpFormatError("invalid_minidump_signature")
    _, version, stream_count, directory_rva, checksum, timestamp, flags = _unpack(
        "<4sIIIIIQ", reader, 0
    )
    if stream_count > 16_384:
        raise MinidumpFormatError("minidump_stream_count_limit")
    _require(reader, directory_rva, stream_count * 12)
    streams: dict[int, tuple[int, int]] = {}
    stream_inventory: list[dict[str, Any]] = []
    for index in range(stream_count):
        stream_type, size, rva = _unpack("<III", reader, directory_rva + index * 12)
        if size and rva:
            _require(reader, rva, size)
        streams.setdefault(stream_type, (rva, size))
        stream_inventory.append({
            "type": stream_type,
            "name": _STREAM_NAMES.get(stream_type, f"Stream{stream_type}"),
            "size": size,
            "rva": rva,
        })

    result: dict[str, Any] = {
        "schema_version": "debug_agent_system.windows_minidump.v1",
        "parser_version": MINIDUMP_PARSER_VERSION,
        "valid": True,
        "file_size": reader.total_size,
        "format_version": f"0x{version:08X}",
        "checksum": f"0x{checksum:08X}",
        "flags": f"0x{flags:016X}",
        "created_time_utc": _unix_time(timestamp),
        "streams": stream_inventory,
        "process": {},
        "exception": {},
        "system": {},
        "modules": [],
        "thread_ids": [],
        "limitations": [
            "no_symbol_loading",
            "no_stack_unwind",
            "exception_module_is_address_range_attribution_only",
        ],
    }

    if 15 in streams:
        result["process"] = _parse_misc_info(reader, *streams[15])
    if 7 in streams:
        result["system"] = _parse_system_info(reader, *streams[7])
    if 4 in streams:
        result["modules"] = _parse_modules(reader, *streams[4], max_modules=max_modules)
    if 3 in streams:
        result["thread_ids"] = _parse_thread_ids(reader, *streams[3])
    if 6 in streams:
        result["exception"] = _parse_exception(reader, *streams[6])
        address = result["exception"].get("address_value")
        module = _module_for_address(result["modules"], address)
        if module:
            result["exception"]["module"] = module["name"]
            result["exception"]["module_path"] = module["path"]
            result["exception"]["module_offset"] = f"0x{address - module['base_address_value']:X}"
    return result


def _parse_misc_info(reader: _MinidumpReader, rva: int, size: int) -> dict[str, Any]:
    if size < 8:
        return {}
    declared_size, flags = _unpack("<II", reader, rva)
    available = min(size, declared_size or size)
    result: dict[str, Any] = {"flags": f"0x{flags:08X}"}
    if flags & 0x1 and available >= 12:
        result["process_id"] = _unpack("<I", reader, rva + 8)[0]
    if flags & 0x2 and available >= 24:
        create_time, user_time, kernel_time = _unpack("<III", reader, rva + 12)
        result.update({
            "create_time_utc": _unix_time(create_time),
            "user_time_seconds": user_time,
            "kernel_time_seconds": kernel_time,
        })
    return result


def _parse_system_info(reader: _MinidumpReader, rva: int, size: int) -> dict[str, Any]:
    if size < 28:
        return {}
    architecture, processor_level, processor_revision = _unpack("<HHH", reader, rva)
    processors = reader.read(rva + 6, 1)[0]
    product_type = reader.read(rva + 7, 1)[0]
    major, minor, build, platform = _unpack("<IIII", reader, rva + 8)
    csd_rva = _unpack("<I", reader, rva + 24)[0]
    result = {
        "architecture": _ARCHITECTURES.get(architecture, f"architecture_{architecture}"),
        "processor_level": processor_level,
        "processor_revision": processor_revision,
        "processor_count": processors,
        "product_type": product_type,
        "os_version": f"{major}.{minor}.{build}",
        "platform_id": platform,
    }
    csd = _read_minidump_string(reader, csd_rva)
    if csd:
        result["service_pack"] = csd
    return result


def _parse_modules(
    reader: _MinidumpReader,
    rva: int,
    size: int,
    *,
    max_modules: int,
) -> list[dict[str, Any]]:
    if size < 4:
        return []
    count = min(_unpack("<I", reader, rva)[0], max_modules)
    entry_size = 108
    available = max(0, (size - 4) // entry_size)
    modules: list[dict[str, Any]] = []
    for index in range(min(count, available)):
        offset = rva + 4 + index * entry_size
        base, image_size, checksum, timestamp, name_rva = _unpack("<QIIII", reader, offset)
        file_ms, file_ls, product_ms, product_ls = _unpack("<IIII", reader, offset + 32)
        path = _read_minidump_string(reader, name_rva)
        modules.append({
            "name": _windows_basename(path),
            "path": path,
            "base_address": f"0x{base:016X}",
            "base_address_value": base,
            "image_size": image_size,
            "checksum": f"0x{checksum:08X}",
            "timestamp_utc": _unix_time(timestamp),
            "file_version": _version(file_ms, file_ls),
            "product_version": _version(product_ms, product_ls),
        })
    return modules


def _parse_thread_ids(reader: _MinidumpReader, rva: int, size: int) -> list[int]:
    if size < 4:
        return []
    count = _unpack("<I", reader, rva)[0]
    available = max(0, (size - 4) // 48)
    return [
        _unpack("<I", reader, rva + 4 + index * 48)[0]
        for index in range(min(count, available, 8192))
    ]


def _parse_exception(reader: _MinidumpReader, rva: int, size: int) -> dict[str, Any]:
    if size < 40:
        return {}
    thread_id = _unpack("<I", reader, rva)[0]
    code, flags = _unpack("<II", reader, rva + 8)
    nested_record, address = _unpack("<QQ", reader, rva + 16)
    parameter_count = min(_unpack("<I", reader, rva + 32)[0], 15)
    parameters = []
    if size >= 40 + parameter_count * 8:
        parameters = list(_unpack(f"<{parameter_count}Q", reader, rva + 40)) if parameter_count else []
    result: dict[str, Any] = {
        "thread_id": thread_id,
        "code": f"0x{code:08X}",
        "name": _EXCEPTION_NAMES.get(code, "unknown_exception"),
        "flags": f"0x{flags:08X}",
        "nested_record": f"0x{nested_record:016X}",
        "address": f"0x{address:016X}",
        "address_value": address,
        "parameters": [f"0x{item:016X}" for item in parameters],
    }
    if code == 0xC0000005 and len(parameters) >= 2:
        result["access_type"] = {0: "read", 1: "write", 8: "execute"}.get(parameters[0], "unknown")
        result["access_address"] = f"0x{parameters[1]:016X}"
    return result


def _module_for_address(modules: list[dict[str, Any]], address: Any) -> dict[str, Any] | None:
    if not isinstance(address, int):
        return None
    for module in modules:
        base = module.get("base_address_value")
        size = module.get("image_size")
        if isinstance(base, int) and isinstance(size, int) and base <= address < base + size:
            return module
    return None


def _read_minidump_string(reader: _MinidumpReader, rva: int) -> str:
    if not rva or rva + 4 > reader.total_size:
        return ""
    byte_length = _unpack("<I", reader, rva)[0]
    if byte_length > 1_048_576 or rva + 4 + byte_length > reader.total_size:
        return ""
    return reader.read(rva + 4, byte_length).decode("utf-16-le", errors="replace").rstrip("\x00")


def _version(ms: int, ls: int) -> str:
    return ".".join(str(item) for item in (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF))


def _windows_basename(value: str) -> str:
    return value.replace("/", "\\").rsplit("\\", 1)[-1]


def _unix_time(value: int) -> str:
    if not value:
        return ""
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _require(reader: _MinidumpReader, offset: int, size: int) -> None:
    if offset < 0 or size < 0 or offset + size > reader.total_size:
        raise MinidumpFormatError("minidump_offset_out_of_bounds")


def _unpack(fmt: str, reader: _MinidumpReader, offset: int) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    _require(reader, offset, size)
    return struct.unpack_from(fmt, reader.read(offset, size), 0)
