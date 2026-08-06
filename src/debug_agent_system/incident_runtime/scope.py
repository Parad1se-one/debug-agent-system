"""Query-derived scope for bounded incident artifact inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import re
from typing import Any, Iterable


_FULL_TIMESTAMP = re.compile(
    r"(?P<year>20\d{2})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})"
    r"(?:[ T]\s*(?P<hour>\d{1,2})[:\uff1a](?P<minute>\d{1,2})(?::(?P<second>\d{1,2}))?)?"
)
_CHINESE_TIMESTAMP = re.compile(
    r"(?:(?P<year>20\d{2})\s*年\s*)?"
    r"(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日?\s*"
    r"(?P<hour>\d{1,2})\s*(?:[:\uff1a时])\s*(?P<minute>\d{1,2})"
    r"(?:\s*(?:[:\uff1a分])\s*(?P<second>\d{1,2}))?"
)
_COMPACT_DATE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
_DASHED_DATE = re.compile(r"(?<!\d)(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
# 紧凑"时.分"格式：query 常见"发生时间3.02左右"、"9.28左右"等写法。
# 小时 0-23、分钟 0-59，且要求附近出现时间语义上下文，避免误判版本号/价格。
_COMPACT_HH_MM = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3])\.(?P<minute>[0-5]\d)(?!\d)"
)
_TIME_CONTEXT_WORDS = re.compile(
    r"时间|左右|凌晨|早上|上午|中午|下午|晚上|夜里|深夜|发生|约|点|分|前|后",
    re.I,
)


@dataclass(slots=True, frozen=True)
class ReferenceTimeWindow:
    reference_time: str
    start_time: str
    end_time: str
    precision: str = "minute"
    source_text: str = ""
    year_inferred: bool = False

    def start(self) -> datetime:
        return datetime.fromisoformat(self.start_time)

    def end(self) -> datetime:
        return datetime.fromisoformat(self.end_time)

    def contains(self, value: datetime) -> bool:
        return self.start() <= value <= self.end()


@dataclass(slots=True)
class IncidentScope:
    reference_windows: list[ReferenceTimeWindow] = field(default_factory=list)
    time_semantics: str = "none"
    timezone: str = "local_unknown"
    inferred_year: int | None = None
    source: str = "query"
    warnings: list[str] = field(default_factory=list)

    @property
    def has_time_scope(self) -> bool:
        return bool(self.reference_windows)

    @property
    def target_dates(self) -> set[str]:
        return {
            datetime.fromisoformat(item.reference_time).date().isoformat()
            for item in self.reference_windows
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_windows": [asdict(item) for item in self.reference_windows],
            "time_semantics": self.time_semantics,
            "timezone": self.timezone,
            "inferred_year": self.inferred_year,
            "source": self.source,
            "warnings": list(self.warnings),
        }


def parse_incident_scope(
    query: str,
    resource_hints: Iterable[str] = (),
    *,
    before_seconds: int = 120,
    after_seconds: int = 180,
) -> IncidentScope:
    """Parse independent reference instants without treating them as a range."""

    text = str(query or "")
    hints = [str(item or "") for item in resource_hints]
    inferred_year = _infer_year([text, *hints])
    recent_date = _infer_recent_date([text, *hints])
    matches: list[tuple[int, str, datetime, bool, str]] = []
    occupied: list[tuple[int, int]] = []

    for pattern in (_FULL_TIMESTAMP, _CHINESE_TIMESTAMP):
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            raw_year = match.groupdict().get("year")
            year = int(raw_year) if raw_year else inferred_year
            if year is None:
                continue
            hour = int(match.groupdict().get("hour") or 0)
            minute = int(match.groupdict().get("minute") or 0)
            second = int(match.groupdict().get("second") or 0)
            try:
                value = datetime(
                    year,
                    int(match.group("month")),
                    int(match.group("day")),
                    hour,
                    minute,
                    second,
                )
            except ValueError:
                continue
            precision = "second" if match.groupdict().get("second") else (
                "minute" if match.groupdict().get("hour") else "date"
            )
            matches.append((match.start(), match.group(0), value, raw_year is None, precision))
            occupied.append((match.start(), match.end()))

    # 紧凑"时.分"格式（如"3.02"、"9.28"）。query 中常伴随"时间/左右/发生/凌晨/
    # 早上"等上下文；日期缺失时回退到资源中最新的日期（通常是打包/采集日）。
    for match in _COMPACT_HH_MM.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        context = text[max(0, match.start() - 12):match.end() + 12]
        if not _TIME_CONTEXT_WORDS.search(context):
            continue
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        if recent_date is None:
            continue
        try:
            value = datetime(
                recent_date.year,
                recent_date.month,
                recent_date.day,
                hour,
                minute,
                0,
            )
        except ValueError:
            continue
        source_text = match.group(0)
        matches.append((match.start(), source_text, value, True, "minute"))
        occupied.append((match.start(), match.end()))

    deduped: dict[str, tuple[int, str, datetime, bool, str]] = {}
    for item in sorted(matches, key=lambda value: value[0]):
        deduped.setdefault(item[2].isoformat(timespec="seconds"), item)
    windows = [
        ReferenceTimeWindow(
            reference_time=value.isoformat(timespec="seconds"),
            start_time=(value - timedelta(seconds=max(0, before_seconds))).isoformat(timespec="seconds"),
            end_time=(value + timedelta(seconds=max(0, after_seconds))).isoformat(timespec="seconds"),
            precision=precision,
            source_text=source_text,
            year_inferred=year_inferred,
        )
        for _, source_text, value, year_inferred, precision in deduped.values()
    ]
    warnings: list[str] = []
    if any(item.year_inferred for item in windows):
        warnings.append("reference_time_year_inferred_from_resource")
    return IncidentScope(
        reference_windows=windows,
        time_semantics=("independent_points" if len(windows) > 1 else ("point" if windows else "none")),
        inferred_year=inferred_year,
        warnings=warnings,
    )


def parse_log_timestamp(line: str) -> datetime | None:
    """Parse the common local timestamps used by AOI diagnostic logs."""

    match = re.search(
        r"(?P<year>20\d{2})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})"
        r"[ T](?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})"
        r"(?:[.,](?P<fraction>\d{1,6}))?",
        str(line),
    )
    if not match:
        return None
    fraction = (match.group("fraction") or "").ljust(6, "0")[:6]
    try:
        return datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            int(fraction or 0),
        )
    except ValueError:
        return None


def name_matches_scope(name: str, scope: IncidentScope) -> bool:
    """Return true when a dated member name overlaps a target date."""

    dates = _dates_in_text(str(name or ""))
    return bool(dates and dates.intersection(scope.target_dates))


def _infer_year(values: Iterable[str]) -> int | None:
    years: list[int] = []
    for value in values:
        for match in _COMPACT_DATE.finditer(value):
            years.append(int(match.group(1)))
        for match in _DASHED_DATE.finditer(value):
            years.append(int(match.group(1)))
    unique = list(dict.fromkeys(years))
    return unique[0] if len(unique) == 1 else (max(unique) if unique else None)


def _infer_recent_date(values: Iterable[str]) -> datetime | None:
    """Infer the most recent concrete date from query/resource text.

    Compact "H.MM" reference times carry no explicit date.  The most recent
    concrete date in the diagnostic package name (typically the collection or
    packaging date) is the least surprising anchor for those times.
    """

    dates: list[datetime] = []
    for value in values:
        for pattern in (_COMPACT_DATE, _DASHED_DATE):
            for match in pattern.finditer(value):
                try:
                    dates.append(datetime(
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                    ))
                except ValueError:
                    continue
    return max(dates) if dates else None


def _dates_in_text(value: str) -> set[str]:
    result: set[str] = set()
    for pattern in (_COMPACT_DATE, _DASHED_DATE):
        for match in pattern.finditer(value):
            try:
                result.add(
                    datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).date().isoformat()
                )
            except ValueError:
                continue
    return result


__all__ = [
    "IncidentScope",
    "ReferenceTimeWindow",
    "name_matches_scope",
    "parse_incident_scope",
    "parse_log_timestamp",
]
