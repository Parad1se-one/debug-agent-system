"""Read-only Jira Data Center synchronizer for the local offline evidence store.

The synchronizer deliberately lives outside TOOL-JIRA.  TOOL-JIRA remains an
offline parser; this module is an explicitly invoked import adapter that needs
user-provided Jira credentials (PAT or account/password).

Raw responses are written to a new versioned root and existing offline exports
are never overwritten.  Authentication values are never serialized or logged.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener


SCHEMA_VERSION = "debug_agent_system.jira_offline_full.v2"
DEFAULT_BASE_URL = "https://jira.example.com"
DEFAULT_PROJECTS = ("TEST", "SMTAOI", "SMTAOITS")
DEFAULT_OUTPUT_ROOT = Path("data/imports/jira_offline/raw")
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class JiraSyncError(RuntimeError):
    """Raised for an unrecoverable Jira synchronization error."""


@dataclass(frozen=True)
class SyncConfig:
    base_url: str
    projects: tuple[str, ...]
    output_root: Path
    token: str = ""
    account: str = ""
    password: str = ""
    workers: int = 4
    page_size: int = 100
    timeout_seconds: float = 45.0
    max_retries: int = 5
    updated_since: str = ""
    max_issues: int = 0
    refresh_existing: bool = False
    fetch_remote_links: bool = True
    fetch_worklogs: bool = True
    use_environment_proxy: bool = False


class JiraClient:
    """Small read-only Jira REST v2 client using Bearer or Basic auth."""

    def __init__(self, config: SyncConfig) -> None:
        self.config = config
        self._local = threading.local()
        # The execution host's generic HTTPS proxy cannot complete TLS with the
        # internal Jira host. Direct access is therefore the safe default; the
        # opt-in switch remains available for environments that need a proxy.
        proxy_handler = ProxyHandler() if config.use_environment_proxy else ProxyHandler({})
        self._opener = build_opener(proxy_handler)

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request_json("GET", path, params=params)

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
        return self._request_json("POST", path, body=body)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json", "User-Agent": "debug-agent-system-jira-offline-sync/2"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        elif self.config.account and self.config.password:
            credential = f"{self.config.account}:{self.config.password}".encode("utf-8")
            headers["Authorization"] = f"Basic {base64.b64encode(credential).decode('ascii')}"
        else:
            raise JiraSyncError("Jira credentials are missing")
        if payload is not None:
            headers["Content-Type"] = "application/json"

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            request = Request(url, data=payload, headers=headers, method=method)
            try:
                with self._opener.open(request, timeout=self.config.timeout_seconds) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8")) if raw else None
            except HTTPError as exc:
                last_error = exc
                if exc.code in {401, 403}:
                    raise JiraSyncError(
                        f"Jira authentication/permission failed with HTTP {exc.code}; "
                        "check PAT scope and issue browse permissions"
                    ) from exc
                if exc.code == 404:
                    raise
                if exc.code not in RETRYABLE_STATUS or attempt >= self.config.max_retries:
                    detail = _http_error_preview(exc)
                    raise JiraSyncError(f"Jira HTTP {exc.code} for {method} {path}: {detail}") from exc
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
                _backoff(attempt, retry_after)
            except (TimeoutError, URLError, OSError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                _backoff(attempt, None)
        raise JiraSyncError(f"Jira request failed for {method} {path}: {type(last_error).__name__}") from last_error

    def verify_identity(self) -> dict[str, Any]:
        value = self.get_json("rest/api/2/myself")
        if not isinstance(value, dict):
            raise JiraSyncError("Jira /myself returned an unexpected payload")
        return value

    def server_info(self) -> dict[str, Any]:
        value = self.get_json("rest/api/2/serverInfo")
        return value if isinstance(value, dict) else {}

    def search_issue_keys(self) -> list[str]:
        project_clause = ", ".join(self.config.projects)
        jql = f"project in ({project_clause})"
        if self.config.updated_since:
            escaped = self.config.updated_since.replace('"', '\\"')
            jql += f' AND updated >= "{escaped}"'
        jql += " ORDER BY key ASC"
        start_at = 0
        keys: list[str] = []
        while True:
            page = self.post_json(
                "rest/api/2/search",
                {
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": self.config.page_size,
                    "fields": ["key"],
                },
            )
            if not isinstance(page, dict):
                raise JiraSyncError("Jira search returned an unexpected payload")
            issues = page.get("issues") if isinstance(page.get("issues"), list) else []
            for issue in issues:
                if isinstance(issue, dict) and issue.get("key"):
                    keys.append(str(issue["key"]))
                    if self.config.max_issues and len(keys) >= self.config.max_issues:
                        return keys
            start_at += len(issues)
            total = int(page.get("total") or 0)
            if not issues or start_at >= total:
                return keys

    def fetch_issue_bundle(self, issue_key: str) -> dict[str, Any]:
        encoded = quote(issue_key, safe="-")
        issue = self.get_json(
            f"rest/api/2/issue/{encoded}",
            {"fields": "*all", "expand": "names,schema,renderedFields,changelog"},
        )
        if not isinstance(issue, dict):
            raise JiraSyncError(f"Jira issue {issue_key} returned an unexpected payload")

        endpoint_errors: dict[str, str] = {}
        fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
        comments = self._embedded_or_paginated(
            fields.get("comment"),
            "comments",
            f"rest/api/2/issue/{encoded}/comment",
            endpoint_errors,
            expand="renderedBody",
        )
        changelog = self._fetch_changelog(encoded, issue, endpoint_errors)
        remote_links: list[dict[str, Any]] = []
        if self.config.fetch_remote_links:
            remote_links = self._fetch_optional_list(
                f"rest/api/2/issue/{encoded}/remotelink", "remote_links", endpoint_errors
            )
        worklogs: list[dict[str, Any]] = []
        if self.config.fetch_worklogs:
            worklogs = self._embedded_or_paginated(
                fields.get("worklog"),
                "worklogs",
                f"rest/api/2/issue/{encoded}/worklog",
                endpoint_errors,
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "issue_key": issue_key,
            "fetched_at": _now_iso(),
            "issue": issue,
            "comments": comments,
            "changelog": changelog,
            "remote_links": remote_links,
            "worklogs": worklogs,
            "endpoint_errors": endpoint_errors,
        }

    def _fetch_changelog(
        self,
        encoded_key: str,
        issue: dict[str, Any],
        endpoint_errors: dict[str, str],
    ) -> list[dict[str, Any]]:
        embedded = issue.get("changelog") if isinstance(issue.get("changelog"), dict) else {}
        embedded_histories = embedded.get("histories") if isinstance(embedded.get("histories"), list) else []
        embedded_total = int(embedded.get("total") or len(embedded_histories))
        if len(embedded_histories) >= embedded_total:
            return [item for item in embedded_histories if isinstance(item, dict)]

        histories = self._fetch_paginated(
            f"rest/api/2/issue/{encoded_key}/changelog", "values", endpoint_errors
        )
        separate_endpoint_error = endpoint_errors.pop("values", "")
        if histories and not separate_endpoint_error:
            return histories

        histories = [item for item in embedded_histories if isinstance(item, dict)]
        total = embedded_total
        if separate_endpoint_error:
            endpoint_errors["changelog"] = (
                f"separate endpoint unavailable ({separate_endpoint_error}); "
                f"used embedded history ({len(histories)}/{total})"
            )
        elif len(histories) < total:
            endpoint_errors["changelog"] = (
                f"separate changelog endpoint unavailable; embedded history is partial ({len(histories)}/{total})"
            )
        return histories

    def _embedded_or_paginated(
        self,
        embedded: Any,
        collection_key: str,
        path: str,
        endpoint_errors: dict[str, str],
        *,
        expand: str = "",
    ) -> list[dict[str, Any]]:
        value = embedded if isinstance(embedded, dict) else {}
        items = value.get(collection_key) if isinstance(value.get(collection_key), list) else []
        total = int(value.get("total") or len(items))
        if len(items) >= total:
            return [item for item in items if isinstance(item, dict)]
        return self._fetch_paginated(path, collection_key, endpoint_errors, expand=expand)

    def _fetch_paginated(
        self,
        path: str,
        collection_key: str,
        endpoint_errors: dict[str, str],
        *,
        expand: str = "",
    ) -> list[dict[str, Any]]:
        start_at = 0
        values: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {"startAt": start_at, "maxResults": self.config.page_size}
            if expand:
                params["expand"] = expand
            try:
                page = self.get_json(path, params)
            except HTTPError as exc:
                if exc.code == 404:
                    endpoint_errors[collection_key] = "endpoint_not_available"
                    return values
                raise
            if not isinstance(page, dict):
                endpoint_errors[collection_key] = "unexpected_payload"
                return values
            items = page.get(collection_key) if isinstance(page.get(collection_key), list) else []
            values.extend(item for item in items if isinstance(item, dict))
            start_at += len(items)
            total = int(page.get("total") or len(values))
            if not items or start_at >= total:
                return values

    def _fetch_optional_list(
        self,
        path: str,
        name: str,
        endpoint_errors: dict[str, str],
    ) -> list[dict[str, Any]]:
        try:
            value = self.get_json(path)
        except HTTPError as exc:
            if exc.code in {403, 404}:
                endpoint_errors[name] = f"http_{exc.code}"
                return []
            raise
        if not isinstance(value, list):
            endpoint_errors[name] = "unexpected_payload"
            return []
        return [item for item in value if isinstance(item, dict)]


def _retry_after_seconds(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value else None
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int, retry_after: float | None) -> None:
    delay = retry_after if retry_after is not None else min(30.0, (2**attempt) + random.random())
    time.sleep(delay)


def _http_error_preview(exc: HTTPError) -> str:
    try:
        return exc.read(800).decode("utf-8", errors="replace").replace("\n", " ")
    except Exception:
        return ""


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("displayName") or value.get("name") or value.get("key") or "")


def _field_name(value: Any) -> str:
    return str(value.get("name") or "") if isinstance(value, dict) else ""


def _version_list(value: Any) -> list[str]:
    return [_field_name(item) for item in value if isinstance(item, dict) and _field_name(item)] if isinstance(value, list) else []


def _normalize_issue_link(link: dict[str, Any]) -> dict[str, Any]:
    link_type = link.get("type") if isinstance(link.get("type"), dict) else {}
    if isinstance(link.get("outwardIssue"), dict):
        issue = link["outwardIssue"]
        direction = "outward"
        description = str(link_type.get("outward") or "")
    else:
        issue = link.get("inwardIssue") if isinstance(link.get("inwardIssue"), dict) else {}
        direction = "inward"
        description = str(link_type.get("inward") or "")
    fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
    return {
        "id": str(link.get("id") or ""),
        "type_id": str(link_type.get("id") or ""),
        "type_name": str(link_type.get("name") or ""),
        "direction": direction,
        "description": description,
        "issue_key": str(issue.get("key") or ""),
        "summary": str(fields.get("summary") or ""),
        "status": _field_name(fields.get("status")),
    }


def _normalize_attachment(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id") or ""),
        "filename": str(item.get("filename") or ""),
        "size": int(item.get("size") or 0),
        "mime_type": str(item.get("mimeType") or ""),
        "author": _name(item.get("author")),
        "created": str(item.get("created") or ""),
        "content_url": str(item.get("content") or ""),
        "thumbnail_url": str(item.get("thumbnail") or ""),
    }


def normalize_issue_bundle(bundle: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    """Create a backwards-compatible detail record plus the newly required fields."""
    issue = bundle.get("issue") if isinstance(bundle.get("issue"), dict) else {}
    fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
    comments = bundle.get("comments") if isinstance(bundle.get("comments"), list) else []
    changelog = bundle.get("changelog") if isinstance(bundle.get("changelog"), list) else []
    links = fields.get("issuelinks") if isinstance(fields.get("issuelinks"), list) else []
    attachments = fields.get("attachment") if isinstance(fields.get("attachment"), list) else []
    components = fields.get("components") if isinstance(fields.get("components"), list) else []
    subtasks = fields.get("subtasks") if isinstance(fields.get("subtasks"), list) else []
    parent = fields.get("parent") if isinstance(fields.get("parent"), dict) else {}
    project = fields.get("project") if isinstance(fields.get("project"), dict) else {}

    normalized_comments = []
    for item in comments:
        if not isinstance(item, dict):
            continue
        normalized_comments.append({
            "id": str(item.get("id") or ""),
            "author": _name(item.get("author")),
            "created": str(item.get("created") or ""),
            "updated": str(item.get("updated") or ""),
            "body": item.get("body") if item.get("body") is not None else "",
            "rendered_body": str(item.get("renderedBody") or ""),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "key": str(issue.get("key") or bundle.get("issue_key") or ""),
        "id": str(issue.get("id") or ""),
        "summary": str(fields.get("summary") or ""),
        "description": fields.get("description") if fields.get("description") is not None else "",
        "environment": fields.get("environment") if fields.get("environment") is not None else "",
        "status": _field_name(fields.get("status")),
        "resolution": _field_name(fields.get("resolution")),
        "assignee": _name(fields.get("assignee")),
        "reporter": _name(fields.get("reporter")),
        "creator": _name(fields.get("creator")),
        "created": str(fields.get("created") or ""),
        "updated": str(fields.get("updated") or ""),
        "resolution_date": str(fields.get("resolutiondate") or ""),
        "due_date": str(fields.get("duedate") or ""),
        "issue_type": _field_name(fields.get("issuetype")),
        "priority": _field_name(fields.get("priority")),
        "project": {"key": str(project.get("key") or ""), "name": str(project.get("name") or "")},
        "labels": [str(item) for item in fields.get("labels", [])] if isinstance(fields.get("labels"), list) else [],
        "components": [_field_name(item) for item in components if _field_name(item)],
        "fix_versions": _version_list(fields.get("fixVersions")),
        "affected_versions": _version_list(fields.get("versions")),
        "parent": {
            "key": str(parent.get("key") or ""),
            "summary": str((parent.get("fields") or {}).get("summary") or "") if isinstance(parent.get("fields"), dict) else "",
        },
        "subtasks": [
            {
                "key": str(item.get("key") or ""),
                "summary": str((item.get("fields") or {}).get("summary") or "") if isinstance(item.get("fields"), dict) else "",
            }
            for item in subtasks
            if isinstance(item, dict)
        ],
        "issue_links": [_normalize_issue_link(item) for item in links if isinstance(item, dict)],
        "attachments": [_normalize_attachment(item) for item in attachments if isinstance(item, dict)],
        "comments": normalized_comments,
        "changelog": changelog,
        "remote_links": bundle.get("remote_links") if isinstance(bundle.get("remote_links"), list) else [],
        "worklogs": bundle.get("worklogs") if isinstance(bundle.get("worklogs"), list) else [],
        "endpoint_errors": bundle.get("endpoint_errors") if isinstance(bundle.get("endpoint_errors"), dict) else {},
        "field_names": issue.get("names") if isinstance(issue.get("names"), dict) else {},
        "field_schema": issue.get("schema") if isinstance(issue.get("schema"), dict) else {},
        "raw_issue_path": str(raw_path),
        "fetched_at": str(bundle.get("fetched_at") or _now_iso()),
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _secret_config(env_file: Path) -> tuple[str, str, str, str]:
    local = _load_env_file(env_file)
    base_url = os.environ.get("JIRA_BASE_URL") or local.get("JIRA_BASE_URL") or DEFAULT_BASE_URL
    token = (
        os.environ.get("JIRA_PAT")
        or os.environ.get("JIRA_TOKEN")
        or local.get("JIRA_PAT")
        or local.get("JIRA_TOKEN")
        or ""
    )
    account = os.environ.get("JIRA_ACCOUNT") or local.get("JIRA_ACCOUNT") or ""
    password = os.environ.get("JIRA_PWD") or local.get("JIRA_PWD") or ""
    return base_url.strip(), token.strip(), account.strip(), password


def _manifest(config: SyncConfig, identity: dict[str, Any], server_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": _now_iso(),
        "completed_at": "",
        "base_url": config.base_url,
        "projects": list(config.projects),
        "updated_since": config.updated_since,
        "authentication_mode": "pat" if config.token else "basic",
        "attachment_content_downloaded": False,
        "identity": {
            "name": str(identity.get("name") or ""),
            "display_name": str(identity.get("displayName") or ""),
        },
        "server": {
            "version": str(server_info.get("version") or ""),
            "deployment_type": str(server_info.get("deploymentType") or ""),
            "title": str(server_info.get("serverTitle") or ""),
        },
        "statistics": {
            "discovered": 0,
            "completed": 0,
            "skipped_existing": 0,
            "failed": 0,
        },
        "failures": [],
    }


def run_sync(config: SyncConfig) -> dict[str, Any]:
    client = JiraClient(config)
    identity = client.verify_identity()
    server_info = client.server_info()
    manifest = _manifest(config, identity, server_info)
    manifest_path = config.output_root / "manifest.json"
    _atomic_write_json(manifest_path, manifest)

    keys = client.search_issue_keys()
    manifest["statistics"]["discovered"] = len(keys)
    _atomic_write_json(config.output_root / "issue_keys.json", keys)
    _atomic_write_json(manifest_path, manifest)

    raw_root = config.output_root / "issues"
    detail_root = config.output_root / "fault_details"

    def process(key: str) -> tuple[str, str, str]:
        raw_path = raw_root / f"{key}.json"
        detail_path = detail_root / f"{key}.json"
        if raw_path.exists() and detail_path.exists() and not config.refresh_existing:
            return key, "skipped_existing", ""
        try:
            bundle = client.fetch_issue_bundle(key)
            normalized = normalize_issue_bundle(bundle, raw_path)
            _atomic_write_json(raw_path, bundle)
            _atomic_write_json(detail_path, normalized)
            return key, "completed", ""
        except Exception as exc:  # keep the full corpus moving; record bounded failure metadata
            return key, "failed", f"{type(exc).__name__}: {str(exc)[:500]}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(process, key): key for key in keys}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            key, status, error = future.result()
            manifest["statistics"][status] += 1
            if error:
                manifest["failures"].append({"issue_key": key, "error": error})
            if index % 25 == 0 or index == len(futures):
                _atomic_write_json(manifest_path, manifest)

    manifest["completed_at"] = _now_iso()
    _atomic_write_json(manifest_path, manifest)
    return manifest


def repair_existing_bundles(output_root: Path) -> dict[str, int]:
    """Repair early v2 bundles from their already-saved embedded Jira data."""
    stats = {"scanned": 0, "repaired": 0, "failed": 0}
    raw_root = output_root / "issues"
    detail_root = output_root / "fault_details"
    for raw_path in sorted(raw_root.glob("*.json")):
        stats["scanned"] += 1
        try:
            bundle = json.loads(raw_path.read_text(encoding="utf-8"))
            issue = bundle.get("issue") if isinstance(bundle.get("issue"), dict) else {}
            embedded = issue.get("changelog") if isinstance(issue.get("changelog"), dict) else {}
            histories = embedded.get("histories") if isinstance(embedded.get("histories"), list) else []
            total = int(embedded.get("total") or len(histories))
            saved = bundle.get("changelog") if isinstance(bundle.get("changelog"), list) else []
            changed = False
            if len(histories) >= total and len(saved) < len(histories):
                bundle["changelog"] = [item for item in histories if isinstance(item, dict)]
                changed = True
            if len(histories) >= total:
                errors = bundle.get("endpoint_errors") if isinstance(bundle.get("endpoint_errors"), dict) else {}
                if "values" in errors or "changelog" in errors:
                    errors.pop("values", None)
                    errors.pop("changelog", None)
                    bundle["endpoint_errors"] = errors
                    changed = True
            detail_path = detail_root / raw_path.name
            if changed:
                _atomic_write_json(raw_path, bundle)
                _atomic_write_json(detail_path, normalize_issue_bundle(bundle, raw_path))
                stats["repaired"] += 1
        except Exception:
            stats["failed"] += 1
    return stats


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--updated-since", default="")
    parser.add_argument("--max-issues", type=int, default=0)
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--no-remote-links", action="store_true")
    parser.add_argument("--no-worklogs", action="store_true")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from the environment (direct Jira access is the default)",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--repair-existing-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.repair_existing_only:
        print(json.dumps(repair_existing_bundles(args.output_root), ensure_ascii=False, indent=2))
        return 0
    base_url, token, account, password = _secret_config(args.env_file)
    if not token and not (account and password):
        raise SystemExit(
            f"Jira credentials are missing. Add JIRA_PAT/JIRA_TOKEN or JIRA_ACCOUNT/JIRA_PWD "
            f"to {args.env_file} (mode 0600) or the process environment."
        )
    config = SyncConfig(
        base_url=base_url,
        token=token,
        account=account,
        password=password,
        projects=tuple(dict.fromkeys(str(item).strip().upper() for item in args.projects if str(item).strip())),
        output_root=args.output_root,
        workers=max(1, args.workers),
        page_size=max(1, min(1000, args.page_size)),
        timeout_seconds=max(1.0, args.timeout_seconds),
        max_retries=max(0, args.max_retries),
        updated_since=str(args.updated_since or "").strip(),
        max_issues=max(0, args.max_issues),
        refresh_existing=bool(args.refresh_existing),
        fetch_remote_links=not args.no_remote_links,
        fetch_worklogs=not args.no_worklogs,
        use_environment_proxy=bool(args.use_env_proxy),
    )
    client = JiraClient(config)
    if args.probe_only:
        identity = client.verify_identity()
        info = client.server_info()
        print(json.dumps({
            "authenticated": True,
            "user": str(identity.get("displayName") or identity.get("name") or ""),
            "server_version": str(info.get("version") or ""),
            "projects": list(config.projects),
        }, ensure_ascii=False, indent=2))
        return 0
    manifest = run_sync(config)
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))
    return 0 if not manifest["statistics"]["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
