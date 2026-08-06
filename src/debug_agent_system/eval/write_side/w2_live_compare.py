from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_agent_system.eval.write_side.w2_live_report import build_live_report


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}


def _ensure_report(path_or_dir: str | Path) -> dict[str, Any]:
    p = Path(path_or_dir)
    if p.is_dir():
        return build_live_report(p)
    return _load_report(p)


def compare_reports(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    bq = (base.get('quality_diagnostics') or {}).get('counters') if isinstance(base.get('quality_diagnostics'), dict) else {}
    cq = (candidate.get('quality_diagnostics') or {}).get('counters') if isinstance(candidate.get('quality_diagnostics'), dict) else {}
    if not isinstance(bq, dict):
        bq = {}
    if not isinstance(cq, dict):
        cq = {}
    bg = (base.get('quality_gate') or {}).get('checks') if isinstance(base.get('quality_gate'), dict) else {}
    cg = (candidate.get('quality_gate') or {}).get('checks') if isinstance(candidate.get('quality_gate'), dict) else {}
    if not isinstance(bg, dict):
        bg = {}
    if not isinstance(cg, dict):
        cg = {}
    metrics = {
        'base_episodes': int((base.get('progress') or {}).get('episodes_completed') or 0),
        'candidate_episodes': int((candidate.get('progress') or {}).get('episodes_completed') or 0),
        'split_required_rate_base': round(float(bg.get('split_required_rate') or 0.0), 6),
        'split_required_rate_candidate': round(float(cg.get('split_required_rate') or 0.0), 6),
        'split_required_rate_delta': round(float(cg.get('split_required_rate') or 0.0) - float(bg.get('split_required_rate') or 0.0), 6),
        'report_noise_rate_base': round(float(bg.get('report_noise_rate') or 0.0), 6),
        'report_noise_rate_candidate': round(float(cg.get('report_noise_rate') or 0.0), 6),
        'report_noise_rate_delta': round(float(cg.get('report_noise_rate') or 0.0) - float(bg.get('report_noise_rate') or 0.0), 6),
        'long_variant_rate_base': round(float(bg.get('long_variant_rate') or 0.0), 6),
        'long_variant_rate_candidate': round(float(cg.get('long_variant_rate') or 0.0), 6),
        'long_variant_rate_delta': round(float(cg.get('long_variant_rate') or 0.0) - float(bg.get('long_variant_rate') or 0.0), 6),
        'empty_case_rate_base': round(float(bg.get('empty_case_rate') or 0.0), 6),
        'empty_case_rate_candidate': round(float(cg.get('empty_case_rate') or 0.0), 6),
        'empty_case_rate_delta': round(float(cg.get('empty_case_rate') or 0.0) - float(bg.get('empty_case_rate') or 0.0), 6),
        'noncanonical_family_rate_base': round(float(bg.get('noncanonical_family_rate') or 0.0), 6),
        'noncanonical_family_rate_candidate': round(float(cg.get('noncanonical_family_rate') or 0.0), 6),
        'noncanonical_family_rate_delta': round(float(cg.get('noncanonical_family_rate') or 0.0) - float(bg.get('noncanonical_family_rate') or 0.0), 6),
    }
    return {
        'schema_version': 'debug_agent_system.w2_live_compare.v1',
        'base_run_dir': base.get('run_dir') or '',
        'candidate_run_dir': candidate.get('run_dir') or '',
        'metrics': metrics,
        'base_status': base.get('status') or '',
        'candidate_status': candidate.get('status') or '',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', default='')
    args = parser.parse_args(argv)
    report = compare_reports(_ensure_report(args.base), _ensure_report(args.candidate))
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
