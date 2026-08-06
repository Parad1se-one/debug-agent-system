from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def build_report(rows: list[dict[str, Any]], *, sample_limit: int = 25) -> dict[str, Any]:
    family_counter: Counter[str] = Counter()
    pair_counter: Counter[str] = Counter()
    variant_counter: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    total_split = 0

    for row in rows:
        card = row.get('case_understanding_card') if isinstance(row.get('case_understanding_card'), dict) else {}
        if not card.get('split_required'):
            continue
        total_split += 1
        cases = [item for item in card.get('cases') or [] if isinstance(item, dict)]
        family_labels = [str((item.get('family_hypothesis') or {}).get('label') or '') for item in cases]
        variant_labels = [str((item.get('variant_hypothesis') or {}).get('label') or '') for item in cases]
        for fam in family_labels:
            if fam:
                family_counter[fam] += 1
        for var in variant_labels:
            if var:
                variant_counter[var] += 1
        norm_pair = ' | '.join(sorted([fam for fam in family_labels if fam]))
        if norm_pair:
            pair_counter[norm_pair] += 1
        if len(samples) < sample_limit:
            samples.append({
                'candidate_id': str(row.get('candidate_id') or ''),
                'label': str(row.get('label') or ''),
                'family_labels': family_labels,
                'variant_labels': variant_labels,
                'case_count': len(cases),
                'issues': list(card.get('schema_issues') or []),
            })

    return {
        'schema_version': 'debug_agent_system.w2_split_diagnostics.v1',
        'episodes': len(rows),
        'split_required_count': total_split,
        'top_split_families': family_counter.most_common(20),
        'top_split_family_pairs': pair_counter.most_common(20),
        'top_split_variants': variant_counter.most_common(20),
        'samples': samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_jsonl')
    parser.add_argument('--out', default='')
    parser.add_argument('--sample-limit', type=int, default=25)
    args = parser.parse_args(argv)
    report = build_report(_load_rows(Path(args.input_jsonl)), sample_limit=args.sample_limit)
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
