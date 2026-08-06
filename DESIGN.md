# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-07-14
- Primary product surface: `data/results/kg_v2_overview.html`
- Evidence reviewed: `src/debug_agent_system/eval/write_side/kg_v2_overview.py`, current generated overview

## Product goals
- Make the current KG readable as `FaultFamily -> FaultVariant -> Action -> Outcome`.
- Let an engineer scan families, filter them, and understand one variant's action results without opening raw JSON.
- Keep provenance and quality signals visible without competing with the diagnostic hierarchy.

## Design principles
- Hierarchy before decoration: every visual grouping must reinforce the KG relationship.
- Scan before inspect: counts, status, and filters are visible before long text.
- Evidence stays subordinate: provenance is available in detail views, not presented as the primary node label.
- Dense but calm: this is an engineering workbench, not a marketing page.

## Visual language
- Color: charcoal blue-gray base with restrained blue for selection, green for verified states, amber for uncertainty, and red for unsafe/high-cost states.
- Typography: system UI stack with compact headings and readable Chinese body text.
- Spacing/layout rhythm: 8px base rhythm, two-column workbench on desktop, stacked layout below 1200px.
- Shape/radius/elevation: 10-16px radius, subtle borders, no decorative floating elements.

## Components
- Family navigator: searchable, filterable list with quality and backlog badges.
- Snapshot metrics: compact counts for the current graph.
- Variant lane: one framed block per variant, with actions and outcome pills inside.
- Family aggregate: secondary view for cross-variant action frequency.

## Accessibility
- Maintain readable contrast and visible focus states.
- Preserve native form controls and keyboard-operable family selection.
- Do not encode outcome meaning by color alone; retain text labels.

## Responsive behavior
- Desktop: fixed-width navigator plus scrollable detail workspace.
- Narrow screens: navigator becomes a bounded top section, then detail content follows.

## Implementation constraints
- The HTML is generated from `kg_v2_overview.py`; generated files must be regenerated after template changes.
- The snapshot remains a sibling JSON file; this change does not alter the data contract.

## Open questions
- [ ] Convert the overview to a standalone downloadable HTML when offline sharing becomes a requirement.
