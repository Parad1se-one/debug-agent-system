# Contributing

Thanks for your interest in `debug-agent-system`!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python3 tests/run_tests.py
```

## Conventions

- **Deterministic core**: the read-side runtime (`DebugAgentSystem`) must stay
  deterministic — no LLM calls in the core loop. Optional LLM paths are opt-in
  and read keys from local env files only.
- **Tests**: the suite uses `tests/run_tests.py` (stdlib runner, no network, no
  API keys). Keep new tests offline and mocked; tests that need proprietary
  data should fail gracefully with a clear message.
- **Data**: the raw proprietary corpus (field chats, tickets, internal docs) is
  **not** distributed. Do not commit it. Graph data under `data/kg_v2/` is a
  sanitized subset — keep it free of names, internal domains, and absolute
  paths.
- **No secrets**: never commit API keys or credentials. Use `.env.example` for
  documented env vars.

## Pull request checklist

- [ ] `PYTHONPATH=src python3 tests/run_tests.py` passes for offline tests
- [ ] No new sensitive content (names / domains / paths / keys)
- [ ] README updated if public API or behavior changed
