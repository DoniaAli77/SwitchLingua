# Stateful Multi-Agent Text Classification Pipeline

Clean Python project skeleton for a stateful, agentic text-classification pipeline.

## Components

1. Primary classifier
2. Router
3. Lexical agent
4. Contextual agent
5. Logic agent
6. Consensus agent
7. Explainability agent
8. Pipeline orchestrator

## Design Principles

- Plain Python only
- Shared dataclass state passed across components
- Type hints and explicit contracts
- Stub-only implementation (no business logic)

## Quickstart

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
```

## Project Tree

See full tree in the generated implementation output.
