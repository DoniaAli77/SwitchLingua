"""
pipeline_wrappers.py
~~~~~~~~~~~~~~~~~~~~
Thin experiment wrappers around the two SwitchLingua pipeline implementations.

Public API
----------
run_original_baseline_generation(task, count, model_name, output_path,
                                  config_overrides, seed)
run_modified_generation(task, count, model_name, output_path,
                         config_overrides, seed)

Both return ``list[dict]`` containing normalised per-sentence records and
write those records incrementally to *output_path* (JSONL, one record per line).

Dry-run / mock mode
-------------------
Set the environment variable ``SWITCHLINGUA_DRY_RUN=1`` **before** importing
or calling these functions to skip all API calls.  Synthetic placeholder data
is returned so downstream analysis scripts can be exercised without spending
API budget.  Can also be toggled at runtime via :func:`set_dry_run`.

Original_baseLine task support
-------------------------------
Original_baseLine only natively supports **topic** generation.  Calling
``run_original_baseline_generation`` with ``task="sentiment"`` or
``task="ner"`` emits a :class:`UserWarning` and falls back to forwarding
the task string as the topic label.  No task-validation agent or
label-conditioning prompts are applied — those are Modified_Version features
only.  If an explicit topic-prompt override is needed, pass it via
``config_overrides={"topics": ["<your topic>"]}``.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import pathlib
import random
import sys
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2]
BASELINE_CORE = ROOT / "Original_baseLine" / "core"
MODIFIED_CORE = ROOT / "Modified_Version" / "core"

# Module names that exist in both core directories — must be evicted from
# sys.modules whenever we switch the active core.
_SHARED_MODULES: frozenset[str] = frozenset([
    "utils", "node_engine", "node_models", "prompt",
    "mcp_tools", "agents", "run_french",
])

# ---------------------------------------------------------------------------
# Dry-run / mock mode
# ---------------------------------------------------------------------------
_DRY_RUN: bool = os.getenv("SWITCHLINGUA_DRY_RUN", "0").strip() == "1"


def set_dry_run(enabled: bool) -> None:
    """Toggle mock mode programmatically (useful in unit tests)."""
    global _DRY_RUN
    _DRY_RUN = enabled


# ---------------------------------------------------------------------------
# Core-switching
# ---------------------------------------------------------------------------

def _activate_core(core_dir: pathlib.Path) -> None:
    """Swap the active core on sys.path and evict all shared cached modules."""
    for d in (BASELINE_CORE, MODIFIED_CORE):
        try:
            sys.path.remove(str(d))
        except ValueError:
            pass
    sys.path.insert(0, str(core_dir))
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# Minimal config / pre_execute builders
# ---------------------------------------------------------------------------

_DEFAULT_CHARACTER: dict[str, Any] = {
    "nationality": {"first_language": "Arabic", "second_language": "English"},
    "gender": ["Male"],
    "age": ["18-25"],
    "education_level": ["College"],
}


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in-place."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _build_baseline_config(task: str, overrides: dict) -> dict:
    """Build a minimal config dict for Original_baseLine's ``generate_scenarios``.

    Original_baseLine's generator iterates over ``topics``, ``tense``,
    ``perspective``, ``character_setting.*``, ``cs_ratio``,
    ``conversation_type``, ``cs_function``, ``cs_type``.  The ``task``
    argument is placed in ``topics`` (the only conditioning dimension the
    baseline supports).
    """
    cfg: dict[str, Any] = {
        "topics": [task],
        "tense": ["Present"],
        "perspective": ["First Person"],
        "cs_ratio": ["70%"],
        "character_setting": {
            "nationality": dict(_DEFAULT_CHARACTER["nationality"]),
            "gender": list(_DEFAULT_CHARACTER["gender"]),
            "age": list(_DEFAULT_CHARACTER["age"]),
            "education_level": list(_DEFAULT_CHARACTER["education_level"]),
        },
        "conversation_type": ["single_turn"],
        "cs_function": ["Expressive"],
        "cs_type": ["Intrasentential"],
        "use_tools": False,
    }
    _deep_merge(cfg, overrides)
    return cfg


def _build_modified_pre_execute(task: str, overrides: dict) -> dict:
    """Build a minimal ``pre_execute`` dict for Modified_Version's ``generate_scenarios``.

    Provides sensible single-scenario defaults for all three task types so
    the scenario list is never empty even without a YAML config file.
    """
    # Use task string as topic label when the task is "topic"; otherwise fall
    # back to a neutral domain so the shared config block is valid.
    default_topic = task if task == "topic" else "tech"

    pre: dict[str, Any] = {
        "task": [task],
        "cs_ratio": ["70%"],
        "shared": {
            "topic": [default_topic],
            "tense": ["Present"],
            "perspective": ["First Person"],
            "cs_function": ["Expressive"],
            "cs_type": ["Intrasentential"],
            "conversation_type": ["single_turn"],
            "character_setting": {
                "nationality": dict(_DEFAULT_CHARACTER["nationality"]),
                "gender": list(_DEFAULT_CHARACTER["gender"]),
                "age": list(_DEFAULT_CHARACTER["age"]),
                "education_level": list(_DEFAULT_CHARACTER["education_level"]),
            },
            "use_tools": False,
            "output_format": "json",
        },
        # task-specific sub-configs (only the requested task will be used)
        "topic": {
            "topics": [task],
        },
        "sentiment": {
            "labels": ["positive", "negative", "neutral"],
            "intensity": ["low"],
            "ambiguity": ["low"],
        },
        "ner": {
            "entity_types": ["PER", "ORG", "LOC"],
            "min_entities": [2],
            "max_entities": [3],
            "must_include_types": ["PER"],
            "allow_code_switched_entities": [True],
        },
    }
    _deep_merge(pre, overrides)
    return pre


# ---------------------------------------------------------------------------
# Conversation ID
# ---------------------------------------------------------------------------

def _make_conversation_id(state: dict) -> str:
    """Return a short deterministic ID derived from the scenario's key fields."""
    parts = ":".join(
        str(state.get(f, ""))
        for f in (
            "task", "label", "topic", "tense", "perspective",
            "gender", "age", "education_level", "cs_ratio",
            "conversation_type", "cs_function", "cs_type",
        )
    )
    return hashlib.sha1(parts.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Mock state factories (dry-run mode)
# ---------------------------------------------------------------------------

def _mock_baseline_state(scenario: dict) -> dict:
    """Return a synthetic state that matches Original_baseLine's output shape."""
    state = dict(scenario)
    state.update({
        "data_generation_result": [
            "[DRY-RUN] هذه جملة تجريبية — mock sentence one.",
            "[DRY-RUN] هذه جملة تجريبية — mock sentence two.",
        ],
        "fluency_result": {"fluency_score": 8.0, "errors": {}, "summary": "mock"},
        "naturalness_result": {"naturalness_score": 8.0, "observations": {}, "summary": "mock"},
        "cs_ratio_result": {"ratio_score": 7.5, "computed_ratio": "70%", "notes": "mock"},
        "social_cultural_result": {"socio_cultural_score": 8.0, "issues": "", "summary": "mock"},
        "score": 7.9,
        "refine_count": 0,
        "summary": "mock summary",
    })
    return state


def _mock_modified_state(scenario: dict) -> dict:
    """Return a synthetic state that matches Modified_Version's output shape."""
    state = _mock_baseline_state(scenario)
    sentences: list[str] = state["data_generation_result"]
    state.update({
        "task": scenario.get("task", "topic"),
        "sentence_records": [
            {
                "index": i,
                "text": s,
                "fluency": {"fluency_score": 8.0, "errors": {}, "summary": "mock"},
                "naturalness": {"naturalness_score": 8.0, "observations": {}, "summary": "mock"},
                "cs_ratio": {"ratio_score": 7.5, "computed_ratio": "70%", "notes": "mock"},
                "socio_cultural": {"socio_cultural_score": 8.0, "issues": "", "summary": "mock"},
                "weighted_score": 7.9,
                "refine_count": 0,
                "status": "pass",
                "task_passed": True,
                "task_validation": {"passed": True, "confidence": 0.9, "notes": "mock"},
            }
            for i, s in enumerate(sentences)
        ],
        "failing_sentence_indices": [],
        "instance_refine_counts": [0] * len(sentences),
        "sentence_scores": [7.9] * len(sentences),
        "task_validation_results_per_instances": [
            {"passed": True, "confidence": 0.9, "notes": "mock"}
            for _ in sentences
        ],
    })
    return state


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------

def _normalize_baseline_state(
    state: dict,
    system_id: str,
    model_used: str,
) -> list[dict]:
    """Expand one Original_baseLine scenario result into per-sentence records.

    Scores are scenario-level aggregates because Original_baseLine does not
    track per-sentence quality metrics.
    """
    sentences: list[str] = state.get("data_generation_result") or []
    conv_id = _make_conversation_id(state)
    task = state.get("task", "topic")
    label = state.get("label") or state.get("topic")

    agg_scores: dict[str, Any] = {
        "fluency": (state.get("fluency_result") or {}).get("fluency_score"),
        "naturalness": (state.get("naturalness_result") or {}).get("naturalness_score"),
        "cs_ratio": (state.get("cs_ratio_result") or {}).get("ratio_score"),
        "socio_cultural": (state.get("social_cultural_result") or {}).get("socio_cultural_score"),
        "weighted": state.get("score"),
    }

    return [
        {
            "system_id": system_id,
            "model_used": model_used,
            "task": task,
            "sentence_id": f"{conv_id}:{i}",
            "conversation_id": conv_id,
            "label": label,
            "tags": None,  # not produced by Original_baseLine
            "text": text,
            "target_cs_ratio": state.get("cs_ratio"),
            "actual_cs_ratio": (state.get("cs_ratio_result") or {}).get("computed_ratio"),
            "scores": agg_scores,
            "accepted": True,  # AcceptanceAgent always writes without rejection
            "rejection_reasons": None,
            "refinement_count": state.get("refine_count"),
            "api_call_count": None,  # not tracked by Original_baseLine
        }
        for i, text in enumerate(sentences)
    ]


def _normalize_modified_state(state: dict, model_used: str) -> list[dict]:
    """Expand one Modified_Version scenario result into per-sentence records.

    Uses per-sentence ``sentence_records`` when available (the preferred
    source); falls back to scenario-level aggregates otherwise.
    """
    sentences: list[str] = state.get("data_generation_result") or []
    conv_id = _make_conversation_id(state)
    task = state.get("task", "topic")
    label = state.get("label") or state.get("topic")

    sentence_records: list[dict] = state.get("sentence_records") or []
    instance_refine_counts: list[int] = state.get("instance_refine_counts") or []
    failing_indices: set[int] = set(state.get("failing_sentence_indices") or [])
    tv_per_instance: list[dict] = state.get("task_validation_results_per_instances") or []

    agg_scores: dict[str, Any] = {
        "fluency": (state.get("fluency_result") or {}).get("fluency_score"),
        "naturalness": (state.get("naturalness_result") or {}).get("naturalness_score"),
        "cs_ratio": (state.get("cs_ratio_result") or {}).get("ratio_score"),
        "socio_cultural": (state.get("social_cultural_result") or {}).get("socio_cultural_score"),
        "weighted": state.get("score"),
    }

    records: list[dict] = []
    for i, text in enumerate(sentences):
        per_rec: dict = sentence_records[i] if i < len(sentence_records) else {}

        if per_rec:
            scores: dict[str, Any] = {
                "fluency": (per_rec.get("fluency") or {}).get("fluency_score"),
                "naturalness": (per_rec.get("naturalness") or {}).get("naturalness_score"),
                "cs_ratio": (per_rec.get("cs_ratio") or {}).get("ratio_score"),
                "socio_cultural": (per_rec.get("socio_cultural") or {}).get("socio_cultural_score"),
                "weighted": per_rec.get("weighted_score"),
            }
            accepted: bool = per_rec.get("status") in ("pass", "refined_pass")
            ref_count: int | None = per_rec.get("refine_count", 0)
        else:
            scores = agg_scores
            accepted = i not in failing_indices
            ref_count = (
                instance_refine_counts[i]
                if i < len(instance_refine_counts)
                else state.get("refine_count")
            )

        tv: dict = (
            tv_per_instance[i]
            if i < len(tv_per_instance)
            else (state.get("task_validation_result") or {})
        )

        records.append({
            "system_id": "modified",
            "model_used": model_used,
            "task": task,
            "sentence_id": f"{conv_id}:{i}",
            "conversation_id": conv_id,
            "label": label if task != "ner" else None,
            "tags": state.get("annotations") if task == "ner" else None,
            "text": text,
            "target_cs_ratio": state.get("cs_ratio"),
            "actual_cs_ratio": (state.get("cs_ratio_result") or {}).get("computed_ratio"),
            "scores": scores,
            "accepted": accepted,
            "rejection_reasons": tv.get("errors") if not accepted else None,
            "refinement_count": ref_count,
            "api_call_count": None,  # not tracked per-call by Modified_Version
        })

    return records


# ---------------------------------------------------------------------------
# JSONL incremental writer
# ---------------------------------------------------------------------------

def _append_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def run_original_baseline_generation(
    task: str,
    count: int,
    model_name: str,
    output_path: str,
    config_overrides: dict | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run Original_baseLine for *count* scenarios and return normalised records.

    Task support
    ------------
    Original_baseLine natively supports **topic-based** generation only.
    Passing ``task="sentiment"`` or ``task="ner"`` emits a ``UserWarning``
    and uses the task string as the topic label for best-effort generation.
    No task-validation agent or label-conditioning is applied.  To provide a
    specific topic prompt, use ``config_overrides={"topics": ["<topic>"]}``.

    Parameters
    ----------
    task:
        Generation task.  ``"topic"`` is fully supported.  ``"sentiment"``
        and ``"ner"`` are passed through as the topic string only (no native
        support; a ``UserWarning`` is raised).
    count:
        Maximum number of scenarios to run.  The full scenario list is built
        from the config, shuffled (when *seed* is set), then truncated.
    model_name:
        Stored in every output record as ``model_used``.  Also patched onto
        ``node_engine.MODEL`` so the actual OpenAI call uses this model.
    output_path:
        Destination JSONL file path.  Records are appended incrementally —
        partial results survive if the run is interrupted.
    config_overrides:
        Deep-merged onto the programmatically generated config before scenario
        expansion.  Useful for overriding ``cs_ratio``, language pair, etc.
    seed:
        RNG seed for Python's ``random`` module.  Affects scenario shuffle
        order so the same *count* subset is reproduced across runs.

    Returns
    -------
    list[dict]
        Flat list of per-sentence records in the common schema.

    Notes
    -----
    * **Dry-run mode**: set ``SWITCHLINGUA_DRY_RUN=1`` (or call
      :func:`set_dry_run`) to get synthetic placeholder records without any
      API calls.
    * **Async context**: this function calls ``asyncio.run()`` internally.
      Running it inside an already-running event loop (e.g. Jupyter) will
      raise ``RuntimeError``.  Install ``nest_asyncio`` and call
      ``nest_asyncio.apply()`` first, or ``await`` the internal coroutine
      directly.
    """
    if task not in ("topic",):
        warnings.warn(
            f"Original_baseLine does not natively support task={task!r}. "
            "The task string will be used as the topic label. "
            "No task validation or label conditioning will be applied. "
            "Use config_overrides={'topics': ['<topic>']} to set an explicit topic.",
            UserWarning,
            stacklevel=2,
        )

    if seed is not None:
        random.seed(seed)

    _activate_core(BASELINE_CORE)
    import node_engine as _ne  # noqa: PLC0415
    import run_french as _rf   # noqa: PLC0415
    import utils as _ut        # noqa: PLC0415

    # Patch the model and redirect the broken "YOUR_OUTPUT_DIR" placeholder
    # so AcceptanceAgent writes to the same directory as output_path.
    _ne.MODEL = model_name
    _ne.OUTPUT_DIR = str(pathlib.Path(output_path).parent)

    cfg = _build_baseline_config(task, config_overrides or {})
    scenarios = _ut.generate_scenarios(cfg)
    if seed is not None:
        random.shuffle(scenarios)
    scenarios = scenarios[:count]

    out = pathlib.Path(output_path)
    all_records: list[dict] = []

    async def _run_all() -> None:
        for i, scenario in enumerate(scenarios):
            print(f"[original_baseline/{task}] scenario {i + 1}/{len(scenarios)}")
            if _DRY_RUN:
                state: dict = _mock_baseline_state(scenario)
            else:
                agent = _rf.CodeSwitchingAgent(scenario)
                state = await agent.run() or {}
            records = _normalize_baseline_state(state, "original_baseline", model_name)
            all_records.extend(records)
            _append_jsonl(out, records)

    asyncio.run(_run_all())
    return all_records


def run_modified_generation(
    task: str,
    count: int,
    model_name: str,
    output_path: str,
    config_overrides: dict | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run Modified_Version for *count* scenarios and return normalised records.

    Task support
    ------------
    Modified_Version natively supports ``"topic"``, ``"sentiment"``, and
    ``"ner"`` tasks via dedicated data-generation and task-validation prompts
    (``TaskValidatorAgent``).  Task validation is enabled by default; set the
    environment variable ``ENABLE_TASK_VALIDATOR=0`` to disable it.

    Parameters
    ----------
    task:
        One of ``"topic"``, ``"sentiment"``, ``"ner"``.
    count:
        Maximum number of scenarios to run after shuffle.
    model_name:
        Stored in every record as ``model_used`` and patched onto
        ``node_engine.MODEL``.
    output_path:
        Destination JSONL file.  Written incrementally.
    config_overrides:
        Deep-merged onto the minimal ``pre_execute`` dict before scenario
        expansion.  For example, override the language pair with::

            config_overrides={"shared": {"character_setting": {
                "nationality": {"first_language": "French",
                                "second_language": "English"}
            }}}

    seed:
        RNG seed for reproducibility.

    Returns
    -------
    list[dict]
        Flat list of per-sentence records in the common schema.  Per-sentence
        quality scores (from ``sentence_records``) are used when available;
        scenario-level aggregates are the fallback.

    Notes
    -----
    * **Dry-run mode**: set ``SWITCHLINGUA_DRY_RUN=1`` (or call
      :func:`set_dry_run`) to get synthetic placeholder records without any
      API calls.
    * **Async context**: same constraint as :func:`run_original_baseline_generation`.
    """
    if task not in ("topic", "sentiment", "ner"):
        raise ValueError(
            f"Unsupported task {task!r}. "
            "Modified_Version supports: 'topic', 'sentiment', 'ner'."
        )

    if seed is not None:
        random.seed(seed)

    _activate_core(MODIFIED_CORE)
    import node_engine as _ne  # noqa: PLC0415
    import run_french as _rf   # noqa: PLC0415
    import utils as _ut        # noqa: PLC0415

    _ne.MODEL = model_name
    _ne.OUTPUT_DIR = str(pathlib.Path(output_path).parent)

    pre_execute = _build_modified_pre_execute(task, config_overrides or {})
    scenarios = _ut.generate_scenarios(pre_execute)
    if seed is not None:
        random.shuffle(scenarios)
    scenarios = scenarios[:count]

    out = pathlib.Path(output_path)
    all_records: list[dict] = []

    async def _run_all() -> None:
        for i, scenario in enumerate(scenarios):
            print(f"[modified/{task}] scenario {i + 1}/{len(scenarios)}")
            if _DRY_RUN:
                state: dict = _mock_modified_state(scenario)
            else:
                agent = _rf.CodeSwitchingAgent(scenario)
                state = await agent.run() or {}
            records = _normalize_modified_state(state, model_name)
            all_records.extend(records)
            _append_jsonl(out, records)

    asyncio.run(_run_all())
    return all_records
