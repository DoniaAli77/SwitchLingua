"""src/config/loader.py

Config-driven task loader.

Reads a pipeline YAML config file (e.g. ``src/config/default.yaml``), selects the
active task, and returns a :class:`TaskBundle` containing:

* a fully populated :class:`~src.state.schema.TaskConfig`
* ``keyword_map``  — ``{label: [token, ...]}``  for :class:`~src.agents.lexical_agent.LexicalAgent`
* ``rule_map``     — ``{label: [pattern, ...]}`` for :class:`~src.agents.logic_agent.LogicAgent`

Maps are derived from the task's ``label_knowledge`` section:
    keyword_map[label] = keywords_l1 + keywords_l2
    rule_map[label]    = regex_rules

Labels with no knowledge entries are silently skipped (agents handle absent
keys gracefully).

Usage
-----
::

    from src.config.loader import load_task_bundle

    bundle = load_task_bundle("src/config/default.yaml")
    # bundle.task_config, bundle.keyword_map, bundle.rule_map

    # Override active task and execution settings:
    bundle = load_task_bundle(
        "src/config/default.yaml",
        active_task="topic_classification",
        threshold=0.7,
        pipeline_mode="paper_style",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.state.schema import TaskConfig


@dataclass(slots=True)
class TaskBundle:
    """All task-level artefacts produced by :func:`load_task_bundle`.

    Attributes
    ----------
    task_config:
        Fully populated :class:`~src.state.schema.TaskConfig` ready to pass
        to the orchestrator.
    keyword_map:
        Mapping of ``label → [token, ...]`` for
        :class:`~src.agents.lexical_agent.LexicalAgent`.
    rule_map:
        Mapping of ``label → [regex_pattern, ...]`` for
        :class:`~src.agents.logic_agent.LogicAgent`.
    active_task:
        The task key that was resolved from the config (after any override).
    """

    task_config: TaskConfig
    keyword_map: Dict[str, List[str]] = field(default_factory=dict)
    rule_map: Dict[str, List[str]] = field(default_factory=dict)
    active_task: str = ""


def load_task_bundle(
    config_path: str | Path,
    *,
    active_task: Optional[str] = None,
    pipeline_mode: Optional[str] = None,
    threshold: Optional[float] = None,
    enable_deliberation: Optional[bool] = None,
    contextual_use_prior_outputs: Optional[bool] = None,
) -> TaskBundle:
    """Load a pipeline config YAML and return a :class:`TaskBundle`.

    Parameters
    ----------
    config_path:
        Path to the YAML config file (e.g. ``src/config/default.yaml``).
    active_task:
        Override for ``config.active_task``.  When ``None`` the YAML value
        is used.
    pipeline_mode:
        Override for ``config.execution.pipeline_mode``.
    threshold:
        Override for ``config.execution.threshold``.
    enable_deliberation:
        Override for ``config.execution.enable_deliberation``.
    contextual_use_prior_outputs:
        Override for ``config.execution.contextual_use_prior_outputs``.

    Returns
    -------
    TaskBundle
        Bundle containing a :class:`~src.state.schema.TaskConfig` and the
        ``keyword_map`` / ``rule_map`` built from the active task's
        ``label_knowledge`` section.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    KeyError
        If the resolved *active_task* is not present under ``tasks:`` in the
        config.
    ValueError
        If ``active_task`` cannot be resolved (neither passed nor set in
        config), or if the YAML is structurally invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    # ── Resolve active task name ─────────────────────────────────────────
    task_name = active_task or raw.get("active_task")
    if not task_name:
        raise ValueError(
            "Cannot resolve active task: set 'active_task' in the config "
            "or pass active_task= to load_task_bundle()."
        )

    tasks: Dict[str, Any] = raw.get("tasks", {})
    if task_name not in tasks:
        available = list(tasks.keys())
        raise KeyError(
            f"Task '{task_name}' not found in config. "
            f"Available tasks: {available}"
        )

    task_def: Dict[str, Any] = tasks[task_name]

    # ── Execution settings — CLI overrides take priority ─────────────────
    exec_cfg: Dict[str, Any] = raw.get("execution", {})
    _threshold = threshold if threshold is not None else float(exec_cfg.get("threshold", 0.65))
    _pipeline_mode: str = pipeline_mode or str(exec_cfg.get("pipeline_mode", "full_agentic"))
    _enable_deliberation: bool = (
        enable_deliberation
        if enable_deliberation is not None
        else bool(exec_cfg.get("enable_deliberation", False))
    )
    _contextual_use_prior: bool = (
        contextual_use_prior_outputs
        if contextual_use_prior_outputs is not None
        else bool(exec_cfg.get("contextual_use_prior_outputs", False))
    )
    _agents_use_primary_signal: bool = bool(
        exec_cfg.get("agents_use_primary_signal", False)
    )

    # ── Labels and descriptions ───────────────────────────────────────────
    labels: List[str] = list(task_def.get("labels", []))
    label_descriptions: Dict[str, str] = dict(task_def.get("label_descriptions", {}))

    # ── Build keyword_map and rule_map from label_knowledge ───────────────
    label_knowledge: Dict[str, Any] = task_def.get("label_knowledge", {}) or {}
    keyword_map: Dict[str, List[str]] = {}
    rule_map: Dict[str, List[str]] = {}

    for lbl in labels:
        knowledge: Dict[str, Any] = label_knowledge.get(lbl, {}) or {}
        kw_l1: List[str] = list(knowledge.get("keywords_l1", None) or [])
        kw_l2: List[str] = list(knowledge.get("keywords_l2", None) or [])
        rules: List[str] = list(knowledge.get("regex_rules", None) or [])

        combined_kws = kw_l1 + kw_l2
        if combined_kws:
            keyword_map[lbl] = combined_kws
        if rules:
            rule_map[lbl] = rules

    task_config = TaskConfig(
        task_name=task_name,
        task_type=str(task_def.get("task_type", "classification")),
        labels=labels,
        label_descriptions=label_descriptions,
        threshold=_threshold,
        enable_deliberation=_enable_deliberation,
        contextual_use_prior_outputs=_contextual_use_prior,
        agents_use_primary_signal=_agents_use_primary_signal,
        pipeline_mode=_pipeline_mode,  # type: ignore[arg-type]
    )

    return TaskBundle(
        task_config=task_config,
        keyword_map=keyword_map,
        rule_map=rule_map,
        active_task=task_name,
    )
