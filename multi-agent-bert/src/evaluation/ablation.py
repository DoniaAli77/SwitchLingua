"""Ablation framework for the multi-agent text-classification pipeline.

Provides three building blocks:

``AblationConfig``
    Declares which agents are active, the consensus weights, and an optional
    router threshold override for a single ablation variant.

``AblationStudy``
    Runs :class:`~src.evaluation.evaluator.Evaluator` for every supplied
    ``AblationConfig``, collects the results, and builds a comparison table.

``AblationReport``
    Returned by :meth:`AblationStudy.run`; contains per-config
    :class:`~src.evaluation.evaluator.EvalReport` objects and the flat
    comparison table.

Config-driven usage
-------------------
Define variants in YAML (one file, ``ablations:`` list) and pass the path to
:meth:`AblationConfig.load_yaml`::

    ablations:
      - name: full_pipeline
        description: All agents enabled
        use_lexical: true
        use_contextual: true
        use_logic: true

      - name: no_lexical
        description: Lexical agent disabled
        use_lexical: false

      - name: deliberation_on
        description: Full pipeline plus deliberation
        use_deliberation: true
        consensus_weights:
          deliberation: 1.5

      - name: contextual_dominant
        description: Contextual agent at triple weight
        consensus_weights:
          lexical: 0.5
          contextual: 3.0
          logic: 0.5

Programmatic usage
------------------
.. code-block:: python

    from src.evaluation.ablation import AblationConfig, AblationStudy

    configs = [
        AblationConfig("full_pipeline"),
        AblationConfig("no_lexical", use_lexical=False),
        AblationConfig("no_contextual", use_contextual=False),
        AblationConfig("no_logic", use_logic=False),
        AblationConfig("deliberation_on", use_deliberation=True),
    ]

    study = AblationStudy(
        task_config=task_config,
        primary_classifier=MockPrimaryClassifier(mode="heuristic"),
        keyword_map=KEYWORD_MAP,
        rule_map=RULE_MAP,
        llm_client=MockLLMClient(mode="heuristic"),
        threshold=0.65,
    )

    report = study.run(configs, dataset)
    study.save(report, output_dir="results/ablation/", run_id="exp_01")
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from src.evaluation.evaluator import EvalReport, Evaluator
from src.state.schema import PipelineState, TaskConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disabled-agent sentinel
# ---------------------------------------------------------------------------

class _DisabledAgent:
    """Lightweight no-op agent used when an ablation disables a component.

    It writes a single history event so the run is traceable, then returns
    state unchanged.  The corresponding state field (e.g.
    ``state.lexical_output``) is deliberately left as ``None`` so
    :class:`~src.agents.consensus_agent.ConsensusAgent` silently skips it.
    """

    def __init__(self, slot_name: str) -> None:
        self._slot = slot_name

    def run(self, state: PipelineState) -> PipelineState:
        state.append_history(
            component=self._slot,
            summary=f"{self._slot} disabled (ablation study)",
            outputs={"disabled": True},
        )
        return state


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Configuration for a single ablation variant.

    Parameters
    ----------
    name:
        Short identifier used in file names and the comparison table.
    use_lexical:
        Include the lexical agent in the escalation path.
    use_contextual:
        Include the contextual agent in the escalation path.
    use_logic:
        Include the logic agent in the escalation path.
    use_deliberation:
        Enable the deliberation stage (requires ``deliberation_agent`` in the
        ``AblationStudy``).
    consensus_weights:
        Override per-agent weights passed to
        :class:`~src.agents.consensus_agent.ConsensusAgent`.  Disabled agents
        are automatically forced to weight ``0.0`` regardless of this mapping.
    threshold:
        Override the router confidence threshold.  Falls back to the
        ``AblationStudy`` base threshold when ``None``.
    description:
        Human-readable description of this variant (shown in the comparison
        table and saved files).
    """

    name: str
    use_lexical: bool = True
    use_contextual: bool = True
    use_logic: bool = True
    use_deliberation: bool = False
    consensus_weights: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None
    description: str = ""

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def effective_weights(self) -> Dict[str, float]:
        """Return the consensus weight dict to use for this ablation.

        Disabled agents are forced to ``0.0`` regardless of any explicit
        ``consensus_weights`` entry.  Deliberation gets a default weight of
        ``1.0`` when enabled but not explicitly set.
        """
        weights: Dict[str, float] = dict(self.consensus_weights or {})
        if not self.use_lexical:
            weights["lexical"] = 0.0
        if not self.use_contextual:
            weights["contextual"] = 0.0
        if not self.use_logic:
            weights["logic"] = 0.0
        if self.use_deliberation:
            weights.setdefault("deliberation", 1.0)
        else:
            weights["deliberation"] = 0.0
        return weights

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "use_lexical": self.use_lexical,
            "use_contextual": self.use_contextual,
            "use_logic": self.use_logic,
            "use_deliberation": self.use_deliberation,
            "consensus_weights": self.consensus_weights,
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AblationConfig":
        """Construct from a plain dict (e.g. loaded from YAML/JSON)."""
        return cls(
            name=str(d["name"]),
            use_lexical=bool(d.get("use_lexical", True)),
            use_contextual=bool(d.get("use_contextual", True)),
            use_logic=bool(d.get("use_logic", True)),
            use_deliberation=bool(d.get("use_deliberation", False)),
            consensus_weights=d.get("consensus_weights") or None,
            threshold=d.get("threshold") or None,
            description=str(d.get("description", "")),
        )

    @classmethod
    def load_yaml(cls, path: str) -> List["AblationConfig"]:
        """Load a list of ablation configs from a YAML file.

        Expected format::

            ablations:
              - name: full_pipeline
                use_lexical: true
                ...
              - name: no_lexical
                use_lexical: false

        Parameters
        ----------
        path:
            Path to the YAML file.

        Returns
        -------
        list of AblationConfig
        """
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict) or "ablations" not in raw:
            raise ValueError(
                f"YAML file '{path}' must have a top-level 'ablations' list."
            )
        items = raw["ablations"]
        if not isinstance(items, list) or not items:
            raise ValueError(
                f"'ablations' in '{path}' must be a non-empty list."
            )
        configs = [cls.from_dict(item) for item in items]
        # Validate uniqueness of names.
        names = [c.name for c in configs]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate ablation names in '{path}': {sorted(duplicates)}"
            )
        return configs

    @classmethod
    def load_json(cls, path: str) -> List["AblationConfig"]:
        """Load a list of ablation configs from a JSON file.

        Expected format::

            {"ablations": [{"name": "full_pipeline", ...}, ...]}
        """
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict) or "ablations" not in raw:
            raise ValueError(
                f"JSON file '{path}' must have a top-level 'ablations' list."
            )
        return [cls.from_dict(item) for item in raw["ablations"]]


# ---------------------------------------------------------------------------
# AblationReport
# ---------------------------------------------------------------------------

@dataclass
class AblationReport:
    """Results from running an :class:`AblationStudy`.

    Attributes
    ----------
    run_id:
        Identifier shared by all output files.
    timestamp:
        ISO-8601 UTC timestamp of when :meth:`AblationStudy.run` completed.
    configs:
        The ablation configs that were run, in the order they were supplied.
    reports:
        One :class:`~src.evaluation.evaluator.EvalReport` per config, in the
        same order as ``configs``.
    comparison:
        Flat list of dicts — one per config — suitable for writing to a CSV.
        Keys: ``name``, ``description``, ``accuracy``, ``macro_f1``,
        ``escalation_rate``, ``escalated_accuracy``, ``escalated_count``,
        plus ``f1_{label}`` for every label in the task's label space.
    meta:
        Additional run metadata (num_samples, label list, eval mode, …).
    """

    run_id: str
    timestamp: str
    configs: List[AblationConfig]
    reports: List[EvalReport]
    comparison: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """Run :class:`~src.evaluation.evaluator.Evaluator` under multiple ablation configs.

    Parameters
    ----------
    task_config:
        :class:`~src.state.schema.TaskConfig` used for every run.  A shallow
        copy is made per variant so the threshold can be overridden per-config
        without mutation.
    primary_classifier:
        Any object with ``run(state) -> state``.  Shared across all variants.
    keyword_map:
        Passed to :class:`~src.agents.lexical_agent.LexicalAgent`.
    rule_map:
        Passed to :class:`~src.agents.logic_agent.LogicAgent`.
    llm_client:
        Passed to :class:`~src.agents.contextual_agent.ContextualAgent`.
    threshold:
        Base router threshold.  Overridden per-config when
        ``AblationConfig.threshold`` is set.
    mode:
        Evaluation mode — ``"primary_only"`` or ``"full_pipeline"``.
        Defaults to ``"full_pipeline"``.
    run_id:
        Identifier prefix for all output files.  Defaults to a UTC timestamp.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        primary_classifier,
        keyword_map: Optional[Dict[str, List[str]]] = None,
        rule_map: Optional[Dict[str, List[str]]] = None,
        llm_client=None,
        threshold: float = 0.65,
        mode: str = "full_pipeline",
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.task_config = task_config
        self.primary_classifier = primary_classifier
        self.keyword_map: Dict[str, List[str]] = keyword_map or {}
        self.rule_map: Dict[str, List[str]] = rule_map or {}
        self.llm_client = llm_client
        self.base_threshold = threshold
        self.mode = mode
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.logger = logger or log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        configs: List[AblationConfig],
        dataset: List[Dict[str, str]],
    ) -> AblationReport:
        """Evaluate the pipeline under each ablation config.

        Parameters
        ----------
        configs:
            List of :class:`AblationConfig` variants to evaluate.  Must be
            non-empty; names must be unique.
        dataset:
            List of ``{"text": "...", "label": "...", "id": "..."}`` dicts.

        Returns
        -------
        AblationReport
        """
        if not configs:
            raise ValueError("configs must be non-empty.")
        names = [c.name for c in configs]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate config names: {sorted(duplicates)}")

        if not dataset:
            raise ValueError("dataset is empty.")

        reports: List[EvalReport] = []
        for cfg in configs:
            self.logger.info(
                "AblationStudy — running config '%s' (%s)",
                cfg.name,
                cfg.description or "no description",
            )
            report = self._run_one_config(cfg, dataset)
            reports.append(report)

        comparison = self._build_comparison(configs, reports)

        return AblationReport(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            configs=configs,
            reports=reports,
            comparison=comparison,
            meta={
                "num_configs": len(configs),
                "num_samples": len(dataset),
                "labels": self.task_config.labels,
                "mode": self.mode,
                "base_threshold": self.base_threshold,
            },
        )

    def save(
        self,
        report: AblationReport,
        output_dir: str = "results",
        run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Persist the full ablation report to disk.

        Saves:

        * ``{run_id}_ablation_comparison.json`` — full comparison table
        * ``{run_id}_ablation_comparison.csv``  — flat comparison CSV
        * Per-config predictions and metrics (via :meth:`Evaluator.save`)

        Parameters
        ----------
        report:
            The :class:`AblationReport` returned by :meth:`run`.
        output_dir:
            Root directory.  Created if absent.
        run_id:
            Override the run id used in file names.

        Returns
        -------
        dict
            Mapping of logical name → absolute path for every written file.
        """
        rid = run_id or report.run_id
        os.makedirs(output_dir, exist_ok=True)

        paths: Dict[str, str] = {}

        # Comparison table.
        comp_json = os.path.join(output_dir, f"{rid}_ablation_comparison.json")
        comp_csv = os.path.join(output_dir, f"{rid}_ablation_comparison.csv")
        self._save_comparison_json(report, comp_json)
        self._save_comparison_csv(report, comp_csv)
        paths["comparison_json"] = comp_json
        paths["comparison_csv"] = comp_csv

        # Per-config detail files.
        for cfg, eval_report in zip(report.configs, report.reports):
            safe_name = cfg.name.replace(" ", "_").replace("/", "-")
            per_cfg_id = f"{rid}__{safe_name}"
            evaluator = self._make_evaluator(cfg, per_cfg_id)
            per_paths = evaluator.save(
                eval_report,
                output_dir=output_dir,
                run_id=per_cfg_id,
            )
            for key, path in per_paths.items():
                paths[f"{safe_name}__{key}"] = path

        self.logger.info(
            "AblationStudy — saved %d files to '%s'", len(paths), output_dir
        )
        return paths

    # ------------------------------------------------------------------
    # Internal — orchestrator construction
    # ------------------------------------------------------------------

    def _make_task_config_for(self, cfg: AblationConfig) -> TaskConfig:
        """Return a task config with the ablation threshold applied."""
        threshold = cfg.threshold if cfg.threshold is not None else self.base_threshold
        return TaskConfig(
            task_name=self.task_config.task_name,
            task_type=self.task_config.task_type,
            labels=list(self.task_config.labels),
            label_descriptions=dict(self.task_config.label_descriptions),
            threshold=threshold,
            contextual_use_prior_outputs=self.task_config.contextual_use_prior_outputs,
            enable_deliberation=cfg.use_deliberation,
            pipeline_mode="full_agentic",
        )

    def _build_orchestrator(self, cfg: AblationConfig):
        """Build a :class:`~src.pipeline.orchestrator.PipelineOrchestrator`
        with agents selectively replaced by :class:`_DisabledAgent` according
        to *cfg*."""
        # Import here to avoid circular deps and to keep this module importable
        # without the full agent tree installed.
        from src.agents.consensus_agent import ConsensusAgent
        from src.agents.contextual_agent import ContextualAgent
        from src.agents.deliberation_agent import DeliberationAgent
        from src.agents.explainability_agent import ExplainabilityAgent
        from src.agents.lexical_agent import LexicalAgent
        from src.agents.logic_agent import LogicAgent
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.router import Router

        lexical = (
            LexicalAgent(keyword_map=self.keyword_map)
            if cfg.use_lexical
            else _DisabledAgent("lexical_agent")
        )
        contextual = (
            ContextualAgent(llm_client=self.llm_client)
            if cfg.use_contextual
            else _DisabledAgent("contextual_agent")
        )
        logic = (
            LogicAgent(rule_map=self.rule_map)
            if cfg.use_logic
            else _DisabledAgent("logic_agent")
        )
        deliberation = (
            DeliberationAgent()
            if cfg.use_deliberation
            else None
        )

        consensus = ConsensusAgent(weights=cfg.effective_weights())

        return PipelineOrchestrator(
            primary_classifier=self.primary_classifier,
            router=Router(),
            lexical_agent=lexical,
            contextual_agent=contextual,
            logic_agent=logic,
            consensus_agent=consensus,
            explainability_agent=ExplainabilityAgent(),
            deliberation_agent=deliberation,
        )

    # ------------------------------------------------------------------
    # Internal — per-config evaluation
    # ------------------------------------------------------------------

    def _make_evaluator(self, cfg: AblationConfig, run_id: str) -> Evaluator:
        task_cfg = self._make_task_config_for(cfg)
        if self.mode == "primary_only":
            return Evaluator(
                task_config=task_cfg,
                primary_classifier=self.primary_classifier,
                mode="primary_only",
                run_id=run_id,
                logger=self.logger,
            )
        orchestrator = self._build_orchestrator(cfg)
        return Evaluator(
            task_config=task_cfg,
            orchestrator=orchestrator,
            mode="full_pipeline",
            run_id=run_id,
            logger=self.logger,
        )

    def _run_one_config(
        self, cfg: AblationConfig, dataset: List[Dict[str, str]]
    ) -> EvalReport:
        safe_name = cfg.name.replace(" ", "_").replace("/", "-")
        run_id = f"{self.run_id}__{safe_name}"
        evaluator = self._make_evaluator(cfg, run_id)
        return evaluator.evaluate(dataset)

    # ------------------------------------------------------------------
    # Internal — comparison table
    # ------------------------------------------------------------------

    def _build_comparison(
        self,
        configs: List[AblationConfig],
        reports: List[EvalReport],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for cfg, rpt in zip(configs, reports):
            per_class_f1 = {m.label: m.f1 for m in rpt.per_class}
            row: Dict[str, Any] = {
                "name":               cfg.name,
                "description":        cfg.description,
                "use_lexical":        cfg.use_lexical,
                "use_contextual":     cfg.use_contextual,
                "use_logic":          cfg.use_logic,
                "use_deliberation":   cfg.use_deliberation,
                "threshold":          cfg.threshold,
                "accuracy":           rpt.accuracy,
                "macro_f1":           rpt.macro_f1,
                "escalation_rate":    rpt.escalation_rate,
                "escalated_count":    rpt.escalated_count,
                "escalated_accuracy": rpt.escalated_accuracy,
                "num_samples":        rpt.num_samples,
                "valid_samples":      rpt.meta.get("valid_samples", rpt.num_samples),
                "error_samples":      rpt.meta.get("error_samples", 0),
            }
            for lbl in self.task_config.labels:
                row[f"f1_{lbl}"] = per_class_f1.get(lbl, 0.0)
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Internal — serialisation
    # ------------------------------------------------------------------

    def _save_comparison_json(self, report: AblationReport, path: str) -> None:
        payload = {
            "run_id":    report.run_id,
            "timestamp": report.timestamp,
            "meta":      report.meta,
            "comparison": report.comparison,
            "configs": [c.to_dict() for c in report.configs],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def _save_comparison_csv(self, report: AblationReport, path: str) -> None:
        if not report.comparison:
            return
        fieldnames = list(report.comparison[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report.comparison)
