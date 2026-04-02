# Current Flow vs Original Baseline

Date: 2026-04-01

## Purpose

This document summarizes the current execution flow in Modified_Version and explains how it differs from Original_baseLine.

Scope:
- Current implementation: Modified_Version
- Reference implementation: Original_baseLine
- Focus: runtime flow, state shape, scoring, outputs, and practical implications

Out of scope:
- old_version and historical folders
- Generated outputs as source-of-truth for implementation unless they reflect code behavior

## Executive Summary

The current pipeline is not a small iteration on the original one. It changes the system at four important levels:

1. Scenario generation is now task-aware.
2. A dedicated task-validation stage was inserted between generation and quality scoring.
3. Most quality evaluation is now tracked per sentence and then aggregated.
4. CS-ratio scoring is no longer purely an LLM judgment; it uses deterministic token statistics plus an LLM scoring step.

The result is that Modified_Version behaves more like an evaluation pipeline for structured task-conditioned generation, while Original_baseLine behaves more like a generic code-switching text generation pipeline with one aggregate quality pass.

## Current Flow: Modified_Version

### 1. Entry Point and Runtime Setup

The current entrypoint is Modified_Version/core/run_french.py.

At startup it:
- applies a LangChain compatibility patch for older assumptions
- enables or disables task validation with ENABLE_TASK_VALIDATOR
- loads scenarios from config/config2.yaml
- shuffles scenarios before execution
- runs them through a LangGraph workflow

Important behavior in the current entrypoint:
- config path is fixed to ../config/config2.yaml
- task validation can be bypassed with an environment flag
- the workflow uses DataGenerationAgent -> TaskValidatorAgent -> evaluation agents -> summarization -> optional refinement -> acceptance
- the current loop advances in steps of 3 but only executes scenarios[i : i + 1], so it is effectively running one scenario at a time even though the outer loop is chunked

## 2. Scenario Construction

Scenario generation is implemented in Modified_Version/core/utils.py.

The current config is task-aware and nested under pre_execute. Instead of producing only style/topic combinations, it produces task-specific scenarios for:
- sentiment
- ner
- topic

Each scenario includes shared code-switching metadata plus task-specific payloads.

Examples:
- sentiment scenarios include label and task_constraints such as intensity and ambiguity
- ner scenarios include annotations plus entity-count/type constraints
- topic scenarios include a target label/topic

This is one of the biggest shifts from the baseline: generation is no longer just “produce code-switched text for a topic”, but “produce code-switched text that also satisfies a downstream NLP task specification”.

## 3. Workflow Graph

The current graph in Modified_Version/core/run_french.py is:

START
-> DataGenerationAgent
-> TaskValidatorAgent
-> FluencyAgent
-> NaturalnessAgent
-> CSRatioAgent
-> SocialCulturalAgent
-> SummarizeResult
-> conditional branch
   - RefinerAgent if score < 8 and refine_count < MAX_REFINER_ITERATIONS
   - AcceptanceAgent otherwise
-> END

Notes:
- MCPAgent is still defined but not wired into the active flow
- the validator runs before all scoring agents
- all four evaluation agents feed into a single summarization node

## 4. Data Generation

Generation still starts from a prompt-driven LLM call, but the input state is richer.

The generator now receives:
- the task type
- task label
- task constraints
- code-switching metadata
- demographic metadata
- language pair

The prompt layer in Modified_Version/core/prompt.py also includes separate task-validation prompts for topic, sentiment, and NER. This indicates that the generation prompt and downstream evaluation are designed around explicit task compliance, not only linguistic quality.

## 5. Task Validation Layer

This layer does not exist in the original baseline.

In Modified_Version/core/node_engine.py, task validation works as follows:
- if the generated output contains multiple sentences, each sentence is validated separately
- the system aggregates per-instance pass/fail, confidence, predicted labels, and errors
- task-specific validators are dispatched based on task type

Current task validators:
- RunSentimentTaskValidatorAgent
- RunNERTaskValidatorAgent
- RunTopicTaskValidatorAgent

The aggregate task_validation_result includes:
- passed
- confidence
- notes
- predicted_label
- errors
- per_instance_results

Practical effect:
- the current pipeline can tell whether the generated text meets the requested task semantics
- this makes the pipeline suitable for controlled dataset generation, not just stylistic generation

## 6. Evaluation Agents

### Fluency

Current behavior:
- evaluates each generated sentence in batch
- stores sentence-level results in fluency_results_per_instances
- computes an aggregate fluency_result as the average across instances

### Naturalness

Current behavior:
- evaluates each sentence in batch
- stores sentence-level results in naturalness_results_per_instances
- computes an aggregate naturalness_result as the average across instances

### CS Ratio

Current behavior is materially different from the baseline.

It now:
- computes deterministic Arabic and English token counts with regex-based logic
- derives actual Arabic/English ratios per sentence
- passes sentence text plus deterministic stats to the LLM prompt
- returns cs_ratio_results_per_instances
- fills missing per-sentence results with fallback records

This is a hybrid scoring design: part deterministic measurement, part LLM interpretation.

### Social / Cultural

Current behavior:
- evaluates each sentence in batch
- stores sentence-level results in social_cultural_results_per_instances
- computes aggregate social_cultural_result as an average summary over instances

## 7. Scoring

The weighting formula is still the same in shape:

- Fluency: 30%
- Naturalness: 25%
- CS ratio: 20%
- Socio-cultural: 25%

But the data source for scoring changed.

Original_baseLine scoring used only aggregate fields:
- fluency_result.fluency_score
- naturalness_result.naturalness_score
- cs_ratio_result.ratio_score
- social_cultural_result.socio_cultural_score

Modified_Version scoring first prefers per-instance lists and averages them:
- fluency_results_per_instances
- naturalness_results_per_instances
- cs_ratio_results_per_instances
- social_cultural_results_per_instances

Only if sentence-level data is absent does it fall back to aggregate fields.

Practical effect:
- the final score now reflects sentence-level consistency rather than a single batch-level judgment
- one very weak sentence can lower the score even if the batch summary looks acceptable

## 8. Summary and Acceptance

Current summarization stores both aggregate and per-instance outputs inside the record summary and computes the final score from the updated weighting scheme.

Acceptance writes the state into:
- Modified_Version/output/<language>.jsonl

In practice for the Arabic-English run, this is:
- Modified_Version/output/Arabic.jsonl

The output schema now contains many more fields than the baseline, including:
- task
- label
- task_constraints
- annotations
- task_validation_result
- fluency_results_per_instances
- naturalness_results_per_instances
- cs_ratio_results_per_instances
- social_cultural_results_per_instances

## 9. Post-Processing Flow

This is another practical difference from the baseline.

Modified_Version includes a separate export utility:
- Modified_Version/convertToExcel.py

That script converts the JSONL output into an analyst-friendly Excel sheet. It flattens both:
- scenario-level metadata
- sentence-level evaluation results

This export step is part of the current working flow even if it is not part of the LangGraph itself.

## Original Flow: Original_baseLine

### 1. Entry Point

The original entrypoint is Original_baseLine/core/run_french.py.

Compared with the current implementation, it is simpler:
- no LangChain compatibility patch
- no task validator toggle
- no task validation node at all
- config path points to ../config/config_augmented_french_eng.yaml

A notable operational detail is that it loops with:
- range(0, 8000, 40)
- tasks = scenarios[i : i + 40]

This means the baseline was designed to process large scenario batches. It assumes a large scenario set and chunks execution into groups of 40.

### 2. Scenario Construction

The original utils.generate_scenarios builds only generic combinations across:
- topic
- tense
- perspective
- gender
- age
- education level
- cs_ratio
- conversation_type
- cs_function
- cs_type

There is no notion of:
- task
- label
- task_constraints
- annotations

This means the original system generates code-switching examples as linguistic content, not as task-conditioned samples for sentiment, NER, or topic classification.

### 3. Workflow Graph

The original graph is:

START
-> DataGenerationAgent
-> FluencyAgent
-> NaturalnessAgent
-> CSRatioAgent
-> SocialCulturalAgent
-> SummarizeResult
-> conditional branch
   - RefinerAgent if score < 8 and refine_count < MAX_REFINER_ITERATIONS
   - AcceptanceAgent otherwise
-> END

This means the baseline uses a direct generation-to-evaluation pipeline with no semantic correctness gate in between.

### 4. Evaluation Model

In Original_baseLine/core/node_engine.py:
- each evaluation agent returns one aggregate object
- fluency_result is a single structured response
- naturalness_result is a single structured response
- cs_ratio_result is a single structured response
- social_cultural_result is a single structured response

The baseline therefore treats the batch as one evaluation unit.

### 5. State Shape

The original AgentRunningState is much smaller. It contains:
- linguistic metadata
- data_generation_result
- aggregate evaluation results
- summary
- score
- housekeeping fields

It does not track:
- task metadata
- validation output
- per-instance quality outputs
- richer output schema for downstream analytics

## Detailed Difference Matrix

### A. Scenario Semantics

Original_baseLine:
- topic-driven, generic code-switching generation

Modified_Version:
- task-aware generation for sentiment, NER, and topic classification

Impact:
- current flow is closer to benchmark or dataset synthesis for supervised NLP tasks
- original flow is closer to broad code-switching sample generation

### B. Validation Stage

Original_baseLine:
- absent

Modified_Version:
- explicit task validator inserted after generation

Impact:
- current flow can reject or diagnose semantically incorrect outputs
- original flow can only score linguistic quality, not task compliance

### C. Evaluation Granularity

Original_baseLine:
- aggregate-only scoring

Modified_Version:
- per-instance scoring plus aggregate summaries

Impact:
- current flow exposes sentence-level weaknesses
- baseline can hide uneven quality inside a strong overall response

### D. CS Ratio Logic

Original_baseLine:
- fully LLM-evaluated cs_ratio_result

Modified_Version:
- deterministic Arabic/English token statistics plus LLM scoring and per-instance output

Impact:
- current flow is more auditable and less opaque
- baseline depends more heavily on model judgment for ratio adherence

### E. Scoring Function

Original_baseLine:
- scores directly from aggregate results

Modified_Version:
- averages per-instance results first, then computes weighted total

Impact:
- current scoring is stricter about consistency across generated sentences

### F. Output Schema

Original_baseLine:
- compact JSONL records with aggregate metrics

Modified_Version:
- richer JSONL records with task metadata, validator output, per-instance metrics, and aggregate summaries

Impact:
- current output is much more useful for debugging and spreadsheet analysis
- it also requires more careful downstream parsing

### G. Operational Behavior

Original_baseLine:
- large-batch execution model
- hard-coded webhook placeholder
- config file referenced from outside the visible folder structure

Modified_Version:
- smaller current scenario set from config2.yaml
- one-scenario-at-a-time effective execution because of the current slice logic
- optional validator enable/disable switch
- webhook effectively disabled by default

Impact:
- current implementation is more controllable for testing
- baseline appears more oriented toward bulk generation runs

### H. Analytics Support

Original_baseLine:
- no built-in Excel conversion utility in scope

Modified_Version:
- includes convertToExcel.py and several test scripts for per-instance scoring and pipeline checks

Impact:
- current implementation supports inspection and regression analysis much better

## Current End-to-End Working Flow

For the way the project is currently being used, the real workflow is:

1. Edit task-aware config in Modified_Version/config/config2.yaml.
2. Run Modified_Version/core/run_french.py.
3. The pipeline generates task-conditioned code-switched outputs.
4. Each output is task-validated.
5. Each sentence is evaluated for fluency, naturalness, cs-ratio, and socio-cultural quality.
6. A weighted score is computed from sentence-level averages.
7. Accepted records are appended to Modified_Version/output/Arabic.jsonl.
8. Modified_Version/convertToExcel.py flattens JSONL into Excel for manual review.

This last export step is operationally important because the current schema is too rich to inspect comfortably by eye in raw JSONL.

## What the Current Version Improves

The current version improves on the baseline in these ways:

- better alignment to supervised NLP task generation
- explicit semantic validation
- sentence-level diagnostics
- more auditable CS-ratio calculation
- richer output for analysis and debugging
- better testing surface

## What the Current Version Makes More Complex

The current version also introduces more complexity:

- larger and sparser JSONL schema
- more fields to keep aligned across runs
- mixed old and new outputs can confuse downstream analysis
- spreadsheet conversion must handle both aggregate and per-instance fields
- operational bugs are easier to introduce because multiple schema versions may coexist in the same output file

## Key Practical Conclusion

Original_baseLine is a simpler linguistic generation pipeline.

Modified_Version is a task-aware, evaluation-heavy dataset construction pipeline with:
- richer state
- more gates
- more diagnostics
- more analyst-facing outputs

That is the core difference.

The baseline asks, “Can we generate code-switched text and score it?”

The current version asks, “Can we generate code-switched text that satisfies an NLP task specification, validate that it does so, score each sentence, and export the result for detailed review?”

## Files Used For This Comparison

- Modified_Version/core/run_french.py
- Modified_Version/core/node_engine.py
- Modified_Version/core/node_models.py
- Modified_Version/core/utils.py
- Modified_Version/core/prompt.py
- Modified_Version/config/config2.yaml
- Modified_Version/convertToExcel.py
- Original_baseLine/core/run_french.py
- Original_baseLine/core/node_engine.py
- Original_baseLine/core/node_models.py
- Original_baseLine/core/utils.py
- Original_baseLine/core/prompt.py