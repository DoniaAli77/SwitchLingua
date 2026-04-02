# SwitchLingua Modified Version - Living Progress Document

**Last Updated:** 2026-04-01 14:00  
**Status:** 🟢 Active Testing Phase

---

## 📋 Project Overview

SwitchLingua is an AI-powered Arabic-English code-switching pipeline that generates, validates, and evaluates realistic code-mixed text scenarios with task-specific constraints (sentiment analysis, named entity recognition, topic classification).

**This Document:** Tracks progress, findings, and improvements across the modified implementation.

---

## 🎯 Current Status

### Pipeline Execution
- ✅ **Last Run:** 2026-04-01 13:59:16
- ✅ **Scenarios Processed:** 18/18 (100%)
- ✅ **Output Generated:** `Modified_Version/output/Arabic.jsonl` (293KB)
- ✅ **System Status:** Fully functional

### Key Metrics Summary
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Average Score | 7.27 | > 8.0 | 🟡 Below target |
| CS Ratio Accuracy | ~50% avg | 70% | 🟡 Below target |
| Task Validation Pass Rate | ~40% | > 80% | 🔴 Below target |
| Per-Instance Evaluation | ✓ Working | ✓ Implemented | ✅ Green |
| Deterministic CS Counting | ✓ Working | ✓ Implemented | ✅ Green |

---

## 📊 Recent Findings

### From Latest Pipeline Run (2026-04-01)

**Sample Output Record 1 (Sentiment/Negative):**
- Sentences Generated: 5
- Fluency Scores: [8.0, 9.0, 7.0, 9.0, 8.0]
- Average Fluency: 8.2
- Task Validation: ❌ FAILED (confidence: 0.83)
  - Error: Generated intensity higher than target
- CS Ratios Observed: [15%, 62.5%, 7.69%, 57.14%, 50%]
- Final Score: 7.15
- Refinement Iterations: 3

**Sample Output Record 2 (Sentiment/Positive):**
- Sentences Generated: 3
- Fluency Scores: [8.0, 9.0, 8.0]
- Average Fluency: 8.67
- Task Validation: ✅ PASSED (confidence: 0.867)
- CS Ratios Observed: [15%, 0%, 22%]
- Final Score: 7.33
- Refinement Iterations: 0

### Key Observations
1. **Per-Instance Results Working:** Fluency, naturalness, and CS ratio now properly tracked per sentence
2. **Task Validation Inconsistent:** ~40-50% pass rate; intensity and ambiguity constraints often violated
3. **CS Ratio Variance:** Generated code-switching ratios vary significantly between sentences (0% to 62.5%)
4. **Score Range:** Most records scoring 7.0-7.8 range; need improvement to reach 8.0+ target
5. **Refinement Loop:** Some scenarios require 2-3 refinement iterations; others accept on first try

---

## 🏗️ Architecture Summary

### Core Components

**Data Generation** (`node_engine.py:RunDataGenerationAgent`)
- Task-aware prompting with specific constraints
- Three parallel task types: sentiment, NER, topic
- Generates 5 code-switched sentences per scenario

**Task Validation** (`node_engine.py:RunTaskValidatorAgent`)
- Validates generated text meets task constraints
- Per-instance validation results
- Returns confidence scores and error details

**Evaluation Agents** (`node_engine.py`)
- Fluency evaluation (per-instance)
- Naturalness evaluation (per-instance)
- CS Ratio calculation (deterministic Unicode-based + scoring)
- Socio-cultural appropriateness (per-instance)

**Scoring** (`utils.py:weighting_scheme`)
- Formula: `fluency×0.30 + naturalness×0.25 + cs_ratio×0.20 + socio_cultural×0.25`
- Aggregates per-instance results via averaging
- Final score range: 0-10

**Config System** (`config/config2.yaml`)
- 18 scenarios generated from: 2 topics × 3 tasks × varying constraints
- Sentiment: 12 scenarios
- NER: 3 scenarios
- Topic: 3 scenarios

---

## 📁 Key Files

### Execution
- core/run_french.py — Main entry point
- config/config2.yaml — Scenario configuration
- .env — API credentials (create locally)

### Implementation
- core/node_engine.py — All pipeline nodes (939 lines)
- core/node_models.py — State/response schemas (181 lines)
- core/prompt.py — Task-aware prompts (611 lines)
- core/utils.py — Config, scenarios, CS ratio, weighting

### Output & Logs
- output/Arabic.jsonl — Primary output records
- core/logs/ — Execution logs

### Documentation
- README_MODIFIED.md — How-to run guide
- mdFiles/MODIFIED_VS_ORIGINAL_IMPLEMENTATION_COMPARISON.md — Baseline comparison

---

## 🧪 Testing & Validation

### Current Testing Approach
1. **Scope:** Compare Modified_Version against Original_baseLine for regressions/improvements
2. **Output Review:** Sample 10-20 records from Arabic.jsonl to assess quality
3. **Metrics Tracked:** Score distribution, validation pass rates, CS ratio adherence, error patterns

### Running Full Pipeline
```powershell
cd Modified_Version/core
python run_french.py
```

### Quick Output Inspection
```powershell
cd Modified_Version/output
Get-Content "Arabic.jsonl" -Tail 5  # Last 5 records
```

---

## 🔄 Changelog / Update Log

| Date | Update | Impact | Status |
|------|--------|--------|--------|
| 2026-04-01 13:59 | Full pipeline execution (18 scenarios) | Fresh outputs generated; validation metrics captured | ✅ Complete |
| 2026-04-01 14:00 | Created living README with metrics tracking | Established progress tracking system | ✅ Complete |
|  |  |  |  |

---

## ⚠️ Known Issues & TODOs

### High Priority
- [ ] **Task Validation Pass Rate Too Low:** Only ~40-50% of scenarios passing validation
- [ ] **CS Ratio Target Miss:** Average ~50% observed; target is 70%
- [ ] **Score Below Target:** Average 7.27 vs target 8.0+

### Medium Priority
- [ ] **Refinement Loop Effectiveness:** Limited improvement gains from refinement
- [ ] **Per-Instance Variance:** High variance in CS ratio between sentences
- [ ] **Naturalness Evaluation:** Needs verification for code-mixed Arabic-English accuracy

### Low Priority
- [ ] **Logging Volume:** Current logs may be verbose
- [ ] **Documentation Sync:** Multiple README files need consolidation

---

## 📈 Next Steps

### Immediate (This Session)
1. Review sample outputs from Arabic.jsonl
2. Run comparison tests using senior-pipeline-tester agent
3. Identify top blockers to prioritize

### Short-term (Next Session)
1. Adjust data generation prompts to enforce constraints better
2. Test different English percentage targets
3. Review scoring weight distribution

### Medium-term
1. Quantify improvements vs Original_baseLine
2. Analyze performance by task type (sentiment/NER/topic)
3. Categorize validation failures for pattern analysis

---

## 🛠️ Troubleshooting

### Pipeline Won't Start
1. Check `.env` file exists in Modified_Version with OPENAI_API_KEY
2. Verify venv activated
3. Check dependencies: `pip install -r Modified_Version/requirements.txt`

### Output File Not Updating
1. Check logs in Modified_Version/core/logs/ for errors
2. Confirm working directory is Modified_Version/core/
3. Verify config2.yaml is valid YAML

### Scores All Below 5.0
1. Check evaluation agent prompts in prompt.py
2. Verify scoring weights in utils.py
3. Consider if LLM is too strict on constraints

---

## 📝 Notes

- This document is **updated continuously** as work progresses
- Update changelog with each session's findings
- Use metrics table to track progress over time
- Archive old findings in mdFiles/ if needed

---

**Last Verified:** 2026-04-01 13:59:16 (Pipeline execution successful)  
**Next Review Due:** After next pipeline execution
