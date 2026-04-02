# Modified Version Run Guide

This guide is the source of truth for running and reviewing the modified pipeline.

## Scope
- Folder: Modified_Version
- Main entrypoint: core/run_french.py
- Main config: config/config2.yaml

## Prerequisites
From workspace root:

```powershell
cd Modified_Version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create file `.env` in `Modified_Version` with:

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_base_url
```

## Full Pipeline Run
Important: run from `Modified_Version/core` because `run_french.py` loads `../config/config2.yaml`.

```powershell
cd Modified_Version/core
python run_french.py
```

## Where Output Is Written
- Accepted pipeline records: `Modified_Version/output/Arabic.jsonl`
- Test artifacts (if test scripts are used): `Modified_Version/output/pipeline_full_real_*.json`
- Runtime logs: `Modified_Version/core/logs/code_switching_agent_*.log`

## What To Review For Improvements
- Overall `score` trend and low-score frequency.
- `task_validation_result.passed` and validation errors.
- `cs_ratio_results_per_instances` closeness to target `cs_ratio`.
- Fluency/naturalness/socio-cultural summaries for recurring quality issues.

## Quick Review Commands
From workspace root:

```powershell
Get-ChildItem "Modified_Version/output" -File | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name,LastWriteTime
Get-Content "Modified_Version/output/Arabic.jsonl" -Tail 3
```
