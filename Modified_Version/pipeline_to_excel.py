"""
Convert pipeline_full_real_*.json output files to a readable Excel workbook.

Usage:
    # Convert all pipeline JSON files found in output/
    python pipeline_to_excel.py

    # Convert a specific file
    python pipeline_to_excel.py output/pipeline_full_real_20260401_131857.json

    # Also convert Arabic.jsonl at the same time
    python pipeline_to_excel.py --all

Output:
    output/pipeline_results_<timestamp>.xlsx
    Two sheets:
      - "Per Sentence"  : one row per generated sentence with all scores
      - "Per Scenario"  : one row per scenario (aggregate view)
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str(v):
    """Convert any value to a clean string; empty for None."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _join_list(lst):
    if not isinstance(lst, list):
        return _str(lst)
    return " | ".join(_str(x) for x in lst)


def _safe(d, *keys, default=None):
    """Nested dict safe-get."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


# ---------------------------------------------------------------------------
# Per-sentence row builder for ONE run (= one scenario)
# ---------------------------------------------------------------------------

def _rows_from_run(run: dict, meta: dict) -> list[dict]:
    state = run.get("state", {})
    checks = run.get("checks", {})

    sentences = state.get("data_generation_result") or []
    if not isinstance(sentences, list):
        sentences = [sentences]

    flu_per = state.get("fluency_results_per_instances") or []
    nat_per = state.get("naturalness_results_per_instances") or []
    cs_per  = state.get("cs_ratio_results_per_instances") or []
    soc_per = state.get("social_cultural_results_per_instances") or []

    agg_flu = state.get("fluency_result") or {}
    agg_nat = state.get("naturalness_result") or {}
    agg_soc = state.get("social_cultural_result") or {}

    tv       = state.get("task_validation_result") or {}
    tv_per   = tv.get("per_instance_results") or []

    # sentence_scores / instance_refine_counts might be present in newer runs
    sent_scores   = state.get("sentence_scores") or []
    refine_counts = state.get("instance_refine_counts") or []

    # Scenario-level metadata
    scenario_meta = {
        "run_file":         meta.get("_source_file", ""),
        "pipeline_ts":      meta.get("timestamp", ""),
        "task":             state.get("task", run.get("task", "")),
        "label":            state.get("label", ""),
        "topic":            state.get("topic", ""),
        "tense":            state.get("tense", ""),
        "perspective":      state.get("perspective", ""),
        "cs_ratio_target":  state.get("cs_ratio", ""),
        "cs_type":          state.get("cs_type", ""),
        "cs_function":      state.get("cs_function", ""),
        "gender":           state.get("gender", ""),
        "age":              state.get("age", ""),
        "education_level":  state.get("education_level", ""),
        "first_language":   state.get("first_language", ""),
        "second_language":  state.get("second_language", ""),
        "conversation_type": state.get("conversation_type", ""),
        "task_constraints": _str(state.get("task_constraints")),
        "scenario_index":   run.get("scenario_index", ""),
        # Aggregate quality
        "overall_score":        state.get("score", ""),
        "agg_fluency_score":    _safe(agg_flu, "fluency_score"),
        "agg_naturalness_score":_safe(agg_nat, "naturalness_score"),
        "agg_socio_score":      _safe(agg_soc, "socio_cultural_score"),
        "agg_fluency_summary":  _safe(agg_flu, "summary", default=""),
        "agg_naturalness_summary": _safe(agg_nat, "summary", default=""),
        "agg_socio_summary":    _safe(agg_soc, "summary", default=""),
        # Task validation (aggregate)
        "task_passed":          _safe(tv, "passed"),
        "task_confidence":      _safe(tv, "confidence"),
        "task_predicted_label": _safe(tv, "predicted_label", default=""),
        "task_notes":           _safe(tv, "notes", default=""),
        # Checks
        "checks_all_passed":    checks.get("all_passed", ""),
    }

    rows = []
    for i, text in enumerate(sentences):
        flu_i = flu_per[i] if i < len(flu_per) and isinstance(flu_per[i], dict) else {}
        nat_i = nat_per[i] if i < len(nat_per) and isinstance(nat_per[i], dict) else {}
        cs_i  = cs_per[i]  if i < len(cs_per)  and isinstance(cs_per[i], dict)  else {}
        soc_i = soc_per[i] if i < len(soc_per) and isinstance(soc_per[i], dict) else {}
        tv_i  = tv_per[i]  if i < len(tv_per)  and isinstance(tv_per[i], dict)  else {}

        # Sentence-level score (if computed by newer pipeline)
        sent_score = sent_scores[i] if i < len(sent_scores) else None
        refine_cnt = refine_counts[i] if i < len(refine_counts) else None

        # CS ratio score: normalize to 0-10 if stored as raw int
        cs_score_raw = cs_i.get("ratio_score")
        cs_score_10 = None
        if cs_score_raw is not None:
            try:
                cs_score_10 = round(float(cs_score_raw) * 10 / 4, 2) if float(cs_score_raw) <= 4 else float(cs_score_raw)
            except (TypeError, ValueError):
                cs_score_10 = cs_score_raw

        row = {
            **scenario_meta,
            "sentence_index": i,
            "sentence":       str(text),
            # Per-sentence computed overall (if available)
            "sentence_score": sent_score,
            "refine_count":   refine_cnt,
            # Fluency
            "fluency_score":   flu_i.get("fluency_score", agg_flu.get("fluency_score")),
            "fluency_summary": flu_i.get("summary", ""),
            "fluency_errors":  _str(flu_i.get("errors", {})),
            # Naturalness
            "naturalness_score":        nat_i.get("naturalness_score", agg_nat.get("naturalness_score")),
            "naturalness_summary":      nat_i.get("summary", ""),
            "naturalness_observations": _str(nat_i.get("observations", {})),
            # CS-ratio
            "cs_ratio_score":    cs_score_raw,
            "cs_ratio_score_10": cs_score_10,
            "cs_ratio_computed": cs_i.get("computed_ratio", ""),
            "cs_ratio_notes":    cs_i.get("notes", ""),
            # Socio-cultural
            "socio_score":   soc_i.get("socio_cultural_score", agg_soc.get("socio_cultural_score")),
            "socio_summary": soc_i.get("summary", ""),
            "socio_issues":  _str(soc_i.get("issues", "")),
            # Per-sentence task validation
            "sent_task_passed":          tv_i.get("passed", tv.get("passed")),
            "sent_task_confidence":      tv_i.get("confidence", tv.get("confidence")),
            "sent_task_predicted_label": tv_i.get("predicted_label", tv.get("predicted_label", "")),
            "sent_task_notes":           tv_i.get("notes", ""),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Parse one pipeline JSON file
# ---------------------------------------------------------------------------

def parse_pipeline_json(filepath: str) -> tuple[list[dict], list[dict]]:
    """Returns (sentence_rows, scenario_rows)."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    meta["_source_file"] = Path(filepath).name

    sent_rows = []
    scen_rows = []

    for run in data.get("runs", []):
        rows = _rows_from_run(run, meta)
        sent_rows.extend(rows)

        state = run.get("state", {})
        checks = run.get("checks", {})
        scen_rows.append({
            "run_file":         meta["_source_file"],
            "pipeline_ts":      meta.get("timestamp", ""),
            "task":             state.get("task", run.get("task", "")),
            "label":            state.get("label", ""),
            "topic":            state.get("topic", ""),
            "tense":            state.get("tense", ""),
            "perspective":      state.get("perspective", ""),
            "cs_ratio_target":  state.get("cs_ratio", ""),
            "gender":           state.get("gender", ""),
            "age":              state.get("age", ""),
            "education_level":  state.get("education_level", ""),
            "first_language":   state.get("first_language", ""),
            "second_language":  state.get("second_language", ""),
            "overall_score":    state.get("score", ""),
            "num_sentences":    len(state.get("data_generation_result") or []),
            "task_passed":      _safe(state.get("task_validation_result") or {}, "passed"),
            "checks_all_passed":checks.get("all_passed", ""),
            "agg_fluency":      _safe(state.get("fluency_result") or {}, "fluency_score"),
            "agg_naturalness":  _safe(state.get("naturalness_result") or {}, "naturalness_score"),
            "agg_socio":        _safe(state.get("social_cultural_result") or {}, "socio_cultural_score"),
            "sentences":        _join_list(state.get("data_generation_result") or []),
        })

    return sent_rows, scen_rows


# ---------------------------------------------------------------------------
# Parse Arabic.jsonl (existing format)
# ---------------------------------------------------------------------------

def parse_jsonl(filepath: str) -> list[dict]:
    """Parse the flat-per-scenario JSONL format into sentence rows."""
    rows = []
    source = Path(filepath).name
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            agg_flu = obj.get("fluency_result") or {}
            agg_nat = obj.get("naturalness_result") or {}
            agg_soc = obj.get("social_cultural_result") or {}
            tv = obj.get("task_validation_result") or {}

            flu_per = obj.get("fluency_results_per_instances") or []
            nat_per = obj.get("naturalness_results_per_instances") or []
            cs_per  = obj.get("cs_ratio_results_per_instances") or []
            soc_per = obj.get("social_cultural_results_per_instances") or []
            tv_per  = (tv.get("per_instance_results") or [])

            sentences = obj.get("data_generation_result") or []
            if not isinstance(sentences, list):
                sentences = [sentences]

            meta_cols = {
                "run_file":         source,
                "task":             obj.get("task", ""),
                "label":            obj.get("label", ""),
                "topic":            obj.get("topic", ""),
                "tense":            obj.get("tense", ""),
                "perspective":      obj.get("perspective", ""),
                "cs_ratio_target":  obj.get("cs_ratio", ""),
                "cs_type":          obj.get("cs_type", ""),
                "cs_function":      obj.get("cs_function", ""),
                "gender":           obj.get("gender", ""),
                "age":              obj.get("age", ""),
                "education_level":  obj.get("education_level", ""),
                "first_language":   obj.get("first_language", ""),
                "second_language":  obj.get("second_language", ""),
                "conversation_type": obj.get("conversation_type", ""),
                "task_constraints": _str(obj.get("task_constraints")),
                "overall_score":    obj.get("score", ""),
                "agg_fluency_score":    agg_flu.get("fluency_score"),
                "agg_naturalness_score":agg_nat.get("naturalness_score"),
                "agg_socio_score":      agg_soc.get("socio_cultural_score"),
                "task_passed":      tv.get("passed"),
                "task_confidence":  tv.get("confidence"),
                "task_predicted_label": tv.get("predicted_label", ""),
            }

            for i, text in enumerate(sentences):
                flu_i = flu_per[i] if i < len(flu_per) and isinstance(flu_per[i], dict) else {}
                nat_i = nat_per[i] if i < len(nat_per) and isinstance(nat_per[i], dict) else {}
                cs_i  = cs_per[i]  if i < len(cs_per)  and isinstance(cs_per[i], dict)  else {}
                soc_i = soc_per[i] if i < len(soc_per) and isinstance(soc_per[i], dict) else {}
                tv_i  = tv_per[i]  if i < len(tv_per)  and isinstance(tv_per[i], dict)  else {}

                cs_score_raw = cs_i.get("ratio_score")
                cs_score_10 = None
                if cs_score_raw is not None:
                    try:
                        cs_score_10 = round(float(cs_score_raw) * 10 / 4, 2) if float(cs_score_raw) <= 4 else float(cs_score_raw)
                    except (TypeError, ValueError):
                        cs_score_10 = cs_score_raw

                rows.append({
                    **meta_cols,
                    "sentence_index": i,
                    "sentence":       str(text),
                    "fluency_score":   flu_i.get("fluency_score", agg_flu.get("fluency_score")),
                    "fluency_summary": flu_i.get("summary", ""),
                    "fluency_errors":  _str(flu_i.get("errors", {})),
                    "naturalness_score":        nat_i.get("naturalness_score", agg_nat.get("naturalness_score")),
                    "naturalness_summary":      nat_i.get("summary", ""),
                    "naturalness_observations": _str(nat_i.get("observations", {})),
                    "cs_ratio_score":    cs_score_raw,
                    "cs_ratio_score_10": cs_score_10,
                    "cs_ratio_computed": cs_i.get("computed_ratio", ""),
                    "cs_ratio_notes":    cs_i.get("notes", ""),
                    "socio_score":   soc_i.get("socio_cultural_score", agg_soc.get("socio_cultural_score")),
                    "socio_summary": soc_i.get("summary", ""),
                    "socio_issues":  _str(soc_i.get("issues", "")),
                    "sent_task_passed":          tv_i.get("passed", tv.get("passed")),
                    "sent_task_confidence":      tv_i.get("confidence", tv.get("confidence")),
                    "sent_task_predicted_label": tv_i.get("predicted_label", tv.get("predicted_label", "")),
                    "sent_task_notes":           tv_i.get("notes", ""),
                })
    return rows


# ---------------------------------------------------------------------------
# Excel writer with column formatting
# ---------------------------------------------------------------------------

_SCORE_COLS = [
    "overall_score", "sentence_score",
    "fluency_score", "naturalness_score", "cs_ratio_score_10", "socio_score",
    "agg_fluency_score", "agg_naturalness_score", "agg_socio_score",
    "agg_fluency", "agg_naturalness", "agg_socio",
    "task_confidence", "sent_task_confidence",
]

_BOOL_COLS = [
    "task_passed", "checks_all_passed",
    "sent_task_passed", "sentence_validation_passed",
]

_WIDE_COLS = {
    "sentence": 60,
    "sentences": 80,
    "fluency_summary": 40,
    "naturalness_summary": 40,
    "socio_summary": 40,
    "agg_fluency_summary": 40,
    "agg_naturalness_summary": 40,
    "agg_socio_summary": 40,
    "fluency_errors": 35,
    "naturalness_observations": 40,
    "socio_issues": 35,
    "sent_task_notes": 40,
    "task_notes": 40,
    "cs_ratio_notes": 30,
}

_DEFAULT_COL_WIDTH = 18


def _write_sheet(ws, df: pd.DataFrame, workbook):
    """Write a DataFrame to an xlsxwriter worksheet with formatting."""
    header_fmt = workbook.add_format({
        "bold": True, "bg_color": "#2E4057", "font_color": "white",
        "border": 1, "text_wrap": True, "valign": "vcenter", "align": "center",
    })
    score_fmt = workbook.add_format({
        "num_format": "0.00", "align": "center", "bg_color": "#EAF4FB",
    })
    bool_true_fmt = workbook.add_format({
        "align": "center", "bg_color": "#D5F5E3", "font_color": "#1D8348",
    })
    bool_false_fmt = workbook.add_format({
        "align": "center", "bg_color": "#FDEDEC", "font_color": "#922B21",
    })
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    default_fmt = workbook.add_format({"valign": "top"})

    # Header row
    for col_i, col_name in enumerate(df.columns):
        ws.write(0, col_i, col_name, header_fmt)

    # Set column widths
    for col_i, col_name in enumerate(df.columns):
        width = _WIDE_COLS.get(col_name, _DEFAULT_COL_WIDTH)
        ws.set_column(col_i, col_i, width)

    # Data rows
    for row_i, row in enumerate(df.itertuples(index=False), start=1):
        for col_i, col_name in enumerate(df.columns):
            val = getattr(row, col_name.replace(" ", "_"), None)

            # Pandas NA → blank
            try:
                import math
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    ws.write_blank(row_i, col_i, None, default_fmt)
                    continue
            except (TypeError, ValueError):
                pass

            if col_name in _SCORE_COLS and val != "":
                try:
                    ws.write_number(row_i, col_i, float(val), score_fmt)
                except (TypeError, ValueError):
                    ws.write(row_i, col_i, str(val) if val is not None else "", default_fmt)
            elif col_name in _BOOL_COLS:
                fmt = bool_true_fmt if val is True else (bool_false_fmt if val is False else default_fmt)
                ws.write(row_i, col_i, str(val) if val is not None else "", fmt)
            elif col_name in _WIDE_COLS:
                ws.write(row_i, col_i, str(val) if val is not None else "", wrap_fmt)
            else:
                ws.write(row_i, col_i, str(val) if val is not None else "", default_fmt)

    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)


def save_excel(sent_rows: list[dict], scen_rows: list[dict], out_path: str):
    sent_df = pd.DataFrame(sent_rows)
    scen_df = pd.DataFrame(scen_rows) if scen_rows else None

    # Preferred column order for sentence sheet
    sent_pref = [
        "run_file", "task", "label", "topic", "sentence_index", "sentence",
        "overall_score", "sentence_score", "refine_count",
        "fluency_score", "naturalness_score", "cs_ratio_score_10", "socio_score",
        "cs_ratio_score", "cs_ratio_computed", "cs_ratio_notes",
        "fluency_summary", "fluency_errors",
        "naturalness_summary", "naturalness_observations",
        "socio_summary", "socio_issues",
        "sent_task_passed", "sent_task_confidence", "sent_task_predicted_label", "sent_task_notes",
        "task_passed", "task_confidence", "task_predicted_label", "task_notes",
        "tense", "perspective", "cs_ratio_target", "cs_type", "cs_function",
        "gender", "age", "education_level", "first_language", "second_language",
        "conversation_type", "task_constraints",
        "agg_fluency_score", "agg_naturalness_score", "agg_socio_score",
        "agg_fluency_summary", "agg_naturalness_summary", "agg_socio_summary",
        "pipeline_ts", "scenario_index", "checks_all_passed",
    ]
    ordered = [c for c in sent_pref if c in sent_df.columns]
    rest = [c for c in sent_df.columns if c not in ordered]
    sent_df = sent_df[ordered + rest]

    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    wb = writer.book

    # Sheet 1 — Per Sentence
    sent_df.to_excel(writer, sheet_name="Per Sentence", index=False)
    _write_sheet(writer.sheets["Per Sentence"], sent_df, wb)

    # Sheet 2 — Per Scenario (only for pipeline JSON runs)
    if scen_df is not None and not scen_df.empty:
        scen_pref = [
            "run_file", "task", "label", "topic", "overall_score", "num_sentences",
            "task_passed", "checks_all_passed",
            "agg_fluency", "agg_naturalness", "agg_socio",
            "tense", "perspective", "cs_ratio_target", "gender", "age",
            "education_level", "first_language", "second_language",
            "pipeline_ts", "sentences",
        ]
        scen_ord = [c for c in scen_pref if c in scen_df.columns]
        scen_rest = [c for c in scen_df.columns if c not in scen_ord]
        scen_df = scen_df[scen_ord + scen_rest]
        scen_df.to_excel(writer, sheet_name="Per Scenario", index=False)
        _write_sheet(writer.sheets["Per Scenario"], scen_df, wb)

    writer.close()
    print(f"Saved {len(sent_df)} sentence rows to: {out_path}")
    if scen_df is not None:
        print(f"  + {len(scen_df)} scenario rows in 'Per Scenario' sheet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert pipeline output to Excel.")
    parser.add_argument(
        "inputs", nargs="*",
        help="JSON/JSONL file(s) to convert. Default: all pipeline_full_real_*.json in output/",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Also include Arabic.jsonl in addition to pipeline JSON files.",
    )
    parser.add_argument(
        "--out", default="",
        help="Output Excel path. Default: output/pipeline_results_<timestamp>.xlsx",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"

    # Collect input files
    files: list[Path] = []
    if args.inputs:
        for p in args.inputs:
            files.extend(Path(f) for f in glob.glob(p))
    else:
        files = sorted(output_dir.glob("pipeline_full_real_*.json"))
        if args.all:
            jsonl = output_dir / "Arabic.jsonl"
            if jsonl.exists():
                files.append(jsonl)

    if not files:
        print("No input files found. Run from Modified_Version/ or pass file paths as arguments.")
        sys.exit(1)

    print(f"Processing {len(files)} file(s)...")

    all_sent: list[dict] = []
    all_scen: list[dict] = []

    for fp in files:
        fp = Path(fp)
        print(f"  {fp.name}")
        if fp.suffix == ".jsonl":
            rows = parse_jsonl(str(fp))
            all_sent.extend(rows)
        elif fp.suffix == ".json":
            s_rows, sc_rows = parse_pipeline_json(str(fp))
            all_sent.extend(s_rows)
            all_scen.extend(sc_rows)
        else:
            print(f"  [SKIP] Unknown format: {fp.name}")

    if not all_sent:
        print("No sentence data extracted.")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out if args.out else str(output_dir / f"pipeline_results_{ts}.xlsx")
    save_excel(all_sent, all_scen, out_path)


if __name__ == "__main__":
    main()
