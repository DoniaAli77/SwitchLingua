import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def extract_numeric_list(values):
    out = []
    for item in values or []:
        if isinstance(item, (int, float)):
            out.append(float(item))
        elif isinstance(item, dict):
            for key in [
                "score",
                "fluency_score",
                "naturalness_score",
                "social_cultural_score",
                "socio_cultural_score",
                "ratio_score",
            ]:
                if key in item and isinstance(item[key], (int, float)):
                    out.append(float(item[key]))
                    break
    return out


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def grouped_to_rows(grouped):
    rows = []
    for key in sorted(grouped.keys()):
        item = grouped[key]
        rows.append(
            {
                "name": key,
                "count": item["count"],
                "passed": item["passed"],
                "pass_rate": (100.0 * item["passed"] / item["count"]) if item["count"] else 0.0,
                "avg_score": safe_mean(item["scores"]),
            }
        )
    return rows


class PipelineOutputReviewer:
    def __init__(self, core_dir: Path, output_file: str = "Arabic.jsonl"):
        self.core_dir = core_dir.resolve()
        self.repo_root = self.core_dir.parent.parent
        self.modified_output_path = (self.core_dir.parent / "output" / output_file).resolve()
        self.default_export_dir = (self.core_dir.parent / "output" / "reviewer_reports").resolve()
        self.default_baseline_path = (self.repo_root / "Original_baseLine" / "output" / "sample.json").resolve()

    def run_pipeline(self, timeout_seconds: int = 7200) -> bool:
        print("=" * 80)
        print("RUNNING FULL PIPELINE")
        print("=" * 80)
        result = subprocess.run(
            [sys.executable, "run_french.py"],
            cwd=str(self.core_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        if result.returncode != 0:
            print("Pipeline failed.")
            print("Last stdout lines:")
            print("\n".join(result.stdout.splitlines()[-20:]))
            print("Last stderr lines:")
            print("\n".join(result.stderr.splitlines()[-20:]))
            return False
        print("Pipeline completed successfully.")
        return True

    @staticmethod
    def _load_jsonl_lines(path: Path):
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
        return records

    @staticmethod
    def _load_json_document(path: Path):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Unsupported JSON payload type in {path}: {type(payload).__name__}")

    def load_records_from_path(self, path: Path):
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Output file not found: {path}")

        try:
            records = self._load_jsonl_lines(path)
            if records:
                return records
        except ValueError:
            pass

        return self._load_json_document(path)

    def analyze(self, records):
        scores = []
        validation_passed = 0
        validation_total = 0

        fluency_scores = []
        naturalness_scores = []
        socio_scores = []
        cs_ratio_scores = []
        refine_counts = []

        by_task = defaultdict(lambda: {"count": 0, "passed": 0, "scores": []})
        by_label = defaultdict(lambda: {"count": 0, "passed": 0, "scores": []})

        for rec in records:
            score = rec.get("score")
            if isinstance(score, (int, float)):
                scores.append(float(score))

            validation = rec.get("task_validation_result", {})
            has_validation = isinstance(validation, dict) and "passed" in validation
            passed = bool(validation.get("passed")) if has_validation else False
            if has_validation:
                validation_total += 1
                if passed:
                    validation_passed += 1

            task = str(rec.get("task", "unknown"))
            label = str(rec.get("label", "unknown"))

            by_task[task]["count"] += 1
            by_label[label]["count"] += 1
            if passed:
                by_task[task]["passed"] += 1
                by_label[label]["passed"] += 1
            if isinstance(score, (int, float)):
                by_task[task]["scores"].append(float(score))
                by_label[label]["scores"].append(float(score))

            record_fluency = extract_numeric_list(rec.get("fluency_results_per_instances", []))
            record_naturalness = extract_numeric_list(rec.get("naturalness_results_per_instances", []))
            record_socio = extract_numeric_list(rec.get("social_cultural_results_per_instances", []))
            record_cs_ratio = extract_numeric_list(rec.get("cs_ratio_results_per_instances", []))

            if not record_fluency and isinstance(rec.get("fluency_result"), dict):
                record_fluency = extract_numeric_list([rec.get("fluency_result")])
            if not record_naturalness and isinstance(rec.get("naturalness_result"), dict):
                record_naturalness = extract_numeric_list([rec.get("naturalness_result")])
            if not record_socio and isinstance(rec.get("social_cultural_result"), dict):
                record_socio = extract_numeric_list([rec.get("social_cultural_result")])
            if not record_cs_ratio and isinstance(rec.get("cs_ratio_result"), dict):
                record_cs_ratio = extract_numeric_list([rec.get("cs_ratio_result")])

            fluency_scores.extend(record_fluency)
            naturalness_scores.extend(record_naturalness)
            socio_scores.extend(record_socio)
            cs_ratio_scores.extend(record_cs_ratio)

            refine = rec.get("refine_count")
            if isinstance(refine, (int, float)):
                refine_counts.append(float(refine))

        summary = {
            "total_records": len(records),
            "validation_pass_rate": (100.0 * validation_passed / validation_total) if validation_total else 0.0,
            "avg_score": safe_mean(scores),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "stdev_score": safe_stdev(scores),
            "avg_fluency": safe_mean(fluency_scores),
            "avg_naturalness": safe_mean(naturalness_scores),
            "avg_socio_cultural": safe_mean(socio_scores),
            "avg_cs_ratio_score": safe_mean(cs_ratio_scores),
            "avg_refine_count": safe_mean(refine_counts),
            "validation_passed": validation_passed,
            "validation_total": validation_total,
        }

        return {
            "summary": summary,
            "by_task": by_task,
            "by_label": by_label,
            "grouped_task_rows": grouped_to_rows(by_task),
            "grouped_label_rows": grouped_to_rows(by_label),
        }

    @staticmethod
    def compare_analyses(modified_analysis, baseline_analysis):
        keys = [
            "total_records",
            "validation_pass_rate",
            "avg_score",
            "min_score",
            "max_score",
            "stdev_score",
            "avg_fluency",
            "avg_naturalness",
            "avg_socio_cultural",
            "avg_cs_ratio_score",
            "avg_refine_count",
        ]
        rows = []
        for key in keys:
            modified_value = modified_analysis["summary"].get(key, 0.0)
            baseline_value = baseline_analysis["summary"].get(key, 0.0)
            if isinstance(modified_value, (int, float)) and isinstance(baseline_value, (int, float)):
                delta = float(modified_value) - float(baseline_value)
            else:
                delta = 0.0
            rows.append(
                {
                    "metric": key,
                    "modified": modified_value,
                    "baseline": baseline_value,
                    "delta": delta,
                }
            )
        return rows

    @staticmethod
    def print_summary(title, summary):
        print("\n" + title)
        print("-" * 80)
        print(f"total_records={summary['total_records']}")
        print(f"validation_pass_rate={summary['validation_pass_rate']:.1f}% ({summary['validation_passed']}/{summary['validation_total']})")
        print(f"avg_score={summary['avg_score']:.2f}, min={summary['min_score']:.2f}, max={summary['max_score']:.2f}, stdev={summary['stdev_score']:.2f}")
        print(f"avg_fluency={summary['avg_fluency']:.2f}")
        print(f"avg_naturalness={summary['avg_naturalness']:.2f}")
        print(f"avg_socio_cultural={summary['avg_socio_cultural']:.2f}")
        print(f"avg_cs_ratio_score={summary['avg_cs_ratio_score']:.2f}")
        print(f"avg_refine_count={summary['avg_refine_count']:.2f}")

    @staticmethod
    def print_breakdown(title, rows):
        print("\n" + title)
        print("-" * 80)
        for row in rows:
            print(f"{row['name']}: count={row['count']}, pass_rate={row['pass_rate']:.1f}%, avg_score={row['avg_score']:.2f}")

    @staticmethod
    def print_samples(records, count=3):
        print("\n" + "=" * 80)
        print(f"SAMPLE OUTPUT RECORDS (first {count})")
        print("=" * 80)
        for idx, rec in enumerate(records[:count], start=1):
            print("\n" + "-" * 80)
            print(f"Record {idx}")
            print(f"task={rec.get('task')} label={rec.get('label')} score={rec.get('score')} refine_count={rec.get('refine_count')}")
            validation = rec.get("task_validation_result", {})
            print(f"validation_passed={validation.get('passed')} confidence={validation.get('confidence')}")
            generated = rec.get("data_generation_result", [])
            print(f"generated_sentences={len(generated)}")
            for i, sent in enumerate(generated[:3], start=1):
                print(f"  [{i}] {sent}")

    @staticmethod
    def print_comparison(rows):
        print("\nCOMPARISON VS BASELINE")
        print("-" * 80)
        for row in rows:
            modified = row["modified"]
            baseline = row["baseline"]
            delta = row["delta"]
            if isinstance(modified, float):
                print(f"{row['metric']}: modified={modified:.2f}, baseline={baseline:.2f}, delta={delta:+.2f}")
            else:
                print(f"{row['metric']}: modified={modified}, baseline={baseline}, delta={delta:+.2f}")

    @staticmethod
    def export_csv(path: Path, rows, fieldnames):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def export_analysis(self, export_dir: Path, analysis, comparison_rows=None, baseline_path=None):
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_json_path = export_dir / f"review_summary_{timestamp}.json"
        summary_csv_path = export_dir / f"review_summary_{timestamp}.csv"
        task_csv_path = export_dir / f"review_by_task_{timestamp}.csv"
        label_csv_path = export_dir / f"review_by_label_{timestamp}.csv"

        summary_payload = {
            "generated_at": datetime.now().isoformat(),
            "modified_output_path": str(self.modified_output_path),
            "baseline_output_path": str(baseline_path) if baseline_path else None,
            "summary": analysis["summary"],
            "by_task": analysis["grouped_task_rows"],
            "by_label": analysis["grouped_label_rows"],
            "comparison": comparison_rows or [],
        }

        with summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        self.export_csv(summary_csv_path, [analysis["summary"]], list(analysis["summary"].keys()))
        self.export_csv(task_csv_path, analysis["grouped_task_rows"], ["name", "count", "passed", "pass_rate", "avg_score"])
        self.export_csv(label_csv_path, analysis["grouped_label_rows"], ["name", "count", "passed", "pass_rate", "avg_score"])

        compare_csv_path = None
        if comparison_rows:
            compare_csv_path = export_dir / f"review_compare_{timestamp}.csv"
            self.export_csv(compare_csv_path, comparison_rows, ["metric", "modified", "baseline", "delta"])

        print("\nEXPORTED FILES")
        print("-" * 80)
        print(summary_json_path)
        print(summary_csv_path)
        print(task_csv_path)
        print(label_csv_path)
        if compare_csv_path:
            print(compare_csv_path)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline, export review summaries, and compare against baseline output.")
    parser.add_argument("--analyze-only", action="store_true", help="Skip pipeline run and only analyze existing output.")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample records to print.")
    parser.add_argument("--export", action="store_true", help="Export summary files to CSV and JSON.")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory for exported review files.")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare modified output against baseline output.")
    parser.add_argument("--baseline-path", type=str, default=None, help="Baseline output path (.json or .jsonl).")
    parser.add_argument("--output-path", type=str, default=None, help="Modified output path override (.json or .jsonl).")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    core_dir = script_path.parent.parent
    reviewer = PipelineOutputReviewer(core_dir=core_dir)

    modified_output_path = Path(args.output_path).resolve() if args.output_path else reviewer.modified_output_path
    baseline_path = Path(args.baseline_path).resolve() if args.baseline_path else reviewer.default_baseline_path
    export_dir = Path(args.export_dir).resolve() if args.export_dir else reviewer.default_export_dir

    print("=" * 80)
    print("SWITCHLINGUA PIPELINE OUTPUT REVIEWER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if not args.analyze_only:
        ok = reviewer.run_pipeline()
        if not ok:
            sys.exit(1)

    modified_records = reviewer.load_records_from_path(modified_output_path)
    modified_analysis = reviewer.analyze(modified_records)

    reviewer.print_summary("SUMMARY", modified_analysis["summary"])
    reviewer.print_breakdown("BY TASK", modified_analysis["grouped_task_rows"])
    reviewer.print_breakdown("BY LABEL", modified_analysis["grouped_label_rows"])
    reviewer.print_samples(modified_records, count=max(1, args.samples))

    comparison_rows = None
    if args.compare_baseline:
        baseline_records = reviewer.load_records_from_path(baseline_path)
        baseline_analysis = reviewer.analyze(baseline_records)
        reviewer.print_summary("BASELINE SUMMARY", baseline_analysis["summary"])
        comparison_rows = reviewer.compare_analyses(modified_analysis, baseline_analysis)
        reviewer.print_comparison(comparison_rows)

    print("\nVERDICT")
    print("-" * 80)
    print("score_target(>=8.0): " + ("PASS" if modified_analysis["summary"]["avg_score"] >= 8.0 else "FAIL"))
    print("validation_target(>=80%): " + ("PASS" if modified_analysis["summary"]["validation_pass_rate"] >= 80.0 else "FAIL"))
    print("cs_ratio_target(>=7.0): " + ("PASS" if modified_analysis["summary"]["avg_cs_ratio_score"] >= 7.0 else "FAIL"))

    if args.export:
        reviewer.export_analysis(export_dir, modified_analysis, comparison_rows=comparison_rows, baseline_path=baseline_path if args.compare_baseline else None)

    print("\nDone.")


if __name__ == "__main__":
    main()

