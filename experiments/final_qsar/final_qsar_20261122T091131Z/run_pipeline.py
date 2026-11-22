"""One-command final evaluation, postprocessing, report and tests."""
from __future__ import annotations

import json
import subprocess
import sys
import platform
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parent


def run(args, log_path=None):
    completed = subprocess.run([sys.executable, *args], cwd=RUN.parents[2], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_path is not None:
        Path(log_path).write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main():
    checkpoint = RUN / "manifests" / "folds_checkpoint.json"
    manifest_path = RUN / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    evaluation_complete = (manifest.get("status") in {"evaluation_complete_internal_only", "complete_internal_evaluation_tests_passed"}
                           and checkpoint.exists()
                           and (RUN / "pooled_oof_predictions.csv").exists())
    if not evaluation_complete:
        train_args = [str(RUN / "train_evaluate.py")]
        if checkpoint.exists():
            train_args.append("--resume")
        run(train_args, RUN / "logs" / "evaluation_stdout.log")
    else:
        print("Evaluation outputs are complete; resuming postprocessing, report, and tests only.")
    run([str(RUN / "postprocess.py")])
    run([str(RUN / "report_builder.py")])
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", str(RUN / "tests")],
                          cwd=RUN.parents[2], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    (RUN / "test_results.txt").write_text(test.stdout, encoding="utf-8")
    print(test.stdout, end="")
    if test.returncode:
        raise SystemExit(test.returncode)
    packages = ["numpy", "pandas", "scikit-learn", "rdkit", "xgboost", "scipy", "matplotlib", "joblib", "pytest"]
    version_lines = [f"Python=={platform.python_version()}"] + [f"{name}=={version(name)}" for name in packages]
    (RUN / "environment_versions.txt").write_text("\n".join(version_lines) + "\n", encoding="utf-8")
    manifest_path = RUN / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "complete_internal_evaluation_tests_passed",
                     "tests_passed": True,
                     "completed_utc": datetime.now(timezone.utc).isoformat(),
                     "entrypoint": "run_pipeline.py",
                     "unique_oof_molecules": int(__import__("pandas").read_csv(RUN / "pooled_oof_predictions.csv").row_index.nunique()),
                     "oof_prediction_records_per_unique_molecule": 2,
                     "environment_versions": "environment_versions.txt",
                     "tests": "tests/"})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    execution_log = RUN / "execution_log.txt"
    existing_log = execution_log.read_text(encoding="utf-8") if execution_log.exists() else ""
    stage_line = "Stages 4-8: evaluation, model fit, postprocessing, report, serialization, and tests completed.\n"
    limitation_line = "Run remains internal scaffold validation; no external/generated-molecule validation.\n"
    if stage_line not in existing_log:
        existing_log += stage_line
    if limitation_line not in existing_log:
        existing_log += limitation_line
    execution_log.write_text(existing_log, encoding="utf-8")


if __name__ == "__main__":
    main()
