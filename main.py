from __future__ import annotations

from pathlib import Path

from analysis_report import build_analysis_report
from data_pipeline import load_dataset
from train_ai_model import train_and_evaluate


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset = load_dataset(base_dir)
    metrics = train_and_evaluate(dataset, base_dir)
    report_path = build_analysis_report(dataset, metrics, base_dir)

    print("Project completed successfully.")
    print(f"Dataset rows: {len(dataset)}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
