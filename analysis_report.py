from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_figures(df: pd.DataFrame, base_dir: Path) -> None:
    figures_dir = base_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["revenue"], bins=30, kde=True, color="teal")
    plt.title("Revenue Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "revenue_distribution.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="marketing_spend",
        y="revenue",
        hue="product_category",
        alpha=0.75,
    )
    plt.title("Marketing Spend vs Revenue")
    plt.tight_layout()
    plt.savefig(figures_dir / "marketing_vs_revenue.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    churn_by_region = (
        df.groupby("region", as_index=False)["churned"].mean().sort_values(by="churned", ascending=False)
    )
    sns.barplot(
        data=churn_by_region,
        x="region",
        y="churned",
        hue="region",
        palette="mako",
        legend=False,
    )
    plt.title("Average Churn Rate by Region")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_by_region.png", dpi=140)
    plt.close()


def build_analysis_report(df: pd.DataFrame, metrics: Dict[str, float], base_dir: Path) -> Path:
    _save_figures(df, base_dir)

    report_path = base_dir / "reports" / "analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    kpis = {
        "total_rows": int(len(df)),
        "avg_revenue": float(df["revenue"].mean()),
        "avg_marketing_spend": float(df["marketing_spend"].mean()),
        "churn_rate": float(df["churned"].mean()),
        "top_region_by_revenue": str(
            df.groupby("region")["revenue"].mean().sort_values(ascending=False).index[0]
        ),
    }

    markdown = f"""# Automated AI + Data Analysis Report

## Key Business KPIs

- Total records: **{kpis["total_rows"]}**
- Average revenue: **{kpis["avg_revenue"]:.2f}**
- Average marketing spend: **{kpis["avg_marketing_spend"]:.2f}**
- Churn rate: **{kpis["churn_rate"]:.2%}**
- Top region by average revenue: **{kpis["top_region_by_revenue"]}**

## Model Performance

- Revenue model R2: **{metrics["regression_r2"]:.3f}**
- Revenue model MAE: **{metrics["regression_mae"]:.2f}**
- Churn model accuracy: **{metrics["classification_accuracy"]:.3f}**
- Customer segment silhouette score: **{metrics["clustering_silhouette"]:.3f}**
- Avg predicted revenue (test split): **{metrics["avg_predicted_revenue"]:.2f}**

## Insights

1. **Marketing investment correlates with revenue**, but product category and region produce clear performance differences.
2. **Churn can be predicted with useful accuracy** using operational, behavioral, and text feedback features.
3. **Customer segmentation reveals distinct behavioral groups** useful for targeted campaigns and retention strategy.

## Generated Visuals

- `reports/figures/revenue_distribution.png`
- `reports/figures/marketing_vs_revenue.png`
- `reports/figures/churn_by_region.png`
"""
    report_path.write_text(markdown, encoding="utf-8")
    return report_path
