from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class DataPaths:
    base_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "business_data.csv"


def _sample_feedback(score: float, rng: np.random.Generator) -> str:
    positive = [
        "Support was fast and helpful",
        "Great experience and smooth onboarding",
        "Very satisfied with service quality",
    ]
    neutral = [
        "Experience was okay, nothing special",
        "Average value for the price",
        "Service quality was acceptable",
    ]
    negative = [
        "Support response took too long",
        "Unhappy with quality and delays",
        "Not satisfied with service consistency",
    ]
    if score >= 0.66:
        return rng.choice(positive)
    if score >= 0.33:
        return rng.choice(neutral)
    return rng.choice(negative)


def generate_dataset(base_dir: Path, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    paths = DataPaths(base_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    regions = np.array(["North", "South", "East", "West"])
    categories = np.array(["Basic", "Standard", "Premium", "Enterprise"])

    marketing_spend = rng.normal(5000, 1800, n_samples).clip(500, None)
    tickets = rng.poisson(7, n_samples)
    usage_score = rng.uniform(0, 100, n_samples)
    satisfaction = rng.uniform(0, 1, n_samples)
    region = rng.choice(regions, size=n_samples, p=[0.25, 0.25, 0.2, 0.3])
    category = rng.choice(categories, size=n_samples, p=[0.3, 0.35, 0.25, 0.1])

    region_effect = pd.Series(region).map(
        {"North": 2500, "South": 900, "East": 1200, "West": 2100}
    ).to_numpy()
    category_effect = pd.Series(category).map(
        {"Basic": 500, "Standard": 1800, "Premium": 3500, "Enterprise": 6000}
    ).to_numpy()

    noise = rng.normal(0, 1300, n_samples)
    revenue = (
        marketing_spend * 1.8
        + usage_score * 90
        + satisfaction * 4000
        - tickets * 120
        + region_effect
        + category_effect
        + noise
    ).clip(300, None)

    churn_prob = (
        0.45
        - 0.0028 * usage_score
        - 0.35 * satisfaction
        + 0.03 * tickets
        + rng.normal(0, 0.03, n_samples)
    )
    churn = (churn_prob > 0.32).astype(int)

    feedback = [_sample_feedback(float(s), rng) for s in satisfaction]

    df = pd.DataFrame(
        {
            "marketing_spend": marketing_spend.round(2),
            "support_tickets": tickets,
            "usage_score": usage_score.round(2),
            "satisfaction_score": satisfaction.round(3),
            "region": region,
            "product_category": category,
            "customer_feedback": feedback,
            "revenue": revenue.round(2),
            "churned": churn,
        }
    )

    df.to_csv(paths.dataset_path, index=False)
    return df


def load_dataset(base_dir: Path) -> pd.DataFrame:
    paths = DataPaths(base_dir)
    if not paths.dataset_path.exists():
        return generate_dataset(base_dir)
    return pd.read_csv(paths.dataset_path)


def get_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = df.drop(columns=["revenue", "churned"])
    y_revenue = df["revenue"]
    y_churn = df["churned"]
    return features, y_revenue, y_churn
