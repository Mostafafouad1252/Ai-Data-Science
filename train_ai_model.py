from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_preprocessor() -> ColumnTransformer:
    numeric_features = ["marketing_spend", "support_tickets", "usage_score", "satisfaction_score"]
    categorical_features = ["region", "product_category"]
    text_feature = "customer_feedback"

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("txt", TfidfVectorizer(max_features=40, ngram_range=(1, 2)), text_feature),
        ],
        remainder="drop",
    )


def train_and_evaluate(df: pd.DataFrame, base_dir: Path) -> Dict[str, float]:
    X = df[
        [
            "marketing_spend",
            "support_tickets",
            "usage_score",
            "satisfaction_score",
            "region",
            "product_category",
            "customer_feedback",
        ]
    ].copy()
    y_reg = df["revenue"]
    y_clf = df["churned"]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    preprocessor = _build_preprocessor()

    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    regressor.fit(X_train, y_reg_train)
    reg_preds = regressor.predict(X_test)

    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    classifier.fit(X_train, y_clf_train)
    clf_preds = classifier.predict(X_test)

    cluster_features = df[["marketing_spend", "usage_score", "satisfaction_score"]]
    scaled_cluster_features = StandardScaler().fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_cluster_features)
    silhouette = silhouette_score(scaled_cluster_features, cluster_labels)

    metrics = {
        "regression_r2": float(r2_score(y_reg_test, reg_preds)),
        "regression_mae": float(mean_absolute_error(y_reg_test, reg_preds)),
        "classification_accuracy": float(accuracy_score(y_clf_test, clf_preds)),
        "clustering_silhouette": float(silhouette),
        "avg_predicted_revenue": float(np.mean(reg_preds)),
    }

    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = cluster_labels
    df_with_clusters.to_csv(reports_dir / "clustered_data.csv", index=False)

    return metrics
