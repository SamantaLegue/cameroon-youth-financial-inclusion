#!/usr/bin/env python3
"""Exploratory analysis and Random Forest model for youth financial inclusion.

The script is intentionally schema-aware. It works with a tidy CSV whose columns
follow data_dictionary_out_of_school_youth.csv, while also tolerating common
World Bank-style aliases such as country_name and region_name.

This is an exploratory portfolio model. It is not an individual credit, fraud,
eligibility, or exclusion model. If the input is country-year aggregate data,
the model describes associations between aggregate indicators; it does not
predict outcomes for individual young people.

Example:
    python analysis/eda_random_forest.py \
        --input data/processed/global_findex_cameroon.csv \
        --output outputs/eda

Dependencies:
    pandas, numpy, matplotlib, seaborn, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LOGGER = logging.getLogger("youth_financial_inclusion")

ALIASES = {
    "country": ["country", "country_name", "economy"],
    "region": ["region", "region_name", "world_region"],
    "year": ["year", "survey_year"],
    "account_ownership": ["account_ownership", "account_own", "account"],
    "mobile_money_ownership": [
        "mobile_money_ownership",
        "mobile_money_account",
        "mobile_money_own",
    ],
    "digital_payment": ["digital_payment", "digital_payments", "made_digital_payment"],
    "phone_access": ["phone_access", "mobile_phone_access", "phone"],
    "internet_access": ["internet_access", "internet_use", "internet"],
    "income_group": ["income_group", "income_classification"],
    "location": ["location", "residence", "urban_rural"],
    "gender": ["gender", "sex"],
    "survey_weight": ["survey_weight", "weight", "sample_weight"],
    "mobile_money_user": ["mobile_money_user", "mobile_money_usage", "mobile_money_use"],
}

BINARY_COLUMNS = {
    "in_school",
    "training_status",
    "out_of_school",
    "neet",
    "account_ownership",
    "mobile_money_ownership",
    "digital_payment",
    "saved_formally",
    "borrowed_formally",
    "phone_access",
    "internet_access",
    "id_barrier",
    "cost_barrier",
    "trust_barrier",
    "distance_barrier",
    "mobile_money_user",
}

MODEL_FEATURES = [
    "account_ownership",
    "digital_payment",
    "phone_access",
    "internet_access",
    "age",
    "income_group",
    "location",
    "gender",
    "region",
    "year",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV file")
    parser.add_argument(
        "--output", default=Path("outputs/eda"), type=Path, help="Output directory"
    )
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--test-size", default=0.25, type=float)
    return parser.parse_args()


def normalise_name(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    return value.strip("_")


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalise_name(column) for column in df.columns]
    rename_map: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            alias = normalise_name(alias)
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def to_binary(series: pd.Series) -> pd.Series:
    """Convert common survey encodings to nullable numeric 0/1 values."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        # Preserve only explicit 0/1 values; other numeric survey encodings are
        # treated as missing until a source-specific mapping is documented.
        return numeric.where(numeric.isin([0, 1]))
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "owned": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "not owned": 0,
    }
    return series.astype("string").str.strip().str.lower().map(mapping)


def prepare_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = standardise_columns(pd.read_csv(path))
    LOGGER.info("Loaded %s rows and %s columns", len(df), len(df.columns))

    for column in BINARY_COLUMNS.intersection(df.columns):
        df[column] = to_binary(df[column])
    for column in ["age", "year", "survey_weight"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "mobile_money_user" not in df.columns:
        source_columns = [
            column
            for column in ["mobile_money_ownership", "digital_payment"]
            if column in df.columns
        ]
        if source_columns:
            df["mobile_money_user"] = (
                df[source_columns].fillna(0).max(axis=1).astype("Int64")
            )
            LOGGER.info("Derived mobile_money_user from %s", source_columns)

    if "out_of_school" in df.columns:
        df["out_of_school"] = to_binary(df["out_of_school"])
        target = df[df["out_of_school"].eq(1)].copy()
        if len(target) >= 10:
            LOGGER.info("EDA/model population restricted to %s out-of-school rows", len(target))
            return target
        LOGGER.warning(
            "out_of_school exists but has fewer than 10 positive rows; using all rows instead"
        )
    return df


def save_bar_chart(df: pd.DataFrame, column: str, output: Path, title: str) -> None:
    if column not in df.columns:
        return
    values = df[column].dropna().value_counts().sort_index()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=values.index.astype(str), y=values.values, color="#d46b3f", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Number of records")
    ax.bar_label(ax.containers[0], fmt="%.0f")
    fig.tight_layout()
    fig.savefig(output / f"distribution_{column}.png", dpi=160)
    plt.close(fig)


def run_eda(df: pd.DataFrame, output: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missingness": {
            column: float(value)
            for column, value in df.isna().mean().sort_values(ascending=False).items()
        },
    }
    df.describe(include="all").transpose().to_csv(output / "descriptive_summary.csv")

    for column, title in [
        ("account_ownership", "Account ownership"),
        ("mobile_money_user", "Mobile-money user indicator"),
        ("out_of_school", "Out-of-school indicator"),
        ("region", "Records by region"),
    ]:
        save_bar_chart(df, column, output, title)

    barrier_columns = [
        column
        for column in ["id_barrier", "cost_barrier", "trust_barrier", "distance_barrier"]
        if column in df.columns
    ]
    if barrier_columns:
        barrier_rates = df[barrier_columns].apply(pd.to_numeric, errors="coerce").mean().sort_values()
        barrier_rates.rename("share_reporting_barrier").to_csv(output / "barrier_rates.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            x=barrier_rates.values,
            y=[column.replace("_", " ").title() for column in barrier_rates.index],
            color="#284b63",
            ax=ax,
        )
        ax.set_title("Reported barriers in the analysis population")
        ax.set_xlabel("Share of records")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(output / "barrier_rates.png", dpi=160)
        plt.close(fig)

    numeric_columns = [
        column
        for column in [
            "account_ownership",
            "mobile_money_user",
            "digital_payment",
            "internet_access",
            "phone_access",
            "age",
        ]
        if column in df.columns
    ]
    if len(numeric_columns) >= 2:
        correlation = df[numeric_columns].apply(pd.to_numeric, errors="coerce").corr()
        correlation.to_csv(output / "numeric_correlations.csv")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap="vlag", center=0, ax=ax)
        ax.set_title("Numeric-variable correlation matrix")
        fig.tight_layout()
        fig.savefig(output / "correlations.png", dpi=160)
        plt.close(fig)

    return summary


def run_random_forest(
    df: pd.DataFrame, output: Path, random_state: int, test_size: float
) -> dict[str, object]:
    result: dict[str, object] = {"status": "skipped"}
    target = "mobile_money_user"
    if target not in df.columns:
        result["reason"] = "No mobile_money_user or derivable mobile-money ownership field was found."
        LOGGER.warning(result["reason"])
        return result

    features = [column for column in MODEL_FEATURES if column in df.columns and column != target]
    if not features:
        result["reason"] = "No supported model features were found."
        LOGGER.warning(result["reason"])
        return result

    model_df = df[features + [target]].copy()
    model_df[target] = to_binary(model_df[target]).astype("float")
    model_df = model_df.dropna(subset=[target])
    class_counts = model_df[target].value_counts()
    if len(class_counts) < 2:
        result["reason"] = "Target contains fewer than two classes after cleaning."
        LOGGER.warning(result["reason"])
        return result
    if len(model_df) < 30:
        LOGGER.warning("Only %s rows available; model metrics may be unstable", len(model_df))

    X = model_df[features]
    y = model_df[target].astype(int)
    numeric_features = [column for column in features if pd.api.types.is_numeric_dtype(X[column])]
    categorical_features = [column for column in features if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", classifier)])

    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics: dict[str, object] = {
        "status": "completed",
        "rows_used": int(len(model_df)),
        "features": features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))

    (output / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(output / "random_forest_feature_importance.csv", index=False)

    # Permutation importance is reported on the held-out set at the original
    # feature level, which is easier to explain than one-hot encoded columns.
    permutation = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    permutation_df = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "mean_importance": permutation.importances_mean,
                "std_importance": permutation.importances_std,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    permutation_df.to_csv(output / "random_forest_permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    top = importance_df.head(12).sort_values("importance")
    sns.barplot(data=top, x="importance", y="feature", color="#d46b3f", ax=ax)
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("Impurity-based importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output / "random_forest_feature_importance.png", dpi=160)
    plt.close(fig)

    LOGGER.info("Random Forest metrics: %s", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sns.set_theme(style="whitegrid", context="notebook")
    args.output.mkdir(parents=True, exist_ok=True)

    df = prepare_data(args.input)
    summary = run_eda(df, args.output)
    model_result = run_random_forest(df, args.output, args.random_state, args.test_size)
    (args.output / "run_summary.json").write_text(
        json.dumps({"eda": summary, "model": model_result}, indent=2, default=float)
    )
    LOGGER.info("Analysis outputs written to %s", args.output.resolve())


if __name__ == "__main__":
    main()
