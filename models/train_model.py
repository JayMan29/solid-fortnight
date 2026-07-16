"""
Walk-forward training: fit only on data available before the test
period, to avoid look-ahead bias. Baseline model is logistic regression
on standardized features -- start simple, add complexity only if it
survives out-of-sample testing.
"""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

FEATURES = [
    "revenue_growth_yoy", "gross_margin", "free_cash_flow_margin",
    "net_debt", "ev_to_revenue", "momentum_6m", "relative_momentum_12m",
    "volatility_6m", "news_count_30d", "news_acceleration",
    "unique_sources_30d", "sentiment_net", "commercial_event_score_90d",
    "rnd_growth_yoy",
]


def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("preprocessing", ColumnTransformer(transformers=[
            ("numeric", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]), FEATURES),
        ])),
        ("model", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
    ])


def train_walk_forward_model(
    dataset: pd.DataFrame,
    training_end: str,
    testing_start: str,
    target_col: str = "outperformed_next_12m",
) -> tuple[Pipeline, pd.DataFrame]:
    """
    `dataset` needs a `date` column plus every column in FEATURES, plus
    the binary target column. Rows with a date after `testing_start`
    are held out entirely from fitting.
    """
    data = dataset.sort_values("date").copy()
    available_features = [f for f in FEATURES if f in data.columns]
    if len(available_features) < 3:
        raise ValueError(
            f"Only {len(available_features)} of {len(FEATURES)} expected features "
            "are present -- check your feature-building step."
        )

    train = data.loc[data["date"] <= pd.Timestamp(training_end)].copy()
    test = data.loc[data["date"] >= pd.Timestamp(testing_start)].copy()
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty -- check your date ranges.")

    pipeline = build_pipeline()
    # rebuild the ColumnTransformer with only the columns actually present
    pipeline.set_params(preprocessing=ColumnTransformer(transformers=[
        ("numeric", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]), available_features),
    ]))

    pipeline.fit(train[available_features], train[target_col])
    test = test.copy()
    test["prediction_probability"] = pipeline.predict_proba(test[available_features])[:, 1]

    auc = roc_auc_score(test[target_col], test["prediction_probability"])
    print(f"Out-of-sample ROC AUC: {auc:.3f}")
    print(classification_report(test[target_col], test["prediction_probability"] >= 0.5))

    return pipeline, test
