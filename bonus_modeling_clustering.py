from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_trader_sentiment import (
    TABLE_DIR,
    OUTPUT_DIR,
    ensure_dirs,
    load_data,
    clean_sentiment,
    clean_trades,
    build_daily_account_metrics,
    join_daily_with_sentiment,
)


def build_next_day_dataset(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df = df.sort_values(["account", "trade_date"]).reset_index(drop=True)

    df["next_day_pnl"] = df.groupby("account")["daily_pnl"].shift(-1)
    df["next_day_profit_bucket"] = np.where(df["next_day_pnl"] > 0, "Profitable", "Non-Profitable")
    df["pnl_volatility_7d"] = (
        df.groupby("account")["daily_pnl"]
        .rolling(7, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["next_day_volatility_bucket"] = np.where(
        df.groupby("account")["pnl_volatility_7d"].shift(-1) > df["pnl_volatility_7d"].median(),
        "High Volatility",
        "Low Volatility",
    )
    return df


def train_profitability_model(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    model_df = df.dropna(subset=["next_day_pnl"]).copy()
    feature_cols_num = [
        "daily_pnl",
        "win_rate",
        "loss_rate",
        "avg_trade_size_usd",
        "gross_volume_usd",
        "trades_per_day",
        "long_short_ratio",
        "avg_effective_leverage",
        "drawdown_proxy",
    ]
    feature_cols_cat = ["sentiment_bucket", "classification"]
    target_col = "next_day_profit_bucket"

    model_df["trade_date"] = pd.to_datetime(model_df["trade_date"])
    split_date = model_df["trade_date"].quantile(0.8)

    train_df = model_df[model_df["trade_date"] <= split_date].copy()
    test_df = model_df[model_df["trade_date"] > split_date].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols_num,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_cols_cat,
            ),
        ]
    )

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    clf.fit(train_df[feature_cols_num + feature_cols_cat], train_df[target_col])
    preds = clf.predict(test_df[feature_cols_num + feature_cols_cat])
    probs = clf.predict_proba(test_df[feature_cols_num + feature_cols_cat])[:, 1]

    out = test_df[["account", "trade_date", target_col]].copy()
    out["predicted_bucket"] = preds
    out["predicted_prob_profitable"] = probs
    out["correct"] = (out["predicted_bucket"] == out[target_col]).astype(int)

    report = []
    report.append(f"Train rows: {len(train_df)}")
    report.append(f"Test rows: {len(test_df)}")
    report.append(f"Accuracy: {accuracy_score(test_df[target_col], preds):.4f}")
    report.append("Classification report:")
    report.append(classification_report(test_df[target_col], preds))
    return out, "\n".join(report)


def build_trader_archetypes(merged: pd.DataFrame) -> pd.DataFrame:
    trader_features = (
        merged.groupby("account", as_index=False)
        .agg(
            avg_daily_pnl=("daily_pnl", "mean"),
            pnl_std=("daily_pnl", "std"),
            avg_win_rate=("win_rate", "mean"),
            avg_trades_per_day=("trades_per_day", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_effective_leverage=("avg_effective_leverage", "mean"),
            avg_long_short_ratio=("long_short_ratio", "mean"),
            active_days=("trade_date", "nunique"),
            fear_exposure=("sentiment_bucket", lambda s: (s == "Fear").mean()),
            greed_exposure=("sentiment_bucket", lambda s: (s == "Greed").mean()),
        )
        .fillna({"pnl_std": 0})
    )

    num_cols = [c for c in trader_features.columns if c != "account"]
    X = trader_features[num_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    labels = kmeans.fit_predict(Xs)
    trader_features["cluster"] = labels

    cluster_stats = (
        trader_features.groupby("cluster", as_index=False)[num_cols]
        .mean()
        .sort_values("avg_daily_pnl", ascending=False)
    )

    rank_map = {}
    names = ["Archetype A", "Archetype B", "Archetype C"]
    for idx, cid in enumerate(cluster_stats["cluster"].tolist()):
        rank_map[cid] = names[idx]
    trader_features["archetype"] = trader_features["cluster"].map(rank_map)

    return trader_features


def main() -> None:
    ensure_dirs()
    sentiment_raw, trades_raw = load_data()
    sentiment = clean_sentiment(sentiment_raw)
    trades = clean_trades(trades_raw)
    daily = build_daily_account_metrics(trades)
    merged = join_daily_with_sentiment(daily, sentiment)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"])

    next_day_df = build_next_day_dataset(merged)
    predictions, model_report = train_profitability_model(next_day_df)
    archetypes = build_trader_archetypes(merged)

    predictions.to_csv(TABLE_DIR / "next_day_profitability_predictions.csv", index=False)
    archetypes.to_csv(TABLE_DIR / "trader_archetypes_clusters.csv", index=False)
    (OUTPUT_DIR / "bonus_model_report.txt").write_text(model_report, encoding="utf-8")

    print("Bonus modeling and clustering complete.")
    print(f"- {TABLE_DIR / 'next_day_profitability_predictions.csv'}")
    print(f"- {TABLE_DIR / 'trader_archetypes_clusters.csv'}")
    print(f"- {OUTPUT_DIR / 'bonus_model_report.txt'}")


if __name__ == "__main__":
    main()
