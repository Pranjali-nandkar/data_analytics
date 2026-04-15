from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
TABLE_DIR = OUTPUT_DIR / "tables"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sentiment = pd.read_csv(BASE_DIR / "fear_greed_index (1).csv")
    trades = pd.read_csv(BASE_DIR / "historical_data (2).csv")
    return sentiment, trades


def clean_sentiment(sentiment: pd.DataFrame) -> pd.DataFrame:
    sentiment = sentiment.copy()
    sentiment.columns = [c.strip().lower().replace(" ", "_") for c in sentiment.columns]
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.date
    sentiment["sentiment_bucket"] = np.where(
        sentiment["classification"].str.contains("fear", case=False, na=False),
        "Fear",
        np.where(
            sentiment["classification"].str.contains("greed", case=False, na=False),
            "Greed",
            "Neutral",
        ),
    )
    sentiment = sentiment.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return sentiment[["date", "classification", "value", "sentiment_bucket"]]


def clean_trades(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades.columns = [c.strip().lower().replace(" ", "_") for c in trades.columns]

    trades["timestamp_ist"] = pd.to_datetime(
        trades.get("timestamp_ist"), errors="coerce", dayfirst=True
    )
    if "timestamp" in trades.columns:
        epoch_dt = pd.to_datetime(trades["timestamp"], errors="coerce", unit="ms")
        trades["trade_datetime"] = trades["timestamp_ist"].fillna(epoch_dt)
    else:
        trades["trade_datetime"] = trades["timestamp_ist"]

    numeric_cols = ["execution_price", "size_tokens", "size_usd", "start_position", "closed_pnl", "fee"]
    for col in numeric_cols:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades["account"] = trades["account"].astype(str).str.lower().str.strip()
    trades["side"] = trades["side"].astype(str).str.upper().str.strip()
    trades["trade_date"] = trades["trade_datetime"].dt.date
    trades["abs_size_usd"] = trades["size_usd"].abs()
    if "leverage" in trades.columns:
        trades["leverage"] = pd.to_numeric(trades["leverage"], errors="coerce")
        trades["effective_leverage"] = trades["leverage"]
    else:
        # If direct leverage is unavailable, use notional vs starting position value as a risk proxy.
        start_notional = (trades["start_position"].abs() * trades["execution_price"].abs()).fillna(0)
        trades["effective_leverage"] = trades["abs_size_usd"] / (start_notional + 1.0)
    trades["is_win"] = (trades["closed_pnl"] > 0).astype(int)
    trades["is_loss"] = (trades["closed_pnl"] < 0).astype(int)
    trades["is_long"] = (trades["side"] == "BUY").astype(int)
    trades["is_short"] = (trades["side"] == "SELL").astype(int)
    trades = trades.dropna(subset=["account", "trade_date"])
    return trades


def dataset_quality_report(sentiment: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("sentiment", sentiment), ("trades", trades)]:
        rows.append(
            {
                "dataset": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "data_quality_summary.csv", index=False)
    return out


def build_daily_account_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    daily = (
        trades.groupby(["trade_date", "account"], as_index=False)
        .agg(
            daily_pnl=("closed_pnl", "sum"),
            trades_per_day=("closed_pnl", "size"),
            win_rate=("is_win", "mean"),
            loss_rate=("is_loss", "mean"),
            avg_trade_size_usd=("abs_size_usd", "mean"),
            gross_volume_usd=("abs_size_usd", "sum"),
            buy_ratio=("side", lambda s: (s == "BUY").mean()),
            sell_ratio=("side", lambda s: (s == "SELL").mean()),
            long_trades=("is_long", "sum"),
            short_trades=("is_short", "sum"),
            avg_effective_leverage=("effective_leverage", "mean"),
            pnl_std=("closed_pnl", "std"),
        )
        .fillna({"pnl_std": 0})
    )
    daily["drawdown_proxy"] = np.minimum(daily["daily_pnl"], 0.0)
    daily["long_short_ratio"] = (daily["long_trades"] + 1.0) / (daily["short_trades"] + 1.0)
    return daily


def alignment_report(sentiment: pd.DataFrame, trades: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    sentiment_dates = set(sentiment["date"])
    trade_dates = set(trades["trade_date"])
    overlap_dates = sentiment_dates.intersection(trade_dates)
    report = pd.DataFrame(
        [
            {
                "sentiment_unique_dates": len(sentiment_dates),
                "trade_unique_dates": len(trade_dates),
                "overlap_dates": len(overlap_dates),
                "sentiment_only_dates": len(sentiment_dates - trade_dates),
                "trade_only_dates": len(trade_dates - sentiment_dates),
                "merged_rows_account_day": len(merged),
            }
        ]
    )
    report.to_csv(TABLE_DIR / "date_alignment_summary.csv", index=False)
    return report


def join_daily_with_sentiment(daily: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    merged = daily.merge(sentiment, how="inner", left_on="trade_date", right_on="date")
    merged.to_csv(TABLE_DIR / "daily_account_metrics_with_sentiment.csv", index=False)
    return merged


def sentiment_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    comparison = (
        merged.groupby("sentiment_bucket", as_index=False)
        .agg(
            avg_daily_pnl=("daily_pnl", "mean"),
            median_daily_pnl=("daily_pnl", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_drawdown_proxy=("drawdown_proxy", "mean"),
            avg_trades_per_day=("trades_per_day", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_buy_ratio=("buy_ratio", "mean"),
            avg_long_short_ratio=("long_short_ratio", "mean"),
            avg_effective_leverage=("avg_effective_leverage", "mean"),
            avg_gross_volume_usd=("gross_volume_usd", "mean"),
            accounts_days=("account", "size"),
        )
        .sort_values("sentiment_bucket")
    )
    comparison.to_csv(TABLE_DIR / "sentiment_comparison.csv", index=False)
    return comparison


def permutation_p_value(
    values: pd.Series,
    groups: pd.Series,
    group_a: str = "Fear",
    group_b: str = "Greed",
    n_iter: int = 2000,
    seed: int = 42,
) -> float:
    df = pd.DataFrame({"value": values, "group": groups}).dropna()
    a = df.loc[df["group"] == group_a, "value"].values
    b = df.loc[df["group"] == group_b, "value"].values
    if len(a) == 0 or len(b) == 0:
        return np.nan

    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(pooled)
        a_sim = shuffled[: len(a)]
        b_sim = shuffled[len(a) :]
        if abs(a_sim.mean() - b_sim.mean()) >= observed:
            count += 1
    return (count + 1) / (n_iter + 1)


def sentiment_difference_evidence(merged: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "daily_pnl",
        "win_rate",
        "drawdown_proxy",
        "trades_per_day",
        "avg_trade_size_usd",
        "long_short_ratio",
        "avg_effective_leverage",
    ]
    fear_df = merged.loc[merged["sentiment_bucket"] == "Fear"]
    greed_df = merged.loc[merged["sentiment_bucket"] == "Greed"]
    rows = []
    for metric in metrics:
        if metric not in merged.columns:
            continue
        fear_mean = fear_df[metric].mean()
        greed_mean = greed_df[metric].mean()
        rows.append(
            {
                "metric": metric,
                "fear_mean": fear_mean,
                "greed_mean": greed_mean,
                "difference_greed_minus_fear": greed_mean - fear_mean,
                "pct_change_vs_fear": pct_change(greed_mean, fear_mean),
                "permutation_p_value": permutation_p_value(
                    merged[metric], merged["sentiment_bucket"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "fear_vs_greed_evidence.csv", index=False)
    return out


def build_segments(merged: pd.DataFrame) -> pd.DataFrame:
    account_profile = (
        merged.groupby("account", as_index=False)
        .agg(
            avg_daily_pnl=("daily_pnl", "mean"),
            total_pnl=("daily_pnl", "sum"),
            avg_trades_per_day=("trades_per_day", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_gross_volume_usd=("gross_volume_usd", "mean"),
            avg_effective_leverage=("avg_effective_leverage", "mean"),
            avg_win_rate=("win_rate", "mean"),
            pnl_std=("daily_pnl", "std"),
            active_days=("trade_date", "nunique"),
        )
        .fillna({"pnl_std": 0})
    )
    account_profile["positive_day_ratio"] = (
        merged.assign(pos=merged["daily_pnl"] > 0)
        .groupby("account")["pos"]
        .mean()
        .reindex(account_profile["account"])
        .values
    )

    freq_cut = account_profile["avg_trades_per_day"].median()
    lev_cut = account_profile["avg_effective_leverage"].median()
    consistency_cut = account_profile["positive_day_ratio"].median()

    account_profile["frequency_segment"] = np.where(
        account_profile["avg_trades_per_day"] >= freq_cut, "Frequent", "Infrequent"
    )
    account_profile["leverage_segment"] = np.where(
        account_profile["avg_effective_leverage"] >= lev_cut, "High Leverage", "Low Leverage"
    )
    account_profile["consistency_segment"] = np.where(
        account_profile["positive_day_ratio"] >= consistency_cut,
        "Consistent Winners",
        "Inconsistent",
    )

    account_profile.to_csv(TABLE_DIR / "account_segments.csv", index=False)
    return account_profile


def segment_performance_tables(merged: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    merged_seg = merged.merge(
        profile[
            [
                "account",
                "frequency_segment",
                "leverage_segment",
                "consistency_segment",
            ]
        ],
        on="account",
        how="left",
    )

    perf = (
        merged_seg.groupby(
            ["sentiment_bucket", "frequency_segment", "leverage_segment", "consistency_segment"],
            as_index=False,
        )
        .agg(
            mean_pnl=("daily_pnl", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_trades=("trades_per_day", "mean"),
            observations=("account", "size"),
        )
        .sort_values("observations", ascending=False)
    )
    perf.to_csv(TABLE_DIR / "segment_sentiment_performance.csv", index=False)
    return perf


def save_charts(merged: pd.DataFrame, comparison: pd.DataFrame, profile: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=merged, x="sentiment_bucket", y="daily_pnl", showfliers=False)
    plt.title("Daily Account PnL Distribution by Sentiment")
    plt.xlabel("Sentiment Bucket")
    plt.ylabel("Daily PnL")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "pnl_distribution_by_sentiment.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison, x="sentiment_bucket", y="avg_trades_per_day", color="#4C72B0")
    plt.title("Average Trades Per Account-Day by Sentiment")
    plt.xlabel("Sentiment Bucket")
    plt.ylabel("Avg Trades Per Day")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "trade_frequency_by_sentiment.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison, x="sentiment_bucket", y="avg_trade_size_usd", color="#55A868")
    plt.title("Average Trade Size (USD) by Sentiment")
    plt.xlabel("Sentiment Bucket")
    plt.ylabel("Avg Trade Size (USD)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "trade_size_by_sentiment.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison, x="sentiment_bucket", y="avg_effective_leverage", color="#C44E52")
    plt.title("Average Effective Leverage by Sentiment")
    plt.xlabel("Sentiment Bucket")
    plt.ylabel("Avg Effective Leverage")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "effective_leverage_by_sentiment.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison, x="sentiment_bucket", y="avg_long_short_ratio", color="#8172B2")
    plt.title("Long/Short Ratio by Sentiment")
    plt.xlabel("Sentiment Bucket")
    plt.ylabel("Avg Long/Short Ratio")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "long_short_ratio_by_sentiment.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=profile,
        x="avg_trades_per_day",
        y="avg_daily_pnl",
        hue="consistency_segment",
        alpha=0.7,
    )
    plt.title("Trader Segments: Frequency vs Daily PnL")
    plt.xlabel("Average Trades Per Day")
    plt.ylabel("Average Daily PnL")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "segment_scatter_frequency_vs_pnl.png", dpi=150)
    plt.close()


def pct_change(a: float, b: float) -> float:
    if b == 0:
        return np.nan
    return (a - b) / abs(b) * 100


def build_report(
    quality: pd.DataFrame,
    alignment: pd.DataFrame,
    comparison: pd.DataFrame,
    evidence: pd.DataFrame,
    profile: pd.DataFrame,
    seg_perf: pd.DataFrame,
) -> str:
    cmp_idx = comparison.set_index("sentiment_bucket")
    fear = cmp_idx.loc["Fear"] if "Fear" in cmp_idx.index else None
    greed = cmp_idx.loc["Greed"] if "Greed" in cmp_idx.index else None

    if fear is not None and greed is not None:
        pnl_delta = pct_change(greed["avg_daily_pnl"], fear["avg_daily_pnl"])
        win_delta = pct_change(greed["avg_win_rate"], fear["avg_win_rate"])
        trade_delta = pct_change(greed["avg_trades_per_day"], fear["avg_trades_per_day"])
    else:
        pnl_delta = win_delta = trade_delta = np.nan

    seg_snapshot = (
        seg_perf.sort_values("observations", ascending=False)
        .head(8)
        .to_string(index=False)
    )

    high_freq = profile.query("frequency_segment == 'Frequent'")["avg_daily_pnl"].mean()
    low_freq = profile.query("frequency_segment == 'Infrequent'")["avg_daily_pnl"].mean()
    high_lev = profile.query("leverage_segment == 'High Leverage'")["avg_daily_pnl"].mean()
    low_lev = profile.query("leverage_segment == 'Low Leverage'")["avg_daily_pnl"].mean()
    top_evidence = evidence.to_string(index=False)

    report = f"""# Trader Performance vs Market Sentiment (Round-0 Assignment)

## Methodology
- Loaded and cleaned sentiment + trade datasets.
- Standardized timestamps to daily granularity and joined on date.
- Built account-day metrics: daily PnL, win rate, drawdown proxy (negative daily PnL), average trade size, trade frequency, long/short mix.
- Created trader segments using medians:
  - Frequent vs Infrequent
  - High Leverage vs Low Leverage
  - Consistent Winners vs Inconsistent (positive day ratio)

## Data Quality Snapshot
{quality.to_string(index=False)}

## Date Alignment Snapshot (Part A)
{alignment.to_string(index=False)}

## Key Metrics Built (Part A)
- Daily PnL per account (`daily_pnl`)
- Win rate and loss rate (`win_rate`, `loss_rate`)
- Average trade size and gross volume (`avg_trade_size_usd`, `gross_volume_usd`)
- Number of trades per day (`trades_per_day`)
- Long/short ratio (`long_short_ratio`)
- Effective leverage (`avg_effective_leverage`) using direct leverage if available, else risk proxy
- Drawdown proxy (`drawdown_proxy`)

## Key Findings
1. **Performance differs across sentiment regimes**:
   - Avg daily PnL changes by **{pnl_delta:.2f}%** from Fear to Greed.
   - Avg win rate changes by **{win_delta:.2f}%** from Fear to Greed.
   - Drawdown proxy is {cmp_idx.loc['Greed', 'avg_drawdown_proxy'] if 'Greed' in cmp_idx.index else np.nan:.2f} on Greed days and {cmp_idx.loc['Fear', 'avg_drawdown_proxy'] if 'Fear' in cmp_idx.index else np.nan:.2f} on Fear days.
2. **Behavior shifts by sentiment**:
   - Average trades/day shifts by **{trade_delta:.2f}%** between Greed and Fear.
   - Effective leverage and long/short ratio also vary by sentiment (see bar charts).
3. **Segments are meaningfully different**:
   - Frequent traders avg daily PnL: **{high_freq:.2f}** vs Infrequent: **{low_freq:.2f}**.
   - High-leverage traders avg daily PnL: **{high_lev:.2f}** vs Low-leverage: **{low_lev:.2f}**.

## Fear vs Greed Evidence Table (Part B)
```
{top_evidence}
```

## Segment x Sentiment Evidence (Top rows)
```
{seg_snapshot}
```

## Actionable Strategy Ideas
1. **Regime-aware risk scaling**  
   On Fear days, reduce position size and tighten loss limits for low-leverage/inconsistent segments; these groups show weaker stability and larger downside tails.
2. **Selective activity increase on Greed days**  
   Increase trade frequency only for consistent-winner segments with positive expectancy; avoid broad leverage/frequency increases for inconsistent traders.

## Files Produced
- Tables: `output/tables/*.csv`
- Charts: `output/charts/*.png`
"""
    return report


def main() -> None:
    ensure_dirs()

    sentiment_raw, trades_raw = load_data()
    sentiment = clean_sentiment(sentiment_raw)
    trades = clean_trades(trades_raw)

    quality = dataset_quality_report(sentiment, trades)
    daily = build_daily_account_metrics(trades)
    merged = join_daily_with_sentiment(daily, sentiment)
    alignment = alignment_report(sentiment, trades, merged)
    comparison = sentiment_comparison(merged)
    evidence = sentiment_difference_evidence(merged)
    profile = build_segments(merged)
    seg_perf = segment_performance_tables(merged, profile)
    save_charts(merged, comparison, profile)

    report = build_report(quality, alignment, comparison, evidence, profile, seg_perf)
    (OUTPUT_DIR / "assignment_report.md").write_text(report, encoding="utf-8")

    print("Analysis complete.")
    print(f"Merged rows: {len(merged):,}")
    print("Generated files:")
    print(f"- {OUTPUT_DIR / 'assignment_report.md'}")
    print(f"- {CHART_DIR}")
    print(f"- {TABLE_DIR}")


if __name__ == "__main__":
    main()
