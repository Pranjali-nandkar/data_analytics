from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
TABLE_DIR = BASE_DIR / "output" / "tables"
CHART_DIR = BASE_DIR / "output" / "charts"
REPORT_FILE = BASE_DIR / "output" / "assignment_report.md"
BONUS_REPORT_FILE = BASE_DIR / "output" / "bonus_model_report.txt"


st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")
st.title("Trader Performance vs Market Sentiment")
st.caption("Lightweight dashboard for Primetrade.ai assignment outputs")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


sentiment_comp = safe_read_csv(TABLE_DIR / "sentiment_comparison.csv")
evidence = safe_read_csv(TABLE_DIR / "fear_vs_greed_evidence.csv")
segments = safe_read_csv(TABLE_DIR / "segment_sentiment_performance.csv")
clusters = safe_read_csv(TABLE_DIR / "trader_archetypes_clusters.csv")
preds = safe_read_csv(TABLE_DIR / "next_day_profitability_predictions.csv")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Core Analysis", "Segments", "Bonus Model", "Archetypes Clustering"]
)

with tab1:
    st.subheader("Fear vs Greed Comparison")
    if not sentiment_comp.empty:
        st.dataframe(sentiment_comp, use_container_width=True)
    else:
        st.warning("Missing `sentiment_comparison.csv`. Run `python analyze_trader_sentiment.py`.")

    st.subheader("Fear vs Greed Evidence")
    if not evidence.empty:
        st.dataframe(evidence, use_container_width=True)
        st.subheader("Fear vs Greed Metric Delta (%)")
        chart_df = evidence.copy()
        chart_df = chart_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pct_change_vs_fear"])
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("metric")["pct_change_vs_fear"])
    else:
        st.warning("Missing `fear_vs_greed_evidence.csv`.")

    for chart_name in [
        "pnl_distribution_by_sentiment.png",
        "trade_frequency_by_sentiment.png",
        "trade_size_by_sentiment.png",
        "effective_leverage_by_sentiment.png",
        "long_short_ratio_by_sentiment.png",
    ]:
        chart_path = CHART_DIR / chart_name
        if chart_path.exists():
            st.image(str(chart_path), caption=chart_name, use_container_width=True)

with tab2:
    st.subheader("Segment-Level Performance")
    if not segments.empty:
        st.dataframe(segments, use_container_width=True)
        st.subheader("Top Segment Performance (Mean PnL)")
        top_seg = segments.sort_values("mean_pnl", ascending=False).head(10).copy()
        top_seg["segment_label"] = (
            top_seg["sentiment_bucket"].astype(str)
            + " | "
            + top_seg["frequency_segment"].astype(str)
            + " | "
            + top_seg["leverage_segment"].astype(str)
            + " | "
            + top_seg["consistency_segment"].astype(str)
        )
        st.bar_chart(top_seg.set_index("segment_label")["mean_pnl"])
    else:
        st.warning("Missing `segment_sentiment_performance.csv`.")

    scatter_path = CHART_DIR / "segment_scatter_frequency_vs_pnl.png"
    if scatter_path.exists():
        st.image(str(scatter_path), caption="Segment scatter", use_container_width=True)

with tab3:
    st.subheader("Next-Day Profitability Prediction")
    if not preds.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction rows", len(preds))
        with col2:
            if "correct" in preds.columns:
                st.metric("Observed test accuracy", f"{preds['correct'].mean():.2%}")
        if "predicted_prob_profitable" in preds.columns:
            st.subheader("Predicted Profitability Probability Distribution")
            st.area_chart(preds["predicted_prob_profitable"].sort_values().reset_index(drop=True))
        st.dataframe(preds.head(300), use_container_width=True)
    else:
        st.warning(
            "Missing `next_day_profitability_predictions.csv`. "
            "Run `python bonus_modeling_clustering.py`."
        )

    if BONUS_REPORT_FILE.exists():
        st.subheader("Model Report")
        st.code(BONUS_REPORT_FILE.read_text(encoding="utf-8"))

with tab4:
    st.subheader("Trader Archetypes (KMeans)")
    if not clusters.empty:
        if "archetype" in clusters.columns:
            st.bar_chart(clusters["archetype"].value_counts())
        if {"avg_trades_per_day", "avg_daily_pnl", "archetype"}.issubset(set(clusters.columns)):
            st.subheader("Archetypes: Trades/Day vs Avg Daily PnL")
            st.scatter_chart(
                clusters,
                x="avg_trades_per_day",
                y="avg_daily_pnl",
                color="archetype",
            )
        st.dataframe(clusters, use_container_width=True)
    else:
        st.warning(
            "Missing `trader_archetypes_clusters.csv`. "
            "Run `python bonus_modeling_clustering.py`."
        )


st.divider()
st.caption("Tip: Run `python analyze_trader_sentiment.py` and `python bonus_modeling_clustering.py` to refresh dashboard data.")
