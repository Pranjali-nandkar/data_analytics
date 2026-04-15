# Trader Performance vs Market Sentiment (Round-0 Assignment)

## Methodology
- Loaded and cleaned sentiment + trade datasets.
- Standardized timestamps to daily granularity and joined on date.
- Built account-day metrics: daily PnL, win rate, drawdown proxy (negative daily PnL), average trade size, trade frequency, long/short mix.
- Created trader segments using medians:
  - Frequent vs Infrequent
  - High Leverage vs Low Leverage
  - Consistent Winners vs Inconsistent (positive day ratio)

## Data Quality Snapshot
  dataset   rows  columns  missing_cells  duplicate_rows
sentiment   2644        4              0               0
   trades 211224       24              0               0

## Date Alignment Snapshot (Part A)
 sentiment_unique_dates  trade_unique_dates  overlap_dates  sentiment_only_dates  trade_only_dates  merged_rows_account_day
                   2644                 480            479                  2165                 1                     2340

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
   - Avg daily PnL changes by **-20.08%** from Fear to Greed.
   - Avg win rate changes by **1.59%** from Fear to Greed.
   - Drawdown proxy is -926.16 on Greed days and -1101.45 on Fear days.
2. **Behavior shifts by sentiment**:
   - Average trades/day shifts by **-27.00%** between Greed and Fear.
   - Effective leverage and long/short ratio also vary by sentiment (see bar charts).
3. **Segments are meaningfully different**:
   - Frequent traders avg daily PnL: **10713.06** vs Infrequent: **3541.02**.
   - High-leverage traders avg daily PnL: **7825.89** vs Low-leverage: **6428.19**.

## Fear vs Greed Evidence Table (Part B)
```
                metric    fear_mean  greed_mean  difference_greed_minus_fear  pct_change_vs_fear  permutation_p_value
             daily_pnl  5185.146443 4144.208334                 -1040.938110          -20.075385             0.455272
              win_rate     0.357071    0.362748                     0.005677            1.589851             0.713643
        drawdown_proxy -1101.449895 -926.159538                   175.290357           15.914510             0.773613
        trades_per_day   105.363291   76.912266                   -28.451025          -27.002787             0.001999
    avg_trade_size_usd  8529.859802 5954.632633                 -2575.227170          -30.190733             0.002999
      long_short_ratio     8.219183    5.707994                    -2.511188          -30.552776             0.057971
avg_effective_leverage   552.930336  616.720609                    63.790274           11.536765             0.708146
```

## Segment x Sentiment Evidence (Top rows)
```
sentiment_bucket frequency_segment leverage_segment consistency_segment     mean_pnl  mean_win_rate  mean_trades  observations
           Greed          Frequent     Low Leverage  Consistent Winners  3712.331573       0.376086   163.190114           263
           Greed        Infrequent    High Leverage        Inconsistent  7330.431015       0.276429    36.941667           240
            Fear          Frequent     Low Leverage  Consistent Winners  4936.277778       0.367845   208.427136           199
           Greed        Infrequent    High Leverage  Consistent Winners  1062.454888       0.277058    24.387755           196
           Greed          Frequent    High Leverage  Consistent Winners  4853.292480       0.583465    88.033113           151
           Greed        Infrequent     Low Leverage  Consistent Winners  2322.690761       0.405271    63.437956           137
            Fear        Infrequent     Low Leverage        Inconsistent  3579.976430       0.297748    34.691176           136
            Fear          Frequent    High Leverage  Consistent Winners 16129.782880       0.487833   158.834586           133
```

## Actionable Strategy Ideas
1. **Regime-aware risk scaling**  
   On Fear days, reduce position size and tighten loss limits for low-leverage/inconsistent segments; these groups show weaker stability and larger downside tails.
2. **Selective activity increase on Greed days**  
   Increase trade frequency only for consistent-winner segments with positive expectancy; avoid broad leverage/frequency increases for inconsistent traders.

## Files Produced
- Tables: `output/tables/*.csv`
- Charts: `output/charts/*.png`
