# Trader Performance vs Market Sentiment

This repository contains my Round-0 assignment for the Primetrade.ai Data Science/Analytics Intern role.

## Files
- `assignment.ipynb` - main notebook (Part A, B, C + bonus sections)
- `app.py` - Streamlit dashboard
- `fear_greed_index (1).csv` - sentiment dataset
- `historical_data (2).csv` - Hyperliquid trade dataset
- `output/` - generated results (tables, charts, reports)

## Environment setup
Use Python 3.10+.

Install dependencies:

`pip install pandas numpy matplotlib seaborn scikit-learn streamlit`

## How to run
### 1) Core assignment (Part A, B, C + bonus sections)
Open `assignment.ipynb` and run all cells.

This step creates:
- cleaned/aligned daily dataset
- Fear vs Greed comparison tables
- segment-level analysis
- charts and written summary

### 2) Streamlit dashboard
Run:

`streamlit run app.py`

Open `http://localhost:8501` in browser.

## Output guide
Main outputs are in `output/`:
- `tables/` - all result tables used in analysis
- `charts/` - all saved charts
- `assignment_report.md` - concise written findings + strategy ideas
- `bonus_model_report.txt` - model metrics

## Notes
- Date alignment is done at daily level using parsed trade timestamps and sentiment dates.
- If direct leverage is not available in raw trade data, a leverage proxy is calculated from notional size and starting position.
