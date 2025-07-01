# Solar Energy Forecasting Pipeline

## Purpose

This pipeline forecasts daily U.S. solar energy production for the next year (in MWh/day) using a simple linear regression model.

## Workflow

- Load hourly solar data from EIA
- Aggregate to daily totals
- Train model on day-of-year patterns
- Predict next 365 days
- Generate summary of forecast
- Save CSV + summary file

## Files

- `ai_module.py` → ML logic
- `summary_module.py` → Generates summary text
- `main_pipeline.py` → End-to-end pipeline
- `cleaned_solar_data.csv` → Input data
- `forecast_next_year.csv` → Forecast output
- `weekly_summary.txt` → Text summary

## Output
