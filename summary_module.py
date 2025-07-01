
import pandas as pd

def generate_summary(df_original, forecast_df):
    """
    Generates a mock summary of the solar energy data and forecast.
    """
    last_full_year = df_original['Month'].dt.year.max()
    if df_original['Month'].dt.month.max() < 12: # if last year is not full
        last_full_year = df_original['Month'].dt.year.max() - 1

    current_year_data = df_original[df_original['Month'].dt.year == last_full_year]['Generation_MWh']
    total_generation_last_full_year = current_year_data.sum()
    avg_monthly_generation_last_full_year = current_year_data.mean()
    max_generation_last_full_year = current_year_data.max()
    min_generation_last_full_year = current_year_data.min()

    predicted_total_next_year = forecast_df['Predicted_Generation_MWh'].sum()
    predicted_avg_monthly_next_year = forecast_df['Predicted_Generation_MWh'].mean()

    summary = f"""
    Weekly Solar Energy Report:

    Overview:
    This report summarizes the net generation of solar energy in the United States and provides a forecast for the upcoming year.

    Historical Data Analysis (Last Full Year: {last_full_year}):
    - Total Generation in {last_full_year}: {total_generation_last_full_year / 1000000:.2f} million MWh
    - Average Monthly Generation in {last_full_year}: {avg_monthly_generation_last_full_year / 1000000:.2f} million MWh
    - Peak Monthly Generation in {last_full_year}: {max_generation_last_full_year / 1000000:.2f} million MWh
    - Minimum Monthly Generation in {last_full_year}: {min_generation_last_full_year / 1000000:.2f} million MWh

    Forecast for the Next 12 Months:
    - Predicted Total Generation: {predicted_total_next_year / 1000000:.2f} million MWh
    - Predicted Average Monthly Generation: {predicted_avg_monthly_next_year / 1000000:.2f} million MWh

    Notes:
    The forecast is based on a Linear Regression model trained on historical data. This model captures the general trend in solar energy generation. For more accurate long-term predictions, advanced time series models considering seasonality and other external factors would be beneficial.
    """
    return summary
