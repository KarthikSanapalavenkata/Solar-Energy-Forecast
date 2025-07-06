# summary_module.py
import pandas as pd # Ensure pandas is imported at the top if not already

def generate_summary(df_cleaned, forecast_df):
    """
    Generates a natural language summary of historical and forecasted solar energy data.

    Args:
        df_cleaned (pd.DataFrame): DataFrame with cleaned historical data.
        forecast_df (pd.DataFrame): DataFrame with forecasted data.

    Returns:
        str: A multi-line string summary.
    """
    summary = []

    # Latest actual generation value
    # Check if df_cleaned is not empty before attempting to access iloc[-1]
    if not df_cleaned.empty:
        # 'Month' column in df_cleaned is expected to be datetime when passed here
        latest = df_cleaned.iloc[-1]
        summary.append(f"Latest available month: {latest['Month'].strftime('%B %Y')} with {latest['Generation_MWh']:.2f} MWh generated.")
    else:
        summary.append("No historical data available for summary.")

    # Forecast statistics
    if not forecast_df.empty:
        # 'Month' column in forecast_df is expected to be datetime when passed here
        avg_forecast = forecast_df['Predicted_Generation_MWh'].mean()

        max_month = forecast_df.loc[forecast_df['Predicted_Generation_MWh'].idxmax()]
        min_month = forecast_df.loc[forecast_df['Predicted_Generation_MWh'].idxmin()]

        summary.append(f"Average forecasted monthly generation for next year: {avg_forecast:.2f} MWh.")
        summary.append(f"Highest forecasted month: {max_month['Month'].strftime('%B %Y')} with {max_month['Predicted_Generation_MWh']:.2f} MWh.")
        summary.append(f"Lowest forecasted month: {min_month['Month'].strftime('%B %Y')} with {min_month['Predicted_Generation_MWh']:.2f} MWh.")
    else:
        summary.append("No forecast data available for summary.")

    return '\n'.join(summary)