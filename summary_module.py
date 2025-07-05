def generate_summary(df_cleaned, forecast_df):
    summary = []

    # Latest actual generation value
    # Check if df_cleaned is not empty before attempting to access iloc[-1]
    if not df_cleaned.empty:
        latest = df_cleaned.iloc[-1]
        summary.append(f"Latest available month: {latest['Month'].strftime('%B %Y')} with {latest['Generation_MWh']:.2f} MWh generated.")
    else:
        summary.append("No historical data available for summary.")

    # Forecast statistics
    if not forecast_df.empty:
        avg_forecast = forecast_df['Predicted_Generation_MWh'].mean()
        # Ensure 'Month' column in forecast_df is datetime to use strftime
        forecast_df['Month'] = pd.to_datetime(forecast_df['Month'])
        
        max_month = forecast_df.loc[forecast_df['Predicted_Generation_MWh'].idxmax()]
        min_month = forecast_df.loc[forecast_df['Predicted_Generation_MWh'].idxmin()]

        summary.append(f"Average forecasted monthly generation for next year: {avg_forecast:.2f} MWh.")
        summary.append(f"Highest forecasted month: {max_month['Month'].strftime('%B %Y')} with {max_month['Predicted_Generation_MWh']:.2f} MWh.")
        summary.append(f"Lowest forecasted month: {min_month['Month'].strftime('%B %Y')} with {min_month['Predicted_Generation_MWh']:.2f} MWh.")
    else:
        summary.append("No forecast data available for summary.")

    return '\n'.join(summary)