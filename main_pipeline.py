import pandas as pd
import matplotlib.pyplot as plt
from ai_module import preprocess_data, train_and_predict
from summary_module import generate_summary
import os
import matplotlib.dates as mdates

def main():
    # Define file paths
    raw_data_path = 'Net Generation of Solar Energy U.csv'
    cleaned_data_path = 'cleaned_solar_data.csv'
    forecast_output_path = 'forecast_next_year.csv'
    summary_output_path = 'weekly_summary.txt'
    plot_output_path = 'solar_generation_trend.png'

    print("--- Starting Data Pipeline ---")

    # Check if raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}. Please ensure it's in the correct directory.")
        return

    # Day 1 & 2: Data Preprocessing
    print("Task 1.1 & 2.1: Preprocessing raw data...")
    try:
        df_cleaned = preprocess_data(raw_data_path, cleaned_data_path)
        print(f"Cleaned data saved to {cleaned_data_path}")
    except ValueError as e:
        print(f"Error during data preprocessing: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return

    # Day 2: Exploratory Analysis (Visualization)
    print("Task 2.2: Generating historical trend plot...")
    if not df_cleaned.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(df_cleaned['Month'], df_cleaned['Generation_MWh'])
        plt.title('Net Generation of Solar Energy in the United States Over Time')
        plt.xlabel('Date')
        plt.ylabel('Net Generation (MWh)')
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(plot_output_path)
        plt.close()
        print(f"Historical trend plot saved to {plot_output_path}")
    else:
        print("Skipping historical trend plot: Cleaned data is empty.")

    # Day 5 Week 2 & Week 3: AI/ML Core Development - Forecasting
    print("Task: Training model and forecasting next year's production...")
    try:
        forecast_df, trained_model = train_and_predict(cleaned_data_path)
        forecast_df.to_csv(forecast_output_path, index=False)
        print(f"Forecast for the next year saved to {forecast_output_path}")
    except Exception as e:
        print(f"Error during model training and forecasting: {e}")
        return

    # Day 4 Week 3: AI-Generated Summary
    print("Task: Generating weekly summary...")
    try:
        # IMPORTANT: Ensure 'Month' is datetime before passing to generate_summary
        # df_cleaned is already datetime from preprocess_data, but if it were reloaded,
        # this would be needed: df_cleaned['Month'] = pd.to_datetime(df_cleaned['Month'])

        # Load forecast_df again to ensure 'Month' column is datetime type
        # This is crucial because .csv save/load loses datetime type
        forecast_df_for_summary = pd.read_csv(forecast_output_path)
        forecast_df_for_summary['Month'] = pd.to_datetime(forecast_df_for_summary['Month'])

        weekly_summary = generate_summary(df_cleaned, forecast_df_for_summary) # Pass the reloaded df
        with open(summary_output_path, 'w') as f:
            f.write(weekly_summary)
        print(f"Weekly summary saved to {summary_output_path}")
    except Exception as e:
        print(f"Error during summary generation: {e}")
        return

    print("--- Data Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()