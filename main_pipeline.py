
import pandas as pd
import matplotlib.pyplot as plt
from ai_module import preprocess_data, train_and_predict
from summary_module import generate_summary

def main():
    # Define file paths
    raw_data_path = 'Net Generation of Solar Energy U.csv'
    cleaned_data_path = 'cleaned_solar_data.csv'
    forecast_output_path = 'forecast_next_year.csv'
    summary_output_path = 'weekly_summary.txt'
    plot_output_path = 'solar_generation_trend.png'

    print("--- Starting Data Pipeline ---")

    # Day 1 & 2: Data Preprocessing
    print("Task 1.1 & 2.1: Preprocessing raw data...")
    df_cleaned = preprocess_data(raw_data_path, cleaned_data_path)
    print(f"Cleaned data saved to {cleaned_data_path}")

    # Day 2: Exploratory Analysis (Visualization)
    print("Task 2.2: Generating historical trend plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(df_cleaned['Month'], df_cleaned['Generation_MWh'])
    plt.title('Net Generation of Solar Energy in the United States Over Time')
    plt.xlabel('Date')
    plt.ylabel('Net Generation (MWh)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close() # Close plot to prevent display in non-interactive environments
    print(f"Historical trend plot saved to {plot_output_path}")

    # Day 5 Week 2 & Week 3: AI/ML Core Development - Forecasting
    print("Task: Training model and forecasting next year's production...")
    forecast_df, trained_model = train_and_predict(cleaned_data_path)
    forecast_df.to_csv(forecast_output_path, index=False)
    print(f"Forecast for the next year saved to {forecast_output_path}")

    # Day 4 Week 3: AI-Generated Summary
    print("Task: Generating weekly summary...")
    weekly_summary = generate_summary(df_cleaned, forecast_df)
    with open(summary_output_path, 'w') as f:
        f.write(weekly_summary)
    print(f"Weekly summary saved to {summary_output_path}")

    print("--- Data Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
