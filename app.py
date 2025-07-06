# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import tempfile # For handling temporary uploaded files

# Import your custom modules
from ai_module import preprocess_data, train_and_predict, run_anomaly_detection
from summary_module import generate_summary

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Solar Energy AI Dashboard")

st.title("☀️ AI-Driven Solar Energy Forecasting & Anomaly Detection")
st.markdown("""
    This application processes historical solar energy generation data, forecasts future output,
    identifies anomalies, and provides a natural language summary.
""")

# --- File Upload Module (Task 1.2: File upload module) ---
st.header("1. Upload Your Energy Data")
uploaded_file = st.file_uploader("Choose a CSV file (e.g., 'Net Generation of Solar Energy U.csv')", type=["csv"])

df_cleaned = pd.DataFrame() # Initialize df_cleaned outside the if block

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    raw_data_df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview:")
    st.dataframe(raw_data_df.head())

    # Create a temporary file to pass to preprocess_data
    # This is necessary because preprocess_data expects a file path, not a BytesIO object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file_path = tmp_file.name
        raw_data_df.to_csv(tmp_file_path, index=False)

    # Define output paths for intermediate files (these will be generated locally)
    cleaned_data_path = 'cleaned_solar_data.csv'
    forecast_output_path = 'forecast_next_year.csv'
    summary_output_path = 'weekly_summary.txt'
    alerts_output_path = 'alerts_today.csv' # For Zapier integration

    if st.button("🚀 Run Analysis"):
        st.info("Running analysis... This might take a moment.")

        # --- Data Preprocessing (Task 2.1: Clean the Dataset) ---
        st.header("2. Data Preprocessing")
        with st.spinner("Preprocessing raw data..."):
            try:
                df_cleaned = preprocess_data(tmp_file_path, cleaned_data_path)
                st.success(f"Data Preprocessed! Cleaned data saved to `{cleaned_data_path}`.")
                st.subheader("Cleaned Data Preview:")
                st.dataframe(df_cleaned.head())
            except ValueError as e:
                st.error(f"Error during data preprocessing: {e}")
                df_cleaned = pd.DataFrame() # Ensure df_cleaned is empty on error
            except Exception as e:
                st.error(f"An unexpected error occurred during preprocessing: {e}")
                df_cleaned = pd.DataFrame()

        # Remove the temporary raw data file
        os.remove(tmp_file_path)

        if not df_cleaned.empty:
            # --- AI/ML Core Development - Forecasting (Task 5.1: Choose Your First Model) ---
            st.header("3. AI-Powered Forecasting")
            with st.spinner("Training model and forecasting next year's production..."):
                try:
                    forecast_df, trained_model = train_and_predict(cleaned_data_path)
                    forecast_df.to_csv(forecast_output_path, index=False)
                    st.success(f"Forecast Generated! Forecast saved to `{forecast_output_path}`.")
                    st.subheader("Next 12-Month Forecast Preview:")
                    st.dataframe(forecast_df.head())
                except Exception as e:
                    st.error(f"Error during model training and forecasting: {e}")
                    forecast_df = pd.DataFrame() # Ensure forecast_df is empty on error

            # --- Anomaly Detection (Day 2: ANOMALY DETECTION + ALERT LOGIC) ---
            st.header("4. Anomaly Detection")
            with st.spinner("Running anomaly detection..."):
                try:
                    # Pass a copy to avoid modifying original df_cleaned in place if not intended
                    anomalies_df = run_anomaly_detection(df_cleaned.copy())
                    if not anomalies_df.empty:
                        st.success(f"Anomalies detected!")
                        st.subheader("Detected Anomalies:")
                        st.dataframe(anomalies_df[['Month', 'Generation_MWh', 'anomaly']])
                        # Prepare alerts for Zapier (Task 4.1)
                        anomalies_df[['Month', 'Generation_MWh']].to_csv(alerts_output_path, index=False)
                        st.info(f"Anomaly alerts saved to `{alerts_output_path}` for Zapier integration.")
                    else:
                        st.info("No significant anomalies detected in the historical data.")
                except Exception as e:
                    st.error(f"Error during anomaly detection: {e}")
                    anomalies_df = pd.DataFrame()

            # --- Charts Section (Task 1.2: Charts section, Task 2.2: Show Anomalies on Chart) ---
            st.header("5. Visualizing Trends & Anomalies")
            st.write("### Net Generation of Solar Energy Over Time")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df_cleaned['Month'], df_cleaned['Generation_MWh'], label='Historical Generation', color='blue')

            # Plot anomalies if detected
            if not anomalies_df.empty:
                # Ensure anomaly dates are datetime for plotting
                anomalies_df['Month'] = pd.to_datetime(anomalies_df['Month'])
                ax.scatter(anomalies_df['Month'], anomalies_df['Generation_MWh'], color='red', s=100, zorder=5, label='Detected Anomaly') # s is size, zorder ensures it's on top

            ax.set_title('Net Generation of Solar Energy in the United States Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Net Generation (MWh)')
            ax.grid(True)
            ax.legend()

            # Improve date spacing on x-axis
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig) # Display plot in Streamlit
            plt.close(fig) # Close the figure to free memory

            # --- AI-Generated Summary (Task 1.1: Display Summary Output) ---
            st.header("6. AI-Generated Summary")
            if not df_cleaned.empty and not forecast_df.empty:
                with st.spinner("Generating weekly summary..."):
                    try:
                        # Ensure forecast_df is passed with correct datetime types for summary generation
                        forecast_df_for_summary = pd.read_csv(forecast_output_path)
                        forecast_df_for_summary['Month'] = pd.to_datetime(forecast_df_for_summary['Month'])

                        weekly_summary = generate_summary(df_cleaned, forecast_df_for_summary)
                        with open(summary_output_path, 'w') as f:
                            f.write(weekly_summary)
                        st.success(f"Weekly summary saved to `{summary_output_path}`.")
                        st.markdown(f"**Weekly Summary:**\n{weekly_summary}")
                    except Exception as e:
                        st.error(f"Error during summary generation: {e}")
            else:
                st.warning("Cannot generate summary without complete historical and forecast data.")

            st.success("🎉 Analysis complete! Check outputs and generated files.")
        else:
            st.error("Cannot proceed with analysis as data preprocessing failed or resulted in empty data.")

# --- Optional: Footer/Instructions for Zapier ---
st.sidebar.header("Zapier Integration")
st.sidebar.markdown("""
    If anomalies are detected, an `alerts_today.csv` file is generated.
    You can configure a Zapier workflow to monitor this file (e.g., via Google Drive)
    and trigger automated alerts (email, Slack, etc.).
""")
st.sidebar.markdown("---")
st.sidebar.info("Upload your CSV and click 'Run Analysis' to begin!")