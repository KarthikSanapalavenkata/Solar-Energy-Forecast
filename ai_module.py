# ai_module.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest # New import for anomaly detection
import numpy as np

def preprocess_data(input_path, output_path):
    """
    Loads raw energy data, cleans it, and saves a processed version.

    Args:
        input_path (str): Path to the raw CSV data file.
        output_path (str): Path to save the cleaned CSV data file.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Load data
    df = pd.read_csv(input_path)

    # Ensure 'Month' column exists and is formatted as datetime
    if 'Month' not in df.columns:
        raise ValueError("Missing 'Month' column in input data. Please ensure your CSV has a 'Month' column.")

    # Attempt to convert 'Month' to datetime. Using errors='coerce' will turn unparseable dates into NaT.
    # Specify format for robustness based on common 'YYYY-MM-DD' assumption.
    # If your data uses 'Jan-01' or 'Aug-18', change format to '%b-%y'.
    # If your data uses '20-Aug', change format to '%d-%b' (and consider adding year if ambiguous).
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m-%d', errors='coerce')

    # Drop rows where 'Month' could not be parsed (NaT values)
    df.dropna(subset=['Month'], inplace=True)

    # Ensure 'Generation_MWh' column exists
    if 'Generation_MWh' not in df.columns:
        raise ValueError("Missing 'Generation_MWh' column in input data. Please ensure your CSV has a 'Generation_MWh' column.")

    # Sort by date and drop missing values in 'Generation_MWh'
    df = df.sort_values('Month')
    df = df.dropna(subset=['Generation_MWh'])

    # Save cleaned data
    df.to_csv(output_path, index=False)
    return df

def train_and_predict(cleaned_path):
    """
    Trains a Linear Regression model on cleaned historical data and forecasts
    solar energy generation for the next 12 months.

    Args:
        cleaned_path (str): Path to the cleaned CSV data file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with forecasted months and predicted generation.
            - LinearRegression model: The trained model.
    """
    df = pd.read_csv(cleaned_path)
    # Re-ensure 'Month' is datetime after reading CSV, as .csv save/load loses type
    df['Month'] = pd.to_datetime(df['Month'])

    # Convert dates to ordinal for regression (numerical representation of date)
    df['OrdinalDate'] = df['Month'].map(lambda x: x.toordinal())

    X = df[['OrdinalDate']] # Features for the model
    y = df['Generation_MWh'] # Target variable

    # Train-test split (optional for simple forecasting but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Forecast for the next 12 months
    last_date = df['Month'].max()
    # Generate future dates starting from the beginning of the month after the last historical date
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=12, freq='MS')

    # Convert future dates to ordinal format, matching the training data's feature
    # Ensure future_ordinals is a DataFrame with the same column name as X_train
    future_ordinals = pd.DataFrame(future_dates.map(lambda x: x.toordinal()).values, columns=['OrdinalDate'])

    # Predict future generation
    future_preds = model.predict(future_ordinals)

    # Ensure predicted values are non-negative, as energy generation cannot be negative
    future_preds[future_preds < 0] = 0

    # Create a DataFrame for the forecast results
    forecast_df = pd.DataFrame({
        'Month': future_dates,
        'Predicted_Generation_MWh': future_preds
    })

    return forecast_df, model

def run_anomaly_detection(df):
    """
    Performs anomaly detection on the 'Generation_MWh' column using Isolation Forest.

    Args:
        df (pd.DataFrame): The cleaned DataFrame containing 'Month' and 'Generation_MWh'.

    Returns:
        pd.DataFrame: A DataFrame containing only the detected anomalies,
                      with 'anomaly' column set to True.
    """
    # Ensure 'Month' is datetime for consistency, though not directly used by IsolationForest
    if not pd.api.types.is_datetime64_any_dtype(df['Month']):
        df['Month'] = pd.to_datetime(df['Month'])

    # Initialize Isolation Forest model
    # contamination: The proportion of outliers in the dataset. Adjust based on expected anomaly rate.
    model = IsolationForest(contamination=0.05, random_state=42) # 5% assumed anomalies

    # Fit the model and predict anomalies. -1 for outliers, 1 for inliers.
    # We fit and predict on the 'Generation_MWh' column
    df['anomaly_score'] = model.fit_predict(df[['Generation_MWh']])
    df['anomaly'] = df['anomaly_score'] == -1 # Mark anomalies as True

    # Filter and return only the rows identified as anomalies
    anomalies = df[df['anomaly'] == True].copy()
    return anomalies[['Month', 'Generation_MWh', 'anomaly_score', 'anomaly']]