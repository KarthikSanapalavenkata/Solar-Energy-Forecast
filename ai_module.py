import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Ensure date column exists and is formatted
    if 'Month' not in df.columns:
        raise ValueError("Missing 'Month' column in input data")

    # Assuming the format is YYYY-MM-DD based on your last provided format,
    # if it's different, adjust format='%Y-%m-%d' accordingly.
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m-%d')

    # Ensure 'Generation_MWh' column exists
    if 'Generation_MWh' not in df.columns:
        raise ValueError("Missing 'Generation_MWh' column in input data")

    # Sort by date and drop missing values
    df = df.sort_values('Month')
    df = df.dropna(subset=['Generation_MWh'])

    # Save cleaned data
    df.to_csv(output_path, index=False)
    return df


def train_and_predict(cleaned_path):
    df = pd.read_csv(cleaned_path)
    df['Month'] = pd.to_datetime(df['Month'])  # Re-ensure datetime type after reading CSV

    # Convert dates to ordinal for regression
    df['OrdinalDate'] = df['Month'].map(lambda x: x.toordinal())

    X = df[['OrdinalDate']]  # X is already a DataFrame here
    y = df['Generation_MWh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Forecast for the next 12 months
    last_date = df['Month'].max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=12, freq='MS')

    # Fix for UserWarning: X does not have valid feature names
    # Ensure future_ordinals is a DataFrame with the same column name as X_train
    future_ordinals = pd.DataFrame(future_dates.map(lambda x: x.toordinal()).values, columns=['OrdinalDate'])

    future_preds = model.predict(future_ordinals)

    future_preds[future_preds < 0] = 0

    forecast_df = pd.DataFrame({
        'Month': future_dates,
        'Predicted_Generation_MWh': future_preds
    })

    return forecast_df, model