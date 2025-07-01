
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def train_and_predict(df_path, num_months_to_predict=12):
    """
    Loads cleaned data, trains a Linear Regression model, and predicts future solar generation.

    Args:
        df_path (str): Path to the cleaned CSV file.
        num_months_to_predict (int): Number of future months to predict.

    Returns:
        pd.DataFrame: DataFrame containing future dates and predicted generation.
        sklearn.linear_model.LinearRegression: The trained model.
    """
    df = pd.read_csv(df_path)
    df['Month'] = pd.to_datetime(df['Month'])
    df['Month_Num'] = (df['Month'].dt.year - df['Month'].min().year) * 12 + df['Month'].dt.month - df['Month'].min().month

    X = df[['Month_Num']]
    y = df['Generation_MWh']

    model = LinearRegression()
    model.fit(X, y)

    last_month_num = df['Month_Num'].max()
    future_month_nums = np.array(range(last_month_num + 1, last_month_num + 1 + num_months_to_predict)).reshape(-1, 1)

    last_date = df['Month'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months_to_predict + 1)]

    future_predictions = model.predict(future_month_nums)

    forecast_df = pd.DataFrame({
        'Month': future_dates,
        'Predicted_Generation_MWh': future_predictions
    })
    return forecast_df, model

def preprocess_data(df_path, output_path='cleaned_solar_data.csv'):
    """
    Loads raw data, cleans it, and saves the cleaned data.
    """
    df = pd.read_csv(df_path)
    df = df.rename(columns={'All utility-scale solar : United States thousand megawatthours': 'Generation_MWh'})
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
    df['Generation_MWh'] = df['Generation_MWh'] * 1000
    df['Generation_MWh'] = df['Generation_MWh'].apply(lambda x: max(0, x))
    df = df.sort_values(by='Month').reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df
