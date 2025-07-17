import prefect
from prefect import flow, task
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
from datetime import datetime

@task
def load_data():
    df = pd.read_csv('data/sales_data.csv', encoding='latin-1')
    return df

@task
def feature_engineering(df):
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['day_of_week'] = df['ORDERDATE'].dt.dayofweek
    df['day_of_month'] = df['ORDERDATE'].dt.day
    df['month'] = df['ORDERDATE'].dt.month
    df['quarter'] = df['ORDERDATE'].dt.quarter
    df['year'] = df['ORDERDATE'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    holiday_dates = ['2003-12-25', '2004-12-25', '2005-12-25', '2003-07-04', '2004-07-04', '2005-07-04']
    df['is_holiday'] = df['ORDERDATE'].isin(pd.to_datetime(holiday_dates)).astype(int)
    df = df.sort_values('ORDERDATE')
    for lag in [7, 30, 90]:
        df[f'sales_lag_{lag}'] = df.groupby(['CUSTOMERNAME'])['SALES'].shift(lag)
    df['rolling_7day_mean'] = df.groupby(['CUSTOMERNAME'])['SALES'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['rolling_30day_std'] = df.groupby(['CUSTOMERNAME'])['SALES'].transform(lambda x: x.rolling(window=30, min_periods=1).std())
    df['sales_7day_growth'] = df.groupby(['CUSTOMERNAME'])['SALES'].pct_change(periods=7)
    df['sales_30day_growth'] = df.groupby(['CUSTOMERNAME'])['SALES'].pct_change(periods=30)
    customer_stats = df.groupby('CUSTOMERNAME').agg({'SALES': ['mean', 'median', 'max', 'min', 'std']})
    customer_stats.columns = ['cust_' + '_'.join(col).strip() for col in customer_stats.columns.values]
    df = df.merge(customer_stats, on='CUSTOMERNAME', how='left')
    df.fillna({'sales_lag_7': 0, 'sales_lag_30': 0, 'sales_lag_90': 0}, inplace=True)
    return df

@task
def prepare_data(df):
    columns_to_drop = ['SALES', 'ORDERDATE', 'CUSTOMERNAME', 'STATUS', 'PRODUCTLINE', 'COUNTRY', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME', 'DEALSIZE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE']
    X = df.drop(columns=columns_to_drop, axis=1, errors='ignore').select_dtypes(include=np.number)
    joblib.dump(X.columns.tolist(), 'feature_columns.joblib')
    y = df['SALES']
    return X, y

@task
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    model = XGBRegressor(random_state=42, objective='reg:squarederror')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    return model

@task
def save_model(model):
    joblib.dump(model, 'sales_forecasting_model.joblib')
    print("Model saved to sales_forecasting_model.joblib")

@flow
def retraining_flow():
    df = load_data()
    df_engineered = feature_engineering(df)
    X, y = prepare_data(df_engineered)
    model = train_model(X, y)
    save_model(model)

if __name__ == "__main__":
    retraining_flow() 