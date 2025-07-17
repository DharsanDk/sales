import pandas as pd

df = pd.read_csv('data/sales_data.csv', encoding='latin-1')
display(df.head())
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())

print("\nData types:")
print(df.dtypes)
# Initial inspection
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())

print("\nData types:")
print(df.dtypes)
# Data quality check
print("\nMissing values per column:")
print(df.isnull().sum())
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df['YEAR'] = df['ORDERDATE'].dt.year
df['MONTH'] = df['ORDERDATE'].dt.month
df['DAY'] = df['ORDERDATE'].dt.day
df['DAY_OF_WEEK'] = df['ORDERDATE'].dt.dayofweek

display(df[['ORDERDATE', 'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK']].head())
import numpy as np
from datetime import datetime

# 1. Time-based feature extraction
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df['day_of_week'] = df['ORDERDATE'].dt.dayofweek  # Monday=0, Sunday=6
df['day_of_month'] = df['ORDERDATE'].dt.day
df['month'] = df['ORDERDATE'].dt.month
df['quarter'] = df['ORDERDATE'].dt.quarter
df['year'] = df['ORDERDATE'].dt.year
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 2. Holiday and special event flags (example - customize for your business)
holiday_dates = ['2003-12-25', '2004-12-25', '2005-12-25', '2003-07-04', '2004-07-04', '2005-07-04']  # Add your relevant dates
df['is_holiday'] = df['ORDERDATE'].isin(pd.to_datetime(holiday_dates)).astype(int)

# 3. Lag features (using 7, 30, 90 day lags as examples)
df = df.sort_values('ORDERDATE')
for lag in [7, 30, 90]:
    df[f'sales_lag_{lag}'] = df.groupby(['CUSTOMERNAME'])['SALES'].shift(lag)

# 4. Rolling statistics (7-day and 30-day windows)
df['rolling_7day_mean'] = df.groupby(['CUSTOMERNAME'])['SALES'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean())
df['rolling_30day_std'] = df.groupby(['CUSTOMERNAME'])['SALES'].transform(
    lambda x: x.rolling(window=30, min_periods=1).std())

# 5. Growth rate features
df['sales_7day_growth'] = df.groupby(['CUSTOMERNAME'])['SALES'].pct_change(periods=7)
df['sales_30day_growth'] = df.groupby(['CUSTOMERNAME'])['SALES'].pct_change(periods=30)

# 6. Customer-specific aggregations
customer_stats = df.groupby('CUSTOMERNAME').agg({
    'SALES': ['mean', 'median', 'max', 'min', 'std']
})
customer_stats.columns = ['cust_' + '_'.join(col).strip() for col in customer_stats.columns.values]
df = df.merge(customer_stats, on='CUSTOMERNAME', how='left')

# 7. Handle missing values from lag features
df.fillna({'sales_lag_7': 0, 'sales_lag_30': 0, 'sales_lag_90': 0}, inplace=True)

# Show new feature set
display(df[['ORDERDATE', 'SALES', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'is_weekend', 'is_holiday', 'sales_lag_7', 'sales_lag_30', 'sales_lag_90', 'rolling_7day_mean', 'rolling_30day_std', 'sales_7day_growth', 'sales_30day_growth', 'cust_SALES_mean', 'cust_SALES_median', 'cust_SALES_max', 'cust_SALES_min', 'cust_SALES_std']].head())
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

# Split data
# Assuming X and y are already created from the df with engineered features
# If not, you would need to re-run the feature engineering step or select columns from the updated df.

# Let's re-create X and y from the df to be sure they include all engineered features
# Define columns to drop - including non-numeric and the target variable
columns_to_drop = ['SALES', 'ORDERDATE', 'CUSTOMERNAME', 'STATUS', 'PRODUCTLINE', 'COUNTRY', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME', 'DEALSIZE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE']

# Create features X by dropping the specified columns from the original dataframe
# Using .copy() to avoid SettingWithCopyWarning
X = df.drop(columns=columns_to_drop, axis=1, errors='ignore').copy()

# Ensure all columns in X are numeric. Drop any remaining non-numeric columns if necessary.
# This is a more robust check in case other non-numeric columns exist.
# Let's explicitly select only numeric types after dropping known columns
X = X.select_dtypes(include=np.number) # Assuming np is imported earlier

import joblib
joblib.dump(X.columns.tolist(), 'feature_columns.joblib')

# Assign the target variable
y = df['SALES']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with XGBoost and hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}
model = XGBRegressor(random_state=42, objective='reg:squarederror')
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Evaluate
predictions = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, predictions)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}")
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is imported if not already
import numpy as np # Ensure numpy is imported

# Separate features and target
# Define columns to drop - including non-numeric and the target variable
columns_to_drop = ['SALES', 'ORDERDATE', 'CUSTOMERNAME', 'STATUS', 'PRODUCTLINE', 'COUNTRY', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME', 'DEALSIZE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE']

# Create features X by dropping the specified columns from the original dataframe
# Using .copy() to avoid SettingWithCopyWarning
X = df.drop(columns=columns_to_drop, axis=1, errors='ignore').copy()

# Ensure all columns in X are numeric. Drop any remaining non-numeric columns if necessary.
# This is a more robust check in case other non-numeric columns exist.
# Let's explicitly select only numeric types after dropping known columns
X = X.select_dtypes(include=np.number)

# Assign the target variable
y = df['SALES']

# Print dtypes of X to verify only numeric columns are present
print("Data types of features (X) before training:")
print(X.dtypes)

# Handle any potential remaining missing values in X after dropping columns
# Using mean imputation as a simple strategy
X = X.fillna(X.mean())

# Convert X to numpy array of float type before fitting
X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32)


# Train model to get feature importance
model.fit(X_np, y_np) # Fit using numpy arrays

# Plot feature importance
# Use X.columns as index since X_np is just array values
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh')
plt.title('Feature Importance')
plt.show()
# 1. Worst predictions analysis
error_df = pd.DataFrame({
    'actual': y_test,
    'predicted': predictions,
    'error': abs(y_test - predictions)
}).sort_values('error', ascending=False)

# Examine top 10 worst predictions
display(error_df.head(10))

# 2. Temporal error patterns
error_df['date'] = df.loc[y_test.index, 'ORDERDATE']
error_df.set_index('date')['error'].resample('W').mean().plot()
# 5. Temporal analysis (assuming there's a date column)
# Convert 'ORDERDATE' to datetime objects
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

print("\nDate range:", df['ORDERDATE'].min(), "to", df['ORDERDATE'].max())

# Monthly sales aggregation example
# Ensure 'ORDERDATE' is the index for resample
df_time = df.set_index('ORDERDATE')
monthly_sales = df_time.resample('ME')['SALES'].sum()

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
monthly_sales.plot(title='Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()
import joblib

# Save the trained model
joblib.dump(model, 'sales_forecasting_model.joblib')

print("Model saved to sales_forecasting_model.joblib")
# Import necessary libraries
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd # Import pandas as it's needed for dataframe operations

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
# Make sure the model file 'sales_forecasting_model.joblib' exists in the same directory
try:
    model = joblib.load('sales_forecasting_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get data from the request
        data = request.get_json(force=True)

        # Convert the received data into a format the model can understand
        # This part needs to match the exact feature structure used during training
        # Assuming the input data is a dictionary where keys are feature names
        # and values are the corresponding feature values.
        # For simplicity, we'll create a DataFrame from the input data.
        # In a real application, you'd need robust input validation and data transformation.

        # Example: Convert single prediction request data to DataFrame row
        # If receiving multiple instances in a list, iterate and predict for each
        if isinstance(data, dict):
            # Assuming a single instance prediction
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
             # Assuming multiple instances prediction
            input_df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input data format. Expected dictionary or list of dictionaries.'}), 400


        # Ensure the input DataFrame has the same columns as the training data features (X)
        # and in the same order. This is a critical step.
        # You might need to store the list of feature columns from your training data (X.columns)
        # and reindex the input_df accordingly, handling potential missing columns (e.g., filling with default values or raising errors).
        # For this example, we'll assume the input data dictionary keys match the expected feature names.

        # Need to make sure the column names and order match X from training
        # A robust API would include checks and potentially reordering/handling missing features
        # For demonstration, assuming input data keys match the required features.
        # If you saved X.columns during training, load it here and use it to reindex.
        # Example (assuming X_columns is available):
        # input_df = input_df.reindex(columns=X_columns, fill_value=0) # Or appropriate fill_value

        # Convert DataFrame to numpy array for prediction
        # Ensure data types match the training data
        input_np = input_df.values.astype(np.float32) # Assuming float32 was used in training


        # Make prediction
        prediction = model.predict(input_np)

        # Return the prediction
        # Convert numpy array output to a list or appropriate JSON format
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# To run the app (for development)
# if __name__ == '__main__':
#     app.run(debug=True)

# Note: For production deployment, use a production-ready WSGI server like Gunicorn or uWSGI.
%%writefile requirements.txt
pandas==1.3.4
numpy==1.21.2
scikit-learn==1.0.1
xgboost==1.5.0
Flask==2.0.2
gunicorn==20.1.0

%%writefile Dockerfile
# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the working directory
COPY app.py .
COPY sales_forecasting_model.joblib . # Copy the saved model

# Expose the port that the app runs on
EXPOSE 5000

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# Build the Docker image
!docker build -t sales-forecasting-api .

# Run the Docker container, mapping port 5000
!docker run -p 5000:5000 sales-forecasting-api

# 1. Components of an automated retraining pipeline:
print("Automated Retraining Pipeline Components:")
print("- Data Ingestion: Collecting new sales data from source systems (e.g., database, data lake).")
print("- Data Preprocessing: Applying the same cleaning and feature engineering steps used for initial training (handling missing values, outliers, creating temporal, lag, and rolling features, customer aggregations).")
print("- Model Training: Training a new model version using the updated, preprocessed dataset.")
print("- Model Evaluation: Evaluating the performance of the new model version on a validation set or using cross-validation to ensure it meets performance criteria (e.g., MAE, RMSE).")
print("- Model Saving: Saving the newly trained and validated model version (e.g., using joblib, pickle, or a model registry).")
print("- Model Deployment/Updating API: Replacing the currently deployed model in the API with the new version, ideally with minimal downtime.")

# 2. How an orchestration tool (Airflow/Prefect) manages the workflow:
print("\nOrchestration with Airflow/Prefect:")
print("- Directed Acyclic Graph (DAG): Define the pipeline as a series of tasks with dependencies (e.g., preprocess depends on ingest, train depends on preprocess, evaluate depends on train, deploy depends on evaluate).")
print("- Task Management: Each component (ingestion, preprocessing, training, etc.) is a distinct task in the DAG.")
print("- Scheduling: Trigger the pipeline run at scheduled intervals (e.g., daily, weekly).")
print("- Monitoring & Logging: Track the execution status of each task, log errors, and set up alerts for failures.")
print("- Dependency Management: Ensure tasks run in the correct order (e.g., don't train before data is preprocessed).")
print("- Retries: Configure tasks to automatically retry on failure.")
print("- Parameterization: Pass configuration (e.g., data source paths, model hyperparameters) to the pipeline run.")

# 3. Potential triggers for retraining:
print("\nRetraining Triggers:")
print("- Scheduled Intervals: Regular retraining (e.g., weekly or monthly) to incorporate new data.")
print("- Data Drift Detection: Monitoring changes in the distribution of input features or the target variable. If significant drift is detected, trigger retraining.")
print("- Performance Degradation: Monitoring the model's performance (e.g., MAE on recent data). If performance drops below a threshold, trigger retraining.")
print("- Manual Trigger: Allowing manual initiation of the retraining process when needed.")
print("- Code Changes: Triggering a retraining process when the model training code or feature engineering logic is updated.")

# 4. Deployment strategies for updating the API:
print("\nModel Deployment Strategies (Minimizing Downtime):")
print("- Rolling Updates: Gradually replace old instances of the API with new ones running the updated model. This is often managed by container orchestration platforms (Kubernetes, Docker Swarm).")
print("- Blue/Green Deployment: Deploy the new version of the API (Green) alongside the old version (Blue). Traffic is gradually shifted from Blue to Green. Once all traffic is on Green and validated, Blue is decommissioned. This allows for quick rollback if issues occur.")
print("- Canary Release: Roll out the new model version to a small subset of users or traffic first. If successful, gradually increase the rollout percentage.")
print("- Atomic Swaps: In simpler setups, this might involve atomically replacing the model file that the API loads, followed by a graceful restart of the API process if necessary (requires careful handling to avoid interrupting ongoing requests).")

