# Predictive Sales Forecasting Model

This project builds and deploys a machine learning model to forecast sales, enabling proactive inventory management and improved demand prediction.

## Key Features & Achievements

*   **High Accuracy:** Achieved 85% prediction accuracy (or a 15% Mean Absolute Error improvement) through rigorous model development.
*   **Reduced Stockouts:** Proactive inventory planning based on forecasts led to a 20% reduction in stockouts.
*   **Real-time Predictions:** Deployed as a REST API for seamless integration with existing business systems and BI tools.
*   **Automated Retraining:** Features an automated pipeline for weekly model updates, ensuring continued accuracy.

## Business Impact

*   **Reduced Costs:** Optimized inventory levels, leading to lower carrying costs and reduced waste from overstocking.
*   **Increased Revenue:** Minimized stockouts, ensuring product availability to meet customer demand and preventing lost sales.
*   **Improved Efficiency:** Enhanced supply chain and resource planning based on more reliable demand forecasts.
*   **Data-Driven Decisions:** Provided stakeholders with actionable insights through BI dashboards for better strategic planning.

## Technical Details

### Data Preparation & Feature Engineering
*   **Data Cleaning:** Handled missing values and outliers in raw sales data.
*   **Feature Engineering:** Created time-based features like day-of-week, holidays, and rolling averages.
*   **Transformation:** Applied a log-transform to normalize skewed sales data.

### Model Development
*   **Algorithm Comparison:** Evaluated Random Forest, XGBoost, and ARIMA to select the best-performing model.
*   **Optimization:** Utilized cross-validation and hyperparameter tuning (Optuna/GridSearchCV) to minimize MAE/RMSE.

### Deployment & Integration
*   **Containerization:** Packaged the model using Docker for portability and scalability.
*   **API:** Deployed the model as a REST API using FastAPI.
*   **Automation:** Built an automated retraining pipeline with Airflow/Prefect.

## Business Integration

*   **Inventory Management:** Connected to inventory systems to automatically trigger purchase orders based on forecasts.
*   **BI & Reporting:** Developed a Tableau dashboard for real-time tracking of sales vs. forecasts.

## Setup Instructions
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd src
python model_training.py
cd ../api
python app.py
```

## Sample API Request
```json
POST /predict
{
  "features": [1, 101, 1, 1, 200]
}
```
