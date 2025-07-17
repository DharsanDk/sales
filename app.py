from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and feature columns
try:
    model = joblib.load('sales_forecasting_model.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    print("Model and features loaded successfully!")
except Exception as e:
    model = None
    feature_columns = None
    print(f"Error loading: {e}")

@app.get("/")
def root():
    return {"message": "Sales Forecasting API is running!"}

@app.post("/predict")
async def predict(request: Request):
    if model is None or feature_columns is None:
        return JSONResponse({"error": "Model or features not loaded"}, status_code=500)
    try:
        data = await request.json()
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return JSONResponse({"error": "Invalid input format"}, status_code=400)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        input_np = input_df.values.astype(np.float32)
        prediction = model.predict(input_np)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500) 