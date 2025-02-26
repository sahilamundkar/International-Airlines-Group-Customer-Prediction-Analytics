from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="IAG Customer Recommendation Predictor",
    description="API for predicting customer recommendations based on IAG data",
    version="1.0.0"
)

# Load the pipeline
try:
    with open('iag_full_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define input data model
class CustomerData(BaseModel):
    iag_business_unit_ug: str
    iag_age_band_auto: str
    iag_tenure_band_enum: str
    iag_site_ug: str
    iag_product_type_auto: str
    iag_region_ug: str
    iag_trust_confidence_scale11: float
    iag_value_price_of_policy_reflects_scale11: float

    class Config:
        schema_extra = {
            "example": {
                "iag_business_unit_ug": "NRMA",
                "iag_age_band_auto": "51-60",
                "iag_tenure_band_enum": "5-10 years",
                "iag_site_ug": "Branch",
                "iag_product_type_auto": "Comprehensive",
                "iag_region_ug": "Metro",
                "iag_trust_confidence_scale11": 8.0,
                "iag_value_price_of_policy_reflects_scale11": 7.0
            }
        }

# Define prediction response model
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    recommendation_likelihood: str

# Define batch input model
class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

# Define batch response model
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

# Prediction endpoint for single customer
@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([customer.dict()])
        
        # Make prediction
        prediction_proba = pipeline.predict_proba(input_data)[0]
        prediction = pipeline.predict(input_data)[0]
        
        # Get probability of the predicted class
        probability = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        # Determine recommendation likelihood
        if prediction == 1:
            likelihood = "Likely to Promote"
        else:
            likelihood = "Likely to be Passive"
        
        return PredictionResponse(
            prediction=str(prediction),
            probability=float(probability),
            recommendation_likelihood=likelihood
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchCustomerData):
    try:
        # Convert batch input to DataFrame
        input_data = pd.DataFrame([customer.dict() for customer in batch.customers])
        
        # Make predictions
        predictions_proba = pipeline.predict_proba(input_data)
        predictions = pipeline.predict(input_data)
        
        # Prepare response
        results = []
        for pred, pred_proba in zip(predictions, predictions_proba):
            probability = pred_proba[1] if pred == 1 else pred_proba[0]
            likelihood = "Likely to Promote" if pred == 1 else "Likely to be Passive"
            
            results.append(PredictionResponse(
                prediction=str(pred),
                probability=float(probability),
                recommendation_likelihood=likelihood
            ))
        
        return BatchPredictionResponse(predictions=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Add model information endpoint
@app.get("/model/info")
async def model_info():
    return {
        "model_type": "Logistic Regression",
        "features": {
            "categorical": [
                "iag_business_unit_ug",
                "iag_age_band_auto",
                "iag_tenure_band_enum",
                "iag_site_ug",
                "iag_product_type_auto",
                "iag_region_ug"
            ],
            "numeric": [
                "iag_trust_confidence_scale11",
                "iag_value_price_of_policy_reflects_scale11"
            ]
        },
        "target": "Likely to recommend (Promote vs Passive)",
        "preprocessing": "Standard scaling for numeric features, One-hot encoding for categorical features"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
