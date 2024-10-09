# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np

# # Load the trained Random Forest model
# model = joblib.load("customer_default_model.pkl")

# # Initialize FastAPI app
# app = FastAPI()

# # Define the input data structure for the API
# class InputData(BaseModel):
#     features: list[float]

# # Define the prediction endpoint
# @app.post("/predict/")
# async def predict(data: InputData):
#     # Convert the list of input features into a numpy array and reshape for a single prediction
#     input_features = np.array([data.features])
    
#     # Ensure the input has 204 features
#     if input_features.shape[1] != 204:
#         return {"error": "The input must contain exactly 204 features."}
    
#     # Make the prediction using the loaded model
#     prediction = model.predict(input_features)
    
#     # Return the result as a JSON response
#     return {"prediction": int(prediction[0])}

import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained Random Forest model
model = joblib.load("customer_default_model.pkl")

# Load feature names
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define the input data structure for the API
class InputData(BaseModel):
    features: list[float]

# Define the prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    # Convert the list of input features into a numpy array and reshape for a single prediction
    input_features = np.array([data.features])

    # Ensure the input has 204 features
    if input_features.shape[1] != 204:
        return {"error": "The input must contain exactly 204 features."}

    # Convert to DataFrame with feature names
    input_features_df = pd.DataFrame(input_features, columns=feature_names)

    # Make the prediction using the loaded model
    prediction = model.predict(input_features_df)

    # Return the result as a JSON response
    return {"prediction": int(prediction[0])}


