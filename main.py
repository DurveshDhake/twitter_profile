import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("model.h5")

# Load the pre-trained StandardScaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the schema for the feature vector
class FeatureVector(BaseModel):
    pos: int
    flw: int
    flg: int
    bl: int
    pic: int
    lin: int
    cl: float
    cz: int
    ni: float
    erl: float
    hc: int

# Define the input model containing a list of feature vectors
class PredictionInput(BaseModel):
    feature_vector: List[FeatureVector]

@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Extract the feature vector from the input
    feature_vectors = input_data.feature_vector

    # Convert feature vectors to a numpy array
    data = np.array([[
        fv.pos, fv.flw, fv.flg, fv.bl, fv.pic, fv.lin,
        fv.cl, fv.cz, fv.ni, fv.erl, fv.hc
    ] for fv in feature_vectors])

    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(data)

    # Make predictions using the loaded model
    predictions = model.predict(scaled_data)

    # Return the predictions
    # return {"predictions": predictions.tolist()}
    labels = ["REAL" if pred[0] >= 0.5 else "FAKE" for pred in predictions]
    
    return {"predictions": labels}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)