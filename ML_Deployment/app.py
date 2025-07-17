#%%

import pickle
import io
import csv
import numpy as np 
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

#%%

with open('stacking_model.pkl', 'rb') as f: 
    loaded_model = pickle.load(f)

app = FastAPI()    


#%%
# create a simple root endpoint
@app.get("/")
def root():
    return {"message": "Customer churn prediction API is running."}

#%%

#Add a prediction endpoint to receive files and return predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)): 
    # Read the uploaded file into a pandas Dataframe 
    contents = await file.read() 
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # store CustomerID variable separately 
    customer_id = df["CustomerID"] 

    # Drop CustomerID 
    df = df.drop(columns='CustomerID') 
    
    # Use your loaded model to predict 
    predictions = loaded_model.predict(df) 

     # Get probability of class 1 (i.e., churn = 1)
    prediction_probibility = loaded_model.predict_proba(df)[:, 1]

    results = [
        {"CustomerID": str(cid), 
         "ChurnStatus": int(pred), 
         "ChurnProbability": round(float(proba), 3), 
         "Risk_Level": (
             "High" if proba > 0.7
             else "Medium" if proba > 0.4 
             else "low"
         )
         }
        for cid, pred, proba in zip(customer_id, predictions, prediction_probibility)
    ]

    return {"predictions": results}

#Add a prediction endpoint to receive files and return csv prediction 
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)): 
    # Read the uploaded file into a pandas Dataframe 
    contents = await file.read() 
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # store CustomerID variable separately 
    customer_id = df["CustomerID"] 

    # Drop CustomerID 
    df = df.drop(columns='CustomerID') 
    
    # Use your loaded model to predict 
    predictions = loaded_model.predict(df) 

     # Get probability of class 1 (i.e., churn = 1)
    prediction_probibility = loaded_model.predict_proba(df)[:, 1]

    results = [
        {"CustomerID": str(cid), 
         "ChurnStatus": int(pred), 
         "ChurnProbability": round(float(proba), 3), 
         "Risk_Level": (
             "High" if proba > 0.7
             else "Medium" if proba > 0.4 
             else "low"
         )
         }
        for cid, pred, proba in zip(customer_id, predictions, prediction_probibility)
    ]

    # create csv in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    output.seek(0)

    # Return the predictions as a list 
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})








