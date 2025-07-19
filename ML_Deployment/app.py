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
    
    # Read raw data
    df = pd.read_csv(io.StringIO(await file.read()))

    # Auto-generate engineered features
    today = datetime.now()
    df['Days_Since_Last_Transaction'] = (today - pd.to_datetime(df['TransactionDate'])).dt.days
    df['Days_Since_last_Interaction'] = (today - pd.to_datetime(df['InteractionDate'])).dt.days 
    df['Days_Since_Last_Login'] = (today - pd.to_datetime(df['LastLoginDate'])).dt.days

    # Prepare for prediction (drop unused columns)
    customer_id = df["CustomerID"]

    df = df.drop([
        'CustomerID', 'TransactionDate', 'InteractionDate', 
        'LastLoginDate', 'TransactionID', 'InteractionID',
        'InteractionType', 'ResolutionStatus'
    ], axis=1, errors='ignore')


     # Use your loaded model to predict 
    predictions = loaded_model.predict(df) 

     # Get probability of class 1 
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

    # Read raw data
    df = pd.read_csv(io.StringIO(await file.read()))

    # Auto-generate engineered features
    today = datetime.now()
    df['Days_Since_Last_Transaction'] = (today - pd.to_datetime(df['TransactionDate'])).dt.days
    df['Days_Since_last_Interaction'] = (today - pd.to_datetime(df['InteractionDate'])).dt.days 
    df['Days_Since_Last_Login'] = (today - pd.to_datetime(df['LastLoginDate'])).dt.days


    # Prepare for prediction (drop unused columns)
    customer_id = df["CustomerID"]
    
    df = df.drop([
        'CustomerID', 'TransactionDate', 'InteractionDate', 
        'LastLoginDate', 'TransactionID', 'InteractionID',
        'InteractionType', 'ResolutionStatus'
    ], axis=1, errors='ignore')
    
    # Use your loaded model to predict 
    predictions = loaded_model.predict(df) 

     # Get probability of class 1
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








