import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from typing import List
from utils import mask_pii

MODEL_PATH = 'data/model/classifier.pkl'
pipeline = load(MODEL_PATH)

app = FastAPI()

class EmailRequest(BaseModel):
    email_body: str

class Entity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[Entity]
    masked_email: str
    category_of_the_email: str

@app.post("/classify", response_model=EmailResponse)
def classify_email(req: EmailRequest):
    try:
        masked, entities = mask_pii(req.email_body)
        pred = pipeline.predict([masked])[0]
        return EmailResponse(
            input_email_body=req.email_body,
            list_of_masked_entities=entities,
            masked_email=masked,
            category_of_the_email=pred
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def greet_json():
    return {"Status": "Running"}