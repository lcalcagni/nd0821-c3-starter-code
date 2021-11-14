import os, sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

sys.path.insert(1, './starter/ml')
sys.path.append('./starter/starter/ml')

from data import process_data
from model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    dvc_output = subprocess.run(
        ["dvc", "pull", "-r", "s3remote"], capture_output=True, text=True)
    if dvc_output.returncode != 0:
        print("dvc pull failed")
    else:
        os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()
path = Path(__file__).parent.absolute()
print(path, "path")


class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age" : 40,
                "workclass" : "Private",
                "fnlgt" : 193524,
                "education" : "Doctorate", 
                "education_num" : 16, 
                "marital_status" : "Married-civ-spouse",
                "occupation" : "Prof-specialty", 
                "relationship" : "Husband", 
                "race" : "White",
                "sex" : "Male", 
                "capital_gain" : 0, 
                "capital_loss" : 0, 
                "hours_per_week" : 60, 
                "native_country" : "United-States"
            }
        }

class Output(BaseModel):
    prediction: str = "Salary is > 50K/year"

@app.get("/")
async def root():
    return {"Welcome to the salary predictions App"}

@app.post("/predict", response_model=Output, status_code=200)
def get_predicition(payload: Input):
    df = pd.DataFrame.from_dict([payload.dict(by_alias=True)])

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"]

    try:
        model = pd.read_pickle(os.path.join(os.getcwd(), "starter/model/model.pkl"))
        Encoder = pd.read_pickle(os.path.join(os.getcwd(), "starter/model/encoder.pkl"))
        lb_ = pd.read_pickle(os.path.join(os.getcwd(),"starter/model/lb.pkl"))
    except:
        model = pd.read_pickle(os.path.join(path,"model/model.pkl"))
        Encoder = pd.read_pickle(os.path.join(path,"model/encoder.pkl"))
        lb_ = pd.read_pickle(os.path.join(path,"model/lb.pkl"))

    X, y, encoder, lb = process_data(df, categorical_features=cat_features, training=False,encoder=Encoder,lb=lb_)

    prediction = inference(model, X)

    if prediction==1: 
        prediction = "Salary is > 50K/year"
    elif prediction==0: 
        prediction = "Salary is <= 50K/year"

    result = {"prediction": prediction}

    return result
