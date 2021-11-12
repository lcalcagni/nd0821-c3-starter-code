from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[0] == "Welcome to the salary predictions App"


def test_post_positive():
    input_data = {
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

    r = client.post("/predict", json=input_data)
    assert r.status_code == 200
    assert r.json() == {"prediction": "Salary is > 50K/year"}


def test_post_negative():
    input_data = {
        "age": 23,
        "workclass": "Local-gov",
        "fnlgt": 190709,
        "education": "Assoc-acdm",
        "education_num": 12,
        "marital_status": "Never-married",
        "occupation": "Protective-serv",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 52,
        "native_country": "United-States",
    }

    r = client.post("/predict", json=input_data)
    assert r.status_code == 200
    assert r.json() == {"prediction": "Salary is <= 50K/year"}
