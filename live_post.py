import requests
import json

url = "https://census-salary-prediction-app.herokuapp.com/predict"

payload = json.dumps({
  "age": 40,
  "workclass": "Private",
  "fnlgt": 193524,
  "education": "Doctorate",
  "education_num": 16,
  "marital_status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 60,
  "native_country": "United-States"
})
headers = {
  'accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
