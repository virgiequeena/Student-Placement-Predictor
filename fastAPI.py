from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
# Load the machine learning models
classifier = joblib.load('artifacts/classifier_model.pkl')
regressor  = joblib.load('artifacts/regressor_model.pkl')
le         = joblib.load('artifacts/label_encoder.pkl')

class StudentInput(BaseModel):
    gender: str
    branch: str
    cgpa: float
    tenth_percentage: float
    twelfth_percentage: float
    backlogs: int
    study_hours_per_day: float
    attendance_percentage: float
    projects_completed: int
    internships_completed: int
    coding_skill_rating: int
    communication_skill_rating: int
    aptitude_skill_rating: int
    hackathons_participated: int
    certifications_count: int
    sleep_hours: float
    stress_level: int
    part_time_job: str
    family_income_level: str
    city_tier: str
    internet_access: str
    extracurricular_involvement: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Placement Prediction API"}

@app.post('/predict/classification')

def predict_classification(data: StudentInput):
    features = pd.DataFrame([data.dict()])
    prediction = classifier.predict(features)
    prediction_label = le.inverse_transform(prediction)[0] #to get original labels
   
    return {'prediction': prediction_label}


@app.post('/predict/regression')
def predict_regression(data: StudentInput):
    features = pd.DataFrame([data.dict()])
    salary = regressor.predict(features)[0]

    return {'predicted_salary_lpa': round(float(salary), 2)}


@app.post('/predict/combined')
def predict_combined(data: StudentInput):
    features = pd.DataFrame([data.dict()])

    # 1. Classify
    prediction = classifier.predict(features)
    prediction_label = le.inverse_transform(prediction)[0]

    # 2. If Placed: Regression
    if prediction_label == 'Placed':
        salary = round(float(regressor.predict(features)[0]), 2)
    else:
        salary = None

    return {
        'placement_prediction': prediction_label,
        'predicted_salary_lpa': salary
    }
