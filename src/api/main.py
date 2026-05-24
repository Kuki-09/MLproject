from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student math scores",
    version="1.0"
)

# Load model once during startup
predict_pipeline = PredictPipeline()


class StudentInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float


@app.get("/")
def home():

    return {
        "message": "Student Performance Prediction API Running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: StudentInput):

    try:

        custom_data = CustomData(
            gender=data.gender,
            race_ethnicity=data.race_ethnicity,
            parental_level_of_education=data.parental_level_of_education,
            lunch=data.lunch,
            test_preparation_course=data.test_preparation_course,
            reading_score=data.reading_score,
            writing_score=data.writing_score
        )

        pred_df = custom_data.get_data_as_data_frame()

        prediction = predict_pipeline.predict(pred_df)

        return {
            "predicted_math_score": round(float(prediction[0]), 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )