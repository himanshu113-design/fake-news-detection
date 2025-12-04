from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, os

app = FastAPI(title="Fake News Detection API – Option A")

# Enable frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained TF-IDF + Logistic Regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tfidf_lr.joblib")
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Backend is running!"}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        return {"error": "No text provided"}

    # Predict probabilities
    probas = model.predict_proba([text])[0]
    fake_score = float(probas[0])
    real_score = float(probas[1])
    prediction = "REAL" if real_score >= fake_score else "FAKE"

    return {
        "prediction": prediction,
        "fake_score": fake_score,
        "real_score": real_score
    }
