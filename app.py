import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "TruthLens API is running"}

@app.post("/predict")
def predict(article: ArticleRequest):
    features = vectorizer.transform([article.text])
    pred = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]

    return {
        "label": pred,
        "prediction": "TRUE NEWS" if pred == 1 else "FAKE NEWS",
        "confidence": round(float(max(probs)) * 100, 2),
        "probabilities": {
            "fake": round(float(probs[0]) * 100, 2),
            "true": round(float(probs[1]) * 100, 2)
        }
    }
