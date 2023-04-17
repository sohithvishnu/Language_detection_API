from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_pipeline
from model.model import __version__

model_version = __version__
app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    language: str


@app.get("/")
async def root():
    return {"health_check": "ok", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)[0]
    return{"language": language}
