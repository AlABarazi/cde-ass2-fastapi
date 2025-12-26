import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

WINE_MODEL_FILENAME = "wine_quality_model.joblib"
WINE_MODEL_PATH = Path(__file__).resolve().parent / WINE_MODEL_FILENAME
WINE_MODEL_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/"
    "MIND-25-26/wine-prediction-demo/main/wine_quality_model.joblib"
)
WINE_MODEL_CACHE_PATH = (
    Path(os.environ.get("WINE_MODEL_CACHE_DIR", "/tmp")) / WINE_MODEL_FILENAME
)
_wine_model = None


def get_wine_model_path() -> Path:
    if WINE_MODEL_PATH.exists():
        return WINE_MODEL_PATH

    if WINE_MODEL_CACHE_PATH.exists():
        return WINE_MODEL_CACHE_PATH

    try:
        urlretrieve(WINE_MODEL_DOWNLOAD_URL, WINE_MODEL_CACHE_PATH)
    except (URLError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing wine model file and download failed. "
                f"Expected at: {WINE_MODEL_PATH} "
                f"or cached at: {WINE_MODEL_CACHE_PATH}. "
                f"Download URL: {WINE_MODEL_DOWNLOAD_URL}. "
                f"Error: {exc}"
            ),
        )

    return WINE_MODEL_CACHE_PATH


def load_wine_model():
    global _wine_model
    if _wine_model is None:
        model_path = get_wine_model_path()
        _wine_model = joblib.load(model_path)

    return _wine_model


class ComputeRequest(BaseModel):
    numbers: list[float]


class ComputeResponse(BaseModel):
    count: int
    total: float
    mean: float
    minimum: float
    maximum: float


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"msg": "Hello from Ala on Render!", "v": "0.2"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"id": item_id, "q": q}


@app.post("/compute")
def compute_stats(payload: ComputeRequest) -> ComputeResponse:
    if not payload.numbers:
        raise HTTPException(status_code=400, detail="numbers must not be empty")

    total = float(sum(payload.numbers))
    count = len(payload.numbers)
    mean = total / count

    return ComputeResponse(
        count=count,
        total=total,
        mean=mean,
        minimum=float(min(payload.numbers)),
        maximum=float(max(payload.numbers)),
    )


@app.post("/predict")
def predict_quality(features: WineFeatures):
    model = load_wine_model()
    data = np.array(
        [
            [
                features.fixed_acidity,
                features.volatile_acidity,
                features.citric_acid,
                features.residual_sugar,
                features.chlorides,
                features.free_sulfur_dioxide,
                features.total_sulfur_dioxide,
                features.density,
                features.pH,
                features.sulphates,
                features.alcohol,
            ]
        ]
    )
    prediction = float(model.predict(data)[0])
    return {"predicted_quality": round(prediction, 2)}


@app.get("/html", response_class=HTMLResponse)
def get_html():
    return "<html><body><h1>Hello, World!</h1></body></html>"