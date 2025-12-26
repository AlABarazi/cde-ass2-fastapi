from functools import lru_cache

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


def get_wine_model_path() -> Path:
    """
    Determine the path to the wine quality model file.
    
    Checks for the model in the local directory first, then in the cache.
    Downloads the model if not found locally or in cache.
    
    Returns:
        Path: The path to the model file.
    
    Raises:
        HTTPException: If the model cannot be found or downloaded.
    """
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


@lru_cache(maxsize=1)
def load_wine_model():
    """
    Load the wine quality prediction model.
    
    Caches the loaded model to avoid reloading on each request.
    
    Returns:
        The loaded scikit-learn model.
    """
    model_path = get_wine_model_path()
    return joblib.load(model_path)


class ComputeRequest(BaseModel):
    """
    Request model for computing statistics on a list of numbers.
    
    Attributes:
        numbers (list[float]): The list of numbers to compute statistics for.
    """
    numbers: list[float]


class ComputeResponse(BaseModel):
    """
    Response model for computed statistics.
    
    Attributes:
        count (int): Number of values.
        total (float): Sum of all values.
        mean (float): Average of values.
        minimum (float): Smallest value.
        maximum (float): Largest value.
    """
    count: int
    total: float
    mean: float
    minimum: float
    maximum: float


class WineFeatures(BaseModel):
    """
    Model for wine chemical features used in quality prediction.
    
    Attributes:
        fixed_acidity (float): Fixed acidity level.
        volatile_acidity (float): Volatile acidity level.
        citric_acid (float): Citric acid level.
        residual_sugar (float): Residual sugar level.
        chlorides (float): Chlorides level.
        free_sulfur_dioxide (float): Free sulfur dioxide level.
        total_sulfur_dioxide (float): Total sulfur dioxide level.
        density (float): Wine density.
        pH (float): pH level.
        sulphates (float): Sulphates level.
        alcohol (float): Alcohol content.
    """
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
    """
    Root endpoint returning a greeting message.
    
    Returns:
        dict: Greeting message with version.
    """
    return {"msg": "Hello from Ala on Render!", "v": "0.2"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    """
    Retrieve an item by ID with optional query parameter.
    
    Args:
        item_id (int): The ID of the item to retrieve.
        q (str | None): Optional query string.
    
    Returns:
        dict: Item information.
    """
    return {"id": item_id, "q": q}


@app.post("/compute")
def compute_stats(payload: ComputeRequest) -> ComputeResponse:
    """
    Compute basic statistics for a list of numbers.
    
    Args:
        payload (ComputeRequest): The request containing the numbers.
    
    Returns:
        ComputeResponse: Computed statistics.
    
    Raises:
        HTTPException: If the numbers list is empty.
    """
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
def predict_quality(features: WineFeatures) -> dict[str, float]:
    """
    Predict wine quality based on chemical features.
    
    Args:
        features (WineFeatures): The wine's chemical features.
    
    Returns:
        dict[str, float]: Predicted quality score.
    """
    model = load_wine_model()
    # Convert features to array in correct order
    feature_values = [
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
    data = np.array([feature_values])
    prediction = float(model.predict(data)[0])
    return {"predicted_quality": round(prediction, 2)}


@app.get("/html", response_class=HTMLResponse)
def get_html() -> str:
    """
    Return a simple HTML page.
    
    Returns:
        str: HTML content.
    """
    return "<html><body><h1>Hello, World!</h1></body></html>"