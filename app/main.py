from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()


class ComputeRequest(BaseModel):
    numbers: list[float]


class ComputeResponse(BaseModel):
    count: int
    total: float
    mean: float
    minimum: float
    maximum: float

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


@app.get("/html", response_class=HTMLResponse)
def get_html():
    return "<html><body><h1>Hello, World!</h1></body></html>"