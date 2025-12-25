from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "Hello from Ala on Render!", "v": "0.2"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"id": item_id, "q": q}
