from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "Hello from Ala on Render!", "v": "0.2"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"id": item_id, "q": q}
@app.get("/html", response_class=HTMLResponse)
def get_html():
    return "<html><body><h1>Hello, World!</h1></body></html>"