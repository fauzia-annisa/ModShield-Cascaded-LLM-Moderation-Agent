from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
app = FastAPI()
clf = pipeline("text-classification", model="model", top_k=None)
class Req(BaseModel): text: str
@app.post("/moderate")
def moderate(r: Req):
    p = {x["label"]: x["score"] for x in clf(r.text[:1000])[0]}["LABEL_1"]
    action = "remove" if p > 0.9 else "human_review" if p > 0.4 else "allow"
    return {"toxicity": round(p,4), "action": action}
@app.get("/health")
def health(): return {"ok": True}
