from fastapi import FastAPI
from pydantic import BaseModel
from agents.web_research import build_graph

app = FastAPI()
graph = build_graph()


class AskReq(BaseModel):
    question: str


class AskRes(BaseModel):
    answer: str


@app.post("/ask", response_model=AskRes)
def ask(req: AskReq):
    result = graph.invoke({"question": req.question})
    return AskRes(answer=result["final_answer"])
