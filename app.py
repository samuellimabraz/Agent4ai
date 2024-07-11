from dotenv import load_dotenv, find_dotenv
from api.agent_api import AgentAPI
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class QueryRequest(BaseModel):
    request: str


load_dotenv(find_dotenv())

agent_api = AgentAPI()

app = FastAPI()


@app.post("/query")
async def query(body: QueryRequest):
    return agent_api.query(body.request)


@app.post("/clear")
async def clear():
    agent_api.clear()
    return "Memory cleared"
