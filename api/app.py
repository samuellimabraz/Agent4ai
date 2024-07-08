import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv, find_dotenv
from api.agent_api import AgentAPI
from fastapi import FastAPI
from fastapi.responses import JSONResponse


load_dotenv(find_dotenv())

agent_api = AgentAPI()

app = FastAPI()


@app.post("/query")
async def query(request: str):
    response = agent_api.query(request)
    return JSONResponse(content=response)


@app.post("/clear")
async def clear():
    agent_api.clear()
    return JSONResponse(content={"status": "Memory cleared"})
