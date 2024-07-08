import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv, find_dotenv
from api.agent_api import AgentAPI
from fastapi import FastAPI


load_dotenv(find_dotenv())
agent_api = AgentAPI()

app = FastAPI()


@app.post("/query")
async def query(request: str):
    return agent_api.query(request)
