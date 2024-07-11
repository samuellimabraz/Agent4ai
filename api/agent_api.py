import os
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Any
import sqlite3

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.tools.embedding_function import CohereEmbedding
from agent.tools.database import MongoDBAtlasVectorDatabase
from agent.tools.base_content_tool import RetrieverContentTool
from agent.tools.calendar_tool import GoogleCalendarTool

from agent.graph.agent import Agent


class AgentAPI:
    def __init__(self):

        model = ChatGroq(temperature=0.0, model="llama3-8b-8192", verbose=True)

        embedding = CohereEmbedding()
        vector_db = MongoDBAtlasVectorDatabase(embedding)
        content_tool = RetrieverContentTool(vector_db)
        # content_tool.create_database(True)
        retriever = content_tool.get_retriever(k=3)

        search_tool = TavilySearchResults(max_results=2)

        google_tool = GoogleCalendarTool()

        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))

        info_id = google_tool.get_user_info()["id"]

        self.agent = Agent(model, retriever, search_tool, google_tool, memory, info_id)

        self.thread_id = -1

    def query(self, request: str) -> dict[str, Any] | Any:
        question = request
        inputs = {"question": question}

        self.thread_id += 1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}

        return self.agent.graph.invoke(inputs, self.thread, debug=True)

    def clear(self):
        self.agent.clear_memory()
