import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pprint import pprint

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
        embedding = CohereEmbedding()
        vector_db = MongoDBAtlasVectorDatabase(embedding)
        content_tool = RetrieverContentTool(vector_db)
        retriever = content_tool.get_retriever(k=3)

        tool = TavilySearchResults(max_results=3, verbose=True)

        model = ChatGroq(temperature=0, model="llama3-70b-8192", verbose=True)

        memory = SqliteSaver.from_conn_string(":memory:")

        go = GoogleCalendarTool()
        info = go.get_user_info()
        pprint(info)
        url_picture = info["picture"]

        self.agent = Agent(model, retriever, tool, go, memory, info["id"])

    def query(self, request: str):
        question = request
        inputs = {"question": question}
        thread = {"configurable": {"thread_id": "1"}}

        logs = []
        final_response = ""

        for output in self.agent.graph.stream(inputs, thread):
            logs.append(output)
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
                pprint(value)

        return {"response": value["generation"], "logs": logs}

    def clear(self):
        self.agent.clear_memory()


if __name__ == "__main__":
    ag = AgentAPI()
    ag.query("Olá, poderia me informar quais são os meus próximos 5 eventos?")
    # ag.clear()
    # ag.query("Você lembra meu nome?")
