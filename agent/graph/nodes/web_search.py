from typing import Any, Dict

from langchain.schema import Document
from langchain_core.tools import BaseTool

from agent.graph.state import GraphState
from agent.graph.nodes.node import Node


class WebSearchNode(Node):
    def __init__(self, search_tool: BaseTool):
        self.web_search_tool = search_tool

    def action(self, state: GraphState) -> Dict[str, Any]:
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        docs = self.web_search_tool.invoke({"query": question})

        web_results = "\n".join([f'Source: {d["url"]} ' + d["content"] for d in docs])

        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}
