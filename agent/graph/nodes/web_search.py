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
        queries = state["queries"]

        for q in queries:
            docs = self.web_search_tool.invoke({"query": q})
            results = ""
            for d in docs:
                url = d["url"]
                content = d["content"][:300]
                results += f"Source: {url} " + content + "\n"
            web_results = Document(page_content=results)
            if documents is not None:
                documents.append(web_results)
            else:
                documents = [web_results]

        return {"documents": documents, "question": question}
