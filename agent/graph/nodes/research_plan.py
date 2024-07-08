from typing import Any, Dict

from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent.graph.state import GraphState
from agent.graph.nodes.node import Node
from agent.graph.chains.planner import ResearchPlanChain


class ResearchPlanNode(Node):
    def __init__(self, model: BaseChatModel, search_tool: BaseTool):
        self.web_search_tool = search_tool
        self.model = model
        self.planner = ResearchPlanChain(model)
        self.plan_chain = self.planner.get_chain()

    def action(self, state: GraphState) -> Dict[str, Any]:
        print("---RESEARCH PLAN---")
        question = state["question"]
        documents = state["documents"]

        queries = self.plan_chain.invoke({"question": question})

        for q in queries.queries:
            docs = self.web_search_tool.invoke({"query": q})
            web_results = "\n".join([d["content"] for d in docs])
            print(docs[0])
            web_results = Document(page_content=web_results)
            if documents is not None:
                documents.append(web_results)
            else:
                documents = [web_results]

        return {"documents": documents, "question": question}
