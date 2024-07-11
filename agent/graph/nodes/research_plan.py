from typing import Any, Dict

from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent.graph.state import GraphState
from agent.graph.nodes.node import Node
from agent.graph.chains.planner import ResearchPlanChain


class ResearchPlanNode(Node):
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.planner = ResearchPlanChain(model)
        self.plan_chain = self.planner.get_chain()

    def action(self, state: GraphState) -> Dict[str, Any]:
        print("---RESEARCH PLAN---")
        question = state["question"]
        documents = state["documents"]

        queries = self.plan_chain.invoke({"question": question})

        return {
            "documents": documents,
            "question": question,
            "queries": queries.queries,
        }
