from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph.chains.router import Router, RouteQuery
from agent.graph.nodes.node import Node
from agent.graph.state import GraphState


class RouterNode(Node):
    def __init__(
        self, model: BaseChatModel, get_chat_history: callable, config: dict
    ) -> None:
        self.router = Router(model, get_chat_history)
        self.router_chain = self.router.get_chain()
        self.config = config

    def action(self, state: GraphState) -> Dict[str, Any]:
        print("---ROUTER---")

        question = state["question"]

        res = self.router_chain.invoke(
            {"question": question},
            config=self.config,
        )

        route = res.datasource

        return {
            "question": question,
            "route": route,
        }
