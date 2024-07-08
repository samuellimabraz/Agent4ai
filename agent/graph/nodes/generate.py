from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph.chains.generation import GenerationChain
from agent.graph.nodes.node import Node
from agent.graph.state import GraphState


class GenerateNode(Node):
    def __init__(
        self, model: BaseChatModel, get_chat_history: callable, config: dict
    ) -> None:
        self.generator = GenerationChain(model, get_chat_history)
        self.generation_chain = self.generator.get_chain()
        self.config = config

    def action(self, state: GraphState) -> Dict[str, Any]:
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        generation = self.generation_chain.invoke(
            {"context": documents, "question": question},
            config=self.config,
        )

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
        }
