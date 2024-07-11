from typing import Any, Dict

from agent.graph.state import GraphState
from agent.graph.nodes.node import Node
from langchain_core.retrievers import BaseRetriever


class RetrieveNode(Node):
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def action(self, state: GraphState) -> Dict[str, Any]:
        print("---RETRIEVE---")
        question = state["question"]

        documents = self.retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents")

        return {"documents": documents, "question": question}
