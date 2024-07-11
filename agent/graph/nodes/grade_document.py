from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph.chains.retrieval_grader import RetrievalChain
from agent.graph.state import GraphState
from agent.graph.nodes.node import Node


class GradeDocumentsNode(Node):

    def __init__(self, model: BaseChatModel) -> None:
        self.retrieval = RetrievalChain(model)
        self.retrieval_chain = self.retrieval.get_chain()

    def action(self, state: GraphState) -> Dict[str, Any]:
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        query = []

        route = "generate"
        for d in documents:
            score = self.retrieval_chain.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        if len(filtered_docs) == 0:
            print("---NO RELEVANT DOCUMENTS FOUND---")
            query = [question]
            route = "websearch"

        return {
            "documents": filtered_docs,
            "question": question,
            "route": route,
            "queries": query,
        }
