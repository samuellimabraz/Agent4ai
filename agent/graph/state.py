from typing import List, TypedDict, TypedDict, Annotated
from langchain_core.messages import AnyMessage
import operator


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    route: str
    documents: List[str]
    queries: List[str]
