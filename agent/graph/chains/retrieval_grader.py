from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalChain:
    def __init__(self, model: BaseChatModel):
        self.model = model

        structured_llm_grader = self.model.with_structured_output(GradeDocuments)

        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, and tech4ai company-related questions about their team organization, responsibilities, values, virtues, etc. grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )

        self.retrieval_grader = self.grade_prompt | structured_llm_grader

    def get_chain(self):
        return self.retrieval_grader
