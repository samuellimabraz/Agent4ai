from typing import TypedDict, Annotated, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool


class Queries(BaseModel):
    queries: List[str]


class ResearchPlanChain:
    def __init__(self, model: BaseChatModel):
        self.model = model

        structured_llm = self.model.with_structured_output(Queries)

        research_plan_prompot = """You are a researcher planning a search for information on a topic. \n
            Provide a list of 3 best queries to search for information on the topic."""
        self.research_plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", research_plan_prompot),
                ("human", "User Question: {question}"),
            ]
        )

        self.research_plan = self.research_plan_prompt | structured_llm

    def get_chain(self):
        return self.research_plan
