from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.history import RunnableWithMessageHistory


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch", "generate", "calendar"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore or a generate response or a calendar.",
    )


class Router:
    def __init__(self, model: BaseChatModel, get_chat_history: callable) -> None:
        self.model = model
        structured_llm_router = self.model.with_structured_output(RouteQuery)

        system = """You are an expert at routing a user question to a vectorstore or web search or a generate response. \
        The vectorstore contains documents related to Tech4ai company, describing your division of teams and responsibilities, \
        your mission, vision, values, culture, internal programs, policies remote work, etc. \
        Use the web search for questions about instructions on internal tools, to provide step-by-step tutorials for accessing, using, \
        and/or installing internal company tools such as Github, Vscode, Jira, and Discord. For all else, use web-search. \
        Use calendar the user input is correlated with timetables, the company's agenda, requires any event information
        Use generate in cases where user input has no association with these themes or can be answered using the history of the conversatiom \
        
        User input: {question}
        History: {history}
        """
        self.route_prompt = ChatPromptTemplate.from_template(template=system)

        self.question_router = self.route_prompt | structured_llm_router
        self.question_router_memory = RunnableWithMessageHistory(
            self.question_router,
            lambda session_id: get_chat_history(session_id),
            input_messages_key="question",
            history_messages_key="history",
        )

    def get_chain(self):
        return self.question_router_memory
