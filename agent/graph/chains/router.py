from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.history import RunnableWithMessageHistory


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch", "generate", "calendar"] = Field(
        "generate",
        description="Given a user question choose to route it to websearch or a vectorstore or a generate response or a calendar.",
    )


class Router:
    def __init__(self, model: BaseChatModel, get_chat_history: callable) -> None:
        self.model = model
        structured_llm_router = self.model.with_structured_output(RouteQuery)
        self.chat_history = get_chat_history
        system = """You are an expert at routing a user question to a vectorstore or websearch or a calendar or a generate response, based on the user's input and history chat. \
        Use vectorstore for inputs related the Tech4ai, tech4ai company, describing your division of teams and responsibilities, your mission, vision, values, culture, skills, internal programs, policies remote work, etc. \
        Use calendar if the user input requires information about user data, email, new events that have not been previously mentioned, is correlated with timetables, the company's agenda
        Use websearch for questions about instructions on internal tools and frameworks, to provide step-by-step tutorials for accessing, using, and/or installing internal company tools such as Github, Vscode, Jira, Discord and others frameworks. \
        Use generate in cases can be answered using the history of the conversation, or the input has no association with others themes. \
        \n\n
        User input in Portuguese: {question}
        \n\n
        History: {history}
        """
        self.route_prompt = ChatPromptTemplate.from_template(template=system)

        self.question_router = self.route_prompt | structured_llm_router
        self.question_router_memory = self.get_chain()

    def get_chain(self):
        return RunnableWithMessageHistory(
            self.question_router,
            lambda session_id: self.chat_history(session_id),
            input_messages_key="question",
            history_messages_key="history",
        )
