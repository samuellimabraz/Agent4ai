from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import Literal


class CalendarQuery(BaseModel):
    """Define a query to interact with Google Calendar."""

    action: Literal["create", "list", "user_info", "other"] = Field(
        ...,
        description="The action to be performed on the calendar: create an event, list events, or get user info.",
    )
    event_summary: str = Field(
        None, description="The summary of the event to be created."
    )
    start_time: str = Field(
        None,
        description="The start time of the event to be created. Format: 'YYYY-MM-DDTHH:MM:SS-03:00'",
    )
    end_time: str = Field(
        None,
        description="The end time of the event to be created. Format: 'YYYY-MM-DDTHH:MM:SS-03:00'",
    )
    location: str = Field(None, description="The location of the event to be created.")
    description: str = Field(
        None, description="The description of the event to be created."
    )
    max_results: int = Field(5, description="The maximum number of events to list.")
    other_message: str = Field(None, description="A message for other informations.")


class CalendarChain:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model
        structured_llm = self.model.with_structured_output(CalendarQuery)
        system = """You are an assistant that helps users manage their Google Calendar, create events, list events, or get user info, or provide other information about conflicts, problems. Based on the Upcoming events and usei input. 
            To create an event, first check event conflicts, if there are, generate an information message (other). If not provide the event_summary, start time, end time, location, and description. The start and end times should be in the format 'YYYY-MM-DDTHH:MM:SS-03:00'. (create)
            If the upcoming events can resolve the query, provide a message with util information (other)
            For other information, provide a message with the information in other_message (other)
            To list events, provide the maximum number of events to list that are you think necessary (default is 5) (list)
            To get user info, provide no additional information (user_info)"""
        self.calendar_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Context: User Info: \n\n {user_info} \n\n Upcoming Events: {events} \n\n User question: {question}",
                ),
            ]
        )
        self.calendar_chain = self.calendar_prompt | structured_llm

    def get_chain(self):
        return self.calendar_chain
