from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.schema import Document

from agent.graph.chains.calendar import CalendarChain, CalendarQuery
from agent.graph.nodes.node import Node
from agent.graph.state import GraphState
from agent.tools.calendar_tool import GoogleCalendarTool


class CalendarNode(Node):
    def __init__(self, model: BaseChatModel, calendar_tool: GoogleCalendarTool) -> None:
        self.calendar_chain = CalendarChain(model).get_chain()
        self.calendar_tool = calendar_tool

    def action(self, state: GraphState) -> Dict[str, Any]:
        print("---CALENDAR---")
        question = state["question"]
        documents = state["documents"]

        user_info = self.calendar_tool.get_user_info()
        events = self.calendar_tool.list_events(max_results=10)
        context = {"question": question, "user_info": user_info, "events": events}
        response: CalendarQuery = self.calendar_chain.invoke(context)

        action = response["action"]

        if action == "create":
            event_summary = response["event_summary"]
            start_time = response["start_time"]
            end_time = response["end_time"]
            location = response["location"]
            description = response["description"]

            result = self.calendar_tool.create_event(
                event_summary,
                start_time,
                end_time,
                location,
                description,
            )
        elif action == "list":
            max_results = response["max_results"]
            result = self.calendar_tool.list_events(max_results)
        elif action == "user_info":
            result = self.calendar_tool.get_user_info()
        elif action == "other":
            result = {"message": response["other_message"]}

        doc = "\n".join([f"{key}: {value}" for key, value in result.items()])

        result = Document(page_content=doc)
        if documents is not None:
            documents.append(result)
        else:
            documents = [result]

        return {"question": question, "documents": documents}
