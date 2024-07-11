from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


def main() -> None:
    load_dotenv()

    tool = TavilySearchResults(max_results=2)
    model = ChatGroq(temperature=0, model="llama3-70b-8192", verbose=True)
    memory = SqliteSaver.from_conn_string(":memory:")
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
        You are allowed to make multiple calls (either together or in sequence). \
        Only look up information when you are sure of what you want. \
        If you need to look up some information before asking a follow up question, you are allowed to do that!
        """
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)

    abot.graph.get_graph().draw_png(output_file_path="graph.png")

    # messages = [HumanMessage(content="What is GitHub?")]
    # thread = {"configurable": {"thread_id": "1"}}
    # for event in abot.graph.stream({"messages": messages}, thread):
    #     for v in event.values():
    #         print(v)


if __name__ == "__main__":
    main()
