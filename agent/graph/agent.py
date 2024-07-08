import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from dotenv import load_dotenv
from pprint import pprint

from agent.graph.chains.answer_grader import GradeAnswerChain
from agent.graph.chains.hallucination_grader import HallucinationChain
from agent.graph.chains.router import RouteQuery, Router
from agent.graph.consts import (
    GENERATE,
    GRADE_DOCUMENTS,
    RETRIEVE,
    WEBSEARCH,
    RESEARCH_PLAN,
    ROUTER,
    CALENDAR,
)
from agent.graph.nodes.calendar_node import CalendarNode
from agent.graph.nodes.generate import GenerateNode
from agent.graph.nodes.grade_document import GradeDocumentsNode
from agent.graph.nodes.generate import GenerateNode
from agent.graph.nodes.retrieve import RetrieveNode
from agent.graph.nodes.router_node import RouterNode
from agent.graph.nodes.research_plan import ResearchPlanNode
from agent.graph.nodes.web_search import WebSearchNode
from agent.graph.state import GraphState

from agent.tools.calendar_tool import GoogleCalendarTool

from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph, START


class Agent:
    def __init__(
        self,
        model: BaseChatModel,
        retriever: VectorStoreRetriever,
        search_tool: BaseTool,
        calendar_tool: GoogleCalendarTool,
        checkpointer,
        user_id,
    ) -> None:
        self.model = model

        self.user_id = user_id
        self.config = {"configurable": {"session_id": self.user_id}}

        self.retrieve_node = RetrieveNode(retriever)
        self.grade_docs_node = GradeDocumentsNode(model)
        self.generate_node = GenerateNode(model, self.get_chat_history, self.config)
        self.planner_node = ResearchPlanNode(model, search_tool)
        self.search_node = WebSearchNode(search_tool)
        self.calendar_node = CalendarNode(model, calendar_tool)
        self.router_node = RouterNode(model, self.get_chat_history, self.config)

        self.hallucination_c = HallucinationChain(model)
        self.hallucination_grader = self.hallucination_c.get_chain()
        self.grader_chain = GradeAnswerChain(model)
        self.answer_grader = self.grader_chain.get_chain()

        self._create_graph(checkpointer)

    def clear_memory(self):
        self.generate_node.generator.get_chat_history(self.user_id).clear()

    def _create_graph(self, checkpointer):
        graph = StateGraph(GraphState)
        graph.add_node(ROUTER, self.router_node.action)
        graph.add_node(RETRIEVE, self.retrieve_node.action)
        graph.add_node(GRADE_DOCUMENTS, self.grade_docs_node.action)
        graph.add_node(GENERATE, self.generate_node.action)
        graph.add_node(RESEARCH_PLAN, self.planner_node.action)
        graph.add_node(WEBSEARCH, self.search_node.action)
        graph.add_node(CALENDAR, self.calendar_node.action)

        graph.add_edge(START, ROUTER)
        graph.add_conditional_edges(
            ROUTER,
            self.route_question,
            {
                RESEARCH_PLAN: RESEARCH_PLAN,
                RETRIEVE: RETRIEVE,
                GENERATE: GENERATE,
                CALENDAR: CALENDAR,
            },
        )
        graph.add_edge(RESEARCH_PLAN, WEBSEARCH)
        graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        graph.add_conditional_edges(
            GRADE_DOCUMENTS,
            self.decide_to_generate,
            {
                WEBSEARCH: WEBSEARCH,
                GENERATE: GENERATE,
            },
        )
        graph.add_edge(WEBSEARCH, GENERATE)
        graph.add_conditional_edges(
            GENERATE,
            self.grade_generation_grounded_in_documents_and_question,
            {
                "useful": END,
                "not useful": WEBSEARCH,
            },
        )
        graph.add_edge(CALENDAR, GENERATE)

        self.graph = graph.compile(checkpointer)
        self.graph.get_graph().draw_mermaid_png(
            output_file_path="images\graph_mermaid.png"
        )
        self.graph.get_graph().draw_png(output_file_path="images\graph.png")

    def route_question(self, state: GraphState) -> str:
        print("---ROUTE QUESTION---")

        if state["route"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return RESEARCH_PLAN
        elif state["route"] == "vectordatabase":
            print("---ROUTE QUESTION TO RAG---")
            return RETRIEVE
        elif state["route"] == "generate":
            print("---ROUTE QUESTION TO GENERATE---")
            return GENERATE
        elif state["route"] == "calendar":
            print("---ROUTE QUESTION TO CALENDAR---")
            return CALENDAR

    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")

        if state["route"] == "web_search":
            print(
                "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return WEBSEARCH
        else:
            print("---DECISION: GENERATE---")
            return GENERATE

    def grade_generation_grounded_in_documents_and_question(
        self, state: GraphState
    ) -> str:
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        print("---GRADE GENERATION vs QUESTION---")
        score = self.answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"

    def get_chat_history(self, session_id: str):
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string="mongodb+srv://samuellimabraz:hibana22@cluster0.bo7cqjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            database_name="langchain",
            collection_name="chat_histories",
        )
