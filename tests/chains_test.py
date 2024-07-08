import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from pprint import pprint

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from agent.graph.chains.generation import GenerationChain
from agent.graph.chains.hallucination_grader import (
    GradeHallucinations,
    HallucinationChain,
)
from agent.graph.chains.retrieval_grader import GradeDocuments, RetrievalChain
from agent.graph.chains.router import RouteQuery, Router
from agent.graph.chains.planner import ResearchPlanChain, Queries
from agent.tools.embedding_function import CohereEmbedding
from agent.tools.database import MongoDBAtlasVectorDatabase
from agent.tools.base_content_tool import RetrieverContentTool


def test_generation_chain(retriever, generation_chain) -> None:
    question = "Quais são os valores e virtudes da tech4ai?"
    docs = retriever.invoke(question)
    print("\nSource documents:")
    pprint(docs)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_planner_chain(question_router, planner_chain) -> None:
    question = "O que é a ferramenta Jira?"

    res: RouteQuery = question_router.invoke({"question": question})
    print(res.datasource)
    assert res.datasource == "websearch"

    res: Queries = planner_chain.invoke({"question": question})
    pprint(res)


def test_retrival_grader_answer_yes(retriever, retrieval_grader) -> None:
    question = "Quais são os valores e virtudes da tech4ai?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print("\nSource document:")
    pprint(doc_txt)

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    print(res.binary_score)
    assert res.binary_score == "yes"


def test_retrival_grader_answer_no(retriever, retrieval_grader) -> None:
    question = "O que é GitHub?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print("\nSource document:")
    pprint(doc_txt)

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    print(res.binary_score)
    assert res.binary_score == "no"


def test_hallucination_grader_answer_yes(
    retriever, generation_chain, hallucination_grader
) -> None:
    question = "O que o time de HyperAutomation desempenha na organização?"
    docs = retriever.invoke(question)
    print("\nSource documents:")
    pprint(docs)

    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    print(res.binary_score)
    assert res.binary_score


def test_hallucination_grader_answer_no(retriever, hallucination_grader) -> None:
    question = "O que o time de HyperAutomation desempenha na organização?"
    docs = retriever.invoke(question)
    print("\nSource documents:")
    pprint(docs)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "Para fazer pizza precisamos primeiro começar pela massa",
        }
    )
    print(res.binary_score)
    assert not res.binary_score


def test_router_to_vectorstore(question_router) -> None:
    question = "O que o time de RH da tech4ai faz?"

    res: RouteQuery = question_router.invoke({"question": question})
    print(res.datasource)
    assert res.datasource == "vectorstore"


def test_router_to_websearch(question_router) -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    print(res.datasource)
    assert res.datasource == "websearch"


def main():
    load_dotenv()

    embedding = CohereEmbedding()
    vector_db = MongoDBAtlasVectorDatabase(embedding)
    content_tool = RetrieverContentTool(vector_db)
    retriever = content_tool.get_retriever(k=4)

    model = ChatGroq(temperature=0, model="llama3-70b-8192", verbose=True)
    router = Router(model)
    question_router = router.get_chain()

    g = GenerationChain(model)
    generation_chain = g.get_chain()

    r = RetrievalChain(model)
    retrieval_grader = r.get_chain()

    h = HallucinationChain(model)
    hallucination_grader = h.get_chain()

    p = ResearchPlanChain(model)
    planner_chain = p.get_chain()

    # test_router_to_websearch(question_router)
    # test_router_to_vectorstore(question_router)
    # test_hallucination_grader_answer_no(retriever, hallucination_grader)
    # test_hallucination_grader_answer_yes(
    #     retriever, generation_chain, hallucination_grader
    # )
    # test_retrival_grader_answer_no(retriever, retrieval_grader)
    # test_retrival_grader_answer_yes(retriever, retrieval_grader)
    # test_generation_chain(retriever, generation_chain)
    test_planner_chain(question_router, planner_chain)


if __name__ == "__main__":
    main()
