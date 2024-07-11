from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

import pprint

from agent.tools.embedding_function import FastEmbedEmbedding, CohereEmbedding
from agent.tools.database import MongoDBAtlasVectorDatabase
from agent.tools.base_content_tool import RetrieverContentTool

load_dotenv()

embedding = CohereEmbedding()
vector_db = MongoDBAtlasVectorDatabase(embedding)
content_manager = RetrieverContentTool(vector_db)
# content_manager.create_database(clear=True)

retriever = content_manager.get_retriever(k=3)


chat = ChatGroq(temperature=0, model="llama3-70b-8192", verbose=True)

custom_prompt_template = """Use the following information to answer the user's question as if you were a conversational agent for a Tech4Humans company, 
willing to help employees with questions about the company and associated technologies. Write the final answer in Brazilian Portuguese

Use this context to answer questions related to the Tech4Humans company and its organization.
If there is information, indicate the pages of the document where the information was found at the end of the answer.
Only use this source of information for company-related questions.

Contexto: {contexto}
Pergunta: {pergunta}
"""

prompt = PromptTemplate(
    template=custom_prompt_template, input_variables=["contexto", "pergunta"]
)

rag_chain = (
    {"contexto": retriever, "pergunta": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

question = "O que o time de HyperAutomation desempenha na organização?"
answer = rag_chain.invoke(question)
print("Question: " + question)
print("Answer: " + answer)

documents = retriever.invoke(question)
print("\nSource documents:")
pprint.pprint(documents)
