import os
import pprint
from time import sleep

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_core.vectorstores import VectorStoreRetriever

from agent.tools.database import VectorDatabaseInterface, MongoDBAtlasVectorDatabase
from agent.tools.embedding_function import CohereEmbedding, FastEmbedEmbedding


class RetrieverContentTool:

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir os caminhos relativos
    DATA_PATH = os.path.join(current_dir, "data")

    def __init__(self, vector_db: VectorDatabaseInterface):
        self.vdb = vector_db

    def create_database(self, clear=False):
        # Create (or update) the data store.
        documents = self.load_documents()
        print(f"Number of documents: {len(documents)}")
        print(documents[0])

        chunks = self.split_documents(documents)
        chunks_with_id = self.calculate_chunk_ids(chunks)
        print(f"Number of chunks: {len(chunks_with_id)}")
        self.vdb.create_database(chunks_with_id, clear)
        sleep(8)

    def get_retriever(self, search_type="similarity", k=10) -> VectorStoreRetriever:
        return self.vdb.get_vector_store().as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )

    def clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def load_documents(self) -> list[Document]:
        print(self.DATA_PATH)
        document_loader = PyPDFDirectoryLoader(self.DATA_PATH)
        docs = document_loader.load()
        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)
        return docs

    def split_documents(self, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def calculate_chunk_ids(self, chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks


if __name__ == "__main__":
    embedding = CohereEmbedding()
    vector_db = MongoDBAtlasVectorDatabase(embedding)
    content_manager = RetrieverContentTool(vector_db)
    # content_manager.create_database(clear=True)

    retriever = content_manager.get_retriever(k=4)

    print(retriever.__class__.__mro__)

    # question = "O que o time de HyperAutomation desempenha na organização?"
    # print("Question: " + question)

    # documents = retriever.invoke(question)
    # print("\nSource documents:")
    # pprint.pprint(documents)

    # vector_store = vector_db.get_vector_store()
    # results = vector_store.similarity_search_with_score(query=question, k=4)
    # pprint.pprint(results)
