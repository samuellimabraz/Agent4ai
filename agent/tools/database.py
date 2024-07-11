from abc import ABC, abstractmethod
import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.schema.document import Document

from agent.tools.embedding_function import EmbeddingInterface


class VectorDatabaseInterface(ABC):
    @abstractmethod
    def create_database(self, docs: list, clear: bool = False):
        pass

    @abstractmethod
    def get_vector_store(self):
        pass


class MongoDBAtlasVectorDatabase(VectorDatabaseInterface):
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_CONNECTION_STRING")
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    def __init__(self, embedding: EmbeddingInterface) -> None:
        self.client = MongoClient(self.MONGODB_ATLAS_CLUSTER_URI)
        print("Connected to MongoDB Atlas")

        self.dbName = "langchain"
        self.collectionName = "multilingual"
        self.collection = self.client[self.dbName][self.collectionName]
        self.embedding = embedding.get_embedding_function()

        print("Creating VectorSearch")
        self.vector_store = MongoDBAtlasVectorSearch(
            embedding=self.embedding,
            collection=self.collection,
            index_name=self.ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )

    def create_database(self, docs: list[Document], clear: bool = False):
        if clear:
            print("Dropping collection")
            self.collection.delete_many({})

        print("Inserting documents")
        self.vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=self.embedding,
            collection=self.collection,
            index_name=self.ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )
        print("Database created")

    def get_vector_store(self) -> MongoDBAtlasVectorSearch:
        return self.vector_store
