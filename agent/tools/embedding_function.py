from abc import ABC, abstractmethod
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_cohere import CohereEmbeddings


class EmbeddingInterface(ABC):
    """
    Interface for creating a generic embedding function
    """

    @abstractmethod
    def get_embedding_function(self):
        pass


class FastEmbedEmbedding(EmbeddingInterface):
    """
    Embedding function using the FastEmbed model from Qdrant, run locally with quantized model and ONNX runtime
    https://cohere.com/blog/introducing-embed-v3
    https://python.langchain.com/v0.2/docs/integrations/text_embedding/fastembed/
    """

    def get_embedding_function(self):
        model_name = "intfloat/multilingual-e5-large"
        return FastEmbedEmbeddings(model_name=model_name, cache_dir="embeddings_cache")


class CohereEmbedding(EmbeddingInterface):
    """
    Embedding function using the Cohere multilingual model
    https://cohere.com/blog/introducing-embed-v3
    https://python.langchain.com/v0.2/docs/integrations/text_embedding/cohere/
    """

    def get_embedding_function(self):
        return CohereEmbeddings(model="embed-multilingual-v3.0")
