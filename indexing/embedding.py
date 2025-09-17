from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

class EmbeddingModel:

    def __init__(self, embedding_provider: str, embedding_model: str):
        if embedding_provider == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        elif embedding_provider == "openai":
            self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        elif embedding_provider == "ollama":
            self.embedding_model = OllamaEmbeddings(model=embedding_model)
        else:
            raise Exception("Embedding model not supported")

    def get(self):
        return self.embedding_model