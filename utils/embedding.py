from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingModel:

    def __init__(self, config: dict[str, str]):
        if config["embedding_provider"] == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model"])
        elif config["embedding_provider"] == "openai":
            self.embedding_model = OpenAIEmbeddings(model=config["embedding_model"])
        elif config["embedding_provider"] == "ollama":
            self.embedding_model = OllamaEmbeddings(model=config["embedding_model"])
        elif config["embedding_provider"] == "google":
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=config["embedding_model"])
        else:
            raise NotImplementedError(f"Embedding provider {config["embedding_provider"]} not supported")

    def get(self):
        return self.embedding_model