from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

class EmbeddingModel:

    def __init__(self, config: dict[str, str]):
        if config["embedding_provider"] == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model"])
        elif config["embedding_provider"] == "openai":
            self.embedding_model = OpenAIEmbeddings(model=config["embedding_model"], base_url=config["embedding_host"])
        elif config["embedding_provider"] == "ollama":
            self.embedding_model = OllamaEmbeddings(model=config["embedding_model"], base_url=config["embedding_host"])
        else:
            raise Exception("Embedding model not supported")

    def get(self):
        return self.embedding_model