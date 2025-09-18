from indexing.vectorstore import VectorStore
from typing import List, Callable
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from pathlib import Path
from indexing.embedding import EmbeddingModel

def get_tools(vector_store_type: str, vector_store_dir: str, k: int, emb_provider: str, emb_name: str, emb_host: str) -> List[Tool]:
    tools = []
    embedding_model = EmbeddingModel(emb_provider, emb_name, emb_host).get()
    store_dir = Path(vector_store_dir)
    for dir in store_dir.iterdir():
        vector_store = VectorStore(vector_store_type, embedding_model, dir.absolute())
        vector_store.load()
        tools.append(create_retriever_tool(
            vector_store.as_retriever(k),
            f"{dir.name}_retriever",
            f"this tool is used to retrieve informations about the topic {dir.name}"
        ))
    return tools