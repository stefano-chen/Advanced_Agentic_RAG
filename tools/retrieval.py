from indexing.vectorstore import VectorStore
from typing import List, Callable
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from pathlib import Path
from indexing.embedding import EmbeddingModel
from langchain_community.tools import DuckDuckGoSearchRun

def get_tools(vector_store_type: str, vector_store_dir: str, k: int, config: dict[str, str]) -> List[Tool]:
    tools = [DuckDuckGoSearchRun()]
    embedding_model = EmbeddingModel(config=config).get()
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