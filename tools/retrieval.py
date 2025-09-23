from indexing.vectorstore import VectorStore
from typing import List, Dict
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from pathlib import Path
from utils.embedding import EmbeddingModel
from langchain_community.tools import DuckDuckGoSearchRun

def get_tools(vector_store_type: str, vector_store_dir: str, k: int, config: Dict[str, str]) -> List[Tool]:
    """
    Generate a list of tools available to the AI Agent

    Parameters:
        vector_store (str): The type of vector store to use (e.g. faiss, chroma, etc.)
        vector_store_dir (str): The path in which the vector store is located
        k (int): a non negative integer reppresenting the number of chunks a tool must retrieve
        config (Dict[str, str]): the embedding model configuration file

    Returns:
        List[Tool]: A list of tools for the agent
    """
    # A web search tool
    tools = [DuckDuckGoSearchRun()]
    embedding_model = EmbeddingModel(config=config).get()
    store_dir = Path(vector_store_dir)
    # For each vectorstore/topic create a retriever tool
    for dir in store_dir.iterdir():
        vector_store = VectorStore(vector_store_type, embedding_model, dir.absolute())
        vector_store.load()
        tools.append(create_retriever_tool(
            vector_store.as_retriever(k),
            f"{dir.name}_retriever",
            f"this tool is used to retrieve informations about the topic {dir.name}"
        ))
    return tools