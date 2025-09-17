import json
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def load_document(type: str, data_path: str):
    if type == "pdf":
        loader = PyPDFLoader(data_path)
    return loader.load()

def apply_chunking(documents: List[Document], strategy: str, strategy_options: dict):
    if strategy == "window":
        text_splitter = RecursiveCharacterTextSplitter(
            **strategy_options
        )
    return text_splitter.split_documents(documents)

def create_vector_store(type: str, documents: List[Document], embedding_model: Embeddings, save_path: str):
    if type == "faiss":
        vector_store = FAISS.from_documents(documents, embedding_model)
        vector_store.save_local(save_path)
    

if __name__ == "__main__":
    
    with open("./config/populate_config.json", "r") as f:
        config = json.load(f)
    
    if config["embedding_provider"] == "huggingface":
        embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model"])
    
    data_dir_path = config["data_dir_path"]
    if not os.path.exists(data_dir_path):
        raise NotADirectoryError(f"{data_dir_path} is not a directory")
    
    for entry in os.scandir(data_dir_path):
        if entry.is_file():
            print(f"loading from {entry.path}")
            pages = load_document(type=entry.path.split(".")[-1], data_path=entry.path)
            print(f"{len(pages)} pages loaded")
            print(f"Apply chunking strategy {config["chunking_strategy"]}")
            chunks = apply_chunking(pages, config["chunking_strategy"], config["chunking_options"])
            print(f"{len(chunks)} chunks extracted")
            save_dir = os.path.join(config["save_dir_path"], entry.name.split(".")[0])
            print(f"saving vector store at {save_dir}\n")
            create_vector_store(config["vector_store_type"], chunks, embedding_model, save_dir)
